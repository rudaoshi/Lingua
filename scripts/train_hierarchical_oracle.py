#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "hierarchical-token-set-baseline"
DEFAULT_UD_UPOS_DIR = DEFAULT_DATA_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lingua.learn.hierarchical.dataset import (
    HierarchicalOracleDataset,
    build_hierarchical_examples,
    build_vocab,
    collate_hierarchical_batch,
    load_ud_upos_by_line,
)
from lingua.learn.hierarchical.model import HierarchicalTokenSetParser
from lingua.learn.hierarchical.training import (
    evaluate_hierarchical_model,
    save_metrics,
    train_one_epoch,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a recursive token-set Lingua oracle model."
    )
    parser.add_argument("--train_path", type=Path, default=DEFAULT_DATA_DIR / "train.jsonl")
    parser.add_argument("--dev_path", type=Path, default=DEFAULT_DATA_DIR / "dev.jsonl")
    parser.add_argument("--test_path", type=Path, default=DEFAULT_DATA_DIR / "test.jsonl")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ud_upos_dir", type=Path, default=DEFAULT_UD_UPOS_DIR)
    parser.add_argument("--disable_ud_upos", action="store_true")
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_dev_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--max_train_units", type=int)
    parser.add_argument("--max_dev_units", type=int)
    parser.add_argument("--max_test_units", type=int)
    parser.add_argument("--min_word_freq", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--encoder_type", choices=("gru", "bert"), default="bert")
    parser.add_argument("--bert_model", default="bert-base-cased")
    parser.add_argument("--bert_lr", type=float, default=2e-5)
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--bert_local_files_only", action="store_true")
    parser.add_argument("--bert_gradient_checkpointing", action="store_true")
    parser.add_argument("--max_subword_length", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--head_pos_weight", "--unit_pos_weight", type=float, default=1.0)
    parser.add_argument("--head_aux_kind_weight", "--aux_kind_weight", type=float, default=1.0)
    parser.add_argument("--decision_weight", type=float, default=0.5)
    parser.add_argument("--role_weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--decode_strategy",
        choices=("two_pass", "exhaustive", "independent"),
        default="two_pass",
        help="Structured decoding strategy for evaluation.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Building hierarchical token-set examples...")
    ud_upos_by_split = load_ud_upos_maps(args)
    train_examples = build_hierarchical_examples(
        args.train_path,
        max_samples=args.max_train_samples,
        max_units=args.max_train_units,
        ud_upos_by_line=ud_upos_by_split.get("train"),
    )
    dev_examples = build_hierarchical_examples(
        args.dev_path,
        max_samples=args.max_dev_samples,
        max_units=args.max_dev_units,
        ud_upos_by_line=ud_upos_by_split.get("dev"),
    )
    test_examples = build_hierarchical_examples(
        args.test_path,
        max_samples=args.max_test_samples,
        max_units=args.max_test_units,
        ud_upos_by_line=ud_upos_by_split.get("test"),
    )
    if not train_examples:
        raise ValueError("No training examples were built")

    vocab = build_vocab(train_examples, min_word_freq=args.min_word_freq)
    vocab.save(args.output_dir / "vocab.json")
    save_run_config(args.output_dir / "config.json", args, vocab, train_examples, dev_examples, test_examples)

    tokenizer = None
    if args.encoder_type == "bert":
        from transformers import AutoTokenizer

        print(f"Loading tokenizer from {args.bert_model}...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.bert_model,
            use_fast=True,
            local_files_only=args.bert_local_files_only,
        )
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("hierarchical BERT encoder requires a fast tokenizer")
        tokenizer.save_pretrained(args.output_dir / "tokenizer")

    train_dataset = HierarchicalOracleDataset(train_examples, vocab)
    dev_dataset = HierarchicalOracleDataset(dev_examples, vocab)
    test_dataset = HierarchicalOracleDataset(test_examples, vocab)
    collate_fn = partial(
        collate_hierarchical_batch,
        pad_id=vocab.pad_id,
        upos_pad_id=vocab.upos_pad_id,
        tokenizer=tokenizer,
        max_subword_length=args.max_subword_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = HierarchicalTokenSetParser(
        vocab_size=len(vocab.word2id),
        num_unit_pos=len(vocab.unit_pos2id),
        num_aux_kinds=len(vocab.aux_kind2id),
        num_role_labels=len(vocab.role_label2id),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        pad_id=vocab.pad_id,
        num_decisions=len(vocab.decision2id),
        num_view_kinds=len(vocab.view_kind2id),
        num_upos=len(vocab.upos2id),
        use_crf=True,
        role_label_names=labels_by_id(vocab.role_label2id),
        unit_pos_names=labels_by_id(vocab.unit_pos2id),
        aux_kind_names=labels_by_id(vocab.aux_kind2id),
        decision_names=labels_by_id(vocab.decision2id),
        encoder_type=args.encoder_type,
        bert_model_name=args.bert_model if args.encoder_type == "bert" else None,
        freeze_bert=args.freeze_bert,
        bert_local_files_only=args.bert_local_files_only,
        bert_gradient_checkpointing=args.bert_gradient_checkpointing,
    ).to(device)
    optimizer = build_optimizer(model, args)

    best_dev_role_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            head_pos_weight=args.head_pos_weight,
            head_aux_kind_weight=args.head_aux_kind_weight,
            decision_weight=args.decision_weight,
            role_weight=args.role_weight,
        )
        dev_metrics = evaluate_hierarchical_model(
            model,
            dev_loader,
            device,
            threshold=args.threshold,
            decode_strategy=args.decode_strategy,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "dev": dev_metrics,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        if dev_metrics["role_f1"] > best_dev_role_f1:
            best_dev_role_f1 = dev_metrics["role_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab.to_dict(),
                    "config": {**serialize_args(args), "use_crf": True},
                    "dev_metrics": dev_metrics,
                },
                args.output_dir / "best.pt",
            )

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_hierarchical_model(
        model,
        test_loader,
        device,
        threshold=args.threshold,
        decode_strategy=args.decode_strategy,
    )
    metrics = {
        "history": history,
        "best_dev_role_f1": best_dev_role_f1,
        "test": test_metrics,
    }
    save_metrics(args.output_dir / "metrics.json", metrics)
    print(json.dumps({"test": test_metrics}, ensure_ascii=False))
    return 0


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.encoder_type != "bert":
        return torch.optim.AdamW(model.parameters(), lr=args.lr)

    bert_parameters = []
    task_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("bert."):
            bert_parameters.append(parameter)
        else:
            task_parameters.append(parameter)

    parameter_groups = []
    if bert_parameters:
        parameter_groups.append({"params": bert_parameters, "lr": args.bert_lr})
    if task_parameters:
        parameter_groups.append({"params": task_parameters, "lr": args.lr})
    return torch.optim.AdamW(parameter_groups)


def load_ud_upos_maps(args: argparse.Namespace) -> dict[str, dict[int, tuple[str, ...]]]:
    if args.disable_ud_upos:
        return {}
    if not args.ud_upos_dir.exists():
        raise FileNotFoundError(
            f"UD UPOS cache directory not found: {args.ud_upos_dir}. "
            "Pass --disable_ud_upos to train without UD features."
        )

    maps = {}
    for split in ("train", "dev", "test"):
        path = ud_upos_path(args.ud_upos_dir, split)
        if not path.exists():
            raise FileNotFoundError(
                f"UD UPOS cache file not found: {path}. "
                "Pass --disable_ud_upos to train without UD features."
            )
        maps[split] = load_ud_upos_by_line(path)
    return maps


def ud_upos_path(base_dir: Path, split: str) -> Path:
    preferred = base_dir / f"{split}_ud.jsonl"
    if preferred.exists():
        return preferred
    return base_dir / f"{split}.jsonl"


def save_run_config(
    path: Path,
    args: argparse.Namespace,
    vocab,
    train_examples,
    dev_examples,
    test_examples,
) -> None:
    config = {
        "args": {
            key: value
            for key, value in serialize_args(args).items()
        },
        "num_train_units": len(train_examples),
        "num_dev_units": len(dev_examples),
        "num_test_units": len(test_examples),
        "vocab_size": len(vocab.word2id),
        "num_view_kinds": len(vocab.view_kind2id),
        "num_upos": len(vocab.upos2id),
        "num_unit_pos": len(vocab.unit_pos2id),
        "num_aux_kinds": len(vocab.aux_kind2id),
        "num_role_labels": len(vocab.role_label2id),
        "use_crf": True,
        "encoder_type": args.encoder_type,
        "bert_model": args.bert_model if args.encoder_type == "bert" else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def serialize_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def labels_by_id(label2id: dict[str, int]) -> list[str]:
    return [
        label
        for label, _index in sorted(label2id.items(), key=lambda item: item[1])
    ]


if __name__ == "__main__":
    raise SystemExit(main())
