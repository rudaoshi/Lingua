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
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lingua.learn.hierarchical.dataset import (
    HierarchicalOracleDataset,
    build_hierarchical_examples,
    build_vocab,
    collate_hierarchical_batch,
)
from lingua.learn.hierarchical.model import HierarchicalTokenSetParser
from lingua.learn.hierarchical.training import (
    evaluate_hierarchical_model,
    save_metrics,
    train_one_epoch,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a lightweight recursive token-set Lingua oracle baseline."
    )
    parser.add_argument("--train_path", type=Path, default=DEFAULT_DATA_DIR / "train.jsonl")
    parser.add_argument("--dev_path", type=Path, default=DEFAULT_DATA_DIR / "dev.jsonl")
    parser.add_argument("--test_path", type=Path, default=DEFAULT_DATA_DIR / "test.jsonl")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_dev_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--max_train_units", type=int)
    parser.add_argument("--max_dev_units", type=int)
    parser.add_argument("--max_test_units", type=int)
    parser.add_argument("--min_word_freq", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unit_pos_weight", type=float, default=1.0)
    parser.add_argument("--aux_kind_weight", type=float, default=1.0)
    parser.add_argument("--decision_weight", type=float, default=0.5)
    parser.add_argument("--role_weight", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Building hierarchical token-set examples...")
    train_examples = build_hierarchical_examples(
        args.train_path,
        max_samples=args.max_train_samples,
        max_units=args.max_train_units,
    )
    dev_examples = build_hierarchical_examples(
        args.dev_path,
        max_samples=args.max_dev_samples,
        max_units=args.max_dev_units,
    )
    test_examples = build_hierarchical_examples(
        args.test_path,
        max_samples=args.max_test_samples,
        max_units=args.max_test_units,
    )
    if not train_examples:
        raise ValueError("No training examples were built")

    vocab = build_vocab(train_examples, min_word_freq=args.min_word_freq)
    vocab.save(args.output_dir / "vocab.json")
    save_run_config(args.output_dir / "config.json", args, vocab, train_examples, dev_examples, test_examples)

    train_dataset = HierarchicalOracleDataset(train_examples, vocab)
    dev_dataset = HierarchicalOracleDataset(dev_examples, vocab)
    test_dataset = HierarchicalOracleDataset(test_examples, vocab)
    collate_fn = partial(collate_hierarchical_batch, pad_id=vocab.pad_id)

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
        num_query_roles=len(vocab.query_role2id),
        num_shared_roles=len(vocab.shared_role2id),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_dev_role_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            unit_pos_weight=args.unit_pos_weight,
            aux_kind_weight=args.aux_kind_weight,
            decision_weight=args.decision_weight,
            role_weight=args.role_weight,
        )
        dev_metrics = evaluate_hierarchical_model(
            model,
            dev_loader,
            device,
            threshold=args.threshold,
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
                    "config": serialize_args(args),
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
        "num_query_roles": len(vocab.query_role2id),
        "num_shared_roles": len(vocab.shared_role2id),
        "num_unit_pos": len(vocab.unit_pos2id),
        "num_aux_kinds": len(vocab.aux_kind2id),
        "num_role_labels": len(vocab.role_label2id),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)


def serialize_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


if __name__ == "__main__":
    raise SystemExit(main())
