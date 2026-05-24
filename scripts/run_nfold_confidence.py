#!/usr/bin/env python3
"""Run n-fold held-out confidence estimation for the Lingua wordgraph parser."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lingua.learn.wordgraph.confidence.merge import merge_fold_results
from lingua.learn.wordgraph.confidence.splits import fold_specs_to_json, prepare_nfold_splits


DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "wordgraph-nfold-confidence"


def run_command(cmd: List[str], *, cwd: Path, log_path: Optional[Path] = None) -> None:
    if log_path is None:
        subprocess.run(cmd, cwd=str(cwd), check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train n wordgraph models and score held-out sample confidence"
    )
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--dev_split", default="dev")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--bert_model", default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--arc_hidden_size", type=int, default=500)
    parser.add_argument("--rel_hidden_size", type=int, default=100)
    parser.add_argument("--node_hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--word_pooling", default="mean", choices=["mean", "first", "last", "max"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--arc_weight", type=float, default=1.0)
    parser.add_argument("--rel_weight", type=float, default=1.0)
    parser.add_argument("--cow_weight", type=float, default=2.0)
    parser.add_argument("--node_type_weight", type=float, default=1.0)
    parser.add_argument("--is_root_weight", type=float, default=1.0)
    parser.add_argument("--device", default=None, help="Device for confidence scoring/eval")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_scoring", action="store_true")
    parser.add_argument("--fail_on_score_error", action="store_true")
    parser.add_argument("--low_confidence_limit", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    folds_dir = output_dir / "folds"
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = prepare_nfold_splits(
        data_dir=data_dir,
        output_dir=output_dir,
        source_split=args.source_split,
        dev_split=args.dev_split,
        n_folds=args.n_folds,
        random_seed=args.random_seed,
        shuffle=not args.no_shuffle,
    )

    config = {
        "data_dir": str(data_dir),
        "source_split": args.source_split,
        "dev_split": args.dev_split,
        "output_dir": str(output_dir),
        "n_folds": args.n_folds,
        "random_seed": args.random_seed,
        "shuffle": not args.no_shuffle,
        "folds": fold_specs_to_json(specs),
        "training": {
            "bert_model": args.bert_model,
            "max_length": args.max_length,
            "arc_hidden_size": args.arc_hidden_size,
            "rel_hidden_size": args.rel_hidden_size,
            "node_hidden_size": args.node_hidden_size,
            "dropout": args.dropout,
            "word_pooling": args.word_pooling,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    train_script = REPO_ROOT / "scripts" / "train_wordgraph.py"
    eval_script = REPO_ROOT / "scripts" / "eval_wordgraph.py"
    score_script = REPO_ROOT / "scripts" / "score_wordgraph_confidence.py"

    for spec in specs:
        model_dir = spec.fold_dir / "model"
        evaluation_dir = spec.fold_dir / "evaluation"
        confidence_dir = spec.fold_dir / "confidence"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        confidence_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            str(train_script),
            "--data_dir",
            str(spec.data_dir),
            "--output_dir",
            str(model_dir),
            "--bert_model",
            args.bert_model,
            "--max_length",
            str(args.max_length),
            "--arc_hidden_size",
            str(args.arc_hidden_size),
            "--rel_hidden_size",
            str(args.rel_hidden_size),
            "--node_hidden_size",
            str(args.node_hidden_size),
            "--dropout",
            str(args.dropout),
            "--word_pooling",
            args.word_pooling,
            "--batch_size",
            str(args.batch_size),
            "--num_epochs",
            str(args.num_epochs),
            "--learning_rate",
            str(args.learning_rate),
            "--weight_decay",
            str(args.weight_decay),
            "--arc_weight",
            str(args.arc_weight),
            "--rel_weight",
            str(args.rel_weight),
            "--cow_weight",
            str(args.cow_weight),
            "--node_type_weight",
            str(args.node_type_weight),
            "--is_root_weight",
            str(args.is_root_weight),
        ]

        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--model_dir",
            str(model_dir),
            "--data_path",
            str(spec.test_path),
            "--output_path",
            str(evaluation_dir / "metrics.json"),
            "--max_length",
            str(args.max_length),
            "--batch_size",
            str(args.eval_batch_size),
        ]
        if args.device:
            eval_cmd.extend(["--device", args.device])

        score_cmd = [
            sys.executable,
            str(score_script),
            "--model_dir",
            str(model_dir),
            "--data_path",
            str(spec.test_path),
            "--manifest_path",
            str(spec.manifest_path),
            "--output_path",
            str(confidence_dir / "detailed_results.jsonl"),
            "--summary_path",
            str(confidence_dir / "summary.json"),
            "--max_length",
            str(args.max_length),
            "--arc_weight",
            str(args.arc_weight),
            "--rel_weight",
            str(args.rel_weight),
            "--cow_weight",
            str(args.cow_weight),
            "--node_type_weight",
            str(args.node_type_weight),
            "--is_root_weight",
            str(args.is_root_weight),
        ]
        if args.device:
            score_cmd.extend(["--device", args.device])
        if args.fail_on_score_error:
            score_cmd.append("--fail_on_error")

        if args.dry_run:
            print(" ".join(train_cmd))
            print(" ".join(eval_cmd))
            print(" ".join(score_cmd))
            continue

        if not args.skip_training:
            run_command(train_cmd, cwd=REPO_ROOT, log_path=spec.fold_dir / "train.log")
        if not args.skip_eval:
            run_command(eval_cmd, cwd=REPO_ROOT, log_path=evaluation_dir / "eval.log")
        if not args.skip_scoring:
            run_command(score_cmd, cwd=REPO_ROOT, log_path=confidence_dir / "score.log")

    if not args.dry_run and not args.skip_scoring:
        merge_fold_results(
            folds_dir=folds_dir,
            output_jsonl=output_dir / f"confidence_{args.source_split}.jsonl",
            output_csv=output_dir / f"confidence_{args.source_split}.csv",
            summary_path=output_dir / "summary.json",
            low_confidence_path=output_dir / f"low_confidence_{args.source_split}.jsonl",
            low_confidence_limit=args.low_confidence_limit,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
