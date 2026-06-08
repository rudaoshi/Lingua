#!/usr/bin/env python3
"""Apply simple leaf Modification normalization to gold Lingua JSONL data."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lingua.utils.normalization import normalize_simple_leaf_modifications  # noqa: E402


DEFAULT_SPLITS = ("train", "dev", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def graph_counts(graph_data: dict[str, Any]) -> dict[str, int]:
    nodes = graph_data["graph"]["nodes"]
    edges = graph_data["graph"]["edges"]
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "modification_aux": sum(1 for node in nodes if node.get("spans") == "Modification"),
        "as_modification_edges": sum(1 for edge in edges if edge.get("label") == "as:modification"),
    }


def process_split(path: Path, split: str, *, apply: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(path)
    output_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        before_lingua = row["lingua"]
        normalized_lingua, stats = normalize_simple_leaf_modifications(before_lingua)
        flattened = stats.get("flattened_simple_leaf_modifications", 0)
        totals["flattened_simple_leaf_modifications"] += flattened

        if normalized_lingua != before_lingua:
            before_counts = graph_counts(before_lingua)
            after_counts = graph_counts(normalized_lingua)
            changes.append(
                {
                    "split": split,
                    "idx": idx,
                    "line": idx + 1,
                    "sentence": row.get("sentence", ""),
                    "flattened_simple_leaf_modifications": flattened,
                    "before": before_counts,
                    "after": after_counts,
                }
            )
            new_row = copy.deepcopy(row)
            new_row["lingua"] = normalized_lingua
            output_rows.append(new_row)
        else:
            output_rows.append(row)

    summary = {
        "split": split,
        "path": str(path),
        "total_examples": len(rows),
        "changed_samples": len(changes),
        "flattened_simple_leaf_modifications": totals["flattened_simple_leaf_modifications"],
        "apply": apply,
    }

    if apply and changes:
        write_jsonl(path, output_rows)

    return changes, summary


def write_report(report_dir: Path, summaries: list[dict[str, Any]], changes: list[dict[str, Any]], *, apply: bool) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "apply": apply,
        "total_changed_samples": sum(item["changed_samples"] for item in summaries),
        "total_flattened_simple_leaf_modifications": sum(
            item["flattened_simple_leaf_modifications"] for item in summaries
        ),
        "splits": summaries,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (report_dir / "changes.jsonl").write_text(
        "".join(json.dumps(change, ensure_ascii=False, separators=(",", ":")) + "\n" for change in changes),
        encoding="utf-8",
    )

    lines = [
        "# Simple leaf Modification normalization",
        "",
        f"Apply: `{apply}`",
        "",
        "| split | total | changed samples | flattened Modification nodes |",
        "|---|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['split']} | {item['total_examples']} | {item['changed_samples']} | "
            f"{item['flattened_simple_leaf_modifications']} |"
        )
    lines.extend(
        [
            f"| **total** | {sum(item['total_examples'] for item in summaries)} | "
            f"{summary['total_changed_samples']} | {summary['total_flattened_simple_leaf_modifications']} |",
            "",
            "## First changed samples",
            "",
            "| split | line | flattened | sentence |",
            "|---|---:|---:|---|",
        ]
    )
    for change in changes[:50]:
        sentence = change["sentence"].replace("|", "\\|")
        lines.append(
            f"| {change['split']} | {change['line']} | "
            f"{change['flattened_simple_leaf_modifications']} | {sentence} |"
        )
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--report_dir", type=Path, default=REPO_ROOT / "reports" / "simple_leaf_modification_normalization")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_changes: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for split in args.splits:
        path = args.data_dir / f"{split}.jsonl"
        changes, summary = process_split(path, split, apply=args.apply)
        all_changes.extend(changes)
        summaries.append(summary)

    write_report(args.report_dir, summaries, all_changes, apply=args.apply)
    print(
        json.dumps(
            {
                "apply": args.apply,
                "total_changed_samples": sum(item["changed_samples"] for item in summaries),
                "total_flattened_simple_leaf_modifications": sum(
                    item["flattened_simple_leaf_modifications"] for item in summaries
                ),
                "splits": summaries,
                "report_dir": str(args.report_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
