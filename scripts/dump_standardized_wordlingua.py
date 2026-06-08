#!/usr/bin/env python3
"""Dump WordLingua graphs from standardized Lingua using the original converter."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lingua.learn.wordgraph.modeler.gp2word import standardized_lingua2wordlingua  # noqa: E402


DEFAULT_SPLITS = ("train", "dev", "test")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                yield line_number, json.loads(line)


def root_count(graph) -> int:
    return sum(1 for node in graph.nodes() if graph.g.in_degree(node.ID) == 0)


def process_split(input_path: Path, output_path: Path, split: str) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    root_counts: Counter[int] = Counter()
    failures: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as out:
        for line_number, row in read_jsonl(input_path):
            totals["examples"] += 1
            try:
                word_graph, _, _ = standardized_lingua2wordlingua(row["lingua"])
            except Exception as exc:  # noqa: BLE001
                totals["failures"] += 1
                if len(failures) < 50:
                    failures.append(
                        {
                            "line": line_number,
                            "sentence": row.get("sentence", ""),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                continue

            totals["converted"] += 1
            if not nx.is_directed_acyclic_graph(word_graph.g):
                totals["cyclic_word_graphs"] += 1
            root_counts[root_count(word_graph)] += 1
            for _, edge, _ in word_graph.edges():
                edge_counts[edge.label] += 1

            out.write(
                json.dumps(
                    {
                        "split": split,
                        "line": line_number,
                        "sentence": row.get("sentence", ""),
                        "words": row["lingua"]["words"],
                        "word_lingua": word_graph.data(),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    return {
        "split": split,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "totals": dict(totals),
        "edge_vocab_size": len(edge_counts),
        "edge_label_counts": dict(edge_counts.most_common()),
        "root_counts": dict(sorted(root_counts.items())),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "standardized_lingua_original_wordlingua",
    )
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for split in args.splits:
        summaries.append(
            process_split(
                args.data_dir / f"{split}.jsonl",
                args.output_dir / f"{split}.wordlingua.jsonl",
                split,
            )
        )

    summary = {"splits": summaries}
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(not item["totals"].get("failures") for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
