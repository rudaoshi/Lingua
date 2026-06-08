#!/usr/bin/env python3
"""Check Lingua relation-instance codec on current JSONL data."""

from __future__ import annotations

import argparse
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

from lingua.learn.wordgraph.modeler.lingua_relation_codec import (  # noqa: E402
    decode_lingua_relation_instances,
    encode_lingua_relation_instances,
    relation_instance_signature,
    relation_instances,
)
from lingua.structure.gpgraph import GPGraph  # noqa: E402
from lingua.utils.normalization import normalize_lingua_edges_to_predicate_nodes  # noqa: E402


DEFAULT_SPLITS = ("train", "dev", "test")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                yield line_number, json.loads(line)


def canonical_graph(graph_data: dict[str, Any]) -> tuple[tuple[Any, ...], Counter[tuple[Any, ...]]]:
    nodes = tuple(
        (
            json.dumps(node.get("spans"), ensure_ascii=False, sort_keys=True),
            node.get("type"),
            node.get("confidence"),
        )
        for node in graph_data["graph"]["nodes"]
    )
    edges = Counter(
        (
            int(edge["start"]),
            int(edge["end"]),
            edge["label"],
            edge.get("confidence"),
        )
        for edge in graph_data["graph"]["edges"]
    )
    return nodes, edges


def process_split(path: Path, split: str) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    relation_label_counts: Counter[str] = Counter()
    relation_role_shape_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []

    for line_number, row in read_jsonl(path):
        totals["examples"] += 1
        graph_data = row["lingua"]
        normalized, norm_stats = normalize_lingua_edges_to_predicate_nodes(graph_data)
        encoded, encode_stats = encode_lingua_relation_instances(graph_data)
        decoded = decode_lingua_relation_instances(encoded)

        # Confirm materialized JSON is parseable by the existing graph class.
        GPGraph.parse(decoded)

        relations = relation_instances(encoded)
        signatures = [relation_instance_signature(item) for item in relations]
        signature_counts = Counter(signatures)
        duplicate_instances = sum(count - 1 for count in signature_counts.values() if count > 1)

        totals["relation_instances"] += len(relations)
        totals["plain_edges"] += len(encoded.get("plain_edges", []))
        totals["duplicate_relation_instances"] += duplicate_instances
        for key, value in encode_stats.items():
            totals[f"encode_stats:{key}"] += int(value)
        for key, value in norm_stats.items():
            totals[f"norm_stats:{key}"] += int(value)

        for item in relations:
            relation_label_counts[item["label"]] += 1
            role_shape = ",".join(role["label"] for role in item.get("roles", []))
            relation_role_shape_counts[f"{item['label']}({role_shape})"] += 1

        if canonical_graph(decoded) != canonical_graph(normalized):
            totals["roundtrip_mismatches"] += 1
            if len(failures) < 20:
                failures.append(
                    {
                        "line": line_number,
                        "sentence": row.get("sentence", ""),
                    }
                )

    return {
        "split": split,
        "path": str(path),
        "totals": dict(totals),
        "relation_label_counts": dict(relation_label_counts),
        "top_relation_role_shapes": dict(relation_role_shape_counts.most_common(30)),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [
        process_split(args.data_dir / f"{split}.jsonl", split)
        for split in args.splits
    ]
    output = {"splits": summaries}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if args.fail_on_mismatch and any(
        split["totals"].get("roundtrip_mismatches", 0)
        for split in summaries
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
