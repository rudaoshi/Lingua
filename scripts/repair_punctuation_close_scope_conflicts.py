#!/usr/bin/env python3
"""Repair punctuation nodes with conflicting close-scope and attachment roles.

Bad pattern:

    parent --punctuation--> p/PunctuationalConstant
    p --as:punctuation--> child

If `p` closes `child`, the outer semantic edge should point at `p`, or the
plain punctuation edge should be removed when `p` already has a semantic host.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    ROOT / "data" / "train.jsonl",
    ROOT / "data" / "dev.jsonl",
    ROOT / "data" / "test.jsonl",
]
REPORT_DIR = ROOT / "reports" / "punctuation_close_scope_repair"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def node_type(sample: dict[str, Any], node_index: int) -> str:
    node = sample["lingua"]["graph"]["nodes"][node_index]
    if isinstance(node["spans"], str):
        return f"AUX:{node['spans']}"
    return node["type"]


def node_text(sample: dict[str, Any], node_index: int) -> str:
    node = sample["lingua"]["graph"]["nodes"][node_index]
    if isinstance(node["spans"], str):
        return node["spans"]
    words = sample["lingua"]["words"]
    return "".join(words[index] for start, end in node["spans"] for index in range(start, end + 1))


def is_punctuation_node(sample: dict[str, Any], node_index: int) -> bool:
    node = sample["lingua"]["graph"]["nodes"][node_index]
    return not isinstance(node["spans"], str) and node["type"] == "PunctuationalConstant"


def reaches(edges: list[dict[str, int | str]], start: int, target: int) -> bool:
    seen: set[int] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node == target:
            return True
        for edge in edges:
            if edge["start"] == node and edge["end"] not in seen:
                seen.add(edge["end"])
                stack.append(edge["end"])
    return False


def conflict_edges(sample: dict[str, Any]) -> list[tuple[int, int, dict[str, Any], int, dict[str, Any]]]:
    edges = sample["lingua"]["graph"]["edges"]
    incoming_punctuation: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    outgoing_close: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)

    for edge_index, edge in enumerate(edges):
        if edge["label"] == "punctuation":
            incoming_punctuation[edge["end"]].append((edge_index, edge))
        elif edge["label"] == "as:punctuation":
            outgoing_close[edge["start"]].append((edge_index, edge))

    conflicts = []
    for node_index in range(len(sample["lingua"]["graph"]["nodes"])):
        if not is_punctuation_node(sample, node_index):
            continue
        for parent_edge_index, parent_edge in incoming_punctuation.get(node_index, []):
            for close_edge_index, close_edge in outgoing_close.get(node_index, []):
                conflicts.append((node_index, parent_edge_index, parent_edge, close_edge_index, close_edge))
    return conflicts


def add_edge(edges: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    if not any(edge["start"] == start and edge["end"] == end and edge["label"] == label for edge in edges):
        edges.append({"start": start, "end": end, "label": label})


def semantic_incoming(edges: list[dict[str, Any]], node_index: int) -> list[dict[str, Any]]:
    return [edge for edge in edges if edge["end"] == node_index and edge["label"] != "punctuation"]


def apply_special(sample: dict[str, Any], split: str, line_no: int, punct_node: int, parent_edge_index: int) -> str | None:
    """Handle cases where the old graph's semantic edge targeted the wrong layer."""

    edges = sample["lingua"]["graph"]["edges"]

    if split == "train.jsonl" and line_no == 1452 and punct_node == 16:
        edges[parent_edge_index]["label"] = "as:modification"
        return "special_aged_modifier"

    if split == "train.jsonl" and line_no == 3700 and punct_node == 17:
        for edge in edges:
            if edge["start"] == 0 and edge["end"] == 18 and edge["label"] == "pred.arg.2":
                edge["end"] = 17
        del edges[parent_edge_index]
        return "special_quote_pred_arg"

    if split == "train.jsonl" and line_no == 3956 and punct_node == 10:
        for edge in edges:
            if edge["start"] == 16 and edge["end"] == 3 and edge["label"] == "variable":
                edge["end"] = 10
            elif edge["start"] == 3 and edge["end"] == 8 and edge["label"] == "body":
                edge["end"] = 9
            elif edge["start"] == 10 and edge["label"] == "as:punctuation":
                edge["end"] = 3
        del edges[parent_edge_index]
        return "special_quote_closed_pp"

    if split == "train.jsonl" and line_no == 4252 and punct_node == 30:
        for edge in edges:
            if edge["start"] == 34 and edge["end"] == 39 and edge["label"] == "pred.arg.2":
                edge["end"] = 30
        del edges[parent_edge_index]
        return "special_clause_final_pp"

    if split == "train.jsonl" and line_no == 6044 and punct_node == 19:
        for edge in edges:
            if edge["start"] == 18 and edge["end"] == 4 and edge["label"] == "pred.arg.2":
                edge["end"] = 19
        add_edge(edges, 17, 4, "as:punctuation")
        del edges[parent_edge_index]
        return "special_quote_period"

    if split == "train.jsonl" and line_no == 6124 and punct_node == 12:
        for edge in edges:
            if edge["start"] == 0 and edge["end"] == 14 and edge["label"] == "pred.arg.2":
                edge["end"] = 12
        add_edge(edges, 10, 14, "as:punctuation")
        del edges[parent_edge_index]
        return "special_quoted_question"

    return None


def repair_first_conflict(sample: dict[str, Any], split: str, line_no: int) -> str | None:
    edges = sample["lingua"]["graph"]["edges"]

    for punct_node, parent_edge_index, parent_edge, close_edge_index, close_edge in conflict_edges(sample):
        parent = parent_edge["start"]
        child = close_edge["end"]

        special = apply_special(sample, split, line_no, punct_node, parent_edge_index)
        if special:
            return special

        direct_edges = [
            edge
            for edge in edges
            if edge["start"] == parent and edge["end"] == child and edge["label"] != "punctuation"
        ]
        if direct_edges:
            for edge in direct_edges:
                edge["end"] = punct_node
            del edges[parent_edge_index]
            return "direct_semantic_edge_to_wrapper"

        if semantic_incoming(edges, punct_node):
            del edges[parent_edge_index]
            return "drop_redundant_punctuation_edge"

        ancestor_edges = [
            edge
            for edge in edges
            if (
                edge["start"] == parent
                and edge["end"] != punct_node
                and edge["label"] != "punctuation"
                and reaches(edges, edge["end"], child)
            )
        ]
        if len(ancestor_edges) == 1:
            old_target = ancestor_edges[0]["end"]
            ancestor_edges[0]["end"] = punct_node
            close_edge["end"] = old_target
            del edges[parent_edge_index]
            return "wrap_existing_descendant_target"

        reverse_edges = [
            edge
            for edge in edges
            if (
                edge["start"] == parent
                and edge["end"] != punct_node
                and edge["label"] != "punctuation"
                and reaches(edges, child, edge["end"])
            )
        ]
        if len(reverse_edges) == 1:
            reverse_edges[0]["end"] = punct_node
            del edges[parent_edge_index]
            return "retarget_edge_to_existing_wrapper"

        if node_type(sample, parent) == "ModificationalConstant":
            incoming = semantic_incoming(edges, parent)
            if incoming:
                for edge in incoming:
                    edge["end"] = punct_node
                add_edge(edges, punct_node, parent, "as:modification")
                del edges[parent_edge_index]
                return "hyphenated_modifier_wrapper"

    return None


def repair_sample(sample: dict[str, Any], split: str, line_no: int) -> list[str]:
    changes: list[str] = []
    while True:
        change = repair_first_conflict(sample, split, line_no)
        if change is None:
            return changes
        changes.append(change)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = Counter()
    changed_samples = []

    for path in DATASETS:
        rows = read_jsonl(path)
        for line_no, sample in enumerate(rows, start=1):
            before = len(conflict_edges(sample))
            changes = repair_sample(sample, path.name, line_no)
            after = len(conflict_edges(sample))
            if changes:
                summary.update(changes)
                changed_samples.append(
                    {
                        "split": path.name,
                        "line": line_no,
                        "sentence": sample["sentence"],
                        "before": before,
                        "after": after,
                        "changes": changes,
                    }
                )
        if args.apply:
            write_jsonl(path, rows)

    report = {
        "applied": args.apply,
        "changed_samples": len(changed_samples),
        "changed_edges": sum(summary.values()),
        "summary": dict(summary),
        "remaining_conflicts": sum(
            len(conflict_edges(sample))
            for path in DATASETS
            for sample in (read_jsonl(path) if args.apply else [])
        )
        if args.apply
        else None,
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (REPORT_DIR / "changes.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in changed_samples) + ("\n" if changed_samples else ""),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
