#!/usr/bin/env python3
"""Repair manually reviewed topic/header dash residuals.

These are sports/news header fragments such as `SOCCER - ...`.  The current
standard treats the first field as discourse metadata closed by the dash, not
as an ordinary List item.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def edge(start: int, end: int, label: str) -> dict[str, Any]:
    return {"start": start, "end": end, "label": label}


def node(spans: Any, node_type: str) -> dict[str, Any]:
    return {"spans": spans, "type": node_type}


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def remove_edges(edges: list[dict[str, Any]], *, start: int | None = None, end: int | None = None, label: str | None = None) -> None:
    edges[:] = [
        item
        for item in edges
        if not (
            (start is None or item["start"] == start)
            and (end is None or item["end"] == end)
            and (label is None or item["label"] == label)
        )
    ]


def add_edge_once(edges: list[dict[str, Any]], item: dict[str, Any]) -> None:
    if item not in edges:
        edges.append(item)


def convert_dash_list_to_discourse(sample: dict[str, Any], *, dash: int, topic: int, content: int) -> None:
    g = graph(sample)
    nodes = g["nodes"]
    edges = g["edges"]
    if nodes[dash]["type"] != "ListFunctor":
        raise ValueError(f"expected dash node {dash} to be ListFunctor")

    discourse = len(nodes)
    nodes[dash]["type"] = "PunctuationalConstant"
    nodes.append(node("Discourse", "ModificationalFunctor"))

    remove_edges(edges, start=dash, label="func.arg")
    for item in edges:
        if item["end"] == dash:
            item["end"] = discourse

    add_edge_once(edges, edge(dash, topic, "as:punctuation"))
    add_edge_once(edges, edge(discourse, content, "variable"))
    add_edge_once(edges, edge(discourse, dash, "body"))


def fix_train_212(sample: dict[str, Any]) -> None:
    convert_dash_list_to_discourse(sample, dash=1, topic=0, content=2)


def fix_train_1254(sample: dict[str, Any]) -> None:
    convert_dash_list_to_discourse(sample, dash=4, topic=0, content=2)


def fix_train_3778(sample: dict[str, Any]) -> None:
    convert_dash_list_to_discourse(sample, dash=6, topic=4, content=7)


def fix_train_4984(sample: dict[str, Any]) -> None:
    convert_dash_list_to_discourse(sample, dash=10, topic=6, content=0)


def fix_train_5278(sample: dict[str, Any]) -> None:
    g = graph(sample)
    nodes = g["nodes"]
    edges = g["edges"]
    modified_results = len(nodes)
    nodes.append(node("Modification", "ModificationalFunctor"))
    remove_edges(edges, start=4, end=2, label="func.arg")
    remove_edges(edges, start=5, end=6, label="func.arg")
    add_edge_once(edges, edge(5, modified_results, "func.arg"))
    add_edge_once(edges, edge(modified_results, 6, "variable"))
    add_edge_once(edges, edge(modified_results, 2, "body"))
    convert_dash_list_to_discourse(sample, dash=4, topic=0, content=5)


def fix_train_10382(sample: dict[str, Any]) -> None:
    convert_dash_list_to_discourse(sample, dash=5, topic=0, content=7)


def fix_test_399(sample: dict[str, Any]) -> None:
    convert_dash_list_to_discourse(sample, dash=4, topic=0, content=2)


FIXES = {
    "train": {
        212: fix_train_212,
        1254: fix_train_1254,
        3778: fix_train_3778,
        4984: fix_train_4984,
        5278: fix_train_5278,
        10382: fix_train_10382,
    },
    "test": {
        399: fix_test_399,
    },
}


def main() -> None:
    changed = []
    for split, fixes in FIXES.items():
        path = DATA / f"{split}.jsonl"
        rows = read_jsonl(path)
        for idx, fix in fixes.items():
            before = json.dumps(rows[idx]["lingua"]["graph"], ensure_ascii=False, sort_keys=True)
            fix(rows[idx])
            after = json.dumps(rows[idx]["lingua"]["graph"], ensure_ascii=False, sort_keys=True)
            if before == after:
                raise ValueError(f"{split}:{idx} was not changed")
            changed.append(f"{split}:{idx}")
        write_jsonl(path, rows)
    print("changed", len(changed), ", ".join(changed))


if __name__ == "__main__":
    main()
