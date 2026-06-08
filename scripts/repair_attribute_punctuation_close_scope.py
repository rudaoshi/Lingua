#!/usr/bin/env python3
"""Repair AttributePredicator delimiter punctuation scope.

This pass handles only:

    AttributePredicator --punctuation--> PunctuationalConstant

For lexical attribute predicates such as ``Email: value``, the delimiter closes
the lexical predicate node. For virtual ``Attribute`` relation nodes such as
``A - B``, the delimiter closes ``pred.arg.1``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = [
    ROOT / "data" / "train.jsonl",
    ROOT / "data" / "dev.jsonl",
    ROOT / "data" / "test.jsonl",
]


def span_range(node: dict[str, Any]) -> tuple[int, int] | None:
    spans = node.get("spans")
    if not isinstance(spans, list):
        return None
    positions = [i for start, end in spans for i in range(start, end + 1)]
    if not positions:
        return None
    return min(positions), max(positions)


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return edge["start"], edge["end"], edge["label"]


def add_edge(edges: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    new_edge = {"start": start, "end": end, "label": label}
    if edge_key(new_edge) not in {edge_key(edge) for edge in edges}:
        edges.append(new_edge)


def remove_first_edge(
    edges: list[dict[str, Any]], start: int, end: int, label: str
) -> bool:
    for index, edge in enumerate(edges):
        if (
            edge["start"] == start
            and edge["end"] == end
            and edge["label"] == label
        ):
            del edges[index]
            return True
    return False


def retarget_attribute_arg1(
    edges: list[dict[str, Any]], source: int, punct: int
) -> bool:
    arg1_edges = [
        edge
        for edge in edges
        if edge["start"] == source and edge["label"] == "pred.arg.1"
    ]
    if len(arg1_edges) != 1:
        return False
    old_arg1 = arg1_edges[0]["end"]
    remove_first_edge(edges, source, punct, "punctuation")
    arg1_edges[0]["end"] = punct
    add_edge(edges, punct, old_arg1, "as:punctuation")
    return True


def wrap_attribute_source(
    edges: list[dict[str, Any]], source: int, punct: int
) -> bool:
    if not remove_first_edge(edges, source, punct, "punctuation"):
        return False
    for edge in edges:
        if edge["end"] == source and edge["start"] != punct:
            edge["end"] = punct
    add_edge(edges, punct, source, "as:punctuation")
    return True


def repair_sample(sample: dict[str, Any]) -> list[str]:
    graph = sample["lingua"]["graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    actions: list[tuple[int, int, str]] = []

    for edge in list(edges):
        if edge["label"] != "punctuation":
            continue
        source = edge["start"]
        punct = edge["end"]
        if not (0 <= source < len(nodes) and 0 <= punct < len(nodes)):
            continue
        if nodes[source]["type"] != "AttributePredicator":
            continue
        if nodes[punct]["type"] != "PunctuationalConstant":
            continue

        source_range = span_range(nodes[source])
        punct_range = span_range(nodes[punct])
        if source_range is None:
            actions.append((source, punct, "arg1"))
        elif (
            punct_range is not None
            and punct_range[1] < source_range[0]
            and any(
                e["start"] == source and e["label"] == "pred.arg.1"
                for e in edges
            )
        ):
            actions.append((source, punct, "arg1"))
        else:
            actions.append((source, punct, "source"))

    changes: list[str] = []
    for source, punct, mode in actions:
        if mode == "arg1":
            changed = retarget_attribute_arg1(edges, source, punct)
        else:
            changed = wrap_attribute_source(edges, source, punct)
        if changed:
            changes.append(f"{mode}:{source}->{punct}")
    return changes


def main() -> None:
    total = 0
    by_file: dict[str, int] = {}
    for path in DATA_FILES:
        samples: list[dict[str, Any]] = []
        file_total = 0
        for line in path.read_text().splitlines():
            sample = json.loads(line)
            changes = repair_sample(sample)
            file_total += len(changes)
            samples.append(sample)
        if file_total:
            path.write_text(
                "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples)
                + "\n"
            )
        by_file[path.name] = file_total
        total += file_total

    print(f"repaired {total} AttributePredicator punctuation edges")
    for name, count in by_file.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
