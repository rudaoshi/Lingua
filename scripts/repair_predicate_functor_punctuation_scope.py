#!/usr/bin/env python3
"""Repair high-confidence predicate/functor punctuation close scope."""

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

SOURCE_TYPES = {
    "AppositionalPredicator",
    "ConjunctionalFunctor",
    "FactualPredicator",
    "GeneralFunctor",
    "InterjectionConstant",
    "LogicalPredicator",
    "ModificationalFunctor",
    "ReferentialPredicator",
}


def node_tokens(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if not isinstance(spans, list):
        return []
    return [i for start, end in spans for i in range(start, end + 1)]


def token_range(tokens: list[int]) -> tuple[int, int] | None:
    if not tokens:
        return None
    return min(tokens), max(tokens)


def descendant_tokens(sample: dict[str, Any], node_index: int) -> list[int]:
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    seen: set[int] = set()
    out: list[int] = []

    def visit(index: int) -> None:
        if index in seen:
            return
        seen.add(index)
        out.extend(node_tokens(nodes[index]))
        for edge in edges:
            if edge["start"] == index and edge["label"] != "punctuation":
                visit(edge["end"])

    visit(node_index)
    return sorted(set(out))


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return edge["start"], edge["end"], edge["label"]


def add_edge(edges: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    new_edge = {"start": start, "end": end, "label": label}
    if edge_key(new_edge) not in {edge_key(edge) for edge in edges}:
        edges.append(new_edge)


def remove_first_edge(edges: list[dict[str, Any]], start: int, end: int, label: str) -> bool:
    for index, edge in enumerate(edges):
        if edge["start"] == start and edge["end"] == end and edge["label"] == label:
            del edges[index]
            return True
    return False


def dedupe_edges(edges: list[dict[str, Any]]) -> None:
    seen: set[tuple[int, int, str]] = set()
    out: list[dict[str, Any]] = []
    for edge in edges:
        key = edge_key(edge)
        if edge["start"] == edge["end"] or key in seen:
            continue
        seen.add(key)
        out.append({k: edge[k] for k in edge if k in {"start", "end", "label"}})
    edges[:] = out


def should_wrap(sample: dict[str, Any], source: int, punct: int) -> tuple[bool, int]:
    nodes = sample["lingua"]["graph"]["nodes"]
    source_node_range = token_range(node_tokens(nodes[source]))
    source_desc_range = token_range(descendant_tokens(sample, source))
    punct_range = token_range(node_tokens(nodes[punct]))
    if punct_range is None:
        return False, 9

    if source_desc_range is not None and punct_range[0] <= source_desc_range[0] and punct_range[1] >= source_desc_range[1]:
        return True, 0
    if source_node_range is not None and punct_range[0] <= source_node_range[0] and punct_range[1] >= source_node_range[1]:
        return True, 0
    if source_desc_range is not None and punct_range[0] == source_desc_range[1] + 1:
        return True, 1
    if source_node_range is not None and punct_range[0] == source_node_range[1] + 1:
        return True, 2
    return False, 9


def wrap_source(sample: dict[str, Any], source: int, punct: int) -> bool:
    edges = sample["lingua"]["graph"]["edges"]
    if not remove_first_edge(edges, source, punct, "punctuation"):
        return False
    for edge in edges:
        if edge["end"] == source and edge["start"] != punct:
            edge["end"] = punct
    add_edge(edges, punct, source, "as:punctuation")
    dedupe_edges(edges)
    return True


def repair_sample(sample: dict[str, Any]) -> list[str]:
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    candidates: list[tuple[int, int, int]] = []
    for edge in list(edges):
        if edge["label"] != "punctuation":
            continue
        source = edge["start"]
        punct = edge["end"]
        if nodes[source]["type"] not in SOURCE_TYPES:
            continue
        if nodes[punct]["type"] != "PunctuationalConstant":
            continue
        ok, priority = should_wrap(sample, source, punct)
        if ok:
            candidates.append((priority, source, punct))

    changes: list[str] = []
    for _, source, punct in sorted(candidates):
        if any(
            edge["start"] == source
            and edge["end"] == punct
            and edge["label"] == "punctuation"
            for edge in edges
        ) and wrap_source(sample, source, punct):
            changes.append(f"wrap:{source}->{punct}")
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
    print(f"repaired {total} predicate/functor punctuation structures")
    for name, count in by_file.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
