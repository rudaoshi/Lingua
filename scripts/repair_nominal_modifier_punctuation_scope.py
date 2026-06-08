#!/usr/bin/env python3
"""Repair nominal/modifier punctuation close scope."""

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


def node_tokens(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if not isinstance(spans, list):
        return []
    return [i for start, end in spans for i in range(start, end + 1)]


def node_range(node: dict[str, Any]) -> tuple[int, int] | None:
    tokens = node_tokens(node)
    if not tokens:
        return None
    return min(tokens), max(tokens)


def descendant_tokens(
    sample: dict[str, Any], node_index: int, *, include_plain_punctuation: bool = False
) -> list[int]:
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
            if edge["start"] != index:
                continue
            if edge["label"] == "punctuation" and not include_plain_punctuation:
                continue
            visit(edge["end"])

    visit(node_index)
    return sorted(set(out))


def descendant_range(sample: dict[str, Any], node_index: int) -> tuple[int, int] | None:
    tokens = descendant_tokens(sample, node_index)
    if not tokens:
        return None
    return min(tokens), max(tokens)


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
        out.append({"start": edge["start"], "end": edge["end"], "label": edge["label"]})
    edges[:] = out


def clean_text(sample: dict[str, Any], node_index: int) -> str:
    words = sample["lingua"]["words"]
    return "".join(words[token] for token in node_tokens(sample["lingua"]["graph"]["nodes"][node_index])).strip()


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


def repair_nfATc_beta(sample: dict[str, Any]) -> bool:
    if sample.get("sentence") != "The amino acid sequence of NFATc.beta differs from that of NFATc .":
        return False
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    source = 11
    dot = 12
    beta = 6
    nodes[dot]["type"] = "ExpressionFunctor"
    remove_first_edge(edges, source, dot, "punctuation")
    remove_first_edge(edges, source, beta, "as:modification")
    for edge in edges:
        if edge["end"] == source:
            edge["end"] = dot
    add_edge(edges, dot, source, "func.arg")
    add_edge(edges, dot, beta, "func.arg")
    dedupe_edges(edges)
    return True


def repair_great_job_vocative(sample: dict[str, Any]) -> bool:
    if sample.get("sentence") != "Great job, guys.":
        return False
    edges = sample["lingua"]["graph"]["edges"]
    source = 0
    comma = 3
    job_phrase = 2
    changed = remove_first_edge(edges, source, comma, "punctuation")
    if changed:
        for edge in edges:
            if edge["start"] == source and edge["end"] == job_phrase and edge["label"] == "as:vocative":
                edge["end"] = comma
        add_edge(edges, comma, job_phrase, "as:punctuation")
        dedupe_edges(edges)
    return changed


def should_wrap(sample: dict[str, Any], source: int, punct: int) -> tuple[bool, int]:
    nodes = sample["lingua"]["graph"]["nodes"]
    source_node_range = node_range(nodes[source])
    source_desc_range = descendant_range(sample, source)
    punct_range = node_range(nodes[punct])
    if source_node_range is None or source_desc_range is None or punct_range is None:
        return False, 9

    wraps_source = (
        punct_range[0] <= source_desc_range[0] and punct_range[1] >= source_desc_range[1]
    ) or (
        punct_range[0] <= source_node_range[0] and punct_range[1] >= source_node_range[1]
    )
    if wraps_source:
        return True, 0

    immediately_after_node = punct_range[0] == source_node_range[1] + 1
    immediately_after_desc = punct_range[0] == source_desc_range[1] + 1
    if immediately_after_node or immediately_after_desc:
        return True, 1

    mark = clean_text(sample, punct)
    if mark in {".", "!", "?", ".?", "!!"} and punct_range[0] > source_desc_range[1]:
        return True, 2

    return False, 9


def repair_sample(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    if repair_nfATc_beta(sample):
        changes.append("nfatc_beta_expression")
    if repair_great_job_vocative(sample):
        changes.append("great_job_vocative")

    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    candidates: list[tuple[int, int, int]] = []
    for edge in list(edges):
        if edge["label"] != "punctuation":
            continue
        source = edge["start"]
        punct = edge["end"]
        if nodes[source]["type"] not in {"NominalConstant", "ModificationalConstant"}:
            continue
        if nodes[punct]["type"] != "PunctuationalConstant":
            continue
        ok, priority = should_wrap(sample, source, punct)
        if ok:
            candidates.append((priority, source, punct))

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
    print(f"repaired {total} nominal/modifier punctuation structures")
    for name, count in by_file.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
