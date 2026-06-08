#!/usr/bin/env python3
"""Repair reviewed parenthetical name qualifiers from Modification to Attribute."""

from __future__ import annotations

import json
from pathlib import Path


TARGETS = {
    "train": {
        885,
        1160,
        1498,
        2409,
        3223,
        3260,
        3509,
        3572,
        4204,
        4247,
        4597,
        4605,
        5085,
        5972,
        6222,
        6341,
        6723,
        6843,
        6903,
        6935,
        7986,
        8011,
        8230,
        10817,
    }
}


def node_text(words: list[str], node: dict) -> str:
    spans = node["spans"]
    if isinstance(spans, str):
        return spans
    parts: list[str] = []
    for start, end in spans:
        parts.extend(words[index].strip() for index in range(start, end + 1))
    return " ".join(parts)


def has_wrapping_parens(words: list[str], node: dict) -> bool:
    spans = node["spans"]
    if isinstance(spans, str):
        return False
    parts: list[str] = []
    for start, end in spans:
        parts.extend(words[index].strip() for index in range(start, end + 1))
    return "(" in parts and ")" in parts


def repair_sample(sample: dict, split: str, idx: int) -> bool:
    words = sample["lingua"]["words"]
    graph = sample["lingua"]["graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    outgoing: dict[int, list[dict]] = {}
    for edge in edges:
        outgoing.setdefault(edge["start"], []).append(edge)

    matches: list[tuple[int, dict, dict]] = []
    for node_id, node in enumerate(nodes):
        if node.get("spans") != "Modification" or node.get("type") != "ModificationalFunctor":
            continue
        variable_edge = None
        body_edge = None
        for edge in outgoing.get(node_id, []):
            if edge["label"] == "variable":
                variable_edge = edge
            elif edge["label"] == "body":
                body_edge = edge
        if variable_edge is None or body_edge is None:
            continue
        variable_node = nodes[variable_edge["end"]]
        body_node = nodes[body_edge["end"]]
        if variable_node.get("spans") != "Expression":
            continue
        if body_node.get("type") != "PunctuationalConstant":
            continue
        if not has_wrapping_parens(words, body_node):
            continue
        if not any(
            edge["start"] == body_edge["end"] and edge["label"] == "as:punctuation"
            for edge in edges
        ):
            continue
        matches.append((node_id, variable_edge, body_edge))

    if len(matches) != 1:
        raise ValueError(
            f"{split}:{idx} expected exactly one reviewed parenthetical Modification, "
            f"found {len(matches)}"
        )

    node_id, variable_edge, body_edge = matches[0]
    nodes[node_id]["spans"] = "Attribute"
    nodes[node_id]["type"] = "AttributePredicator"
    variable_edge["label"] = "pred.arg.1"
    body_edge["label"] = "pred.arg.2"
    return True


def main() -> None:
    total = 0
    for split, indices in TARGETS.items():
        path = Path("data") / f"{split}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        repaired = 0
        for idx in sorted(indices):
            sample = json.loads(lines[idx])
            if repair_sample(sample, split, idx):
                repaired += 1
                lines[idx] = json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{split}: repaired {repaired}")
        total += repaired
    print(f"total: repaired {total}")


if __name__ == "__main__":
    main()
