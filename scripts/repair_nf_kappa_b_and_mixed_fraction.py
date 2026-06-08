#!/usr/bin/env python3
"""Repair two reviewed train mismatch patterns.

1. Mixed standings fields such as "14 1/2" should be one Expression field.
2. The biomedical name "NF-kappa B" should be one Expression, not
   Modification(variable=B, body=hyphen-expression).
"""

from __future__ import annotations

import json
from pathlib import Path


MIXED_FRACTION_TARGETS = {
    ("train", 2157),
    ("train", 7714),
}


def node_text(words: list[str], node: dict) -> str:
    spans = node["spans"]
    if isinstance(spans, str):
        return spans
    parts: list[str] = []
    for start, end in spans:
        parts.extend(words[index].strip() for index in range(start, end + 1))
    return " ".join(parts)


def first_pos(node: dict) -> int:
    spans = node["spans"]
    if isinstance(spans, str):
        return 10**9
    return min(start for start, _ in spans)


def is_integer_node(words: list[str], node: dict) -> bool:
    return node_text(words, node).strip().isdigit()


def outgoing(edges: list[dict]) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {}
    for edge in edges:
        result.setdefault(edge["start"], []).append(edge)
    return result


def add_edge_once(edges: list[dict], start: int, end: int, label: str) -> None:
    if any(edge["start"] == start and edge["end"] == end and edge["label"] == label for edge in edges):
        return
    edges.append({"start": start, "end": end, "label": label})


def remove_node(graph: dict, node_id: int) -> None:
    nodes = graph["nodes"]
    edges = graph["edges"]
    del nodes[node_id]
    new_edges: list[dict] = []
    for edge in edges:
        if edge["start"] == node_id or edge["end"] == node_id:
            continue
        new_edge = dict(edge)
        if new_edge["start"] > node_id:
            new_edge["start"] -= 1
        if new_edge["end"] > node_id:
            new_edge["end"] -= 1
        new_edges.append(new_edge)
    graph["edges"] = new_edges


def repair_mixed_fraction(sample: dict) -> int:
    words = sample["lingua"]["words"]
    graph = sample["lingua"]["graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    out = outgoing(edges)
    repairs = 0

    for list_id, node in enumerate(nodes):
        if node.get("type") != "ListFunctor":
            continue
        list_args = [edge["end"] for edge in out.get(list_id, []) if edge["label"] == "func.arg"]
        for slash_id in list_args:
            slash = nodes[slash_id]
            if slash.get("type") != "ExpressionFunctor" or node_text(words, slash).strip() != "/":
                continue
            slash_args = [edge["end"] for edge in out.get(slash_id, []) if edge["label"] == "func.arg"]
            if len(slash_args) != 2 or not all(is_integer_node(words, nodes[arg]) for arg in slash_args):
                continue
            whole_candidates = [
                arg
                for arg in list_args
                if arg != slash_id
                and is_integer_node(words, nodes[arg])
                and first_pos(nodes[arg]) < first_pos(slash)
            ]
            if not whole_candidates:
                continue
            whole_id = max(whole_candidates, key=lambda arg: first_pos(nodes[arg]))
            graph["edges"] = [
                edge
                for edge in graph["edges"]
                if not (edge["start"] == list_id and edge["end"] == whole_id and edge["label"] == "func.arg")
            ]
            add_edge_once(graph["edges"], slash_id, whole_id, "func.arg")
            repairs += 1
    return repairs


def repair_nf_kappa_b(sample: dict) -> int:
    words = sample["lingua"]["words"]
    graph = sample["lingua"]["graph"]
    repairs = 0

    while True:
        nodes = graph["nodes"]
        edges = graph["edges"]
        out = outgoing(edges)
        target: tuple[int, int, int] | None = None
        for mod_id, node in enumerate(nodes):
            if node.get("spans") != "Modification" or node.get("type") != "ModificationalFunctor":
                continue
            variable_id = None
            body_id = None
            for edge in out.get(mod_id, []):
                if edge["label"] == "variable":
                    variable_id = edge["end"]
                elif edge["label"] == "body":
                    body_id = edge["end"]
            if variable_id is None or body_id is None:
                continue
            if node_text(words, nodes[variable_id]).strip() != "B":
                continue
            body = nodes[body_id]
            if body.get("type") != "ExpressionFunctor" or node_text(words, body).strip() != "-":
                continue
            body_args = [
                node_text(words, nodes[edge["end"]]).strip().lower()
                for edge in out.get(body_id, [])
                if edge["label"] == "func.arg"
            ]
            if set(body_args) != {"nf", "kappa"}:
                continue
            target = (mod_id, variable_id, body_id)
            break

        if target is None:
            break

        mod_id, variable_id, body_id = target
        for edge in graph["edges"]:
            if edge["end"] == mod_id:
                edge["end"] = body_id
        graph["edges"] = [
            edge
            for edge in graph["edges"]
            if not (edge["start"] == mod_id and edge["label"] in {"variable", "body"})
        ]
        add_edge_once(graph["edges"], body_id, variable_id, "func.arg")
        remove_node(graph, mod_id)
        repairs += 1

    return repairs


def main() -> None:
    totals = {"mixed_fraction": 0, "nf_kappa_b": 0}
    for split in ("train", "dev", "test"):
        path = Path("data") / f"{split}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for idx, line in enumerate(lines):
            sample = json.loads(line)
            mixed_repairs = 0
            if (split, idx) in MIXED_FRACTION_TARGETS:
                mixed_repairs = repair_mixed_fraction(sample)
                if mixed_repairs != 1:
                    raise ValueError(f"{split}:{idx} expected one mixed fraction repair, got {mixed_repairs}")
            nf_repairs = repair_nf_kappa_b(sample)
            if mixed_repairs or nf_repairs:
                print(f"{split}:{idx} mixed_fraction={mixed_repairs} nf_kappa_b={nf_repairs}")
            totals["mixed_fraction"] += mixed_repairs
            totals["nf_kappa_b"] += nf_repairs
            new_lines.append(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"totals: {totals}")


if __name__ == "__main__":
    main()
