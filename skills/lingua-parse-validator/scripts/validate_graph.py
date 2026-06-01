#!/usr/bin/env python3
"""Validate Lingua graph structure and schema for JSON or JSONL samples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


NODE_TYPES = {
    "AttributePredicator",
    "AppositionalPredicator",
    "FactualPredicator",
    "LogicalPredicator",
    "ReferentialPredicator",
    "ConjunctionalFunctor",
    "ExpressionFunctor",
    "GeneralFunctor",
    "ListFunctor",
    "ModificationalFunctor",
    "DeterminerConstant",
    "InterjectionConstant",
    "ModificationalConstant",
    "NominalConstant",
    "OtherConstant",
    "PunctuationalConstant",
    "SymbolConstant",
}

AUX_LABELS = {
    "Apposition",
    "Attribute",
    "Copula",
    "Discourse",
    "Expression",
    "List",
    "Modification",
    "Parataxis",
    "Ref",
    "Vocative",
    "Whether",
}

BASE_EDGES = {
    "body",
    "variable",
    "pred.arg.1",
    "pred.arg.2",
    "pred.arg.3",
    "pred.arg.4",
    "func.arg",
    "appositive",
    "attribute",
    "discourse",
    "index",
    "modification",
    "nonsense",
    "punctuation",
    "ref",
    "repeat",
    "vocative",
}
EDGE_LABELS = BASE_EDGES | {f"as:{edge}" for edge in BASE_EDGES}


def load_items(path: Path) -> list[tuple[int | None, dict[str, Any]]]:
    text = path.read_text().strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [(line_no, json.loads(line)) for line_no, line in enumerate(text.splitlines(), 1)]
    data = json.loads(text)
    if isinstance(data, list):
        return [(index + 1, item) for index, item in enumerate(data)]
    return [(None, data)]


def extract_graph(sample: dict[str, Any]) -> tuple[list[str] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    if "lingua" in sample:
        lingua = sample["lingua"]
        graph = lingua["graph"]
        return lingua.get("words"), graph["nodes"], graph["edges"]
    if "graph" in sample:
        graph = sample["graph"]
        return sample.get("words") or sample.get("tokens"), graph["nodes"], graph["edges"]
    return sample.get("words") or sample.get("tokens"), sample["nodes"], sample["edges"]


def normalize_edge(edge: dict[str, Any]) -> dict[str, Any]:
    if {"start", "end", "label"} <= set(edge):
        return edge
    if {"source", "target", "relation"} <= set(edge):
        return {"start": edge["source"], "end": edge["target"], "label": edge["relation"]}
    return edge


def add(errors: list[dict[str, Any]], code: str, location: str, message: str) -> None:
    errors.append({"code": code, "location": location, "message": message})


def validate_spans(
    line: int | None,
    words: list[str] | None,
    nodes: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    if words is None:
        return
    coverage = [0] * len(words)
    for index, node in enumerate(nodes):
        spans = node.get("spans")
        if isinstance(spans, str):
            continue
        if not isinstance(spans, list):
            add(errors, "TOKEN_PARTITION_ERROR", f"{line}:nodes[{index}].spans", "spans must be a list or auxiliary string")
            continue
        for span_index, span in enumerate(spans):
            if not (
                isinstance(span, list)
                and len(span) == 2
                and all(isinstance(value, int) for value in span)
            ):
                add(errors, "TOKEN_PARTITION_ERROR", f"{line}:nodes[{index}].spans[{span_index}]", "span must be [start, end]")
                continue
            start, end = span
            if not 0 <= start <= end < len(words):
                add(errors, "TOKEN_PARTITION_ERROR", f"{line}:nodes[{index}].spans[{span_index}]", "span is outside token bounds")
                continue
            for token_index in range(start, end + 1):
                coverage[token_index] += 1
    if coverage != [1] * len(words):
        bad = [index for index, count in enumerate(coverage) if count != 1]
        add(errors, "TOKEN_PARTITION_ERROR", f"{line}:tokens", f"token coverage is not exactly once at token indexes {bad[:20]}")


def validate_graph(line: int | None, sample: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    try:
        words, nodes, raw_edges = extract_graph(sample)
    except Exception as exc:  # noqa: BLE001
        return [{"code": "FORM_ERROR", "location": str(line), "message": f"cannot extract graph: {exc}"}]

    if not isinstance(nodes, list):
        return [{"code": "FORM_ERROR", "location": f"{line}:nodes", "message": "nodes must be a list"}]
    if not isinstance(raw_edges, list):
        return [{"code": "FORM_ERROR", "location": f"{line}:edges", "message": "edges must be a list"}]

    edges = [normalize_edge(edge) for edge in raw_edges]
    for index, node in enumerate(nodes):
        node_type = node.get("type")
        spans = node.get("spans")
        if node_type not in NODE_TYPES:
            add(errors, "SCHEMA_ERROR", f"{line}:nodes[{index}].type", f"illegal node type {node_type!r}")
        if isinstance(spans, str) and spans not in AUX_LABELS:
            add(errors, "SCHEMA_ERROR", f"{line}:nodes[{index}].spans", f"illegal auxiliary label {spans!r}")

    validate_spans(line, words, nodes, errors)

    adjacency = [[] for _ in nodes]
    indegree = [0] * len(nodes)
    outgoing: dict[int, list[str]] = defaultdict(list)
    outgoing_edges: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for index, edge in enumerate(edges):
        label = edge.get("label")
        start = edge.get("start")
        end = edge.get("end")
        if label not in EDGE_LABELS:
            add(errors, "SCHEMA_ERROR", f"{line}:edges[{index}].label", f"illegal edge label {label!r}")
        if not isinstance(start, int) or not 0 <= start < len(nodes):
            add(errors, "SCHEMA_ERROR", f"{line}:edges[{index}].start", f"invalid source {start!r}")
            continue
        if not isinstance(end, int) or not 0 <= end < len(nodes):
            add(errors, "SCHEMA_ERROR", f"{line}:edges[{index}].end", f"invalid target {end!r}")
            continue
        adjacency[start].append(end)
        indegree[end] += 1
        outgoing[start].append(label)
        outgoing_edges[start].append((index, end, label))

    roots = [index for index, degree in enumerate(indegree) if degree == 0]
    if len(nodes) and len(roots) != 1:
        add(errors, "GRAPH_FORM_ERROR", f"{line}:graph", f"expected one root, found {roots}")

    if len(roots) == 1:
        reachable = set()
        stack = roots[:]
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            stack.extend(adjacency[node])
        if len(reachable) != len(nodes):
            missing = sorted(set(range(len(nodes))) - reachable)
            add(errors, "GRAPH_FORM_ERROR", f"{line}:graph", f"unreachable node indexes {missing[:20]}")

    color = [0] * len(nodes)

    def visit(node: int) -> bool:
        color[node] = 1
        for child in adjacency[node]:
            if color[child] == 1:
                return False
            if color[child] == 0 and not visit(child):
                return False
        color[node] = 2
        return True

    for node in range(len(nodes)):
        if color[node] == 0 and not visit(node):
            add(errors, "GRAPH_FORM_ERROR", f"{line}:graph", "cycle detected")
            break

    for index, node in enumerate(nodes):
        spans = node.get("spans")
        labels = set(outgoing.get(index, []))
        if spans == "Modification" and not {"variable", "body"} <= labels:
            add(errors, "AUX_STRUCTURE_ERROR", f"{line}:nodes[{index}]", "Modification requires variable and body")
        if spans == "Ref" and not {"pred.arg.1", "pred.arg.2"} <= labels:
            add(errors, "AUX_STRUCTURE_ERROR", f"{line}:nodes[{index}]", "Ref requires pred.arg.1 and pred.arg.2")

        as_edges = [edge for edge in outgoing_edges.get(index, []) if edge[2].startswith("as:")]
        if as_edges and len(outgoing_edges.get(index, [])) != 1:
            edge_summary = ", ".join(
                f"edges[{edge_index}] {label}->nodes[{end}]"
                for edge_index, end, label in outgoing_edges[index]
            )
            add(
                errors,
                "AS_SHORTHAND_STRUCTURE_ERROR",
                f"{line}:nodes[{index}]",
                f"node with outgoing as:* shorthand must have exactly one outgoing edge; found {edge_summary}",
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-errors", type=int, default=200)
    args = parser.parse_args()

    all_errors: list[dict[str, Any]] = []
    items = load_items(args.path)
    for line, sample in items:
        all_errors.extend(validate_graph(line, sample))

    result = {
        "ok": not all_errors,
        "samples": len(items),
        "error_count": len(all_errors),
        "errors": all_errors[: args.max_errors],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not all_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
