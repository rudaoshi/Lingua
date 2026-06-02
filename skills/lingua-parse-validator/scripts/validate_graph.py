#!/usr/bin/env python3
"""Validate Lingua graph structure and schema for JSON or JSONL samples."""

from __future__ import annotations

import argparse
import json
import re
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

AUX_NODE_TYPES = {
    "Apposition": "AppositionalPredicator",
    "Attribute": "AttributePredicator",
    "Copula": "FactualPredicator",
    "Discourse": "ModificationalFunctor",
    "Expression": "ExpressionFunctor",
    "List": "ListFunctor",
    "Modification": "ModificationalFunctor",
    "Parataxis": "ConjunctionalFunctor",
    "Ref": "ReferentialPredicator",
    "Vocative": "ModificationalFunctor",
    "Whether": "GeneralFunctor",
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
NUMERIC_TEXT_RE = re.compile(r"\d+(?:\.\d+)?")


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


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if isinstance(spans, str) or not isinstance(spans, list):
        return []
    indexes: list[int] = []
    for span in spans:
        if isinstance(span, list) and len(span) == 2:
            start, end = span
            if isinstance(start, int) and isinstance(end, int):
                indexes.extend(range(start, end + 1))
    return indexes


def node_span_bounds(node: dict[str, Any]) -> tuple[int, int] | None:
    indexes = node_token_indexes(node)
    if not indexes:
        return None
    return min(indexes), max(indexes)


def node_text(node: dict[str, Any], words: list[str] | None) -> str:
    spans = node.get("spans")
    if isinstance(spans, str):
        return spans
    if words is None:
        return ""
    return "".join(words[index] for index in node_token_indexes(node)).strip()


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
        if isinstance(spans, str) and spans in AUX_NODE_TYPES and node_type != AUX_NODE_TYPES[spans]:
            add(
                errors,
                "AUX_NODE_TYPE_ERROR",
                f"{line}:nodes[{index}]",
                f"auxiliary label {spans!r} requires node type {AUX_NODE_TYPES[spans]!r}, got {node_type!r}",
            )

    validate_spans(line, words, nodes, errors)

    adjacency = [[] for _ in nodes]
    indegree = [0] * len(nodes)
    outgoing: dict[int, list[str]] = defaultdict(list)
    outgoing_edges: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    incoming_edges: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    valid_edges: list[tuple[int, int, int, str]] = []
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
        incoming_edges[end].append((index, start, label))
        valid_edges.append((index, start, end, label))

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

    def descendant_token_indexes(start: int) -> tuple[int, ...]:
        seen: set[int] = set()
        stack = [start]
        token_indexes: list[int] = []
        while stack:
            node_index = stack.pop()
            if node_index in seen:
                continue
            seen.add(node_index)
            token_indexes.extend(node_token_indexes(nodes[node_index]))
            stack.extend(adjacency[node_index])
        return tuple(sorted(set(token_indexes)))

    def descendant_token_text(start: int) -> str:
        if words is None:
            return ""
        return "".join(words[index] for index in descendant_token_indexes(start)).strip()

    def is_reachable(start: int, target: int) -> bool:
        seen: set[int] = set()
        stack = [start]
        while stack:
            node_index = stack.pop()
            if node_index == target:
                return True
            if node_index in seen:
                continue
            seen.add(node_index)
            stack.extend(adjacency[node_index])
        return False

    for index, node in enumerate(nodes):
        spans = node.get("spans")
        labels = set(outgoing.get(index, []))
        if spans == "Modification" and not {"variable", "body"} <= labels:
            add(errors, "AUX_STRUCTURE_ERROR", f"{line}:nodes[{index}]", "Modification requires variable and body")
        if spans == "Modification":
            variable_edges = [
                edge
                for edge in outgoing_edges.get(index, [])
                if edge[2] == "variable"
            ]
            body_edges = [
                edge
                for edge in outgoing_edges.get(index, [])
                if edge[2] == "body"
            ]
            if len(variable_edges) == 1 and len(body_edges) == 1:
                variable_index = variable_edges[0][1]
                body_index = body_edges[0][1]
                variable_desc = descendant_token_indexes(variable_index)
                body_desc = descendant_token_indexes(body_index)
                if variable_desc and variable_desc == body_desc:
                    add(
                        errors,
                        "REDUNDANT_MODIFICATION_ERROR",
                        f"{line}:nodes[{index}]",
                        "Modification.variable and Modification.body recover the same token set; retarget incoming edges to the shared child and remove the aux wrapper",
                    )
                if nodes[body_index].get("type") == "ModificationalFunctor":
                    for _, child_index, child_label in outgoing_edges.get(body_index, []):
                        if child_label == "variable" and descendant_token_indexes(child_index) == variable_desc:
                            add(
                                errors,
                                "REDUNDANT_MODIFICATION_ERROR",
                                f"{line}:nodes[{index}]",
                                "Modification.body already anchors the same variable; retarget incoming edges to the body functor and remove the aux wrapper",
                            )
                            break
                if is_reachable(variable_index, body_index):
                    add(
                        errors,
                        "REDUNDANT_MODIFICATION_ERROR",
                        f"{line}:nodes[{index}]",
                        "Modification.body is already reachable from Modification.variable; retarget incoming edges to the variable root and remove the aux wrapper",
                    )
        if spans == "Ref" and not {"pred.arg.1", "pred.arg.2"} <= labels:
            add(errors, "AUX_STRUCTURE_ERROR", f"{line}:nodes[{index}]", "Ref requires pred.arg.1 and pred.arg.2")
        if spans == "Apposition" and not {"pred.arg.1", "pred.arg.2"} <= labels:
            add(errors, "AUX_STRUCTURE_ERROR", f"{line}:nodes[{index}]", "Apposition requires pred.arg.1 and pred.arg.2")
        if (
            isinstance(spans, str)
            and spans in {"Expression", "List", "Parataxis"}
            and outgoing.get(index, []).count("func.arg") < 2
        ):
            add(
                errors,
                "UNARY_FUNCTOR_AUX_ERROR",
                f"{line}:nodes[{index}]",
                f"{spans} auxiliary node requires at least two func.arg children",
            )
        if node.get("type") == "ListFunctor":
            for edge_index, child_index, child_label in outgoing_edges.get(index, []):
                if child_label != "func.arg":
                    continue
                child = nodes[child_index]
                if child.get("type") == "OtherConstant":
                    add(
                        errors,
                        "LIST_ITEM_STRUCTURE_ERROR",
                        f"{line}:edges[{edge_index}]",
                        "List item should be a recoverable field; do not use OtherConstant as a sibling item",
                    )

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

        if spans == "Apposition":
            arg1_edges = [
                edge
                for edge in outgoing_edges.get(index, [])
                if edge[2] == "pred.arg.1"
            ]
            for edge_index, arg1_index, _ in arg1_edges:
                arg1 = nodes[arg1_index]
                if arg1.get("type") not in {"PunctuationalConstant", "SymbolConstant"}:
                    continue
                close_edges = [
                    edge
                    for edge in outgoing_edges.get(arg1_index, [])
                    if edge[2] == "as:punctuation"
                ]
                if not close_edges:
                    add(
                        errors,
                        "APPOSITION_PUNCTUATION_ARG1_ERROR",
                        f"{line}:edges[{edge_index}]",
                        "Apposition.pred.arg.1 punctuation/symbol target must be a close wrapper with as:punctuation",
                    )
                if any(child_index == index for _, child_index, _ in close_edges):
                    add(
                        errors,
                        "APPOSITION_PUNCTUATION_ARG1_ERROR",
                        f"{line}:edges[{edge_index}]",
                        "Apposition.pred.arg.1 must not point to the punctuation wrapper that closes the whole Apposition",
                    )
                if node_text(arg1, words).strip() == "-" and len(close_edges) == 1:
                    value_text = descendant_token_text(close_edges[0][1])
                    if NUMERIC_TEXT_RE.fullmatch(value_text):
                        add(
                            errors,
                            "VALUE_LABEL_DASH_APPOSITION_ERROR",
                            f"{line}:edges[{edge_index}]",
                            "numeric value-label fragments like '42 - Number ...' should use dash ExpressionFunctor func.arg edges, not Apposition",
                        )

    if words and len(words) >= 3:
        stripped_words = [word.strip() for word in words]
        if (
            NUMERIC_TEXT_RE.fullmatch(stripped_words[0])
            and stripped_words[1] == "-"
            and stripped_words[2] in {"Number", "Percentage"}
        ):
            dash_owners = [
                node_index
                for node_index, node in enumerate(nodes)
                if 1 in node_token_indexes(node)
            ]
            if len(dash_owners) != 1:
                add(
                    errors,
                    "VALUE_LABEL_DASH_EXPRESSION_ERROR",
                    f"{line}:tokens[1]",
                    f"value-label dash token must have exactly one owner, found {dash_owners}",
                )
            else:
                dash_index = dash_owners[0]
                dash = nodes[dash_index]
                func_arg_count = outgoing.get(dash_index, []).count("func.arg")
                if (
                    dash.get("type") != "ExpressionFunctor"
                    or node_text(dash, words).strip() != "-"
                    or func_arg_count < 2
                ):
                    add(
                        errors,
                        "VALUE_LABEL_DASH_EXPRESSION_ERROR",
                        f"{line}:nodes[{dash_index}]",
                        "numeric value-label fragments like '42 - Number ...' require dash ExpressionFunctor with at least two func.arg edges",
                    )

    compound_blocked_right_types = {
        "FactualPredicator",
        "LogicalPredicator",
        "ModificationalFunctor",
        "GeneralFunctor",
        "ConjunctionalFunctor",
    }
    for edge_index, start, end, label in valid_edges:
        source = nodes[start]
        target = nodes[end]
        if (
            label != "body"
            or source.get("spans") != "Modification"
            or target.get("type") not in {"PunctuationalConstant", "SymbolConstant"}
            or node_text(target, words) != "-"
        ):
            continue
        variable_edges = [
            (child_index, child_end)
            for child_index, child_end, child_label in outgoing_edges.get(start, [])
            if child_label == "variable"
        ]
        close_edges = [
            (child_index, child_end)
            for child_index, child_end, child_label in outgoing_edges.get(end, [])
            if child_label == "as:punctuation"
        ]
        if len(variable_edges) != 1 or len(close_edges) != 1:
            continue
        right = nodes[variable_edges[0][1]]
        left = nodes[close_edges[0][1]]
        left_bounds = node_span_bounds(left)
        hyphen_bounds = node_span_bounds(target)
        right_bounds = node_span_bounds(right)
        if not (left_bounds and hyphen_bounds and right_bounds):
            continue
        has_punctuation_parent = any(
            nodes[parent_start].get("type") == "PunctuationalConstant"
            and parent_label == "as:punctuation"
            for _, parent_start, parent_label in incoming_edges.get(start, [])
        )
        if (
            left_bounds[1] < hyphen_bounds[0] < right_bounds[0]
            and right.get("type") not in compound_blocked_right_types
            and left.get("type") not in {"PunctuationalConstant", "SymbolConstant"}
            and not has_punctuation_parent
        ):
            add(
                errors,
                "COMPOUND_HYPHEN_MODIFICATION_ERROR",
                f"{line}:edges[{edge_index}]",
                "compound/name/measure hyphen should be ExpressionFunctor with func.arg children, not Modification.body punctuation",
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
