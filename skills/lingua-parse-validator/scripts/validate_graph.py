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
RELATIVE_REF_ARG2_WRAPPED_MARKERS = {
    "which",
    "who",
    "whom",
    "whose",
}
RELATIVE_REF_ARG2_DIRECT_ADVERBS = {
    "when",
    "where",
}
RELATIVE_REF_ARG2_ARGUMENT_MARKERS = {
    "that",
    "which",
    "who",
    "whom",
    "whose",
}
RELATIVE_REF_OVERT_MARKERS = {
    "that",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
}
AUX_FUNC_ARG_PARALLEL_SPANS = {
    "Expression",
    "List",
    "Parataxis",
}
REF_ARG2_NON_CLAUSAL_AUX_SPANS = {
    "Apposition",
    "Attribute",
}
PUNCTUATION_TOKEN_TEXTS = {
    ",",
    ".",
    ";",
    ":",
    "--",
    "-",
    "(",
    ")",
    "[",
    "]",
    '"',
    "'",
}
DISCOURSE_MARKER_MODIFICATION_BODIES = {
    "however",
}


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


def add(
    errors: list[dict[str, Any]],
    code: str,
    location: str,
    message: str,
    *,
    severity: str = "error",
) -> None:
    errors.append(
        {
            "code": code,
            "severity": severity,
            "location": location,
            "message": message,
        }
    )


def is_blocking_finding(finding: dict[str, Any]) -> bool:
    return finding.get("severity", "error") != "warning"


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
        return [
            {
                "code": "FORM_ERROR",
                "severity": "error",
                "location": str(line),
                "message": f"cannot extract graph: {exc}",
            }
        ]

    if not isinstance(nodes, list):
        return [
            {
                "code": "FORM_ERROR",
                "severity": "error",
                "location": f"{line}:nodes",
                "message": "nodes must be a list",
            }
        ]
    if not isinstance(raw_edges, list):
        return [
            {
                "code": "FORM_ERROR",
                "severity": "error",
                "location": f"{line}:edges",
                "message": "edges must be a list",
            }
        ]

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

    def descendant_token_words(start: int) -> set[str]:
        if words is None:
            return set()
        return {
            words[index].strip().lower()
            for index in descendant_token_indexes(start)
            if words[index].strip()
        }

    def descendant_non_punctuation_words(start: int) -> tuple[str, ...]:
        if words is None:
            return ()
        return tuple(
            words[index].strip().lower()
            for index in descendant_token_indexes(start)
            if words[index].strip() and words[index].strip() not in PUNCTUATION_TOKEN_TEXTS
        )

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
                body_non_punctuation_words = descendant_non_punctuation_words(body_index)
                if (
                    len(body_non_punctuation_words) == 1
                    and body_non_punctuation_words[0] in DISCOURSE_MARKER_MODIFICATION_BODIES
                ):
                    add(
                        errors,
                        "DISCOURSE_MARKER_MODIFICATION_ERROR",
                        f"{line}:nodes[{index}]",
                        (
                            "parenthetical discourse marker `however` should use a "
                            "Discourse auxiliary node, not Modification"
                        ),
                    )
                body = nodes[body_index]
                body_spans = body.get("spans")
                if body_spans == "Ref":
                    ref_arg1_edges = [
                        child_edge
                        for child_edge in outgoing_edges.get(body_index, [])
                        if child_edge[2] == "pred.arg.1"
                    ]
                    variable_desc_set = set(variable_desc)
                    for _, ref_arg1_index, _ in ref_arg1_edges:
                        ref_arg1_desc_set = set(descendant_token_indexes(ref_arg1_index))
                        if variable_desc and ref_arg1_desc_set == variable_desc_set:
                            add(
                                errors,
                                "MODIFICATION_BODY_REF_DUPLICATE_VARIABLE_ERROR",
                                f"{line}:nodes[{index}]",
                                "Modification.body is a Ref whose pred.arg.1 duplicates Modification.variable; attach the Ref directly or remove the redundant Modification shell",
                            )
                        elif (
                            ref_arg1_desc_set
                            and ref_arg1_desc_set <= variable_desc_set
                            and is_reachable(variable_index, ref_arg1_index)
                        ):
                            add(
                                errors,
                                "MODIFICATION_BODY_REF_DESCENDANT_ANTECEDENT_ERROR",
                                f"{line}:nodes[{index}]",
                                "Modification.body is a Ref whose pred.arg.1 is already inside Modification.variable; attach the Ref at the inner relation instead of wrapping the outer variable",
                            )
                if body_spans == "Discourse":
                    add(
                        errors,
                        "MODIFICATION_BODY_DISCOURSE_WARNING",
                        f"{line}:nodes[{index}]",
                        "Modification.body is a Discourse auxiliary; parenthetical or connective discourse should usually be represented as Discourse over the modified unit, not Modification wrapping Discourse",
                        severity="warning",
                    )
                if body_spans == "Modification":
                    inner_variable_edges = [
                        child_edge
                        for child_edge in outgoing_edges.get(body_index, [])
                        if child_edge[2] == "variable"
                    ]
                    for _, inner_variable_index, _ in inner_variable_edges:
                        if is_reachable(variable_index, inner_variable_index) or is_reachable(
                            inner_variable_index, variable_index
                        ):
                            add(
                                errors,
                                "NESTED_MODIFICATION_SCOPE_WARNING",
                                f"{line}:nodes[{index}]",
                                "nested Modification variables are in an ancestor/descendant relation; check whether one modifier should attach to the larger recoverable unit instead of stacking wrappers",
                                severity="warning",
                            )
        if spans == "Ref" and not {"pred.arg.1", "pred.arg.2"} <= labels:
            add(errors, "AUX_STRUCTURE_ERROR", f"{line}:nodes[{index}]", "Ref requires pred.arg.1 and pred.arg.2")
        if spans == "Ref":
            arg1_edges = [
                edge
                for edge in outgoing_edges.get(index, [])
                if edge[2] == "pred.arg.1"
            ]
            for arg1_edge_index, arg1_index, _ in arg1_edges:
                if node_text(nodes[arg1_index], words).lower() == "what":
                    add(
                        errors,
                        "WHAT_REF_ANTECEDENT_ERROR",
                        f"{line}:edges[{arg1_edge_index}]",
                        "`what` heads a nominal/free wh-clause and should not be used as Ref.pred.arg.1 antecedent",
                    )
            has_expletive_it_arg1 = any(
                node_text(nodes[arg1_index], words).strip().lower() == "it"
                for _, arg1_index, _ in arg1_edges
            )
            arg2_edges = [
                edge
                for edge in outgoing_edges.get(index, [])
                if edge[2] == "pred.arg.2"
            ]
            for edge_index, arg2_index, _ in arg2_edges:
                arg2 = nodes[arg2_index]
                arg2_text = node_text(arg2, words).lower()
                arg2_spans = arg2.get("spans")
                if arg2.get("type") in {"PunctuationalConstant", "SymbolConstant"}:
                    close_modification_edges = [
                        child_edge
                        for child_edge in outgoing_edges.get(arg2_index, [])
                        if child_edge[2] == "as:punctuation"
                        and nodes[child_edge[1]].get("spans") == "Modification"
                    ]
                    for child_edge_index, _, _ in close_modification_edges:
                        add(
                            errors,
                            "REF_ARG2_PUNCTUATION_CLOSED_MODIFICATION_WARNING",
                            f"{line}:edges[{child_edge_index}]",
                            "Ref.pred.arg.2 points to punctuation that closes only a Modification; punctuation should usually close the whole Ref while Ref.pred.arg.2 points to the relative-clause unit",
                            severity="warning",
                        )
                if isinstance(arg2_spans, str) and arg2_spans in REF_ARG2_NON_CLAUSAL_AUX_SPANS:
                    add(
                        errors,
                        "REF_ARG2_NON_CLAUSAL_AUX_ERROR",
                        f"{line}:edges[{edge_index}]",
                        (
                            "Ref.pred.arg.2 should recover a relative-clause unit; "
                            f"found non-clausal aux {arg2_spans!r}"
                        ),
                    )
                if arg2_text in RELATIVE_REF_ARG2_DIRECT_ADVERBS:
                    non_func_arg_edges = [
                        child_edge
                        for child_edge in outgoing_edges.get(arg2_index, [])
                        if child_edge[2] != "func.arg"
                    ]
                    for child_edge_index, _, child_label in non_func_arg_edges:
                        add(
                            errors,
                            "RELATIVE_ADVERB_EXTRA_CHILD_ERROR",
                            f"{line}:edges[{child_edge_index}]",
                            (
                                "when/where used as Ref.pred.arg.2 must only use "
                                f"func.arg children; found extra {child_label!r} edge"
                            ),
                        )
                if arg2_text in RELATIVE_REF_ARG2_WRAPPED_MARKERS:
                    add(
                        errors,
                        "REF_ARG2_DIRECT_WH_MARKER_ERROR",
                        f"{line}:edges[{edge_index}]",
                        "Ref.pred.arg.2 must point to the relative-clause unit, not directly to a wh-word marker",
                    )
                if (
                    arg2.get("type") == "FactualPredicator"
                    and not has_expletive_it_arg1
                    and not (descendant_token_words(arg2_index) & RELATIVE_REF_OVERT_MARKERS)
                ):
                    add(
                        errors,
                        "ZERO_RELATIVE_REF_WARNING",
                        f"{line}:edges[{edge_index}]",
                        "Ref.pred.arg.2 points directly to a predicate; if this is a zero relative clause, remove the Ref wrapper and connect the antecedent as the recoverable gap",
                        severity="warning",
                    )
                if arg2.get("spans") != "Modification":
                    continue
                variable_edges = [
                    child_edge
                    for child_edge in outgoing_edges.get(arg2_index, [])
                    if child_edge[2] == "variable"
                ]
                for variable_edge_index, variable_index, _ in variable_edges:
                    arg1_descendant_sets = [
                        set(descendant_token_indexes(arg1_index))
                        for _, arg1_index, _ in arg1_edges
                    ]
                    variable_descendant_set = set(descendant_token_indexes(variable_index))
                    if variable_descendant_set and any(
                        arg1_descendant_set
                        and arg1_descendant_set <= variable_descendant_set
                        for arg1_descendant_set in arg1_descendant_sets
                    ):
                        add(
                            errors,
                            "REF_ARG2_VARIABLE_CONTAINS_ANTECEDENT_WARNING",
                            f"{line}:edges[{variable_edge_index}]",
                            (
                                "Ref.pred.arg.2 Modification.variable recovers the antecedent; "
                                "argument relative clauses should put the explicit wh/that marker "
                                "in variable, while zero or reduced relatives should usually be "
                                "direct Modification without a Ref wrapper"
                            ),
                            severity="warning",
                        )
                relative_adverb_wrapper_edges = [
                    child_edge
                    for child_edge in outgoing_edges.get(arg2_index, [])
                    if child_edge[2] in {"variable", "body"}
                    and node_text(nodes[child_edge[1]], words).lower()
                    in RELATIVE_REF_ARG2_DIRECT_ADVERBS
                ]
                for child_edge_index, child_index, child_label in relative_adverb_wrapper_edges:
                    child = nodes[child_index]
                    variable_marker_indexes = [
                        variable_index
                        for _, variable_index, _ in variable_edges
                        if node_text(nodes[variable_index], words).lower()
                        in RELATIVE_REF_ARG2_ARGUMENT_MARKERS
                    ]
                    is_temporal_body_inside_argument_relative = False
                    if child_label == "body" and child.get("type") == "LogicalPredicator":
                        for _, arg_index, arg_label in outgoing_edges.get(child_index, []):
                            if arg_label != "pred.arg.1":
                                continue
                            arg_descendant_tokens = set(descendant_token_indexes(arg_index))
                            if any(
                                set(descendant_token_indexes(variable_index))
                                <= arg_descendant_tokens
                                for variable_index in variable_marker_indexes
                            ):
                                is_temporal_body_inside_argument_relative = True
                                break
                    if is_temporal_body_inside_argument_relative:
                        continue
                    add(
                        errors,
                        "RELATIVE_ADVERB_REF_WRAPPER_WARNING",
                        f"{line}:edges[{child_edge_index}]",
                        (
                            "when/where appears inside Ref.pred.arg.2 Modification via "
                            f"{child_label!r}; relative adverbs should usually be "
                            "Ref.pred.arg.2 directly with func.arg to the clause, or the "
                            "relative-clause body should point to its real clause root"
                        ),
                        severity="warning",
                    )
                for variable_edge_index, variable_index, _ in variable_edges:
                    variable = nodes[variable_index]
                    variable_text = node_text(variable, words).lower()
                    if variable.get("type") == "LogicalPredicator":
                        add(
                            errors,
                            "REF_ARG2_LOGICAL_VARIABLE_ERROR",
                            f"{line}:edges[{variable_edge_index}]",
                            "Ref.pred.arg.2 Modification.variable should not be a LogicalPredicator; use the relative marker or relative adverb root",
                        )
        if spans == "Apposition" and not {"pred.arg.1", "pred.arg.2"} <= labels:
            add(errors, "AUX_STRUCTURE_ERROR", f"{line}:nodes[{index}]", "Apposition requires pred.arg.1 and pred.arg.2")
        if spans == "Attribute":
            arg1_edges = [
                edge
                for edge in outgoing_edges.get(index, [])
                if edge[2] == "pred.arg.1"
            ]
            arg2_edges = [
                edge
                for edge in outgoing_edges.get(index, [])
                if edge[2] == "pred.arg.2"
            ]
            for edge_index, arg1_index, _ in arg1_edges:
                arg1 = nodes[arg1_index]
                if arg1.get("type") not in {"PunctuationalConstant", "SymbolConstant"}:
                    continue
                if any(nodes[arg2_index].get("spans") == "Modification" for _, arg2_index, _ in arg2_edges):
                    add(
                        errors,
                        "ATTRIBUTE_PUNCTUATION_MODIFICATION_WARNING",
                        f"{line}:edges[{edge_index}]",
                        "Attribute connects punctuation/symbol to a Modification; check whether this is really a parenthetical Modification or Apposition rather than an Attribute relation",
                        severity="warning",
                    )
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
        if isinstance(spans, str) and spans in AUX_FUNC_ARG_PARALLEL_SPANS:
            func_arg_edges = [
                edge
                for edge in outgoing_edges.get(index, [])
                if edge[2] == "func.arg"
            ]
            for left_offset, (left_edge_index, left_index, _) in enumerate(func_arg_edges):
                for right_edge_index, right_index, _ in func_arg_edges[left_offset + 1 :]:
                    if is_reachable(left_index, right_index):
                        code = (
                            "PARATAXIS_CHILD_ANCESTOR_ERROR"
                            if spans == "Parataxis"
                            else "AUX_FUNCTOR_CHILD_ANCESTOR_ERROR"
                        )
                        add(
                            errors,
                            code,
                            f"{line}:edges[{left_edge_index}]",
                            (
                                f"{spans} func.arg children must be parallel; "
                                f"nodes[{right_index}] is reachable from sibling nodes[{left_index}]"
                            ),
                        )
                    if is_reachable(right_index, left_index):
                        code = (
                            "PARATAXIS_CHILD_ANCESTOR_ERROR"
                            if spans == "Parataxis"
                            else "AUX_FUNCTOR_CHILD_ANCESTOR_ERROR"
                        )
                        add(
                            errors,
                            code,
                            f"{line}:edges[{right_edge_index}]",
                            (
                                f"{spans} func.arg children must be parallel; "
                                f"nodes[{left_index}] is reachable from sibling nodes[{right_index}]"
                            ),
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

        if words is not None and node.get("type") == "ModificationalConstant" and isinstance(spans, list):
            token_indices = [
                token_index
                for span in spans
                if isinstance(span, list) and len(span) == 2
                for token_index in range(span[0], span[1] + 1)
            ]
            if len(token_indices) > 1 and any(words[token_index].strip() == "-" for token_index in token_indices):
                add(
                    errors,
                    "HYPHENATED_MULTIWORD_MODIFICATIONAL_CONSTANT_ERROR",
                    f"{line}:nodes[{index}]",
                    "hyphenated multiword ModificationalConstant should be split into ExpressionFunctor or explicit Modification structure",
                )

        if words is not None and node.get("type") == "PunctuationalConstant":
            token_indices = node_token_indexes(node)
            if len(token_indices) == 1 and words[token_indices[0]].strip() == ".":
                punctuation_parents = [
                    parent_index
                    for _, parent_index, parent_label in incoming_edges.get(index, [])
                    if parent_label == "as:punctuation"
                    and nodes[parent_index].get("type") == "PunctuationalConstant"
                ]
                if not punctuation_parents:
                    chain = [index]
                    current = index
                    while True:
                        punctuation_children = [
                            child_index
                            for _, child_index, child_label in outgoing_edges.get(current, [])
                            if child_label == "as:punctuation"
                            and nodes[child_index].get("type") == "PunctuationalConstant"
                            and len(node_token_indexes(nodes[child_index])) == 1
                            and words[node_token_indexes(nodes[child_index])[0]].strip() == "."
                        ]
                        if len(punctuation_children) != 1:
                            break
                        current = punctuation_children[0]
                        chain.append(current)
                    chain_token_indices = [
                        node_token_indexes(nodes[chain_index])[0]
                        for chain_index in chain
                    ]
                    if len(chain) > 1 and sorted(chain_token_indices) == list(
                        range(min(chain_token_indices), max(chain_token_indices) + 1)
                    ):
                        add(
                            errors,
                            "CONSECUTIVE_PERIOD_PUNCTUATION_CHAIN_WARNING",
                            f"{line}:nodes[{index}]",
                            "consecutive `.` punctuation tokens should usually be one PunctuationalConstant span, not a chain of single-period wrappers",
                            severity="warning",
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
        modifier_hyphen = (
            left.get("type") == "ModificationalConstant"
            and target.get("type") == "PunctuationalConstant"
        )
        has_punctuation_parent = any(
            nodes[parent_start].get("type") == "PunctuationalConstant"
            and parent_label == "as:punctuation"
            for _, parent_start, parent_label in incoming_edges.get(start, [])
        )
        if (
            left_bounds[1] < hyphen_bounds[0] < right_bounds[0]
            and right.get("type") not in compound_blocked_right_types
            and left.get("type") not in {"PunctuationalConstant", "SymbolConstant"}
            and not modifier_hyphen
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
    blocking_errors = [error for error in all_errors if is_blocking_finding(error)]
    warnings = [error for error in all_errors if error.get("severity") == "warning"]

    result = {
        "ok": not blocking_errors,
        "samples": len(items),
        "error_count": len(blocking_errors),
        "warning_count": len(warnings),
        "finding_count": len(all_errors),
        "errors": all_errors[: args.max_errors],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not blocking_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
