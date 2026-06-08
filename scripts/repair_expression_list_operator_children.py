#!/usr/bin/env python3
"""Repair Expression/List wrappers around explicit operator children."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = [
    ROOT / "data" / "train.jsonl",
    ROOT / "data" / "dev.jsonl",
    ROOT / "data" / "test.jsonl",
]

FUNCTOR_TYPES = {"ExpressionFunctor", "ListFunctor"}
WRAPPER_SPANS = {"Expression", "List"}
LIST_SEPARATOR_TEXTS = {",", "/", "-", "–", "—", ";", "..."}


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return edge["start"], edge["end"], edge["label"]


def dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, str]] = set()
    result: list[dict[str, Any]] = []
    for edge in edges:
        key = edge_key(edge)
        if key in seen:
            continue
        seen.add(key)
        result.append(edge)
    return result


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if not isinstance(spans, list):
        return []
    return [index for start, end in spans for index in range(start, end + 1)]


def spans_for_tokens(tokens: list[int]) -> list[list[int]]:
    tokens = sorted(set(tokens))
    spans: list[list[int]] = []
    start = prev = tokens[0]
    for token in tokens[1:]:
        if token == prev + 1:
            prev = token
            continue
        spans.append([start, prev])
        start = prev = token
    spans.append([start, prev])
    return spans


def node_text(sample: dict[str, Any], node_index: int) -> str:
    node = graph(sample)["nodes"][node_index]
    spans = node.get("spans")
    if isinstance(spans, str):
        return spans
    words = sample["lingua"]["words"]
    return "".join(words[index] for index in node_token_indexes(node)).strip()


def descendant_tokens(sample: dict[str, Any], node_index: int) -> list[int]:
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    seen: set[int] = set()
    tokens: list[int] = []

    def visit(index: int) -> None:
        if index in seen:
            return
        seen.add(index)
        tokens.extend(node_token_indexes(nodes[index]))
        for edge in edges:
            if edge["start"] == index:
                visit(edge["end"])

    visit(node_index)
    return sorted(set(tokens))


def descendant_nodes(sample: dict[str, Any], node_index: int) -> set[int]:
    edges = graph(sample)["edges"]
    seen: set[int] = set()

    def visit(index: int) -> None:
        if index in seen:
            return
        seen.add(index)
        for edge in edges:
            if edge["start"] == index:
                visit(edge["end"])

    visit(node_index)
    return seen


def sorted_func_args(sample: dict[str, Any], source: int) -> list[int]:
    def key(index: int) -> tuple[int, int]:
        tokens = descendant_tokens(sample, index)
        if not tokens:
            return 10**9, 10**9
        return min(tokens), max(tokens)

    return sorted(
        [edge["end"] for edge in graph(sample)["edges"] if edge["start"] == source and edge["label"] == "func.arg"],
        key=key,
    )


def add_edge(edges: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    edge = {"start": start, "end": end, "label": label}
    if edge_key(edge) not in {edge_key(existing) for existing in edges}:
        edges.append(edge)


def remove_edges(
    edges: list[dict[str, Any]],
    *,
    start: int | None = None,
    end: int | None = None,
    label: str | None = None,
) -> None:
    kept: list[dict[str, Any]] = []
    for edge in edges:
        if start is not None and edge["start"] != start:
            kept.append(edge)
            continue
        if end is not None and edge["end"] != end:
            kept.append(edge)
            continue
        if label is not None and edge["label"] != label:
            kept.append(edge)
            continue
        # Drop this edge.
    edges[:] = kept


def redirect_incoming(sample: dict[str, Any], old: int, new: int) -> None:
    for edge in graph(sample)["edges"]:
        if edge["end"] == old:
            edge["end"] = new


def only_incoming_from_parent(sample: dict[str, Any], node_index: int, parent: int) -> bool:
    return all(
        edge["start"] == parent and edge["label"] == "func.arg"
        for edge in graph(sample)["edges"]
        if edge["end"] == node_index
    )


def is_aux_expression_or_list(node: dict[str, Any]) -> bool:
    spans = node.get("spans")
    return node.get("type") in FUNCTOR_TYPES and isinstance(spans, str) and spans in WRAPPER_SPANS


def is_visible_operator(node: dict[str, Any]) -> bool:
    return node.get("type") in FUNCTOR_TYPES and not isinstance(node.get("spans"), str)


def is_collapsible_child(node: dict[str, Any]) -> bool:
    return is_visible_operator(node) or is_aux_expression_or_list(node)


def delete_nodes(sample: dict[str, Any], indexes: set[int]) -> None:
    if not indexes:
        return
    nodes = graph(sample)["nodes"]
    remap: dict[int, int] = {}
    new_nodes: list[dict[str, Any]] = []
    for old_index, node in enumerate(nodes):
        if old_index in indexes:
            continue
        remap[old_index] = len(new_nodes)
        new_nodes.append(node)

    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        if edge["start"] in indexes or edge["end"] in indexes:
            continue
        new_edges.append(
            {
                "start": remap[edge["start"]],
                "end": remap[edge["end"]],
                "label": edge["label"],
            }
        )
    graph(sample)["nodes"] = new_nodes
    graph(sample)["edges"] = dedupe_edges(new_edges)


def collapse_single_child_operator_wrappers(sample: dict[str, Any]) -> list[str]:
    """Remove Expression/List aux nodes that only wrap one explicit operator."""
    changes: list[str] = []
    while True:
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        made_change = False
        for node_index, node in enumerate(nodes):
            if not is_aux_expression_or_list(node):
                continue
            outgoing = [edge for edge in edges if edge["start"] == node_index]
            func_args = [edge["end"] for edge in outgoing if edge["label"] == "func.arg"]
            if len(outgoing) != 1 or len(func_args) != 1:
                continue
            child = func_args[0]
            if child == node_index or child >= len(nodes):
                continue
            if not is_collapsible_child(nodes[child]):
                continue

            for edge in edges:
                if edge["end"] == node_index:
                    edge["end"] = child
            changes.append(f"collapse:{node.get('spans')}:{node_index}->{child}")
            delete_nodes(sample, {node_index})
            made_change = True
            break
        if not made_change:
            break
    return changes


def collapse_redundant_operator_wrappers(sample: dict[str, Any]) -> list[str]:
    """Remove aux wrappers whose arguments are already contained by one operator child."""
    changes: list[str] = []
    while True:
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        made_change = False
        for node_index, node in enumerate(nodes):
            if not is_aux_expression_or_list(node):
                continue
            outgoing = [edge for edge in edges if edge["start"] == node_index]
            if any(edge["label"] != "func.arg" for edge in outgoing):
                continue
            args = [edge["end"] for edge in outgoing]
            for child in args:
                if not is_visible_operator(nodes[child]):
                    continue
                if set(args).issubset(descendant_nodes(sample, child)):
                    redirect_incoming(sample, node_index, child)
                    delete_nodes(sample, {node_index})
                    changes.append(f"collapse_redundant:{node.get('spans')}")
                    made_change = True
                    break
            if made_change:
                break
        if not made_change:
            break
    return changes


def repair_rank_marker_punctuation_children(sample: dict[str, Any]) -> list[str]:
    """Turn list child punctuation such as `8.` or `#4)` into expression operators."""
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    for parent, node in enumerate(nodes):
        if node.get("type") != "ListFunctor" or node.get("spans") != "List":
            continue
        for arg in sorted_func_args(sample, parent):
            if nodes[arg].get("type") != "PunctuationalConstant":
                continue
            mark = node_text(sample, arg)
            if mark not in {".", ")"}:
                continue
            if not only_incoming_from_parent(sample, arg, parent):
                continue
            outgoing = [edge for edge in edges if edge["start"] == arg]
            close_targets = [edge["end"] for edge in outgoing if edge["label"] == "as:punctuation"]
            if len(outgoing) != len(close_targets) or len(close_targets) != 1:
                continue
            close_text = node_text(sample, close_targets[0])
            if not re.fullmatch(r"#?\d+", close_text):
                continue
            nodes[arg]["type"] = "ExpressionFunctor"
            remove_edges(edges, start=arg, label="as:punctuation")
            add_edge(edges, arg, close_targets[0], "func.arg")
            changes.append(f"rank_marker:{mark}")
    if changes:
        graph(sample)["edges"] = dedupe_edges(edges)
    return changes


def repair_bare_abbreviation_dot_children(sample: dict[str, Any]) -> list[str]:
    """Turn bare dot children in expressions (`p. 36`, `vs. Team`) into operators."""
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    for parent, node in enumerate(nodes):
        if node.get("type") != "ExpressionFunctor" or node.get("spans") != "Expression":
            continue
        args = sorted_func_args(sample, parent)
        for pos, arg in enumerate(args):
            if nodes[arg].get("type") != "PunctuationalConstant" or node_text(sample, arg) != ".":
                continue
            if pos == 0:
                continue
            if not only_incoming_from_parent(sample, arg, parent):
                continue
            if any(edge["start"] == arg for edge in edges):
                continue
            left = args[pos - 1]
            left_tokens = descendant_tokens(sample, left)
            dot_tokens = node_token_indexes(nodes[arg])
            if not left_tokens or len(dot_tokens) != 1 or max(left_tokens) + 1 != dot_tokens[0]:
                continue
            left_text = node_text(sample, left)
            if not re.fullmatch(r"[A-Za-z]{1,4}|No|vs", left_text, flags=re.IGNORECASE):
                continue
            nodes[arg]["type"] = "ExpressionFunctor"
            remove_edges(edges, start=parent, end=left, label="func.arg")
            add_edge(edges, arg, left, "func.arg")
            changes.append("abbrev_dot")
    if changes:
        graph(sample)["edges"] = dedupe_edges(edges)
    return changes


def repair_closed_abbreviation_dot_children(sample: dict[str, Any]) -> list[str]:
    """Turn close-punctuation abbreviation dots (`No.`) into expression operators."""
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    for parent, node in enumerate(nodes):
        if node.get("type") != "ExpressionFunctor" or node.get("spans") != "Expression":
            continue
        for arg in sorted_func_args(sample, parent):
            if nodes[arg].get("type") != "PunctuationalConstant" or node_text(sample, arg) != ".":
                continue
            if not only_incoming_from_parent(sample, arg, parent):
                continue
            outgoing = [edge for edge in edges if edge["start"] == arg]
            close_targets = [edge["end"] for edge in outgoing if edge["label"] == "as:punctuation"]
            if len(outgoing) != len(close_targets) or len(close_targets) != 1:
                continue
            close_text = node_text(sample, close_targets[0])
            if not re.fullmatch(r"[A-Za-z]{1,4}|No|vs", close_text, flags=re.IGNORECASE):
                continue
            nodes[arg]["type"] = "ExpressionFunctor"
            remove_edges(edges, start=arg, label="as:punctuation")
            add_edge(edges, arg, close_targets[0], "func.arg")
            changes.append("closed_abbrev_dot")
    if changes:
        graph(sample)["edges"] = dedupe_edges(edges)
    return changes


def repair_bare_expression_comma_children(sample: dict[str, Any]) -> list[str]:
    """Turn date/place expression commas from punctuation operands into operators."""
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    for parent, node in enumerate(nodes):
        if node.get("type") != "ExpressionFunctor" or node.get("spans") != "Expression":
            continue
        args = sorted_func_args(sample, parent)
        for pos, arg in enumerate(args):
            if nodes[arg].get("type") != "PunctuationalConstant" or node_text(sample, arg) != ",":
                continue
            if pos == 0 or pos + 1 >= len(args):
                continue
            if not only_incoming_from_parent(sample, arg, parent):
                continue
            if any(edge["start"] == arg for edge in edges):
                continue
            left = args[pos - 1]
            right = args[pos + 1]
            comma_tokens = node_token_indexes(nodes[arg])
            left_tokens = descendant_tokens(sample, left)
            right_tokens = descendant_tokens(sample, right)
            if not comma_tokens or not left_tokens or not right_tokens:
                continue
            if not (max(left_tokens) < comma_tokens[0] < min(right_tokens)):
                continue
            # Keep this conservative: these are the residual date/place-like
            # expression commas, not ordinary clause punctuation.
            if not (
                re.fullmatch(r"[A-Z][A-Za-z]+|[A-Z]{2,}", node_text(sample, left))
                or re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)?", node_text(sample, left))
            ):
                continue
            nodes[arg]["type"] = "ExpressionFunctor"
            remove_edges(edges, start=parent, end=left, label="func.arg")
            remove_edges(edges, start=parent, end=right, label="func.arg")
            add_edge(edges, arg, left, "func.arg")
            add_edge(edges, arg, right, "func.arg")
            changes.append("expression_comma")
    if changes:
        graph(sample)["edges"] = dedupe_edges(edges)
    return changes


def repair_closed_expression_comma_children(sample: dict[str, Any]) -> list[str]:
    """Turn close-punctuation address/place commas inside expressions into operators."""
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    for parent, node in enumerate(nodes):
        if node.get("type") != "ExpressionFunctor" or node.get("spans") != "Expression":
            continue
        args = sorted_func_args(sample, parent)
        for pos, arg in enumerate(args):
            if nodes[arg].get("type") != "PunctuationalConstant" or node_text(sample, arg) != ",":
                continue
            if pos + 1 >= len(args):
                continue
            if not only_incoming_from_parent(sample, arg, parent):
                continue
            outgoing = [edge for edge in edges if edge["start"] == arg]
            close_targets = [edge["end"] for edge in outgoing if edge["label"] == "as:punctuation"]
            if len(outgoing) != len(close_targets) or len(close_targets) != 1:
                continue
            right = args[pos + 1]
            comma_tokens = node_token_indexes(nodes[arg])
            left_tokens = descendant_tokens(sample, close_targets[0])
            right_tokens = descendant_tokens(sample, right)
            if not comma_tokens or not left_tokens or not right_tokens:
                continue
            if not (max(left_tokens) < comma_tokens[0] < min(right_tokens)):
                continue
            nodes[arg]["type"] = "ExpressionFunctor"
            remove_edges(edges, start=arg, label="as:punctuation")
            remove_edges(edges, start=parent, end=right, label="func.arg")
            add_edge(edges, arg, close_targets[0], "func.arg")
            add_edge(edges, arg, right, "func.arg")
            changes.append("closed_expression_comma")
    if changes:
        graph(sample)["edges"] = dedupe_edges(edges)
    return changes


def repair_trailing_expression_comma_children(sample: dict[str, Any]) -> list[str]:
    """Make a trailing comma close the preceding expression instead of being its operand."""
    changes: list[str] = []
    while True:
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        made_change = False
        for parent, node in enumerate(nodes):
            if node.get("type") != "ExpressionFunctor" or node.get("spans") != "Expression":
                continue
            outgoing = [edge for edge in edges if edge["start"] == parent]
            if any(edge["label"] != "func.arg" for edge in outgoing):
                continue
            args = sorted_func_args(sample, parent)
            if len(args) != 2:
                continue
            left, comma = args
            if not is_visible_operator(nodes[left]):
                continue
            if nodes[comma].get("type") != "PunctuationalConstant" or node_text(sample, comma) != ",":
                continue
            if not only_incoming_from_parent(sample, comma, parent):
                continue
            if any(edge["start"] == comma for edge in edges):
                continue
            left_tokens = descendant_tokens(sample, left)
            comma_tokens = node_token_indexes(nodes[comma])
            if not left_tokens or len(comma_tokens) != 1 or max(left_tokens) >= comma_tokens[0]:
                continue
            redirect_incoming(sample, parent, comma)
            remove_edges(edges, start=parent)
            add_edge(edges, comma, left, "as:punctuation")
            delete_nodes(sample, {parent})
            changes.append("trailing_expression_comma")
            made_change = True
            break
        if not made_change:
            break
    return changes


def repair_unclosed_pair_punctuation_children(sample: dict[str, Any]) -> list[str]:
    """Attach paired punctuation children to the expression material they wrap."""
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    for parent, node in enumerate(list(nodes)):
        if node.get("type") != "ExpressionFunctor" or node.get("spans") != "Expression":
            continue
        args = sorted_func_args(sample, parent)
        for arg in list(args):
            if nodes[arg].get("type") != "PunctuationalConstant":
                continue
            mark = node_text(sample, arg)
            if mark not in {'" "', "<>", "( )"}:
                continue
            if any(edge["start"] == arg for edge in edges):
                continue
            punct_tokens = node_token_indexes(nodes[arg])
            if len(punct_tokens) != 2:
                continue
            open_token, close_token = min(punct_tokens), max(punct_tokens)
            enclosed = [
                candidate
                for candidate in args
                if candidate != arg
                and descendant_tokens(sample, candidate)
                and min(descendant_tokens(sample, candidate)) > open_token
                and max(descendant_tokens(sample, candidate)) < close_token
            ]
            if not enclosed:
                continue
            for child in enclosed:
                remove_edges(edges, start=parent, end=child, label="func.arg")
            if len(enclosed) == 1:
                target = enclosed[0]
            else:
                target = len(nodes)
                nodes.append({"spans": "Expression", "type": "ExpressionFunctor"})
                for child in enclosed:
                    add_edge(edges, target, child, "func.arg")
            add_edge(edges, arg, target, "as:punctuation")
            changes.append(f"pair_close:{mark}")
    if changes:
        graph(sample)["edges"] = dedupe_edges(edges)
    return changes


def repair_possessive_apostrophe_children(sample: dict[str, Any]) -> list[str]:
    """Use apostrophe as the visible operator for tokenized possessives like Chad ' s."""
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    for parent, node in enumerate(nodes):
        if node.get("type") != "ExpressionFunctor" or node.get("spans") != "Expression":
            continue
        args = sorted_func_args(sample, parent)
        for pos, arg in enumerate(args):
            if nodes[arg].get("type") != "PunctuationalConstant" or node_text(sample, arg) != "'":
                continue
            if pos == 0 or pos + 1 >= len(args):
                continue
            if not only_incoming_from_parent(sample, arg, parent):
                continue
            if any(edge["start"] == arg for edge in edges):
                continue
            left = args[pos - 1]
            right = args[pos + 1]
            if node_text(sample, right).lower() != "s":
                continue
            nodes[arg]["type"] = "ExpressionFunctor"
            remove_edges(edges, start=parent, end=left, label="func.arg")
            remove_edges(edges, start=parent, end=right, label="func.arg")
            add_edge(edges, arg, left, "func.arg")
            add_edge(edges, arg, right, "func.arg")
            changes.append("possessive_apostrophe")
    if changes:
        graph(sample)["edges"] = dedupe_edges(edges)
    return changes


def repair_aux_list_separator_children(sample: dict[str, Any]) -> list[str]:
    """Use explicit list separators as the ListFunctor root instead of List aux children."""
    changes: list[str] = []
    while True:
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        made_change = False
        for parent, node in enumerate(nodes):
            if node.get("type") != "ListFunctor" or node.get("spans") != "List":
                continue
            outgoing = [edge for edge in edges if edge["start"] == parent]
            if any(edge["label"] != "func.arg" for edge in outgoing):
                continue
            args = sorted_func_args(sample, parent)
            punct_args = [
                arg
                for arg in args
                if nodes[arg].get("type") == "PunctuationalConstant"
                and node_text(sample, arg) in LIST_SEPARATOR_TEXTS
            ]
            if not punct_args:
                continue
            punct_texts = {node_text(sample, arg) for arg in punct_args}
            if len(punct_texts) != 1:
                continue
            if not all(only_incoming_from_parent(sample, arg, parent) for arg in punct_args):
                continue

            item_args: list[int] = []
            for arg in args:
                if arg in punct_args:
                    punct_out = [edge for edge in edges if edge["start"] == arg]
                    close_targets = [edge["end"] for edge in punct_out if edge["label"] == "as:punctuation"]
                    if len(punct_out) != len(close_targets) or not close_targets:
                        item_args = []
                        break
                    item_args.extend(close_targets)
                else:
                    item_args.append(arg)
            if len(item_args) < 2:
                continue

            root = punct_args[0]
            nodes[root]["type"] = "ListFunctor"
            separator_tokens: list[int] = []
            for punct in punct_args:
                separator_tokens.extend(node_token_indexes(nodes[punct]))
            nodes[root]["spans"] = spans_for_tokens(separator_tokens)

            redirect_incoming(sample, parent, root)
            remove_edges(edges, start=parent)
            for punct in punct_args:
                remove_edges(edges, start=punct)
            for item in item_args:
                if item != root and item not in punct_args[1:]:
                    add_edge(edges, root, item, "func.arg")

            delete_nodes(sample, {parent, *punct_args[1:]})
            changes.append(f"list_separator:{next(iter(punct_texts))}")
            made_change = True
            break
        if not made_change:
            break
    return changes


def repair_leading_marker_list_child(sample: dict[str, Any]) -> list[str]:
    """Move leading decorative markers such as `--` outside the list as wrappers."""
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    for parent, node in enumerate(nodes):
        if node.get("type") != "ListFunctor" or node.get("spans") != "List":
            continue
        args = sorted_func_args(sample, parent)
        if not args:
            continue
        marker = args[0]
        if nodes[marker].get("type") != "PunctuationalConstant" or node_text(sample, marker) != "--":
            continue
        if not only_incoming_from_parent(sample, marker, parent):
            continue
        if any(edge["start"] == marker for edge in edges):
            continue
        marker_tokens = node_token_indexes(nodes[marker])
        other_tokens = [token for arg in args[1:] for token in descendant_tokens(sample, arg)]
        if not marker_tokens or not other_tokens or min(marker_tokens) > min(other_tokens):
            continue
        remove_edges(edges, start=parent, end=marker, label="func.arg")
        redirect_incoming(sample, parent, marker)
        add_edge(edges, marker, parent, "as:punctuation")
        changes.append("leading_marker_wrapper")
    if changes:
        graph(sample)["edges"] = dedupe_edges(edges)
    return changes


def repair_nested_address_comma_child(sample: dict[str, Any]) -> list[str]:
    """Repair address chains `A, B, C` where the inner comma is still punctuation."""
    changes: list[str] = []
    while True:
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        made_change = False
        for parent, node in enumerate(nodes):
            if node.get("type") != "ExpressionFunctor" or node_text(sample, parent) != ",":
                continue
            args = sorted_func_args(sample, parent)
            punct_args = [
                arg
                for arg in args
                if nodes[arg].get("type") == "PunctuationalConstant" and node_text(sample, arg) == ","
            ]
            left_args = [arg for arg in args if arg not in punct_args]
            if len(punct_args) != 1 or len(left_args) != 1:
                continue
            punct = punct_args[0]
            close_targets = [
                edge["end"] for edge in edges if edge["start"] == punct and edge["label"] == "as:punctuation"
            ]
            if len(close_targets) != 1:
                continue
            mod_indexes = [
                index
                for index, candidate in enumerate(nodes)
                if candidate.get("type") == "ModificationalFunctor"
                and candidate.get("spans") == "Modification"
                and any(edge["start"] == index and edge["end"] == parent and edge["label"] == "variable" for edge in edges)
            ]
            if len(mod_indexes) != 1:
                continue
            mod_index = mod_indexes[0]
            bodies = [edge["end"] for edge in edges if edge["start"] == mod_index and edge["label"] == "body"]
            if len(bodies) != 1:
                continue

            left = left_args[0]
            middle = close_targets[0]
            right = bodies[0]
            nodes[punct]["type"] = "ExpressionFunctor"
            remove_edges(edges, start=punct, label="as:punctuation")
            add_edge(edges, punct, left, "func.arg")
            add_edge(edges, punct, middle, "func.arg")
            remove_edges(edges, start=parent, end=left, label="func.arg")
            add_edge(edges, parent, right, "func.arg")
            redirect_incoming(sample, mod_index, parent)
            remove_edges(edges, start=mod_index)
            delete_nodes(sample, {mod_index})
            changes.append("nested_address_comma")
            made_change = True
            break
        if not made_change:
            break
    return changes


def repair_letter_child_mistyped_as_punctuation(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    for index, node in enumerate(graph(sample)["nodes"]):
        if node.get("type") != "PunctuationalConstant":
            continue
        text = node_text(sample, index)
        if re.fullmatch(r"[A-Za-z]", text):
            node["type"] = "NominalConstant"
            changes.append("letter_punct_type")
    return changes


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n" for sample in samples),
        encoding="utf-8",
    )


def main() -> None:
    total = Counter()
    examples: list[str] = []
    for path in DATA_FILES:
        samples = load_jsonl(path)
        file_changes = 0
        for line_no, sample in enumerate(samples, 1):
            changes = []
            changes.extend(collapse_single_child_operator_wrappers(sample))
            changes.extend(collapse_redundant_operator_wrappers(sample))
            changes.extend(repair_rank_marker_punctuation_children(sample))
            changes.extend(repair_bare_abbreviation_dot_children(sample))
            changes.extend(repair_closed_abbreviation_dot_children(sample))
            changes.extend(repair_bare_expression_comma_children(sample))
            changes.extend(repair_closed_expression_comma_children(sample))
            changes.extend(repair_trailing_expression_comma_children(sample))
            changes.extend(repair_unclosed_pair_punctuation_children(sample))
            changes.extend(repair_possessive_apostrophe_children(sample))
            changes.extend(repair_aux_list_separator_children(sample))
            changes.extend(repair_leading_marker_list_child(sample))
            changes.extend(repair_nested_address_comma_child(sample))
            changes.extend(repair_letter_child_mistyped_as_punctuation(sample))
            if not changes:
                continue
            file_changes += len(changes)
            total[path.name] += len(changes)
            if len(examples) < 20:
                examples.append(f"{path.name}:{line_no}: {', '.join(changes[:4])}")
        if file_changes:
            write_jsonl(path, samples)
    print("collapsed wrappers:", dict(total))
    if examples:
        print("examples:")
        for example in examples:
            print(example)


if __name__ == "__main__":
    main()
