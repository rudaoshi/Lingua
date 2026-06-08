#!/usr/bin/env python3
"""Repair high-confidence Expression/List consistency issues in gold lingua data."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = [
    ROOT / "data" / "train.jsonl",
    ROOT / "data" / "dev.jsonl",
    ROOT / "data" / "test.jsonl",
]


def load_expression_repair_module():
    path = ROOT / "scripts" / "repair_expression_format_functors.py"
    spec = importlib.util.spec_from_file_location("expr_repair", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPR_REPAIR = load_expression_repair_module()


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if not isinstance(spans, list):
        return []
    return [index for start, end in spans for index in range(start, end + 1)]


def spans_for_tokens(tokens: list[int]) -> list[list[int]]:
    tokens = sorted(tokens)
    spans: list[list[int]] = []
    start = prev = tokens[0]
    for token in tokens[1:]:
        if token == prev + 1:
            prev = token
        else:
            spans.append([start, prev])
            start = prev = token
    spans.append([start, prev])
    return spans


def token_text(sample: dict[str, Any], token_index: int) -> str:
    return sample["lingua"]["words"][token_index].strip()


def node_text(sample: dict[str, Any], node_index: int) -> str:
    node = graph(sample)["nodes"][node_index]
    spans = node["spans"]
    if isinstance(spans, str):
        return spans
    words = sample["lingua"]["words"]
    return "".join(words[index] for index in node_token_indexes(node))


def clean_node_text(sample: dict[str, Any], node_index: int) -> str:
    return node_text(sample, node_index).strip()


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return edge["start"], edge["end"], edge["label"]


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
) -> int:
    kept: list[dict[str, Any]] = []
    removed = 0
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
        removed += 1
    edges[:] = kept
    return removed


def dedupe_edges(sample: dict[str, Any]) -> None:
    graph(sample)["edges"] = EXPR_REPAIR.remove_duplicate_edges(graph(sample)["edges"])


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


def descendant_range(sample: dict[str, Any], node_index: int) -> tuple[int, int] | None:
    tokens = descendant_tokens(sample, node_index)
    if not tokens:
        return None
    return min(tokens), max(tokens)


def outgoing_edges(sample: dict[str, Any], node_index: int) -> list[dict[str, Any]]:
    return [edge for edge in graph(sample)["edges"] if edge["start"] == node_index]


def list_args(sample: dict[str, Any], source: int) -> list[int]:
    return [
        edge["end"]
        for edge in graph(sample)["edges"]
        if edge["start"] == source and edge["label"] == "func.arg"
    ]


def sorted_list_args(sample: dict[str, Any], source: int) -> list[int]:
    def sort_key(index: int) -> tuple[int, int]:
        rng = descendant_range(sample, index)
        return rng if rng is not None else (10**9, 10**9)

    return sorted(list_args(sample, source), key=sort_key)


def has_edge(sample: dict[str, Any], start: int, end: int, label: str) -> bool:
    return any(
        edge["start"] == start and edge["end"] == end and edge["label"] == label
        for edge in graph(sample)["edges"]
    )


def is_numeric_node(sample: dict[str, Any], node_index: int) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", clean_node_text(sample, node_index)))


def repair_decimal_dot_list_item(sample: dict[str, Any]) -> list[str]:
    """Convert standings-style `.535` dots from close punctuation to expression functors."""
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    changes: list[str] = []

    for list_index, node in enumerate(nodes):
        if node["type"] != "ListFunctor":
            continue
        args = sorted_list_args(sample, list_index)
        for pos, arg in enumerate(args):
            if clean_node_text(sample, arg) != ".":
                continue
            if nodes[arg]["type"] != "PunctuationalConstant":
                continue
            arg_tokens = node_token_indexes(nodes[arg])
            if len(arg_tokens) != 1 or pos + 1 >= len(args):
                continue
            next_arg = args[pos + 1]
            if descendant_range(sample, next_arg) != (arg_tokens[0] + 1, arg_tokens[0] + 1):
                continue
            if not is_numeric_node(sample, next_arg):
                continue
            closed = [
                edge["end"]
                for edge in edges
                if edge["start"] == arg and edge["label"] == "as:punctuation"
            ]
            if not closed or not all(is_numeric_node(sample, closed_arg) for closed_arg in closed):
                continue

            nodes[arg]["type"] = "ExpressionFunctor"
            for closed_arg in closed:
                add_edge(edges, list_index, closed_arg, "func.arg")
            remove_edges(edges, start=arg, label="as:punctuation")
            remove_edges(edges, start=list_index, end=next_arg, label="func.arg")
            add_edge(edges, arg, next_arg, "func.arg")
            changes.append(f"decimal_dot:{list_index}->{arg}")

    if changes:
        dedupe_edges(sample)
    return changes


def repair_decimal_dot_orphan_left_arg(sample: dict[str, Any]) -> list[str]:
    """Repair data already converted by the decimal-dot rule before the left item patch."""
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    changes: list[str] = []

    for list_index, node in enumerate(nodes):
        if node["type"] != "ListFunctor":
            continue
        args = sorted_list_args(sample, list_index)
        arg_set = set(args)
        candidate_dots = set(args)
        for arg in args:
            candidate_dots.update(
                edge["end"]
                for edge in edges
                if edge["start"] == arg and edge["label"] == "as:punctuation"
            )
        for arg in sorted(candidate_dots):
            if nodes[arg]["type"] != "ExpressionFunctor":
                continue
            if clean_node_text(sample, arg) != ".":
                continue
            tokens = node_token_indexes(nodes[arg])
            if len(tokens) != 1 or tokens[0] == 0:
                continue
            left_token = tokens[0] - 1
            owner = EXPR_REPAIR.owner_by_token(nodes).get(left_token)
            if owner is None or owner in arg_set:
                continue
            if not is_numeric_node(sample, owner):
                continue
            if any(edge["end"] == owner for edge in edges):
                continue
            add_edge(edges, list_index, owner, "func.arg")
            changes.append(f"decimal_left_arg:{list_index}->{owner}")

    if changes:
        dedupe_edges(sample)
    return changes


LIST_CLOSE_MARKS = {"-", "--", "–", "—", "...", "/", "|", ":", ";"}


def repair_list_delimiter_close(sample: dict[str, Any]) -> list[str]:
    """Make list delimiters close the preceding item instead of standing alone."""
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    changes: list[str] = []

    for list_index, node in enumerate(nodes):
        if node["type"] != "ListFunctor":
            continue
        args = sorted_list_args(sample, list_index)
        for pos, arg in enumerate(args):
            if pos == 0:
                continue
            mark = clean_node_text(sample, arg)
            if mark not in LIST_CLOSE_MARKS:
                continue
            if any(edge["start"] == arg for edge in edges):
                continue
            if nodes[arg]["type"] not in {
                "OtherConstant",
                "PunctuationalConstant",
                "SymbolConstant",
                "ExpressionFunctor",
            }:
                continue
            prev_arg = args[pos - 1]
            if has_edge(sample, arg, prev_arg, "as:punctuation"):
                continue
            remove_edges(edges, start=list_index, end=prev_arg, label="func.arg")
            nodes[arg]["type"] = "PunctuationalConstant"
            add_edge(edges, arg, prev_arg, "as:punctuation")
            changes.append(f"list_delimiter_close:{list_index}->{arg}")

    if changes:
        dedupe_edges(sample)
    return changes


def repair_hyphen_slash_format_functors(sample: dict[str, Any]) -> list[str]:
    """Clean visible hyphen/slash expression functors that still use modification edges."""
    nodes = graph(sample)["nodes"]
    words = sample["lingua"]["words"]
    changes: list[str] = []

    for node_index, node in enumerate(list(nodes)):
        if node_index >= len(nodes):
            continue
        if node["type"] != "ExpressionFunctor" or isinstance(node["spans"], str):
            continue
        tokens = node_token_indexes(node)
        if len(tokens) != 1:
            continue
        token_index = tokens[0]
        mark = token_text(sample, token_index)
        if mark not in {"-", "–", "—", "/"}:
            continue
        # A leading-space operator is usually a delimiter or unary sign in this
        # tokenization, not an internal hyphen/slash format functor.
        if sample["lingua"]["words"][token_index] != mark:
            continue
        labels = [edge["label"] for edge in outgoing_edges(sample, node_index)]
        if labels.count("func.arg") >= 2 and all(label == "func.arg" for label in labels):
            continue
        group_tokens = EXPR_REPAIR.hyphen_sequence_around(words, token_index)
        if not group_tokens:
            continue
        compact = "".join(token_text(sample, token) for token in group_tokens)
        if not EXPR_REPAIR.high_confidence_hyphen_expression(compact, allow_proper=True):
            continue
        before = json.dumps(graph(sample), ensure_ascii=False, sort_keys=True)
        if EXPR_REPAIR.rewrite_existing_token_group(sample, group_tokens):
            after = json.dumps(graph(sample), ensure_ascii=False, sort_keys=True)
            if before == after:
                continue
            changes.append(f"hyphen_slash_functor:{node_index}")

    if changes:
        dedupe_edges(sample)
    return changes


MONTHS = EXPR_REPAIR.MONTHS


def repair_date_comma_aux_expression(sample: dict[str, Any]) -> list[str]:
    """Nest month-day-year commas as format functors under the enclosing Expression."""
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    changes: list[str] = []

    for expr_index, node in enumerate(nodes):
        if node["type"] != "ExpressionFunctor" or not isinstance(node["spans"], str):
            continue
        args = sorted_list_args(sample, expr_index)
        for pos, arg in enumerate(args):
            if clean_node_text(sample, arg) != ",":
                continue
            if nodes[arg]["type"] != "PunctuationalConstant":
                continue
            if pos == 0 or pos + 1 >= len(args):
                continue
            prev_arg = args[pos - 1]
            next_arg = args[pos + 1]
            if not re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)?", clean_node_text(sample, prev_arg)):
                continue
            if not re.fullmatch(r"\d{4}", clean_node_text(sample, next_arg)):
                continue
            has_month = any(clean_node_text(sample, earlier).rstrip(".") in MONTHS for earlier in args[:pos])
            if not has_month:
                continue
            nodes[arg]["type"] = "ExpressionFunctor"
            remove_edges(edges, start=expr_index, end=prev_arg, label="func.arg")
            remove_edges(edges, start=expr_index, end=next_arg, label="func.arg")
            add_edge(edges, arg, prev_arg, "func.arg")
            add_edge(edges, arg, next_arg, "func.arg")
            changes.append(f"date_comma:{expr_index}->{arg}")

    if changes:
        dedupe_edges(sample)
    return changes


def repair_fractional_time_dot(sample: dict[str, Any]) -> list[str]:
    """Convert the fractional dot in time/score values such as `1:18.52`."""
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    changes: list[str] = []

    for expr_index, node in enumerate(nodes):
        if node["type"] != "ExpressionFunctor" or not isinstance(node["spans"], str):
            continue
        args = sorted_list_args(sample, expr_index)
        for pos, arg in enumerate(args):
            if clean_node_text(sample, arg) != ".":
                continue
            if nodes[arg]["type"] != "PunctuationalConstant":
                continue
            if pos + 1 >= len(args):
                continue
            next_arg = args[pos + 1]
            if not is_numeric_node(sample, next_arg):
                continue
            closed = [
                edge["end"]
                for edge in edges
                if edge["start"] == arg and edge["label"] == "as:punctuation"
            ]
            if len(closed) != 1:
                continue
            closed_arg = closed[0]
            if nodes[closed_arg]["type"] != "ExpressionFunctor":
                continue
            if clean_node_text(sample, closed_arg) != ":":
                continue

            nodes[arg]["type"] = "ExpressionFunctor"
            remove_edges(edges, start=arg, end=closed_arg, label="as:punctuation")
            remove_edges(edges, start=expr_index, end=next_arg, label="func.arg")
            add_edge(edges, arg, closed_arg, "func.arg")
            add_edge(edges, arg, next_arg, "func.arg")
            changes.append(f"fractional_time_dot:{expr_index}->{arg}")

    if changes:
        dedupe_edges(sample)
    return changes


def repair_sample(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    changes.extend(repair_decimal_dot_list_item(sample))
    changes.extend(repair_decimal_dot_orphan_left_arg(sample))
    changes.extend(repair_list_delimiter_close(sample))
    changes.extend(repair_hyphen_slash_format_functors(sample))
    changes.extend(repair_date_comma_aux_expression(sample))
    changes.extend(repair_fractional_time_dot(sample))
    return changes


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    total = 0
    by_kind: dict[str, int] = {}
    by_file: dict[str, int] = {}
    for path in DATA_FILES:
        rows = read_jsonl(path)
        file_total = 0
        for sample in rows:
            before = json.dumps(sample["lingua"]["graph"], ensure_ascii=False, sort_keys=True)
            changes = repair_sample(sample)
            after = json.dumps(sample["lingua"]["graph"], ensure_ascii=False, sort_keys=True)
            if before == after:
                continue
            file_total += len(changes)
            for change in changes:
                kind = change.split(":", 1)[0]
                by_kind[kind] = by_kind.get(kind, 0) + 1
        if file_total:
            write_jsonl(path, rows)
        by_file[path.name] = file_total
        total += file_total

    print(f"repaired {total} gold Expression/List consistency issues")
    for name, count in by_file.items():
        print(f"{name}: {count}")
    for kind, count in sorted(by_kind.items()):
        print(f"{kind}: {count}")


if __name__ == "__main__":
    main()
