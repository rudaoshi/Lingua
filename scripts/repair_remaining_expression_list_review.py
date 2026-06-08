#!/usr/bin/env python3
"""Manual-reviewed repairs for remaining Expression/List consistency cases."""

from __future__ import annotations

import importlib.util
import csv
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = [
    ROOT / "data" / "train.jsonl",
    ROOT / "data" / "dev.jsonl",
    ROOT / "data" / "test.jsonl",
]
CANDIDATE_TSV = ROOT / "reports" / "expression_separator_remaining_candidates.tsv"


def load_expression_repair_module():
    path = ROOT / "scripts" / "repair_expression_format_functors.py"
    spec = importlib.util.spec_from_file_location("expr_repair", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPR_REPAIR = load_expression_repair_module()


def load_candidate_tokens(category: str) -> set[tuple[str, int, int]]:
    if not CANDIDATE_TSV.exists():
        return set()
    tokens: set[tuple[str, int, int]] = set()
    with CANDIDATE_TSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("category") != category:
                continue
            token_index = row.get("token_index", "")
            if not token_index.isdigit():
                continue
            tokens.add((row["split"], int(row["line"]), int(token_index)))
    return tokens


def load_candidate_nodes(categories: set[str]) -> set[tuple[str, int, int]]:
    if not CANDIDATE_TSV.exists():
        return set()
    nodes: set[tuple[str, int, int]] = set()
    with CANDIDATE_TSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("category") not in categories:
                continue
            node_index = row.get("node_index", "")
            if not node_index.isdigit():
                continue
            nodes.add((row["split"], int(row["line"]), int(node_index)))
    return nodes


ADDRESS_COMMA_TOKENS = load_candidate_tokens("address_place_comma")
DATE_COMMA_TOKENS = load_candidate_tokens("date_comma")
CURRENCY_SYMBOL_TOKENS = load_candidate_tokens("currency_percent_rank_symbol")
INITIAL_DOT_TOKENS = load_candidate_tokens("initial_abbrev_dot")
HYPHEN_SEPARATOR_TOKENS = load_candidate_tokens("hyphenated_name_code") | load_candidate_tokens(
    "numeric_date_range_dash"
)
MERGED_HYPHEN_NODES = load_candidate_nodes({"merged_hyphen_slash_name_code", "hyphenated_name_code"})


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if isinstance(spans, str):
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
    if isinstance(node["spans"], str):
        return node["spans"]
    return "".join(sample["lingua"]["words"][index] for index in node_token_indexes(node))


def clean_node_text(sample: dict[str, Any], node_index: int) -> str:
    return node_text(sample, node_index).strip()


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


def descendant_text(sample: dict[str, Any], node_index: int) -> str:
    return "".join(sample["lingua"]["words"][index] for index in descendant_tokens(sample, node_index))


def first_descendant_token(sample: dict[str, Any], node_index: int) -> int | None:
    tokens = descendant_tokens(sample, node_index)
    return tokens[0] if tokens else None


def first_own_token(sample: dict[str, Any], node_index: int) -> int | None:
    tokens = node_token_indexes(graph(sample)["nodes"][node_index])
    return tokens[0] if tokens else None


def last_descendant_token(sample: dict[str, Any], node_index: int) -> int | None:
    tokens = descendant_tokens(sample, node_index)
    return tokens[-1] if tokens else None


def outgoing_edges(sample: dict[str, Any], node_index: int) -> list[dict[str, Any]]:
    return [edge for edge in graph(sample)["edges"] if edge["start"] == node_index]


def incoming_edges(sample: dict[str, Any], node_index: int) -> list[dict[str, Any]]:
    return [edge for edge in graph(sample)["edges"] if edge["end"] == node_index]


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
    graph(sample)["edges"] = EXPR_REPAIR.remove_duplicate_edges(new_edges)


def redirect_incoming(sample: dict[str, Any], old: int, new: int) -> None:
    for edge in graph(sample)["edges"]:
        if edge["end"] == old:
            edge["end"] = new


def arg_edges(sample: dict[str, Any], source: int, label: str = "func.arg") -> list[int]:
    return [edge["end"] for edge in graph(sample)["edges"] if edge["start"] == source and edge["label"] == label]


def sorted_args(sample: dict[str, Any], source: int, label: str = "func.arg") -> list[int]:
    def key(index: int) -> tuple[int, int]:
        tokens = descendant_tokens(sample, index)
        if not tokens:
            return 10**9, 10**9
        return tokens[0], tokens[-1]

    return sorted(arg_edges(sample, source, label), key=key)


def owner_by_token(sample: dict[str, Any]) -> dict[int, int]:
    owner: dict[int, int] = {}
    for index, node in enumerate(graph(sample)["nodes"]):
        if isinstance(node["spans"], str):
            continue
        for token in node_token_indexes(node):
            owner[token] = index
    return owner


def clean_desc(sample: dict[str, Any], node_index: int) -> str:
    return re.sub(r"\s+", " ", descendant_text(sample, node_index).strip())


def looks_month_day(sample: dict[str, Any], node_index: int) -> bool:
    text = clean_desc(sample, node_index)
    return bool(
        re.search(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|"
            r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?$",
            text,
        )
    )


def looks_year(sample: dict[str, Any], node_index: int) -> bool:
    return bool(re.fullmatch(r"\d{4}", clean_node_text(sample, node_index)))


def looks_place_comma(sample: dict[str, Any], comma: int, left: int, right: int) -> bool:
    if clean_node_text(sample, comma) != ",":
        return False
    left_text = clean_desc(sample, left)
    right_text = clean_desc(sample, right)
    if not left_text or not right_text:
        return False
    if len(left_text.split()) > 5 or len(right_text.split()) > 4:
        return False
    if not re.search(r"[A-ZÀ-ÖØ-Þ]", left_text):
        return False
    if not re.search(r"[A-ZÀ-ÖØ-Þ]", right_text):
        return False
    if re.search(r"\b(and|or|but|which|that|who|when|because)\b", left_text, re.I):
        return False
    return True


def expressionify_separator_with_args(
    sample: dict[str, Any],
    separator: int,
    args: list[int],
    *,
    old_labels_to_remove: set[str] | None = None,
) -> bool:
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    nodes[separator]["type"] = "ExpressionFunctor"
    old_labels_to_remove = old_labels_to_remove or {"as:punctuation", "as:modification"}
    for label in old_labels_to_remove:
        remove_edges(edges, start=separator, label=label)
    for arg in args:
        remove_edges(edges, start=separator, end=arg)
    for arg in args:
        if arg != separator:
            add_edge(edges, separator, arg, "func.arg")
    dedupe_edges(sample)
    return True


def repair_merged_hyphen_nodes(sample: dict[str, Any], sample_key: tuple[str, int]) -> list[str]:
    changes: list[str] = []
    for index, node in enumerate(list(graph(sample)["nodes"])):
        if index >= len(graph(sample)["nodes"]):
            break
        if node["type"] not in {"NominalConstant", "ModificationalConstant"}:
            continue
        if (sample_key[0], sample_key[1], index) not in MERGED_HYPHEN_NODES:
            continue
        if isinstance(node["spans"], str):
            continue
        tokens = node_token_indexes(node)
        if len(tokens) < 3:
            continue
        separator_tokens = [
            token
            for token in tokens
            if token_text(sample, token) in {"-", "–", "—", "/"}
        ]
        if not separator_tokens:
            continue
        if any(sample["lingua"]["words"][token] != token_text(sample, token) for token in separator_tokens):
            continue
        compact = "".join(token_text(sample, token) for token in tokens)
        if not re.search(r"[-/–—]", compact):
            continue
        if not re.fullmatch(r"[A-Za-z0-9À-ÖØ-öø-ÿ]+(?:[-/–—][A-Za-z0-9À-ÖØ-öø-ÿ]+)+", compact):
            continue
        before = json.dumps(graph(sample), ensure_ascii=False, sort_keys=True)
        if EXPR_REPAIR.rewrite_merged_node(sample, index):
            after = json.dumps(graph(sample), ensure_ascii=False, sort_keys=True)
            if before != after:
                changes.append(f"merged_hyphen:{compact}")
    if changes:
        dedupe_edges(sample)
    return changes


def repair_modifier_close_separator(sample: dict[str, Any], sample_key: tuple[str, int]) -> list[str]:
    """Turn `A-` close punctuation plus Modification into an A-B expression."""
    changes: list[str] = []

    while True:
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        made_change = False
        for separator, node in enumerate(nodes):
            mark = clean_node_text(sample, separator)
            if mark not in {"-", "–", "—", "/"}:
                continue
            close_edges = [
                edge for edge in edges if edge["start"] == separator and edge["label"] == "as:punctuation"
            ]
            if len(close_edges) != 1:
                continue
            left = close_edges[0]["end"]
            sep_token = first_own_token(sample, separator)
            if sep_token is None:
                continue
            if (sample_key[0], sample_key[1], sep_token) not in HYPHEN_SEPARATOR_TOKENS:
                continue
            modifiers = []
            for mod_index, mod_node in enumerate(nodes):
                if mod_node.get("spans") != "Modification":
                    continue
                labels = {(edge["label"], edge["end"]) for edge in edges if edge["start"] == mod_index}
                if ("body", separator) not in labels:
                    continue
                variables = [edge["end"] for edge in edges if edge["start"] == mod_index and edge["label"] == "variable"]
                if len(variables) != 1:
                    continue
                modifiers.append((mod_index, variables[0]))
            if not modifiers:
                continue
            mod_index, right = modifiers[0]
            left_end = last_descendant_token(sample, left)
            right_start = first_descendant_token(sample, right)
            if left_end is None or right_start is None or not (left_end < sep_token < right_start):
                continue

            redirect_incoming(sample, mod_index, separator)
            remove_edges(edges, start=mod_index)
            remove_edges(edges, start=separator, end=left, label="as:punctuation")
            expressionify_separator_with_args(sample, separator, [left, right])
            delete_nodes(sample, {mod_index})
            changes.append(f"modifier_close:{mark}")
            made_change = True
            break
        if not made_change:
            break
    return changes


def repair_variable_close_place_comma(sample: dict[str, Any], sample_key: tuple[str, int]) -> list[str]:
    """Repair `City , State` residuals where comma is the Modification variable."""
    changes: list[str] = []

    while True:
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        made_change = False
        for comma, node in enumerate(nodes):
            if clean_node_text(sample, comma) != ",":
                continue
            comma_token = first_own_token(sample, comma)
            if comma_token is None or (sample_key[0], sample_key[1], comma_token) not in ADDRESS_COMMA_TOKENS:
                continue
            close = [
                edge["end"]
                for edge in edges
                if edge["start"] == comma and edge["label"] == "as:punctuation"
            ]
            if len(close) != 1:
                continue
            left = close[0]
            modifiers = []
            for mod_index, mod_node in enumerate(nodes):
                if mod_node.get("spans") != "Modification":
                    continue
                variables = [edge["end"] for edge in edges if edge["start"] == mod_index and edge["label"] == "variable"]
                bodies = [edge["end"] for edge in edges if edge["start"] == mod_index and edge["label"] == "body"]
                if variables == [comma] and len(bodies) == 1:
                    modifiers.append((mod_index, bodies[0]))
            if not modifiers:
                continue
            mod_index, right = modifiers[0]
            if not looks_place_comma(sample, comma, left, right):
                continue
            redirect_incoming(sample, mod_index, comma)
            remove_edges(edges, start=mod_index)
            remove_edges(edges, start=comma, end=left, label="as:punctuation")
            expressionify_separator_with_args(sample, comma, [left, right])
            delete_nodes(sample, {mod_index})
            changes.append("place_comma_variable")
            made_change = True
            break
        if not made_change:
            break
    return changes


def repair_right_prefixed_hyphen_modifier(sample: dict[str, Any]) -> list[str]:
    """Repair shapes such as `I kappa B-alpha`, where the right node owns `-alpha`."""
    changes: list[str] = []

    while True:
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        owner = owner_by_token(sample)
        made_change = False
        for mod_index, mod_node in enumerate(nodes):
            if mod_node.get("spans") != "Modification":
                continue
            variables = [edge["end"] for edge in edges if edge["start"] == mod_index and edge["label"] == "variable"]
            bodies = [edge["end"] for edge in edges if edge["start"] == mod_index and edge["label"] == "body"]
            if len(variables) != 1 or len(bodies) != 1:
                continue
            right = variables[0]
            left = bodies[0]
            right_tokens = node_token_indexes(nodes[right])
            if len(right_tokens) < 2:
                continue
            first = right_tokens[0]
            if token_text(sample, first) not in {"-", "–", "—", "/"}:
                continue
            if first == 0 or owner.get(first - 1) != left:
                continue
            rest_tokens = right_tokens[1:]
            nodes[right]["spans"] = spans_for_tokens([first])
            nodes[right]["type"] = "ExpressionFunctor"
            nodes.append({"spans": spans_for_tokens(rest_tokens), "type": "NominalConstant"})
            rest = len(nodes) - 1
            redirect_incoming(sample, mod_index, right)
            remove_edges(edges, start=mod_index)
            add_edge(edges, right, left, "func.arg")
            add_edge(edges, right, rest, "func.arg")
            delete_nodes(sample, {mod_index})
            changes.append("right_prefixed_hyphen")
            made_change = True
            break
        if not made_change:
            break
    return changes


def repair_date_commas(sample: dict[str, Any], sample_key: tuple[str, int]) -> list[str]:
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]

    while True:
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        made_change = False
        for comma, node in enumerate(nodes):
            if clean_node_text(sample, comma) != ",":
                continue
            comma_token = first_own_token(sample, comma)
            if comma_token is None or (sample_key[0], sample_key[1], comma_token) not in DATE_COMMA_TOKENS:
                continue
            if node["type"] == "ExpressionFunctor":
                continue
            left_candidates = [
                index
                for index in range(len(nodes))
                if index != comma
                and looks_month_day(sample, index)
                and (last_descendant_token(sample, index) or -1) < comma_token
            ]
            year_candidates = [
                index
                for index in range(len(nodes))
                if index != comma
                and looks_year(sample, index)
                and first_own_token(sample, index) == comma_token + 1
            ]
            if not left_candidates or not year_candidates:
                continue
            left = max(left_candidates, key=lambda index: last_descendant_token(sample, index) or -1)
            year = min(year_candidates, key=lambda index: first_descendant_token(sample, index) or 10**9)
            close_targets = [
                edge["end"]
                for edge in edges
                if edge["start"] == comma and edge["label"] == "as:punctuation"
            ]
            if close_targets and any(target != left for target in close_targets):
                continue
            date_tokens = set(descendant_tokens(sample, left)) | set(descendant_tokens(sample, comma)) | set(
                descendant_tokens(sample, year)
            )
            containers = []
            for index, candidate in enumerate(nodes):
                if index in {left, comma, year}:
                    continue
                if candidate["type"] != "ExpressionFunctor" or not isinstance(candidate["spans"], str):
                    continue
                candidate_tokens = set(descendant_tokens(sample, index))
                if candidate_tokens and candidate_tokens <= date_tokens and comma_token in candidate_tokens:
                    containers.append(index)
            for container in containers:
                redirect_incoming(sample, container, comma)
            redirect_incoming(sample, year, comma)
            remove_edges(edges, start=year, end=comma, label="as:modification")
            remove_edges(edges, start=comma, label="as:punctuation")
            expressionify_separator_with_args(sample, comma, [left, year])
            delete_nodes(sample, set(containers))
            changes.append("date_comma_container")
            made_change = True
            break
        if not made_change:
            break

    for expr_index, node in enumerate(list(nodes)):
        if expr_index >= len(nodes):
            break
        if node["type"] != "ExpressionFunctor" or not isinstance(node["spans"], str):
            continue
        args = sorted_args(sample, expr_index)
        for pos, arg in enumerate(args):
            if clean_node_text(sample, arg) != ",":
                continue
            comma_token = first_own_token(sample, arg)
            if comma_token is None or (sample_key[0], sample_key[1], comma_token) not in DATE_COMMA_TOKENS:
                continue
            if pos == 0 or pos + 1 >= len(args):
                continue
            prev_arg = args[pos - 1]
            next_arg = args[pos + 1]
            if not looks_year(sample, next_arg):
                continue
            close_targets = [
                edge["end"]
                for edge in edges
                if edge["start"] == arg and edge["label"] == "as:punctuation"
            ]
            if re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)?", clean_node_text(sample, prev_arg)):
                day_arg = prev_arg
            elif len(close_targets) == 1 and re.fullmatch(
                r"\d{1,2}(?:st|nd|rd|th)?", clean_node_text(sample, close_targets[0])
            ):
                day_arg = close_targets[0]
            else:
                continue
            has_month = any(
                looks_month_day(sample, earlier) or clean_node_text(sample, earlier).rstrip(".") in EXPR_REPAIR.MONTHS
                for earlier in args[:pos]
            )
            has_month = has_month or any(
                edge["end"] == expr_index
                and edge["label"] == "as:modification"
                and clean_node_text(sample, edge["start"]).rstrip(".") in EXPR_REPAIR.MONTHS
                for edge in edges
            )
            if not has_month:
                continue
            nodes[arg]["type"] = "ExpressionFunctor"
            remove_edges(edges, start=arg, label="as:punctuation")
            if day_arg == prev_arg:
                remove_edges(edges, start=expr_index, end=prev_arg, label="func.arg")
            remove_edges(edges, start=expr_index, end=next_arg, label="func.arg")
            add_edge(edges, arg, day_arg, "func.arg")
            add_edge(edges, arg, next_arg, "func.arg")
            changes.append("date_comma_expr")

    while True:
        made_change = False
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        for comma, node in enumerate(nodes):
            if clean_node_text(sample, comma) != ",":
                continue
            comma_token = first_own_token(sample, comma)
            if comma_token is None or (sample_key[0], sample_key[1], comma_token) not in DATE_COMMA_TOKENS:
                continue
            close = [
                edge["end"]
                for edge in edges
                if edge["start"] == comma and edge["label"] == "as:punctuation"
            ]
            if len(close) != 1 or not re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)?", clean_node_text(sample, close[0])):
                continue
            day = close[0]
            direct_years = [
                edge["start"]
                for edge in edges
                if edge["end"] == comma and edge["label"] == "as:modification" and looks_year(sample, edge["start"])
            ]
            if direct_years:
                year = direct_years[0]
                redirect_incoming(sample, year, comma)
                remove_edges(edges, start=year, end=comma, label="as:modification")
                remove_edges(edges, start=comma, end=day, label="as:punctuation")
                expressionify_separator_with_args(sample, comma, [day, year])
                changes.append("date_comma_day_year")
                made_change = True
                break
            containers = []
            for mod_index, mod_node in enumerate(nodes):
                if mod_node.get("spans") != "Modification":
                    continue
                variables = [edge["end"] for edge in edges if edge["start"] == mod_index and edge["label"] == "variable"]
                bodies = [edge["end"] for edge in edges if edge["start"] == mod_index and edge["label"] == "body"]
                if bodies == [comma] and len(variables) == 1 and looks_year(sample, variables[0]):
                    containers.append((mod_index, variables[0]))
            if not containers:
                continue
            mod_index, year = containers[0]
            redirect_incoming(sample, mod_index, comma)
            for edge in list(edges):
                if edge["end"] == mod_index and edge["label"] == "as:modification":
                    edge["end"] = comma
            remove_edges(edges, start=mod_index)
            remove_edges(edges, start=comma, end=day, label="as:punctuation")
            expressionify_separator_with_args(sample, comma, [day, year])
            delete_nodes(sample, {mod_index})
            changes.append("date_comma_mod_container")
            made_change = True
            break
        if not made_change:
            break

    while True:
        made_change = False
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        for comma, node in enumerate(nodes):
            if clean_node_text(sample, comma) != ",":
                continue
            comma_token = first_own_token(sample, comma)
            if comma_token is None or (sample_key[0], sample_key[1], comma_token) not in DATE_COMMA_TOKENS:
                continue
            close = [
                edge["end"]
                for edge in edges
                if edge["start"] == comma and edge["label"] == "as:punctuation"
            ]
            if len(close) != 1:
                continue
            left = close[0]
            if not looks_month_day(sample, left):
                continue
            years = [
                edge["start"]
                for edge in edges
                if edge["end"] == comma and edge["label"] == "as:modification" and looks_year(sample, edge["start"])
            ]
            if len(years) != 1:
                continue
            year = years[0]
            final_punct = [
                edge["start"]
                for edge in edges
                if edge["end"] == year and edge["label"] == "as:punctuation"
            ]
            # In this residual date shape the year used to be the root of the
            # whole date ("March 28, 2005").  After the comma becomes the
            # format functor, outside edges must point at the comma root.
            redirect_incoming(sample, year, comma)
            remove_edges(edges, start=comma, end=left, label="as:punctuation")
            remove_edges(edges, start=year, end=comma, label="as:modification")
            for punct in final_punct:
                add_edge(edges, punct, comma, "as:punctuation")
            expressionify_separator_with_args(sample, comma, [left, year])
            changes.append("date_comma_close")
            made_change = True
            break
        if not made_change:
            break

    if changes:
        dedupe_edges(sample)
    return changes


def repair_address_commas_in_expression(sample: dict[str, Any], sample_key: tuple[str, int]) -> list[str]:
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]

    for expr_index, node in enumerate(list(nodes)):
        if expr_index >= len(nodes):
            break
        if node["type"] != "ExpressionFunctor" or not isinstance(node["spans"], str):
            continue
        args = sorted_args(sample, expr_index)
        for pos, arg in enumerate(args):
            if clean_node_text(sample, arg) != ",":
                continue
            comma_token = first_own_token(sample, arg)
            if comma_token is None or (sample_key[0], sample_key[1], comma_token) not in ADDRESS_COMMA_TOKENS:
                continue
            close = [
                edge["end"]
                for edge in edges
                if edge["start"] == arg and edge["label"] == "as:punctuation"
            ]
            if pos + 1 >= len(args):
                continue
            if len(close) == 1:
                left = close[0]
            elif pos > 0:
                left = args[pos - 1]
            else:
                continue
            right = args[pos + 1]
            if not looks_place_comma(sample, arg, left, right):
                continue
            nodes[arg]["type"] = "ExpressionFunctor"
            remove_edges(edges, start=arg, end=left, label="as:punctuation")
            if left == args[pos - 1]:
                remove_edges(edges, start=expr_index, end=left, label="func.arg")
            remove_edges(edges, start=expr_index, end=right, label="func.arg")
            add_edge(edges, arg, left, "func.arg")
            add_edge(edges, arg, right, "func.arg")
            changes.append("address_comma_expr")

    if changes:
        dedupe_edges(sample)
    return changes


def repair_place_comma_as_modification(sample: dict[str, Any], sample_key: tuple[str, int]) -> list[str]:
    changes: list[str] = []

    while True:
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        made_change = False
        for comma, node in enumerate(nodes):
            if clean_node_text(sample, comma) != ",":
                continue
            comma_token = first_own_token(sample, comma)
            if comma_token is None or (sample_key[0], sample_key[1], comma_token) not in ADDRESS_COMMA_TOKENS:
                continue
            close = [
                edge["end"]
                for edge in edges
                if edge["start"] == comma and edge["label"] == "as:punctuation"
            ]
            if len(close) != 1:
                continue
            left = close[0]
            right_candidates = [
                edge["start"]
                for edge in edges
                if edge["end"] == comma and edge["label"] == "as:modification"
            ]
            if len(right_candidates) != 1:
                continue
            right = right_candidates[0]
            if not looks_place_comma(sample, comma, left, right):
                continue
            redirect_incoming(sample, right, comma)
            remove_edges(edges, start=right, end=comma, label="as:modification")
            remove_edges(edges, start=comma, end=left, label="as:punctuation")
            final_punct = [
                edge["start"]
                for edge in edges
                if edge["end"] == right and edge["label"] == "as:punctuation"
            ]
            for punct in final_punct:
                remove_edges(edges, start=punct, end=right, label="as:punctuation")
                add_edge(edges, punct, comma, "as:punctuation")
            expressionify_separator_with_args(sample, comma, [left, right])
            changes.append("place_comma_as_mod")
            made_change = True
            break
        if not made_change:
            break
    return changes


def repair_currency_prefix_symbols(sample: dict[str, Any], sample_key: tuple[str, int]) -> list[str]:
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]

    for symbol, node in enumerate(nodes):
        if clean_node_text(sample, symbol) not in {"$", "#"}:
            continue
        symbol_token = first_own_token(sample, symbol)
        if symbol_token is None:
            continue
        if (sample_key[0], sample_key[1], symbol_token) not in CURRENCY_SYMBOL_TOKENS:
            continue
        if node["type"] == "ExpressionFunctor":
            continue
        outgoing = outgoing_edges(sample, symbol)
        as_mod = [edge["end"] for edge in outgoing if edge["label"] == "as:modification"]
        if len(as_mod) == 1:
            target = as_mod[0]
            sym_token = symbol_token
            target_token = first_own_token(sample, target)
            if sym_token is not None and target_token is not None and sym_token < target_token:
                remove_edges(edges, start=symbol, end=target, label="as:modification")
                nodes[symbol]["type"] = "ExpressionFunctor"
                add_edge(edges, symbol, target, "func.arg")
                changes.append(f"prefix_symbol:{clean_node_text(sample, symbol)}")
                continue
        incoming_as_mod = [edge["start"] for edge in edges if edge["end"] == symbol and edge["label"] == "as:modification"]
        if len(incoming_as_mod) == 1:
            value = incoming_as_mod[0]
            sym_token = symbol_token
            value_token = first_own_token(sample, value)
            if sym_token is not None and value_token is not None and sym_token < value_token:
                redirect_incoming(sample, value, symbol)
                remove_edges(edges, start=value, end=symbol, label="as:modification")
                nodes[symbol]["type"] = "ExpressionFunctor"
                add_edge(edges, symbol, value, "func.arg")
                changes.append(f"prefix_symbol_value:{clean_node_text(sample, symbol)}")

    if changes:
        dedupe_edges(sample)
    return changes


def repair_currency_aux_expressions(sample: dict[str, Any], sample_key: tuple[str, int]) -> list[str]:
    changes: list[str] = []

    while True:
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        made_change = False
        for expr_index, node in enumerate(nodes):
            if node["type"] != "ExpressionFunctor" or not isinstance(node["spans"], str):
                continue
            args = sorted_args(sample, expr_index)
            sign_args = []
            for arg in args:
                token = first_own_token(sample, arg)
                if token is None:
                    continue
                if (sample_key[0], sample_key[1], token) in CURRENCY_SYMBOL_TOKENS and clean_node_text(sample, arg) in {
                    "#",
                    "$",
                }:
                    sign_args.append(arg)
            if len(sign_args) != 1 or len(args) < 2:
                continue
            sign = sign_args[0]
            values = [arg for arg in args if arg != sign]
            if not values:
                continue
            sign_text = clean_node_text(sample, sign)
            nodes[sign]["type"] = "ExpressionFunctor"
            remove_edges(edges, start=sign)
            for edge in list(edges):
                if edge["start"] == expr_index and edge["label"] != "func.arg":
                    add_edge(edges, sign, edge["end"], edge["label"])
            redirect_incoming(sample, expr_index, sign)
            for value in values:
                add_edge(edges, sign, value, "func.arg")
            delete_nodes(sample, {expr_index})
            changes.append(f"currency_aux:{sign_text}")
            made_change = True
            break
        if not made_change:
            break
    return changes


def repair_initial_abbrev_dot(sample: dict[str, Any], sample_key: tuple[str, int]) -> list[str]:
    changes: list[str] = []

    while True:
        nodes = graph(sample)["nodes"]
        edges = graph(sample)["edges"]
        made_change = False
        for dot, node in enumerate(nodes):
            if clean_node_text(sample, dot) != ".":
                continue
            token = first_own_token(sample, dot)
            if token is None or (sample_key[0], sample_key[1], token) not in INITIAL_DOT_TOKENS:
                continue
            close = [
                edge["end"]
                for edge in edges
                if edge["start"] == dot and edge["label"] == "as:punctuation"
            ]
            if len(close) != 1:
                continue
            initial = close[0]
            modifiers = []
            for mod_index, mod_node in enumerate(nodes):
                if mod_node.get("spans") != "Modification":
                    continue
                variables = [edge["end"] for edge in edges if edge["start"] == mod_index and edge["label"] == "variable"]
                bodies = [edge["end"] for edge in edges if edge["start"] == mod_index and edge["label"] == "body"]
                if bodies == [dot] and len(variables) == 1:
                    modifiers.append((mod_index, variables[0]))
            if not modifiers:
                continue
            mod_index, name = modifiers[0]
            redirect_incoming(sample, mod_index, dot)
            remove_edges(edges, start=mod_index)
            remove_edges(edges, start=dot, end=initial, label="as:punctuation")
            expressionify_separator_with_args(sample, dot, [initial, name])
            delete_nodes(sample, {mod_index})
            changes.append("initial_abbrev_dot")
            made_change = True
            break
        if not made_change:
            break
    return changes


def repair_email_domain_dots(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]

    for expr_index, node in enumerate(list(nodes)):
        if node["type"] != "ExpressionFunctor" or not isinstance(node["spans"], str):
            continue
        args = sorted_args(sample, expr_index)
        for pos, arg in enumerate(args):
            if clean_node_text(sample, arg) != "." or nodes[arg]["type"] == "ExpressionFunctor":
                continue
            if pos == 0 or pos + 1 >= len(args):
                continue
            prev_arg = args[pos - 1]
            next_arg = args[pos + 1]
            if clean_node_text(sample, next_arg).lower() not in {"com", "net", "org", "edu", "gov", "uk"}:
                continue
            nodes[arg]["type"] = "ExpressionFunctor"
            remove_edges(edges, start=expr_index, end=prev_arg, label="func.arg")
            remove_edges(edges, start=expr_index, end=next_arg, label="func.arg")
            add_edge(edges, arg, prev_arg, "func.arg")
            add_edge(edges, arg, next_arg, "func.arg")
            changes.append("email_domain_dot")

    if changes:
        dedupe_edges(sample)
    return changes


def repair_time_range_dash(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]

    for dash, node in enumerate(nodes):
        if clean_node_text(sample, dash) not in {"-", "–", "—"}:
            continue
        if node["type"] == "ExpressionFunctor":
            continue
        close = [
            edge["end"]
            for edge in edges
            if edge["start"] == dash and edge["label"] == "as:punctuation"
        ]
        if len(close) != 1:
            continue
        left = close[0]
        left_text = clean_desc(sample, left)
        if not re.fullmatch(r"\d{1,2}:\d{2}", left_text):
            continue
        modifiers = [
            mod_index
            for mod_index, mod_node in enumerate(nodes)
            if mod_node.get("spans") == "Modification"
            and any(edge["start"] == mod_index and edge["end"] == dash and edge["label"] == "body" for edge in edges)
        ]
        if not modifiers:
            continue
        mod_index = modifiers[0]
        variables = [edge["end"] for edge in edges if edge["start"] == mod_index and edge["label"] == "variable"]
        if len(variables) != 1:
            continue
        right = variables[0]
        right_text = clean_desc(sample, right)
        if not re.fullmatch(r"\d{1,2}:\d{2}", right_text):
            continue
        redirect_incoming(sample, mod_index, dash)
        remove_edges(edges, start=mod_index)
        remove_edges(edges, start=dash, end=left, label="as:punctuation")
        expressionify_separator_with_args(sample, dash, [left, right])
        delete_nodes(sample, {mod_index})
        changes.append("time_range_dash")
        break

    return changes


def repair_sample(sample: dict[str, Any], sample_key: tuple[str, int]) -> list[str]:
    changes: list[str] = []
    changes.extend(repair_merged_hyphen_nodes(sample, sample_key))
    changes.extend(repair_right_prefixed_hyphen_modifier(sample))
    changes.extend(repair_modifier_close_separator(sample, sample_key))
    changes.extend(repair_variable_close_place_comma(sample, sample_key))
    changes.extend(repair_date_commas(sample, sample_key))
    changes.extend(repair_address_commas_in_expression(sample, sample_key))
    changes.extend(repair_place_comma_as_modification(sample, sample_key))
    changes.extend(repair_currency_prefix_symbols(sample, sample_key))
    changes.extend(repair_currency_aux_expressions(sample, sample_key))
    changes.extend(repair_initial_abbrev_dot(sample, sample_key))
    changes.extend(repair_email_domain_dots(sample))
    changes.extend(repair_time_range_dash(sample))
    return changes


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    total_samples = 0
    total_changes = 0
    by_file: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}

    for path in DATA_FILES:
        rows = read_jsonl(path)
        changed = False
        for line_number, sample in enumerate(rows, 1):
            before = deepcopy(sample["lingua"]["graph"])
            changes = repair_sample(sample, (path.name, line_number))
            if sample["lingua"]["graph"] == before:
                continue
            changed = True
            total_samples += 1
            total_changes += len(changes)
            by_file[path.name] += 1
            for change in changes:
                kind = change.split(":", 1)[0]
                by_kind[kind] += 1
                examples.setdefault(kind, []).append(f"{path.name}:{line_number} {sample['sentence']}")
        if changed:
            write_jsonl(path, rows)

    print(f"changed_samples {total_samples}")
    print(f"changed_items {total_changes}")
    for name, count in sorted(by_file.items()):
        print(f"{name} {count}")
    for kind, count in sorted(by_kind.items()):
        print(f"{kind} {count}")
        for example in examples[kind][:3]:
            print(f"  {example}")


if __name__ == "__main__":
    main()
