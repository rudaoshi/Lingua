#!/usr/bin/env python3
"""Repair high-confidence expression-internal format separators.

This intentionally skips mixed semantic cases such as hyphenated lexical
adjectives and address/place commas unless they match a narrower deterministic
pattern.  The goal is to remove obvious merged format strings and punctuation
separators inside dates, numbers, URLs, files, abbreviations, and compact
formula-like expressions while preserving graph shape.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CANDIDATES = ROOT / "reports" / "expression_separator_candidates.tsv"
REPORT_DIR = ROOT / "reports" / "expression_format_functor_repair"

DATASETS = {
    "train.jsonl": DATA / "train.jsonl",
    "dev.jsonl": DATA / "dev.jsonl",
    "test.jsonl": DATA / "test.jsonl",
}

DETERMINISTIC_CATEGORIES = {
    "merged_numeric_date_time_version",
    "decimal_version_duration_dot",
    "date_fraction_slash",
    "numeric_date_range_dash",
    "numeric_thousands_comma",
    "merged_numeric_thousands",
    "date_comma",
    "initial_abbrev_dot",
    "merged_initial_abbrev",
    "compact_paren_bracket_format",
    "merged_compact_paren_code",
    "hyphenated_name_code",
    "merged_hyphen_slash_name_code",
    "math_code_operator",
    "merged_math_code_symbol",
}

PARSER_CATEGORIES = {
    "url_email_file_separator",
    "merged_url_email_file",
    "currency_percent_rank_symbol",
}

SKIPPED_MIXED_CATEGORIES = {
    "address_place_comma",
}

OPERATOR_TOKENS = {
    ":",
    "://",
    "/",
    "-",
    "–",
    "—",
    ".",
    ",",
    "@",
    "?",
    "=",
    "&",
    "#",
    "%",
    "$",
    "+",
    "*",
    "<",
    ">",
    "(",
    ")",
    "[",
    "]",
}

MONTHS = {
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Sept",
    "Oct",
    "Nov",
    "Dec",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def token_text(word: str) -> str:
    return word.strip()


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    indexes: list[int] = []
    for start, end in node["spans"]:
        indexes.extend(range(start, end + 1))
    return indexes


def spans_for_tokens(tokens: list[int]) -> list[list[int]]:
    if not tokens:
        raise ValueError("cannot make spans for empty token list")
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


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def node_text(sample: dict[str, Any], node_index: int) -> str:
    words = sample["lingua"]["words"]
    node = graph(sample)["nodes"][node_index]
    if isinstance(node["spans"], str):
        return node["spans"]
    return "".join(words[index] for index in node_token_indexes(node))


def compact_node_text(sample: dict[str, Any], node_index: int) -> str:
    words = sample["lingua"]["words"]
    node = graph(sample)["nodes"][node_index]
    if isinstance(node["spans"], str):
        return node["spans"]
    return "".join(token_text(words[index]) for index in node_token_indexes(node))


def owner_by_token(nodes: list[dict[str, Any]]) -> dict[int, int]:
    owner: dict[int, int] = {}
    for node_index, node in enumerate(nodes):
        if isinstance(node["spans"], str):
            continue
        for token in node_token_indexes(node):
            owner[token] = node_index
    return owner


def infer_constant_type(words: list[str], tokens: list[int], fallback: str = "NominalConstant") -> str:
    text = "".join(token_text(words[token]) for token in tokens)
    if not text:
        return fallback
    if re.fullmatch(r"[\W_]+", text, flags=re.UNICODE):
        return "SymbolConstant"
    return fallback if fallback.endswith("Constant") else "NominalConstant"


def append_node(nodes: list[dict[str, Any]], spans: list[list[int]], node_type: str) -> int:
    nodes.append({"spans": spans, "type": node_type})
    return len(nodes) - 1


class TreeBuilder:
    def __init__(
        self,
        sample: dict[str, Any],
        existing_owner: dict[int, int] | None = None,
        reusable_root: int | None = None,
        reusable_nodes: set[int] | None = None,
        fallback_type: str = "NominalConstant",
        url_mode: bool = False,
    ) -> None:
        self.sample = sample
        self.words = sample["lingua"]["words"]
        self.nodes = graph(sample)["nodes"]
        self.edges: list[dict[str, Any]] = []
        self.existing_owner = existing_owner or {}
        self.reusable_root = reusable_root
        self.reusable_nodes = reusable_nodes or set()
        self.fallback_type = fallback_type
        self.url_mode = url_mode

    def build(self, tokens: list[int], prefer_root: int | None = None) -> int | None:
        tokens = sorted(tokens)
        if not tokens:
            return None
        op_group = self.choose_operator(tokens)
        if op_group is None:
            return self.node_for_operand(tokens)

        root = self.node_for_operator(op_group, prefer_root)
        op_set = set(op_group)
        left = [token for token in tokens if token < op_group[0]]
        right = [token for token in tokens if token > op_group[-1]]

        for child_tokens in (left, right):
            child = self.build(child_tokens)
            if child is not None and child != root:
                self.edges.append({"start": root, "end": child, "label": "func.arg"})
        return root

    def node_for_operator(self, op_group: list[int], prefer_root: int | None) -> int:
        if prefer_root is not None:
            node_index = prefer_root
            if self.reusable_root == prefer_root:
                self.reusable_root = None
        elif self.reusable_root is not None:
            node_index = self.reusable_root
            self.reusable_root = None
        else:
            owners = {self.existing_owner[token] for token in op_group if token in self.existing_owner}
            if len(owners) == 1:
                candidate = next(iter(owners))
                candidate_tokens = node_token_indexes(self.nodes[candidate])
                if sorted(candidate_tokens) == sorted(op_group) or candidate in self.reusable_nodes:
                    node_index = candidate
                else:
                    node_index = append_node(self.nodes, spans_for_tokens(op_group), "ExpressionFunctor")
            else:
                node_index = append_node(self.nodes, spans_for_tokens(op_group), "ExpressionFunctor")

        self.nodes[node_index]["spans"] = spans_for_tokens(op_group)
        self.nodes[node_index]["type"] = "ExpressionFunctor"
        return node_index

    def node_for_operand(self, tokens: list[int]) -> int:
        owners = {self.existing_owner[token] for token in tokens if token in self.existing_owner}
        if len(owners) == 1:
            candidate = next(iter(owners))
            candidate_tokens = sorted(node_token_indexes(self.nodes[candidate]))
            if candidate_tokens == sorted(tokens) or candidate in self.reusable_nodes:
                return candidate
        if len(owners) > 1 and owners <= self.reusable_nodes:
            return self.local_root_for_existing_operand(owners, tokens)
        return append_node(
            self.nodes,
            spans_for_tokens(tokens),
            infer_constant_type(self.words, tokens, self.fallback_type),
        )

    def local_root_for_existing_operand(self, owners: set[int], tokens: list[int]) -> int:
        indegree = {node: 0 for node in owners}
        for edge in graph(self.sample)["edges"]:
            if edge["start"] in owners and edge["end"] in owners:
                indegree[edge["end"]] += 1
        roots = [node for node, degree in indegree.items() if degree == 0]
        if len(roots) == 1:
            return roots[0]
        # Fall back to the owner of the rightmost token.  This matches many
        # local head-final numeric/name fragments and avoids adding duplicate
        # lexical coverage.
        for token in reversed(tokens):
            if token in self.existing_owner:
                return self.existing_owner[token]
        return sorted(owners)[0]

    def choose_operator(self, tokens: list[int]) -> list[int] | None:
        token_set = set(tokens)
        ops = [token for token in tokens if token_text(self.words[token]) in OPERATOR_TOKENS]
        if not ops:
            return None

        # Compact parenthetical formulas such as 25(OH)D3 use paired brackets as
        # one local functor when both bracket tokens are inside this token set.
        for open_mark, close_mark in (("(", ")"), ("[", "]")):
            opens = [token for token in ops if token_text(self.words[token]) == open_mark]
            closes = [token for token in ops if token_text(self.words[token]) == close_mark]
            if opens and closes and opens[0] < closes[-1] and opens[0] != tokens[0]:
                # Only choose the pair when it is not the whole outer expression.
                inner = [token for token in tokens if opens[0] < token < closes[-1]]
                if inner and closes[-1] != tokens[-1]:
                    return [opens[0], closes[-1]]

        if self.url_mode:
            for mark in ("?", "#", "&", "="):
                for token in ops:
                    if token_text(self.words[token]) == mark:
                        return [token]
            for token in ops:
                if token_text(self.words[token]) == "://":
                    return [token]
            for pos, token in enumerate(tokens[:-2]):
                if (
                    token_text(self.words[token]) == ":"
                    and token_text(self.words[tokens[pos + 1]]) == "/"
                    and token_text(self.words[tokens[pos + 2]]) == "/"
                ):
                    return [token, tokens[pos + 1], tokens[pos + 2]]
            for mark in ("/", "@", "."):
                for token in ops:
                    if token_text(self.words[token]) == mark:
                        return [token]

        # Prefix and suffix format operators.
        if token_text(self.words[tokens[0]]) in {"$", "#", "<"}:
            return [tokens[0]]
        if token_text(self.words[tokens[-1]]) in {"%", "+", ">", "."}:
            return [tokens[-1]]

        # Binary separators: use the leftmost separator to form a stable nested
        # hierarchy, e.g. 06 / (04 / 01) and 08 : (36 : 00).
        for token in ops:
            if token != tokens[0] and token != tokens[-1]:
                return [token]
        return [ops[0]]


def remove_duplicate_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, str]] = set()
    out: list[dict[str, Any]] = []
    for edge in edges:
        key = (edge["start"], edge["end"], edge["label"])
        if edge["start"] == edge["end"]:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append({k: edge[k] for k in ("start", "end", "label")})
    return out


def replace_group_edges(
    sample: dict[str, Any],
    group_nodes: set[int],
    root: int,
    internal_edges: list[dict[str, Any]],
) -> None:
    old_edges = graph(sample)["edges"]
    new_edges: list[dict[str, Any]] = []
    operator_nodes = {edge["start"] for edge in internal_edges if edge["start"] in group_nodes}
    for edge in old_edges:
        start = edge["start"]
        end = edge["end"]
        if start in group_nodes and end in group_nodes:
            if start in operator_nodes or end in operator_nodes or edge["label"] == "func.arg":
                continue
            new_edges.append({"start": start, "end": end, "label": edge["label"]})
            continue
        if start in group_nodes:
            start = root
        if end in group_nodes:
            end = root
        if start == end:
            continue
        new_edges.append({"start": start, "end": end, "label": edge["label"]})
    new_edges.extend(internal_edges)
    graph(sample)["edges"] = remove_duplicate_edges(new_edges)


def is_aux_node(sample: dict[str, Any], node_index: int) -> bool:
    return isinstance(graph(sample)["nodes"][node_index]["spans"], str)


def rewrite_merged_node(
    sample: dict[str, Any],
    node_index: int,
    *,
    url_mode: bool = False,
) -> bool:
    nodes = graph(sample)["nodes"]
    if node_index >= len(nodes) or is_aux_node(sample, node_index):
        return False
    tokens = node_token_indexes(nodes[node_index])
    if len(tokens) <= 1:
        return False
    compact = "".join(token_text(sample["lingua"]["words"][token]) for token in tokens)
    if not any(token_text(sample["lingua"]["words"][token]) in OPERATOR_TOKENS for token in tokens):
        return False

    old_type = nodes[node_index]["type"]
    builder = TreeBuilder(sample, reusable_root=node_index, fallback_type=old_type, url_mode=url_mode)
    root = builder.build(tokens, prefer_root=node_index)
    if root != node_index:
        return False
    graph(sample)["edges"].extend(builder.edges)
    graph(sample)["edges"] = remove_duplicate_edges(graph(sample)["edges"])
    return compact_node_text(sample, node_index) != compact or nodes[node_index]["type"] == "ExpressionFunctor"


def token_group_is_owned_exactly(sample: dict[str, Any], tokens: list[int]) -> tuple[dict[int, int], set[int]] | None:
    nodes = graph(sample)["nodes"]
    owner = owner_by_token(nodes)
    if any(token not in owner for token in tokens):
        return None
    group_nodes = {owner[token] for token in tokens}
    token_set = set(tokens)
    for node_index in group_nodes:
        node = nodes[node_index]
        if isinstance(node["spans"], str):
            return None
        node_tokens = set(node_token_indexes(node))
        if not node_tokens <= token_set:
            return None
    return owner, group_nodes


def rewrite_existing_token_group(
    sample: dict[str, Any],
    tokens: list[int],
    *,
    url_mode: bool = False,
) -> bool:
    tokens = sorted(set(tokens))
    owned = token_group_is_owned_exactly(sample, tokens)
    if owned is None:
        return False
    owner, group_nodes = owned
    builder = TreeBuilder(
        sample,
        existing_owner=owner,
        reusable_nodes=group_nodes,
        fallback_type="NominalConstant",
        url_mode=url_mode,
    )
    root = builder.build(tokens)
    if root is None or root not in group_nodes:
        return False
    replace_group_edges(sample, group_nodes, root, builder.edges)
    return True


def numeric_sequence_around(words: list[str], token_index: int, separators: set[str]) -> list[int] | None:
    tokens = [token_index]
    left = token_index - 1
    right = token_index + 1
    if left < 0 or right >= len(words):
        return None
    if not re.fullmatch(r"\d+(?:\.\d+)?", token_text(words[left])):
        return None
    if not re.fullmatch(r"\d+(?:\.\d+)?", token_text(words[right])):
        return None
    tokens = [left, token_index, right]
    cursor = left - 1
    while cursor - 1 >= 0 and token_text(words[cursor]) in separators and re.fullmatch(
        r"\d+(?:\.\d+)?", token_text(words[cursor - 1])
    ):
        tokens = [cursor - 1, cursor] + tokens
        cursor -= 2
    cursor = right + 1
    while cursor + 1 < len(words) and token_text(words[cursor]) in separators and re.fullmatch(
        r"\d+(?:\.\d+)?", token_text(words[cursor + 1])
    ):
        tokens.extend([cursor, cursor + 1])
        cursor += 2
    return tokens


def date_comma_tokens(words: list[str], comma_index: int) -> list[int] | None:
    if comma_index < 2 or comma_index + 1 >= len(words):
        return None
    day = token_text(words[comma_index - 1])
    year = token_text(words[comma_index + 1])
    if not re.fullmatch(r"\d{1,2}", day) or not re.fullmatch(r"\d{4}", year):
        return None
    for month_index in range(comma_index - 2, max(-1, comma_index - 5), -1):
        if token_text(words[month_index]).rstrip(".") in MONTHS:
            return list(range(month_index, comma_index + 2))
    return None


def currency_percent_tokens(words: list[str], token_index: int) -> list[int] | None:
    sep = token_text(words[token_index])
    if sep in {"$", "#"}:
        right = token_index + 1
        if right >= len(words) or not re.fullmatch(r"\d+(?:\.\d+)?", token_text(words[right])):
            return None
        tokens = [token_index, right]
        cursor = right + 1
        while cursor + 1 < len(words) and token_text(words[cursor]) in {",", "."} and re.fullmatch(
            r"\d+", token_text(words[cursor + 1])
        ):
            tokens.extend([cursor, cursor + 1])
            cursor += 2
        return tokens
    if sep in {"%", "+"}:
        left = token_index - 1
        if left < 0 or not re.fullmatch(r"\d+(?:\.\d+)?", token_text(words[left])):
            return None
        tokens = [left, token_index]
        cursor = left - 1
        while cursor - 1 >= 0 and token_text(words[cursor]) in {",", "."} and re.fullmatch(
            r"\d+", token_text(words[cursor - 1])
        ):
            tokens = [cursor - 1, cursor] + tokens
            cursor -= 2
        # Include compact biomedical prefixes such as CD + 4 + only when the
        # token before the number is directly glued to it.
        if left - 1 >= 0 and not words[left].startswith(" ") and re.fullmatch(r"[A-Za-z]+", token_text(words[left - 1])):
            tokens = [left - 1] + tokens
        return tokens
    return None


def has_lexical_expression_parent(sample: dict[str, Any], node_index: int) -> bool:
    nodes = graph(sample)["nodes"]
    for edge in graph(sample)["edges"]:
        if edge["end"] != node_index:
            continue
        parent = nodes[edge["start"]]
        if parent["type"] == "ExpressionFunctor" and not isinstance(parent["spans"], str):
            return True
    return False


def has_aux_outgoing(sample: dict[str, Any], node_index: int) -> bool:
    nodes = graph(sample)["nodes"]
    return any(
        edge["start"] == node_index and isinstance(nodes[edge["end"]]["spans"], str)
        for edge in graph(sample)["edges"]
    )


def has_outgoing_outside_nodes(sample: dict[str, Any], node_index: int, allowed_nodes: set[int]) -> bool:
    return any(
        edge["start"] == node_index and edge["end"] not in allowed_nodes
        for edge in graph(sample)["edges"]
    )


def compact_paren_tokens(words: list[str], token_index: int) -> list[int] | None:
    mark = token_text(words[token_index])
    if mark not in {"(", ")", "[", "]"}:
        return None
    open_mark, close_mark = ("(", ")") if mark in {"(", ")"} else ("[", "]")
    if mark == open_mark:
        open_index = token_index
        close_index = next(
            (idx for idx in range(open_index + 1, min(len(words), open_index + 6)) if token_text(words[idx]) == close_mark),
            None,
        )
    else:
        close_index = token_index
        open_index = next(
            (idx for idx in range(close_index - 1, max(-1, close_index - 6), -1) if token_text(words[idx]) == open_mark),
            None,
        )
    if open_index is None or close_index is None or not (open_index < close_index):
        return None
    if open_index == 0 or close_index + 1 >= len(words):
        return None
    if words[open_index].startswith(" ") or words[close_index + 1].startswith(" "):
        return None
    if not re.search(r"[A-Za-z0-9]", token_text(words[open_index - 1]) + token_text(words[close_index + 1])):
        return None
    return list(range(open_index - 1, close_index + 2))


def hyphen_sequence_around(words: list[str], token_index: int) -> list[int] | None:
    if token_text(words[token_index]) not in {"-", "–", "—", "/"}:
        return None
    if token_index == 0 or token_index + 1 >= len(words):
        return None
    if not re.fullmatch(r"[A-Za-z0-9À-ÖØ-öø-ÿ]+", token_text(words[token_index - 1])):
        return None
    if not re.fullmatch(r"[A-Za-z0-9À-ÖØ-öø-ÿ]+", token_text(words[token_index + 1])):
        return None

    tokens = [token_index - 1, token_index, token_index + 1]
    cursor = token_index - 2
    while (
        cursor - 1 >= 0
        and token_text(words[cursor]) in {"-", "–", "—", "/"}
        and re.fullmatch(r"[A-Za-z0-9À-ÖØ-öø-ÿ]+", token_text(words[cursor - 1]))
    ):
        tokens = [cursor - 1, cursor] + tokens
        cursor -= 2
    cursor = token_index + 2
    while (
        cursor + 1 < len(words)
        and token_text(words[cursor]) in {"-", "–", "—", "/"}
        and re.fullmatch(r"[A-Za-z0-9À-ÖØ-öø-ÿ]+", token_text(words[cursor + 1]))
    ):
        tokens.extend([cursor, cursor + 1])
        cursor += 2
    return tokens


def high_confidence_hyphen_expression(compact: str, *, allow_proper: bool) -> bool:
    compact = re.sub(r"\s+", "", compact)
    if not re.search(r"[-/–—]", compact):
        return False
    parts = [part for part in re.split(r"[-/–—]", compact) if part]
    if len(parts) < 2:
        return False
    if any(not re.fullmatch(r"[A-Za-z0-9À-ÖØ-öø-ÿ]+", part) for part in parts):
        return False

    # Biomedical/model/code names: NF-kappaB, VCAM-1, IL-2, HTLV-I, TR-201.
    if re.search(r"\d", compact) and re.search(r"[A-Za-z]", compact):
        return True
    if any(re.search(r"[A-Z]{2,}", part) for part in parts):
        return True

    # Proper-name compounds: Epstein-Barr, Saint-Jérôme, Pyrénées-Orientales.
    if allow_proper and re.search(r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ]+[-/–—][A-ZÀ-ÖØ-Þ]", compact):
        return True
    return False


def repair_hyphen_candidate(sample: dict[str, Any], node_index: int, token_index: int | None = None) -> bool:
    node = graph(sample)["nodes"][node_index]
    if is_aux_node(sample, node_index):
        return False

    if len(node_token_indexes(node)) > 1:
        if node["type"] != "NominalConstant":
            return False
        compact = compact_node_text(sample, node_index)
        if not high_confidence_hyphen_expression(compact, allow_proper=True):
            return False
        return rewrite_merged_node(sample, node_index)

    if token_index is None:
        return False
    words = sample["lingua"]["words"]
    tokens = hyphen_sequence_around(words, token_index)
    if not tokens:
        return False
    compact = "".join(token_text(words[token]) for token in tokens)
    # Token-level numeric-unit ranges such as 1,452-km, 20-korun, or
    # 15- to 20-fold are often already woven into modifier scope.  Merged code
    # nodes with digits (IL-2, VCAM-1) are still handled above.
    if re.search(r"\d", compact):
        return False
    if not high_confidence_hyphen_expression(compact, allow_proper=True):
        return False
    owned = token_group_is_owned_exactly(sample, tokens)
    if owned is None:
        return False
    _, group_nodes = owned
    if any(has_lexical_expression_parent(sample, group_node) for group_node in group_nodes):
        return False
    if has_outgoing_outside_nodes(sample, node_index, group_nodes):
        return False
    allowed_types = {"NominalConstant", "PunctuationalConstant", "ExpressionFunctor"}
    if any(graph(sample)["nodes"][group_node]["type"] not in allowed_types for group_node in group_nodes):
        return False
    return rewrite_existing_token_group(sample, tokens)


def repair_token_candidate(sample: dict[str, Any], row: dict[str, str]) -> bool:
    words = sample["lingua"]["words"]
    token_index = int(row["token_index"])
    node_index = int(row["node_index"])
    category = row["category"]
    node = graph(sample)["nodes"][node_index]

    if category == "hyphenated_name_code":
        return repair_hyphen_candidate(sample, node_index, token_index)

    # If the flagged separator is inside a multi-token lexical node, treat that
    # node as a merged expression.  Rewriting it as an existing token group would
    # otherwise reuse the same node for nested operators and leave uncovered
    # tokens.
    if not isinstance(node["spans"], str) and len(node_token_indexes(node)) > 1:
        return rewrite_merged_node(sample, node_index, url_mode=category == "url_email_file_separator")

    if category == "decimal_version_duration_dot":
        # Standalone decimal dots are often already inside a larger time/score
        # expression (e.g. 1:18.52).  Rewriting them locally can create cycles
        # with the existing colon expression, so only merged decimal nodes are
        # handled automatically.
        return False

    if category in {"date_fraction_slash", "numeric_date_range_dash", "numeric_thousands_comma"}:
        if category == "numeric_date_range_dash":
            owner = owner_by_token(graph(sample)["nodes"])
            for neighbor in (token_index - 1, token_index + 1):
                if neighbor in owner and has_lexical_expression_parent(sample, owner[neighbor]):
                    return False
        separators = {token_text(words[token_index])}
        tokens = numeric_sequence_around(words, token_index, separators)
        return bool(tokens and rewrite_existing_token_group(sample, tokens))

    if category == "date_comma":
        # Month-day-year commas are structurally diverse in current graphs; many
        # are tied into prepositional or close-punctuation scope.  Leave them to
        # the review bucket until their surrounding date expression is repaired.
        return False

    if category == "currency_percent_rank_symbol":
        if token_text(words[token_index]) == "#":
            return False
        if has_aux_outgoing(sample, node_index):
            return False
        tokens = currency_percent_tokens(words, token_index)
        if not tokens:
            return False
        owned = token_group_is_owned_exactly(sample, tokens)
        if owned is None:
            return False
        _, group_nodes = owned
        if has_outgoing_outside_nodes(sample, node_index, group_nodes):
            return False
        return rewrite_existing_token_group(sample, tokens)

    if category == "initial_abbrev_dot":
        # Only repair when the abbreviation is already a merged node.  Separate
        # dots often sit inside a larger name expression and are handled through
        # the merged-node path.
        return False

    if category in {"compact_paren_bracket_format", "math_code_operator"}:
        tokens = compact_paren_tokens(words, token_index)
        if tokens:
            return rewrite_existing_token_group(sample, tokens)
        if category == "math_code_operator":
            tokens = currency_percent_tokens(words, token_index)
            return bool(tokens and rewrite_existing_token_group(sample, tokens))
        return False

    return False


def repair_merged_candidate(sample: dict[str, Any], row: dict[str, str]) -> bool:
    node_index = int(row["node_index"])
    category = row["category"]
    if is_aux_node(sample, node_index):
        return False
    compact = compact_node_text(sample, node_index)

    if category == "merged_url_email_file":
        # Avoid spaced slash group names like "/ IV" unless the compact string
        # looks like a genuine URL/email/file/code path.
        if not re.search(
            r"https?://|www\.|@\w|[A-Za-z0-9_.-]+\.(?:com|org|net|edu|gov|html|asp|php|pdf|doc|DOC)\b|[?&][A-Za-z0-9_]+=",
            compact,
            flags=re.IGNORECASE,
        ):
            return False
        return rewrite_merged_node(sample, node_index, url_mode=True)

    if category in {
        "merged_numeric_date_time_version",
        "merged_numeric_thousands",
        "merged_initial_abbrev",
        "merged_hyphen_slash_name_code",
        "merged_math_code_symbol",
        "merged_compact_paren_code",
    }:
        if category == "merged_hyphen_slash_name_code":
            return repair_hyphen_candidate(sample, node_index)
        return rewrite_merged_node(sample, node_index)

    return False


def sample_key(row: dict[str, str]) -> tuple[str, int]:
    return row["split"], int(row["line"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    args = parser.parse_args()

    candidate_rows = read_candidates(args.candidates)
    rows_by_sample: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_sample[sample_key(row)].append(row)

    datasets = {name: read_jsonl(path) for name, path in DATASETS.items()}
    changes: list[dict[str, Any]] = []
    skipped = Counter()
    repaired_nodes: set[tuple[str, int, int]] = set()

    for (split, line_number), rows in sorted(rows_by_sample.items()):
        sample = datasets[split][line_number - 1]
        before = deepcopy(sample["lingua"]["graph"])
        sample_changes: list[str] = []

        # Merged nodes first; token-level rows for the same node become stale
        # after this, so they are skipped by repaired_nodes.
        for row in rows:
            category = row["category"]
            if category in SKIPPED_MIXED_CATEGORIES:
                skipped[category] += 1
                continue
            if row["kind"] != "merged_expression_node":
                continue
            if category not in DETERMINISTIC_CATEGORIES | PARSER_CATEGORIES:
                skipped[category] += 1
                continue
            node_index = int(row["node_index"])
            node_key = (split, line_number, node_index)
            if node_key in repaired_nodes:
                continue
            if repair_merged_candidate(sample, row):
                repaired_nodes.add(node_key)
                sample_changes.append(f"{category}: node {node_index} `{row['node_text']}`")
            else:
                skipped[f"shape:{category}"] += 1

        for row in rows:
            category = row["category"]
            if category in SKIPPED_MIXED_CATEGORIES:
                continue
            if row["kind"] != "separator_token_not_expression_functor":
                continue
            if category not in DETERMINISTIC_CATEGORIES | PARSER_CATEGORIES:
                skipped[category] += 1
                continue
            node_index = int(row["node_index"])
            if (split, line_number, node_index) in repaired_nodes:
                continue
            if repair_token_candidate(sample, row):
                # Mark every node now touched by this token's group as done only
                # at the original separator node level; duplicate rows are still
                # harmless because exact group rewrites become no-ops.
                repaired_nodes.add((split, line_number, node_index))
                sample_changes.append(f"{category}: token {row['token_index']} `{row['sep']}`")
            else:
                skipped[f"shape:{category}"] += 1

        after = sample["lingua"]["graph"]
        if after != before:
            changes.append(
                {
                    "split": split,
                    "line": line_number,
                    "sentence": sample["sentence"],
                    "changes": sample_changes,
                    "node_count_before": len(before["nodes"]),
                    "node_count_after": len(after["nodes"]),
                    "edge_count_before": len(before["edges"]),
                    "edge_count_after": len(after["edges"]),
                }
            )

    if not args.dry_run:
        for name, path in DATASETS.items():
            write_jsonl(path, datasets[name])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "changes.jsonl").write_text(
        "\n".join(json.dumps(change, ensure_ascii=False, separators=(",", ":")) for change in changes) + ("\n" if changes else ""),
        encoding="utf-8",
    )
    summary = {
        "dry_run": args.dry_run,
        "changed_samples": len(changes),
        "changed_by_split": Counter(change["split"] for change in changes),
        "changed_items": sum(len(change["changes"]) for change in changes),
        "skipped": skipped,
    }
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
