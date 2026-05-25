#!/usr/bin/env python3
"""Normalize Lingua be/copula structures according to the current standard.

The script applies only graph-local repairs that are recoverable from the
existing annotation:

1. Split `there be` predicates so `there` is `pred.arg.1` and the existent is
   `pred.arg.2`.
2. Merge split simple copulas (`be -> pred.arg.2 -> adjective/state`) into a
   single factual predicate, preserving adverbial modifier scope.
3. Merge split direct `be + prep` predicates.
4. Split non-direct merged `be ... prep` predicates, keeping the intervening
   state complement with `be` and representing the prep as a separate modifier.
5. Split recoverable adverbial modifiers out of merged simple copular
   adjective/state predicates.

Ambiguous candidates are reported but not changed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

BE_FORMS = {
    "'m",
    "'re",
    "'s",
    "am",
    "are",
    "be",
    "been",
    "being",
    "is",
    "m",
    "re",
    "s",
    "was",
    "were",
}

MODALS_AND_AUX = {
    "'d",
    "'ll",
    "'ve",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "will",
    "would",
}

NEGATION_PARTS = {"not", "n't", "nt", "n", "t"}

ADVERBIAL_MODIFIERS = {
    "absolutely",
    "actually",
    "already",
    "also",
    "almost",
    "basically",
    "clearly",
    "completely",
    "currently",
    "deeply",
    "definitely",
    "entirely",
    "especially",
    "even",
    "extremely",
    "fairly",
    "fully",
    "generally",
    "hardly",
    "highly",
    "immaculately",
    "just",
    "kind",
    "largely",
    "less",
    "mainly",
    "more",
    "mostly",
    "nearly",
    "never",
    "no",
    "not",
    "obviously",
    "only",
    "particularly",
    "perfectly",
    "pretty",
    "primarily",
    "probably",
    "quite",
    "rather",
    "really",
    "relatively",
    "scarcely",
    "simply",
    "slightly",
    "so",
    "somewhat",
    "still",
    "terribly",
    "too",
    "totally",
    "truly",
    "usually",
    "very",
}

DIRECT_PREPOSITIONS = {
    "about",
    "above",
    "across",
    "after",
    "against",
    "along",
    "among",
    "around",
    "at",
    "before",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "beyond",
    "by",
    "down",
    "for",
    "from",
    "in",
    "inside",
    "into",
    "near",
    "of",
    "off",
    "on",
    "onto",
    "out",
    "outside",
    "over",
    "through",
    "throughout",
    "to",
    "toward",
    "towards",
    "under",
    "up",
    "upon",
    "with",
    "within",
    "without",
}

MULTIWORD_PREPOSITIONS = {
    ("out", "of"),
    ("up", "for"),
}

STATE_ADJECTIVES = {
    "able",
    "acceptable",
    "accessible",
    "accurate",
    "afraid",
    "alive",
    "alone",
    "amazing",
    "angry",
    "available",
    "aware",
    "awesome",
    "bad",
    "basic",
    "beautiful",
    "big",
    "boring",
    "busy",
    "careful",
    "cheap",
    "clean",
    "clear",
    "close",
    "comfortable",
    "complete",
    "concerned",
    "confident",
    "confusing",
    "convenient",
    "correct",
    "corrupt",
    "crazy",
    "curious",
    "cute",
    "dangerous",
    "different",
    "difficult",
    "disappointed",
    "done",
    "easy",
    "effective",
    "efficient",
    "empty",
    "enough",
    "excellent",
    "excited",
    "exciting",
    "expensive",
    "fair",
    "fantastic",
    "fine",
    "free",
    "friendly",
    "full",
    "fun",
    "funny",
    "glad",
    "good",
    "great",
    "happy",
    "hard",
    "helpful",
    "high",
    "hopeful",
    "hot",
    "important",
    "impossible",
    "incredible",
    "insufficient",
    "interested",
    "interesting",
    "kind",
    "large",
    "late",
    "likely",
    "little",
    "long",
    "low",
    "mad",
    "necessary",
    "new",
    "nice",
    "open",
    "perfect",
    "pleased",
    "possible",
    "pretty",
    "quick",
    "ready",
    "real",
    "reasonable",
    "relevant",
    "responsible",
    "right",
    "sad",
    "safe",
    "same",
    "satisfied",
    "scared",
    "serious",
    "short",
    "significant",
    "simple",
    "slow",
    "small",
    "sorry",
    "special",
    "stressed",
    "strong",
    "subject",
    "successful",
    "sure",
    "terrible",
    "tired",
    "true",
    "useful",
    "valuable",
    "warm",
    "well",
    "willing",
    "wonderful",
    "worried",
    "worth",
    "wrong",
}

PARTICIPLE_STATE_WHITELIST = {
    "amazed",
    "annoyed",
    "closed",
    "concerned",
    "confused",
    "convinced",
    "delighted",
    "depressed",
    "disappointed",
    "done",
    "embarrassed",
    "excited",
    "finished",
    "interested",
    "pleased",
    "prepared",
    "satisfied",
    "scared",
    "stressed",
    "surprised",
    "tired",
    "worried",
}

VERBAL_PARTICIPLE_BLOCKLIST = {
    "achieved",
    "blocked",
    "charged",
    "cleaned",
    "completed",
    "found",
    "given",
    "issued",
    "left",
    "made",
    "picked",
    "recovered",
    "redeveloped",
    "set",
    "studied",
    "surpassed",
}


def norm(token: str) -> str:
    return token.strip().lower().replace("’", "'")


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def words(sample: dict[str, Any]) -> list[str]:
    return sample["lingua"]["words"]


def node_tokens(sample: dict[str, Any], node_index: int) -> list[tuple[int, str, str]]:
    node = graph(sample)["nodes"][node_index]
    spans = node.get("spans")
    if isinstance(spans, str):
        return []
    tokens: list[tuple[int, str, str]] = []
    for start, end in spans:
        for index in range(start, end + 1):
            token = words(sample)[index]
            tokens.append((index, token, norm(token)))
    return sorted(tokens)


def descendant_token_indexes(sample: dict[str, Any], node_index: int) -> list[int]:
    seen: set[int] = set()
    stack = [node_index]
    indexes: list[int] = []
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        indexes.extend(index for index, _, _ in node_tokens(sample, current))
        stack.extend(edge["end"] for edge in edge_out(sample, current))
    return indexes


def node_text(sample: dict[str, Any], node_index: int) -> str:
    text = " ".join(token.strip() for _, token, _ in node_tokens(sample, node_index)).strip()
    if text:
        return text
    return str(graph(sample)["nodes"][node_index].get("spans"))


def edge_out(sample: dict[str, Any], node_index: int, label: str | None = None) -> list[dict[str, Any]]:
    return [
        edge
        for edge in graph(sample)["edges"]
        if edge["start"] == node_index and (label is None or edge["label"] == label)
    ]


def edge_in(sample: dict[str, Any], node_index: int, label: str | None = None) -> list[dict[str, Any]]:
    return [
        edge
        for edge in graph(sample)["edges"]
        if edge["end"] == node_index and (label is None or edge["label"] == label)
    ]


def add_node(sample: dict[str, Any], spans: Any, node_type: str) -> int:
    nodes = graph(sample)["nodes"]
    nodes.append({"spans": spans, "type": node_type})
    return len(nodes) - 1


def add_edge(sample: dict[str, Any], start: int, end: int, label: str) -> None:
    graph(sample)["edges"].append({"start": start, "end": end, "label": label})


def spans_from_token_indexes(indexes: list[int]) -> list[list[int]]:
    indexes = sorted(indexes)
    if not indexes:
        return []
    spans: list[list[int]] = []
    start = previous = indexes[0]
    for index in indexes[1:]:
        if index == previous + 1:
            previous = index
        else:
            spans.append([start, previous])
            start = previous = index
    spans.append([start, previous])
    return spans


def set_node_token_indexes(sample: dict[str, Any], node_index: int, indexes: list[int]) -> None:
    graph(sample)["nodes"][node_index]["spans"] = spans_from_token_indexes(indexes)


def remove_nodes(sample: dict[str, Any], removed: set[int]) -> dict[int, int]:
    g = graph(sample)
    mapping: dict[int, int] = {}
    new_nodes: list[dict[str, Any]] = []
    for old_index, node in enumerate(g["nodes"]):
        if old_index in removed:
            continue
        mapping[old_index] = len(new_nodes)
        new_nodes.append(node)

    new_edges: list[dict[str, Any]] = []
    for edge in g["edges"]:
        if edge["start"] in removed or edge["end"] in removed:
            continue
        new_edge = dict(edge)
        new_edge["start"] = mapping[edge["start"]]
        new_edge["end"] = mapping[edge["end"]]
        new_edges.append(new_edge)

    g["nodes"] = new_nodes
    g["edges"] = new_edges
    return mapping


def remove_edges(sample: dict[str, Any], predicate) -> None:
    graph(sample)["edges"] = [edge for edge in graph(sample)["edges"] if not predicate(edge)]


def replace_edge_endpoints(sample: dict[str, Any], old: int, new: int, incoming_only: bool = False) -> None:
    for edge in graph(sample)["edges"]:
        if edge["end"] == old:
            edge["end"] = new
        if not incoming_only and edge["start"] == old:
            edge["start"] = new


def is_be_only(sample: dict[str, Any], node_index: int) -> bool:
    tokens = [normalized for _, _, normalized in node_tokens(sample, node_index) if normalized != "'"]
    return bool(tokens) and all(token in BE_FORMS for token in tokens)


def has_be(sample: dict[str, Any], node_index: int) -> bool:
    return any(normalized in BE_FORMS for _, _, normalized in node_tokens(sample, node_index))


def is_factual(sample: dict[str, Any], node_index: int) -> bool:
    return graph(sample)["nodes"][node_index].get("type") == "FactualPredicator"


def is_prep_token(normalized: str) -> bool:
    return normalized in DIRECT_PREPOSITIONS


def is_be_apostrophe(tokens: list[tuple[int, str, str]], position: int) -> bool:
    if tokens[position][2] != "'":
        return False
    return position + 1 < len(tokens) and tokens[position + 1][2] in {"m", "re", "s"}


def is_negation_apostrophe(tokens: list[tuple[int, str, str]], position: int) -> bool:
    if tokens[position][2] != "'":
        return False
    return (
        position > 0
        and position + 1 < len(tokens)
        and tokens[position - 1][2] in {"n"}
        and tokens[position + 1][2] in {"t"}
    )


def looks_like_state_word(normalized: str) -> bool:
    if normalized in STATE_ADJECTIVES or normalized in PARTICIPLE_STATE_WHITELIST:
        return True
    if normalized.endswith("ing") and normalized in STATE_ADJECTIVES:
        return True
    if normalized.endswith("ed") and normalized not in PARTICIPLE_STATE_WHITELIST:
        return False
    if normalized in VERBAL_PARTICIPLE_BLOCKLIST:
        return False
    return False


def modifier_chain_to_state_head(sample: dict[str, Any], start: int) -> tuple[list[int], int] | None:
    """Return modifier chain and state head for a split copular complement."""
    nodes = graph(sample)["nodes"]
    chain = [start]
    current = start
    seen = {start}
    while True:
        outgoing = [
            edge
            for edge in edge_out(sample, current, "as:modification")
            if nodes[edge["end"]].get("type") == "ModificationalConstant"
        ]
        if any(
            edge["label"] == "as:modification"
            and nodes[edge["end"]].get("type") != "ModificationalConstant"
            for edge in edge_out(sample, current)
        ):
            # The complement is an internal modifier of a nominal expression
            # (`strong pullers`, `bad news`), not a simple predicative state.
            return None
        if not outgoing:
            break
        if len(outgoing) != 1:
            return None
        current = outgoing[0]["end"]
        if current in seen:
            return None
        seen.add(current)
        chain.append(current)

    head = chain[-1]
    head_tokens = node_tokens(sample, head)
    if not head_tokens:
        return None
    head_norms = [normalized for _, _, normalized in head_tokens]
    if len(head_norms) > 2:
        return None
    if not looks_like_state_word(head_norms[-1]):
        return None
    return chain, head


def repair_there_be_merged(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    for node_index in range(len(nodes)):
        if not is_factual(sample, node_index):
            continue
        tokens = node_tokens(sample, node_index)
        normalized = [item[2] for item in tokens]
        if "there" not in normalized or not any(token in BE_FORMS for token in normalized):
            continue
        there_indexes = [index for index, _, token in tokens if token == "there"]
        be_indexes = [index for index, _, token in tokens if token in BE_FORMS]
        if len(there_indexes) != 1 or not be_indexes:
            continue

        existing_arg1 = edge_out(sample, node_index, "pred.arg.1")
        existing_arg2 = edge_out(sample, node_index, "pred.arg.2")
        if existing_arg2:
            continue
        there_index = there_indexes[0]
        earliest_be = min(be_indexes)
        if there_index > earliest_be:
            # Question inversion such as "is there any cure" is existential,
            # but locative "I was there" has an ordinary subject before `be`.
            before_be_arg1 = False
            for edge in existing_arg1:
                descendant_indexes = descendant_token_indexes(sample, edge["end"])
                if descendant_indexes and min(descendant_indexes) < earliest_be:
                    before_be_arg1 = True
                    break
            if before_be_arg1:
                continue

        there_node = add_node(sample, [[there_index, there_index]], "NominalConstant")
        keep_indexes = [index for index, _, _ in tokens if index not in set(there_indexes)]
        set_node_token_indexes(sample, node_index, keep_indexes)
        for edge in existing_arg1:
            edge["label"] = "pred.arg.2"
        add_edge(sample, node_index, there_node, "pred.arg.1")
        changes.append("there_be_merged_split")
    return changes


def repair_there_be_arg4(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    for node_index in range(len(graph(sample)["nodes"])):
        if not is_factual(sample, node_index) or not is_be_only(sample, node_index):
            continue
        arg1_texts = [node_text(sample, edge["end"]).lower() for edge in edge_out(sample, node_index, "pred.arg.1")]
        if "there" not in {text.strip() for text in arg1_texts}:
            continue
        if edge_out(sample, node_index, "pred.arg.2"):
            continue
        arg4 = edge_out(sample, node_index, "pred.arg.4")
        if len(arg4) != 1:
            continue
        arg4[0]["label"] = "pred.arg.2"
        changes.append("there_be_arg4_to_arg2")
    return changes


def repair_split_simple_copula(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    removed: set[int] = set()

    for node_index in range(len(nodes)):
        if node_index in removed or not is_factual(sample, node_index) or not is_be_only(sample, node_index):
            continue
        arg2_edges = edge_out(sample, node_index, "pred.arg.2")
        if len(arg2_edges) != 1:
            continue
        complement = arg2_edges[0]["end"]
        if complement in removed or nodes[complement].get("type") != "ModificationalConstant":
            continue

        chain_result = modifier_chain_to_state_head(sample, complement)
        if chain_result is None:
            continue
        chain, head = chain_result
        if any(edge for edge in edge_in(sample, head) if edge["start"] not in set(chain) | {node_index}):
            continue
        if any(edge for edge in edge_out(sample, head) if edge["label"] != "as:modification"):
            continue

        be_indexes = [index for index, _, _ in node_tokens(sample, node_index)]
        head_indexes = [index for index, _, _ in node_tokens(sample, head)]
        set_node_token_indexes(sample, node_index, be_indexes + head_indexes)

        chain_set = set(chain)
        remove_edges(
            sample,
            lambda edge, n=node_index, c=complement, h=head: (
                edge["start"] == n and edge["end"] == c and edge["label"] == "pred.arg.2"
            )
            or (edge["start"] == h),
        )
        if len(chain) > 1:
            previous_modifier = chain[-2]
            for edge in graph(sample)["edges"]:
                if (
                    edge["start"] == previous_modifier
                    and edge["end"] == head
                    and edge["label"] == "as:modification"
                ):
                    edge["end"] = node_index
            transfer_incoming_to_new_node(sample, node_index, chain[0], excluded_starts=set(chain))
        removed.add(head)
        changes.append("split_simple_copula_merged")

    if removed:
        remove_nodes(sample, removed)
    return changes


def repair_split_direct_be_prep(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    nodes = graph(sample)["nodes"]
    removed: set[int] = set()

    for node_index in range(len(nodes)):
        if node_index in removed or not is_factual(sample, node_index) or not is_be_only(sample, node_index):
            continue
        arg2_edges = edge_out(sample, node_index, "pred.arg.2")
        if len(arg2_edges) != 1:
            continue
        prep = arg2_edges[0]["end"]
        if prep in removed or nodes[prep].get("type") != "ModificationalFunctor":
            continue
        prep_tokens = node_tokens(sample, prep)
        if not prep_tokens or not all(is_prep_token(token) for _, _, token in prep_tokens):
            continue
        be_indexes = [index for index, _, _ in node_tokens(sample, node_index)]
        prep_indexes = [index for index, _, _ in prep_tokens]
        if min(prep_indexes) != max(be_indexes) + 1:
            continue
        bodies = edge_out(sample, prep, "body")
        variables = edge_out(sample, prep, "variable")
        if len(bodies) != 1 or variables:
            continue

        set_node_token_indexes(sample, node_index, be_indexes + prep_indexes)
        body = bodies[0]["end"]
        remove_edges(
            sample,
            lambda edge, n=node_index, p=prep: (
                edge["start"] == n and edge["end"] == p and edge["label"] == "pred.arg.2"
            )
            or edge["start"] == p
            or edge["end"] == p,
        )
        add_edge(sample, node_index, body, "pred.arg.2")
        removed.add(prep)
        changes.append("split_direct_be_prep_merged")

    if removed:
        remove_nodes(sample, removed)
    return changes


def partition_merged_be_prep(sample: dict[str, Any], node_index: int) -> tuple[list[int], list[int], list[int], int] | None:
    tokens = node_tokens(sample, node_index)
    if not tokens:
        return None
    normalized = [token for _, _, token in tokens]
    be_positions = [pos for pos, token in enumerate(normalized) if token in BE_FORMS]
    if not be_positions:
        return None
    be_pos = be_positions[-1]
    prep_positions = [pos for pos, token in enumerate(normalized) if token in DIRECT_PREPOSITIONS and pos > be_pos]
    if not prep_positions:
        return None

    prep_pos = prep_positions[-1]
    if prep_pos == be_pos + 1:
        return None
    if prep_pos != len(tokens) - 1:
        return None
    if normalized[prep_pos] == "to" and any(token in {"going", "likely", "scheduled"} for token in normalized):
        return None
    if normalized[prep_pos] == "as" and any(token in {"good", "hopeful", "useful"} for token in normalized):
        return None

    between = tokens[be_pos + 1 : prep_pos]
    if not between:
        return None
    modifier_indexes = [
        index
        for index, _, token in between
        if token in ADVERBIAL_MODIFIERS or token in NEGATION_PARTS or token == "'"
    ]
    state_indexes = [
        index
        for index, _, token in between
        if index not in set(modifier_indexes)
    ]
    if state_indexes:
        state_norms = [token for index, _, token in between if index in set(state_indexes)]
        if not all(looks_like_state_word(token) or token in {"all", "longer"} for token in state_norms):
            return None

    be_indexes = [
        index
        for position, (index, _, token) in enumerate(tokens[:prep_pos])
        if token in BE_FORMS or is_be_apostrophe(tokens, position)
    ]
    predicate_indexes = be_indexes + state_indexes
    if not predicate_indexes:
        return None
    prep_indexes = [tokens[prep_pos][0]]
    return predicate_indexes, modifier_indexes, prep_indexes, prep_pos


def transfer_incoming_to_new_node(
    sample: dict[str, Any],
    old: int,
    new: int,
    excluded_starts: set[int] | None = None,
) -> None:
    excluded_starts = excluded_starts or set()
    for edge in graph(sample)["edges"]:
        if edge["end"] == old and edge["start"] != new and edge["start"] not in excluded_starts:
            edge["end"] = new


def grouped_modifier_spans(tokens: list[tuple[int, str, str]], modifier_indexes: list[int]) -> list[list[int]]:
    modifier_set = set(modifier_indexes)
    groups: list[list[int]] = []
    position = 0
    while position < len(tokens):
        index, _, token = tokens[position]
        if index not in modifier_set:
            position += 1
            continue
        if token == "n" and position + 2 < len(tokens):
            second = tokens[position + 1]
            third = tokens[position + 2]
            if (
                second[0] in modifier_set
                and third[0] in modifier_set
                and second[2] == "'"
                and third[2] == "t"
            ):
                groups.append([index, third[0]])
                position += 3
                continue
        groups.append([index, index])
        position += 1
    return groups


def add_modifier_chain(sample: dict[str, Any], modifier_spans: list[list[int]], target: int) -> int:
    current_target = target
    for start, end in reversed(modifier_spans):
        modifier = add_node(sample, [[start, end]], "ModificationalConstant")
        add_edge(sample, modifier, current_target, "as:modification")
        current_target = modifier
    return current_target


def repair_merged_be_intervened_prep(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    original_node_count = len(graph(sample)["nodes"])
    for node_index in range(original_node_count):
        if not is_factual(sample, node_index) or not has_be(sample, node_index):
            continue
        arg2_edges = edge_out(sample, node_index, "pred.arg.2")
        if len(arg2_edges) != 1:
            continue
        partition = partition_merged_be_prep(sample, node_index)
        if partition is None:
            continue
        predicate_indexes, modifier_indexes, prep_indexes, _ = partition
        if not modifier_indexes and len(predicate_indexes) <= 1:
            continue

        object_node = arg2_edges[0]["end"]
        prep_node = add_node(sample, spans_from_token_indexes(prep_indexes), "ModificationalFunctor")
        set_node_token_indexes(sample, node_index, predicate_indexes)
        transfer_incoming_to_new_node(sample, node_index, prep_node)
        modifier_tokens = [(index, words(sample)[index], norm(words(sample)[index])) for index in sorted(modifier_indexes)]
        modifier_root = add_modifier_chain(
            sample,
            grouped_modifier_spans(modifier_tokens, modifier_indexes) if modifier_indexes else [],
            node_index,
        )
        remove_edges(
            sample,
            lambda edge, n=node_index, obj=object_node: (
                edge["start"] == n and edge["end"] == obj and edge["label"] == "pred.arg.2"
            ),
        )
        add_edge(sample, prep_node, modifier_root, "variable")
        add_edge(sample, prep_node, object_node, "body")
        changes.append("merged_intervened_be_prep_split")
    return changes


def partition_merged_simple_copula(sample: dict[str, Any], node_index: int) -> tuple[list[int], list[int]] | None:
    if edge_out(sample, node_index, "pred.arg.2"):
        return None
    tokens = node_tokens(sample, node_index)
    if not tokens:
        return None
    normalized = [token for _, _, token in tokens if token != "'"]
    if any(token in MODALS_AND_AUX for token in normalized):
        return None
    if any(token in DIRECT_PREPOSITIONS for token in normalized):
        return None
    if not any(token in BE_FORMS for token in normalized):
        return None

    modifier_indexes: list[int] = []
    predicate_indexes: list[int] = []
    state_tokens: list[str] = []
    for position, (index, _, token) in enumerate(tokens):
        if is_be_apostrophe(tokens, position):
            predicate_indexes.append(index)
        elif is_negation_apostrophe(tokens, position):
            modifier_indexes.append(index)
        elif token == "'":
            predicate_indexes.append(index)
        elif token in BE_FORMS:
            predicate_indexes.append(index)
        elif token in ADVERBIAL_MODIFIERS or token in NEGATION_PARTS:
            modifier_indexes.append(index)
        else:
            predicate_indexes.append(index)
            state_tokens.append(token)

    if not modifier_indexes or not state_tokens:
        return None
    if len(state_tokens) > 2:
        return None
    if not looks_like_state_word(state_tokens[-1]):
        return None
    return predicate_indexes, modifier_indexes


def repair_merged_simple_copula_adverbs(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    original_node_count = len(graph(sample)["nodes"])
    for node_index in range(original_node_count):
        if not is_factual(sample, node_index) or not has_be(sample, node_index):
            continue
        partition = partition_merged_simple_copula(sample, node_index)
        if partition is None:
            continue
        predicate_indexes, modifier_indexes = partition
        modifier_spans = grouped_modifier_spans(node_tokens(sample, node_index), modifier_indexes)
        set_node_token_indexes(sample, node_index, predicate_indexes)
        old_node_count = len(graph(sample)["nodes"])
        modifier_root = add_modifier_chain(sample, modifier_spans, node_index)
        new_nodes = set(range(old_node_count, len(graph(sample)["nodes"])))
        transfer_incoming_to_new_node(sample, node_index, modifier_root, excluded_starts=new_nodes)
        changes.append("merged_simple_copula_adverb_split")
    return changes


def repair_sample(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    # Order matters: normalize existential and already split structures before
    # splitting merged predicates.
    changes.extend(repair_there_be_merged(sample))
    changes.extend(repair_there_be_arg4(sample))
    changes.extend(repair_split_direct_be_prep(sample))
    changes.extend(repair_split_simple_copula(sample))
    changes.extend(repair_merged_be_intervened_prep(sample))
    changes.extend(repair_merged_simple_copula_adverbs(sample))
    return changes


def compact_graph(sample: dict[str, Any]) -> str:
    lines = [sample["sentence"], "NODES"]
    for index, node in enumerate(graph(sample)["nodes"]):
        lines.append(f"{index}: {node_text(sample, index)!r} {node.get('spans')!r} {node.get('type')}")
    lines.append("EDGES")
    for edge in graph(sample)["edges"]:
        lines.append(
            f"{edge['start']}->{edge['end']} {edge['label']} | "
            f"{node_text(sample, edge['start'])!r}->{node_text(sample, edge['end'])!r}"
        )
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def run(apply: bool, data_dir: Path, report_dir: Path) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    all_changes: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        new_rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(rows, 1):
            before = deepcopy(row)
            after = deepcopy(row)
            changes = repair_sample(after)
            if changes:
                for change in changes:
                    counts[f"{split}:{change}"] += 1
                    counts[change] += 1
                all_changes.append(
                    {
                        "split": split,
                        "line_number": line_number,
                        "sentence": row.get("sentence", ""),
                        "changes": changes,
                        "before": compact_graph(before),
                        "after": compact_graph(after),
                        "sample": after,
                    }
                )
                new_rows.append(after)
            else:
                new_rows.append(row)
        if apply:
            write_jsonl(path, new_rows)

    changes_path = report_dir / "changes.jsonl"
    changes_path.write_text(
        "\n".join(json.dumps(change, ensure_ascii=False) for change in all_changes) + ("\n" if all_changes else ""),
        encoding="utf-8",
    )

    summary_lines = [
        "# Be/Copula Rule Repair Summary",
        "",
        f"- mode: {'apply' if apply else 'dry-run'}",
        f"- changed samples: {len(all_changes)}",
        "",
        "## Change Counts",
        "",
        "| key | count |",
        "|---|---:|",
    ]
    for key, count in sorted(counts.items()):
        summary_lines.append(f"| {key} | {count} |")
    summary_lines.extend(
        [
            "",
            "## First 40 Changes",
            "",
        ]
    )
    for change in all_changes[:40]:
        summary_lines.extend(
            [
                f"### {change['split']}:{change['line_number']} `{', '.join(change['changes'])}`",
                "",
                change["sentence"],
                "",
                "```text",
                change["before"],
                "",
                "--- after ---",
                change["after"],
                "```",
                "",
            ]
        )
    summary_path = report_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(json.dumps({"apply": apply, "changed_samples": len(all_changes), "counts": counts}, ensure_ascii=False, indent=2))
    print(f"wrote {summary_path}")
    print(f"wrote {changes_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="rewrite data/train.jsonl, data/dev.jsonl, and data/test.jsonl")
    parser.add_argument("--data-dir", type=Path, default=DATA, help="directory containing train/dev/test JSONL files")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORTS / "be_copula_rule_repair",
        help="directory for summary.md and changes.jsonl",
    )
    args = parser.parse_args()
    return run(args.apply, args.data_dir, args.report_dir)


if __name__ == "__main__":
    raise SystemExit(main())
