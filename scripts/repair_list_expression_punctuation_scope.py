#!/usr/bin/env python3
"""Repair high-confidence List/Expression punctuation scope leftovers."""

from __future__ import annotations

import importlib.util
import json
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
    module.OPERATOR_TOKENS.add("_")
    return module


EXPR_REPAIR = load_expression_repair_module()


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if not isinstance(spans, list):
        return []
    return [i for start, end in spans for i in range(start, end + 1)]


def token_text(sample: dict[str, Any], node_index: int) -> str:
    words = sample["lingua"]["words"]
    tokens = node_token_indexes(sample["lingua"]["graph"]["nodes"][node_index])
    return "".join(words[token] for token in tokens)


def clean_token_text(sample: dict[str, Any], node_index: int) -> str:
    return token_text(sample, node_index).strip()


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return edge["start"], edge["end"], edge["label"]


def add_edge(edges: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    new_edge = {"start": start, "end": end, "label": label}
    if edge_key(new_edge) not in {edge_key(edge) for edge in edges}:
        edges.append(new_edge)


def remove_first_edge(
    edges: list[dict[str, Any]], start: int, end: int, label: str
) -> bool:
    for index, edge in enumerate(edges):
        if (
            edge["start"] == start
            and edge["end"] == end
            and edge["label"] == label
        ):
            del edges[index]
            return True
    return False


def dedupe_edges(sample: dict[str, Any]) -> None:
    sample["lingua"]["graph"]["edges"] = EXPR_REPAIR.remove_duplicate_edges(
        sample["lingua"]["graph"]["edges"]
    )


def descendant_tokens(sample: dict[str, Any], node_index: int) -> list[int]:
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    seen: set[int] = set()
    out: list[int] = []

    def visit(index: int) -> None:
        if index in seen:
            return
        seen.add(index)
        out.extend(node_token_indexes(nodes[index]))
        for edge in edges:
            if edge["start"] == index:
                visit(edge["end"])

    visit(node_index)
    return sorted(set(out))


def descendant_range(sample: dict[str, Any], node_index: int) -> tuple[int, int] | None:
    tokens = descendant_tokens(sample, node_index)
    if not tokens:
        return None
    return min(tokens), max(tokens)


def list_args(sample: dict[str, Any], source: int) -> list[int]:
    return [
        edge["end"]
        for edge in sample["lingua"]["graph"]["edges"]
        if edge["start"] == source and edge["label"] == "func.arg"
    ]


def surrounding_args(
    sample: dict[str, Any], source: int, punct: int
) -> tuple[int | None, int | None]:
    punct_range = descendant_range(sample, punct)
    if punct_range is None:
        return None, None
    before: list[tuple[int, int]] = []
    after: list[tuple[int, int]] = []
    for arg in list_args(sample, source):
        arg_range = descendant_range(sample, arg)
        if arg_range is None:
            continue
        if arg_range[1] < punct_range[0]:
            before.append((arg_range[1], arg))
        elif arg_range[0] > punct_range[1]:
            after.append((arg_range[0], arg))
    prev_arg = max(before)[1] if before else None
    next_arg = min(after)[1] if after else None
    return prev_arg, next_arg


def convert_list_separator_to_attribute(
    sample: dict[str, Any], source: int, punct: int, arg1: int, arg2: int
) -> bool:
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    if not remove_first_edge(edges, source, punct, "punctuation"):
        return False
    nodes[source]["type"] = "AttributePredicator"
    nodes[source]["spans"] = "Attribute"
    for edge in edges:
        if edge["start"] != source or edge["label"] != "func.arg":
            continue
        if edge["end"] == arg1:
            edge["label"] = "pred.arg.1"
            edge["end"] = punct
        elif edge["end"] == arg2:
            edge["label"] = "pred.arg.2"
    add_edge(edges, punct, arg1, "as:punctuation")
    dedupe_edges(sample)
    return True


def close_previous_list_arg(
    sample: dict[str, Any], source: int, punct: int, arg: int
) -> bool:
    edges = sample["lingua"]["graph"]["edges"]
    if not remove_first_edge(edges, source, punct, "punctuation"):
        return False
    for edge in edges:
        if edge["start"] == source and edge["end"] == arg and edge["label"] == "func.arg":
            edge["end"] = punct
            add_edge(edges, punct, arg, "as:punctuation")
            dedupe_edges(sample)
            return True
    return False


def wrap_source_with_punctuation(
    sample: dict[str, Any], source: int, punct: int
) -> bool:
    edges = sample["lingua"]["graph"]["edges"]
    if not remove_first_edge(edges, source, punct, "punctuation"):
        return False
    for edge in edges:
        if edge["end"] == source and edge["start"] != punct:
            edge["end"] = punct
    add_edge(edges, punct, source, "as:punctuation")
    dedupe_edges(sample)
    return True


def repair_pin_hash_field(sample: dict[str, Any]) -> bool:
    if sample.get("sentence") != "Pin #for Paul Kaufman ONLY: 362416":
        return False
    graph = sample["lingua"]["graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    source = 3
    colon = 0
    value = 2
    nodes.append({"spans": "Attribute", "type": "AttributePredicator"})
    attribute = len(nodes) - 1
    edges[:] = [
        edge
        for edge in edges
        if not (
            edge["start"] == source
            and (
                (edge["end"] == colon and edge["label"] == "punctuation")
                or (edge["end"] == value and edge["label"] == "pred.arg.2")
            )
        )
    ]
    add_edge(edges, attribute, colon, "pred.arg.1")
    add_edge(edges, attribute, value, "pred.arg.2")
    add_edge(edges, colon, source, "as:punctuation")
    dedupe_edges(sample)
    return True


def repair_vote_result_colon(sample: dict[str, Any]) -> bool:
    if not sample.get("sentence", "").startswith(
        "The result of the vote of the 15 member countries"
    ):
        return False
    edges = sample["lingua"]["graph"]["edges"]
    changed = remove_first_edge(edges, 30, 15, "punctuation")
    if changed:
        for edge in edges:
            if edge["start"] == 29 and edge["end"] == 0 and edge["label"] == "as:punctuation":
                edge["end"] = 15
        add_edge(edges, 15, 0, "as:punctuation")
        dedupe_edges(sample)
    return changed


def repair_list_punctuation(sample: dict[str, Any]) -> list[str]:
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    changes: list[str] = []
    candidates = [
        (edge["start"], edge["end"])
        for edge in list(edges)
        if edge["label"] == "punctuation"
        and nodes[edge["start"]]["type"] == "ListFunctor"
        and nodes[edge["end"]]["type"] == "PunctuationalConstant"
    ]
    for source, punct in candidates:
        if not any(
            edge["start"] == source
            and edge["end"] == punct
            and edge["label"] == "punctuation"
            for edge in edges
        ):
            continue
        mark = clean_token_text(sample, punct)
        prev_arg, next_arg = surrounding_args(sample, source, punct)
        args = list_args(sample, source)
        if mark in {"-", "–", "—", "."} and prev_arg is not None and next_arg is not None and len(args) == 2:
            if convert_list_separator_to_attribute(sample, source, punct, prev_arg, next_arg):
                changes.append(f"list_to_attribute:{source}->{punct}")
        elif mark in {",", "/", ";"} and prev_arg is not None:
            if close_previous_list_arg(sample, source, punct, prev_arg):
                changes.append(f"list_close_arg:{source}->{punct}")
    return changes


def repair_aux_expression_punctuation(sample: dict[str, Any]) -> list[str]:
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    changes: list[str] = []
    sources = sorted(
        {
            edge["start"]
            for edge in edges
            if edge["label"] == "punctuation"
            and nodes[edge["start"]]["type"] == "ExpressionFunctor"
            and isinstance(nodes[edge["start"]]["spans"], str)
        }
    )
    for source in sources:
        tokens: set[int] = set()
        for edge in edges:
            if edge["start"] == source and edge["label"] in {"func.arg", "punctuation"}:
                tokens.update(descendant_tokens(sample, edge["end"]))
        if not tokens:
            continue
        if EXPR_REPAIR.rewrite_existing_token_group(sample, sorted(tokens), url_mode=True):
            for edge in sample["lingua"]["graph"]["edges"]:
                if edge["start"] == source and edge["label"] == "punctuation":
                    edge["label"] = "func.arg"
            dedupe_edges(sample)
            changes.append(f"aux_expression:{source}")
    return changes


def repair_lexical_expression_punctuation(sample: dict[str, Any]) -> list[str]:
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    changes: list[str] = []
    candidates = [
        (edge["start"], edge["end"])
        for edge in list(edges)
        if edge["label"] == "punctuation"
        and nodes[edge["start"]]["type"] == "ExpressionFunctor"
        and not isinstance(nodes[edge["start"]]["spans"], str)
        and nodes[edge["end"]]["type"] == "PunctuationalConstant"
    ]
    for source, punct in candidates:
        if not any(
            edge["start"] == source
            and edge["end"] == punct
            and edge["label"] == "punctuation"
            for edge in edges
        ):
            continue
        source_range = descendant_range(sample, source)
        punct_range = descendant_range(sample, punct)
        if source_range is None or punct_range is None:
            continue
        mark = clean_token_text(sample, punct)
        wraps_source = punct_range[0] <= source_range[0] and punct_range[1] >= source_range[1]
        closes_after_source = punct_range[0] > source_range[1]
        if wraps_source or closes_after_source or mark in {'"', "'", "()", "[]"}:
            if wrap_source_with_punctuation(sample, source, punct):
                changes.append(f"expression_wrap:{source}->{punct}")
    return changes


def repair_sample(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    if repair_pin_hash_field(sample):
        changes.append("pin_hash_attribute")
    changes.extend(repair_list_punctuation(sample))
    if repair_vote_result_colon(sample):
        changes.append("vote_result_colon")
    changes.extend(repair_aux_expression_punctuation(sample))
    changes.extend(repair_lexical_expression_punctuation(sample))
    return changes


def main() -> None:
    total = 0
    by_file: dict[str, int] = {}
    for path in DATA_FILES:
        samples: list[dict[str, Any]] = []
        file_total = 0
        for line in path.read_text().splitlines():
            sample = json.loads(line)
            changes = repair_sample(sample)
            file_total += len(changes)
            samples.append(sample)
        if file_total:
            path.write_text(
                "\n".join(json.dumps(sample, ensure_ascii=False) for sample in samples)
                + "\n"
            )
        by_file[path.name] = file_total
        total += file_total
    print(f"repaired {total} List/Expression punctuation structures")
    for name, count in by_file.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
