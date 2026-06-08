#!/usr/bin/env python3
"""Repair prefixal hyphen modifiers such as non-nuclear and pro-choice.

The repaired structure follows docs/standard.md:

    a/ModificationalConstant
    -/PunctuationalConstant --as:punctuation--> a
    Modification/ModificationalFunctor
      variable -> b
      body -> -

Any external edge that previously pointed at the whole `a-b` expression points
at the new Modification wrapper after the repair.
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
REPORTS = ROOT / "reports" / "hyphen_prefix_modifier_repair"

PREFIX_MODIFIERS = {"non", "pro", "anti"}

RIGHT_MODIFIER_WORDS = {
    "american",
    "commercial",
    "crumbly",
    "cultured",
    "democratic",
    "essential",
    "human",
    "israeli",
    "pakistani",
    "proliferative",
    "serbian",
    "working",
}

MOD_CONST = "ModificationalConstant"
MOD_FUNCTOR = "ModificationalFunctor"
NOM_CONST = "NominalConstant"
PUNCT_CONST = "PunctuationalConstant"
EXPR_FUNCTOR = "ExpressionFunctor"


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def words(sample: dict[str, Any]) -> list[str]:
    return sample["lingua"]["words"]


def node_text(sample: dict[str, Any], node_index: int) -> str:
    node = graph(sample)["nodes"][node_index]
    spans = node.get("spans")
    if isinstance(spans, str):
        return spans
    pieces: list[str] = []
    for span in spans or []:
        if isinstance(span, list) and len(span) == 2:
            pieces.extend(words(sample)[span[0] : span[1] + 1])
    return "".join(pieces).strip()


def token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    indexes: list[int] = []
    if isinstance(spans, list):
        for span in spans:
            if isinstance(span, list) and len(span) == 2:
                indexes.extend(range(span[0], span[1] + 1))
    return indexes


def add_node(sample: dict[str, Any], spans: Any, node_type: str) -> int:
    graph(sample)["nodes"].append({"spans": spans, "type": node_type})
    return len(graph(sample)["nodes"]) - 1


def add_edge_once(sample: dict[str, Any], start: int, end: int, label: str) -> None:
    for edge in graph(sample)["edges"]:
        if edge["start"] == start and edge["end"] == end and edge["label"] == label:
            return
    graph(sample)["edges"].append({"start": start, "end": end, "label": label})


def dedupe_edges(sample: dict[str, Any]) -> None:
    seen: set[tuple[int, int, str]] = set()
    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        key = (edge["start"], edge["end"], edge["label"])
        if key in seen:
            continue
        seen.add(key)
        new_edges.append(edge)
    graph(sample)["edges"] = new_edges


def right_type_for_word(right_word: str, whole_type: str) -> str:
    lowered = right_word.strip().lower()
    if lowered in RIGHT_MODIFIER_WORDS:
        return MOD_CONST
    if lowered.endswith(("able", "ible", "al", "ary", "ic", "ive", "less", "ous")):
        return MOD_CONST
    if whole_type == NOM_CONST:
        return NOM_CONST
    return NOM_CONST


def split_expression_candidates(sample: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = graph(sample)["nodes"]
    edges = graph(sample)["edges"]
    rows: list[dict[str, Any]] = []
    for dash_index, node in enumerate(nodes):
        if node.get("type") != EXPR_FUNCTOR or node_text(sample, dash_index) != "-":
            continue
        children = [edge["end"] for edge in edges if edge["start"] == dash_index and edge["label"] == "func.arg"]
        if len(children) < 2:
            continue
        children = sorted(children, key=lambda child: token_indexes(nodes[child])[:1] or [10**9])
        left, right = children[0], children[1]
        left_word = node_text(sample, left).strip().lower()
        if left_word not in PREFIX_MODIFIERS:
            continue
        rows.append({"kind": "split_expression", "dash": dash_index, "left": left, "right": right})
    return rows


def whole_node_candidates(sample: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    toks = words(sample)
    for node_index, node in enumerate(graph(sample)["nodes"]):
        indexes = token_indexes(node)
        if len(indexes) != 3:
            continue
        left_i, dash_i, right_i = indexes
        if toks[dash_i].strip() != "-":
            continue
        left_word = toks[left_i].strip().lower()
        if left_word not in PREFIX_MODIFIERS:
            continue
        rows.append(
            {
                "kind": "whole_node",
                "node": node_index,
                "left_i": left_i,
                "dash_i": dash_i,
                "right_i": right_i,
                "right_type": right_type_for_word(toks[right_i], node.get("type", "")),
            }
        )
    return rows


def repair_split_expression(sample: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    dash = cand["dash"]
    left = cand["left"]
    right = cand["right"]
    before_type = graph(sample)["nodes"][left].get("type")

    graph(sample)["nodes"][left]["type"] = MOD_CONST
    graph(sample)["nodes"][dash]["type"] = PUNCT_CONST
    mod = add_node(sample, "Modification", MOD_FUNCTOR)

    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        if edge["start"] == dash and edge["label"] == "func.arg" and edge["end"] in {left, right}:
            continue
        new_edge = dict(edge)
        if new_edge["end"] == dash:
            new_edge["end"] = mod
        elif new_edge["start"] == dash:
            new_edge["start"] = mod
        new_edges.append(new_edge)
    graph(sample)["edges"] = new_edges

    add_edge_once(sample, dash, left, "as:punctuation")
    add_edge_once(sample, mod, right, "variable")
    add_edge_once(sample, mod, dash, "body")
    dedupe_edges(sample)

    return {
        "kind": "split_expression",
        "left": node_text(sample, left),
        "right": node_text(sample, right),
        "left_type_before": before_type,
        "left_type_after": MOD_CONST,
        "dash_type_after": PUNCT_CONST,
    }


def repair_whole_node(sample: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    old = cand["node"]
    toks = words(sample)
    left_word = toks[cand["left_i"]].strip()
    right_word = toks[cand["right_i"]].strip()
    old_text = node_text(sample, old)
    old_type = graph(sample)["nodes"][old].get("type")
    old_outgoing_mod_heads = [
        edge["end"]
        for edge in graph(sample)["edges"]
        if edge["start"] == old and edge["label"] == "as:modification"
    ]

    graph(sample)["nodes"][old] = {"spans": [[cand["left_i"], cand["left_i"]]], "type": MOD_CONST}
    dash = add_node(sample, [[cand["dash_i"], cand["dash_i"]]], PUNCT_CONST)
    right = add_node(sample, [[cand["right_i"], cand["right_i"]]], cand["right_type"])
    inner_mod = add_node(sample, "Modification", MOD_FUNCTOR)
    outer_mods: list[int] = []
    for head in old_outgoing_mod_heads:
        outer = add_node(sample, "Modification", MOD_FUNCTOR)
        outer_mods.append(outer)
        add_edge_once(sample, outer, head, "variable")
        add_edge_once(sample, outer, inner_mod, "body")

    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        if edge["start"] == old and edge["label"] == "as:modification":
            continue
        new_edge = dict(edge)
        if new_edge["start"] == old:
            new_edge["start"] = inner_mod
        if new_edge["end"] == old:
            if outer_mods:
                new_edge["end"] = outer_mods[0]
            else:
                new_edge["end"] = inner_mod
        new_edges.append(new_edge)
    graph(sample)["edges"] = new_edges

    add_edge_once(sample, dash, old, "as:punctuation")
    add_edge_once(sample, inner_mod, right, "variable")
    add_edge_once(sample, inner_mod, dash, "body")
    dedupe_edges(sample)

    return {
        "kind": "whole_node",
        "old_text": old_text,
        "old_type": old_type,
        "left": left_word,
        "right": right_word,
        "left_type_after": MOD_CONST,
        "right_type_after": cand["right_type"],
        "dash_type_after": PUNCT_CONST,
        "outer_modifications_added": len(outer_mods),
    }


def repair_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []

    # Repair existing split expressions first. Recompute after each change because
    # appended nodes change indexes but existing candidate node ids stay stable.
    while True:
        candidates = split_expression_candidates(sample)
        if not candidates:
            break
        changes.append(repair_split_expression(sample, candidates[0]))

    while True:
        candidates = whole_node_candidates(sample)
        if not candidates:
            break
        changes.append(repair_whole_node(sample, candidates[0]))

    return changes


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "changes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (REPORTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    summary_counts: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()

    for split in ["train", "dev", "test"]:
        path = args.data_dir / f"{split}.jsonl"
        samples = read_jsonl(path)
        out_samples = deepcopy(samples)
        for idx, sample in enumerate(out_samples):
            changes = repair_sample(sample)
            if not changes:
                continue
            by_split[split] += 1
            summary_counts["samples_changed"] += 1
            summary_counts["changes"] += len(changes)
            for change in changes:
                by_kind[change["kind"]] += 1
                all_rows.append(
                    {
                        "split": split,
                        "idx": idx,
                        "line": idx + 1,
                        "sentence": sample["sentence"],
                        **change,
                    }
                )
        if args.apply:
            write_jsonl(path, out_samples)

    summary = {
        "apply": args.apply,
        "samples_changed": summary_counts["samples_changed"],
        "changes": summary_counts["changes"],
        "by_split": dict(by_split),
        "by_kind": dict(by_kind),
    }
    write_report(all_rows, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
