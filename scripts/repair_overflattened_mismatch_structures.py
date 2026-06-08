#!/usr/bin/env python3
"""Repair reviewed over-flattened expression/list structures.

The repairs here are intentionally narrow:
- date + time + AM/PM expressions: attach AM/PM to the time expression;
- range endpoints with a local modifier/status: attach that modifier to the
  endpoint instead of keeping it as a third range argument;
- hyphenated number + percent: keep the hyphenated number as the modifier of
  the unit;
- determiner + hyphen range: let the determiner wrap the range expression;
- list items duplicated with their punctuation close wrapper: keep the wrapper.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def node_text(words: list[str], node: dict) -> str:
    spans = node["spans"]
    if isinstance(spans, str):
        return spans
    parts: list[str] = []
    for start, end in spans:
        parts.extend(words[index].strip() for index in range(start, end + 1))
    return " ".join(parts)


def first_pos(node: dict) -> int:
    spans = node["spans"]
    if isinstance(spans, str):
        return 10**9
    return min(start for start, _ in spans)


def structural_first_pos(graph: dict, node_id: int, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    if node_id in seen:
        return 10**9
    seen.add(node_id)
    node = graph["nodes"][node_id]
    pos = first_pos(node)
    if pos != 10**9:
        return pos
    child_positions = [
        structural_first_pos(graph, edge["end"], seen)
        for edge in graph["edges"]
        if edge["start"] == node_id
    ]
    return min(child_positions, default=10**9)


def outgoing(edges: list[dict]) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {}
    for edge in edges:
        result.setdefault(edge["start"], []).append(edge)
    return result


def add_edge_once(edges: list[dict], start: int, end: int, label: str) -> None:
    if any(edge["start"] == start and edge["end"] == end and edge["label"] == label for edge in edges):
        return
    edges.append({"start": start, "end": end, "label": label})


def remove_edge_once(edges: list[dict], start: int, end: int, label: str) -> bool:
    for index, edge in enumerate(edges):
        if edge["start"] == start and edge["end"] == end and edge["label"] == label:
            del edges[index]
            return True
    return False


def func_args(graph: dict, node_id: int) -> list[int]:
    return [edge["end"] for edge in outgoing(graph["edges"]).get(node_id, []) if edge["label"] == "func.arg"]


def retarget_incoming(graph: dict, old_end: int, new_end: int) -> int:
    changed = 0
    for edge in graph["edges"]:
        if edge["end"] == old_end:
            edge["end"] = new_end
            changed += 1
    return changed


def repair_timestamp_ampm(sample: dict) -> int:
    words = sample["lingua"]["words"]
    graph = sample["lingua"]["graph"]
    nodes = graph["nodes"]
    repairs = 0
    for expr_id, node in enumerate(nodes):
        if node.get("type") != "ExpressionFunctor":
            continue
        args = func_args(graph, expr_id)
        date_args = [
            arg
            for arg in args
            if nodes[arg].get("type") == "ExpressionFunctor" and node_text(words, nodes[arg]).strip() == "/"
        ]
        time_args = [
            arg
            for arg in args
            if nodes[arg].get("type") == "ExpressionFunctor" and node_text(words, nodes[arg]).strip() == ":"
        ]
        ampm_args = [
            arg
            for arg in args
            if nodes[arg].get("type") == "NominalConstant" and node_text(words, nodes[arg]).strip() in {"AM", "PM"}
        ]
        if not date_args or not time_args or not ampm_args:
            continue
        time_id = min(time_args, key=lambda arg: first_pos(nodes[arg]))
        for ampm_id in ampm_args:
            if remove_edge_once(graph["edges"], expr_id, ampm_id, "func.arg"):
                add_edge_once(graph["edges"], time_id, ampm_id, "func.arg")
                repairs += 1
    return repairs


def repair_hyphen_with_determiner(sample: dict) -> int:
    words = sample["lingua"]["words"]
    graph = sample["lingua"]["graph"]
    nodes = graph["nodes"]
    repairs = 0
    for hyphen_id, node in enumerate(nodes):
        if node.get("type") != "ExpressionFunctor" or node_text(words, node).strip() not in {"-", "–", "—"}:
            continue
        args = func_args(graph, hyphen_id)
        determiner_args = [
            arg
            for arg in args
            if nodes[arg].get("type") == "DeterminerConstant"
            and node_text(words, nodes[arg]).strip().lower() in {"the", "a", "an"}
        ]
        for det_id in determiner_args:
            if not remove_edge_once(graph["edges"], hyphen_id, det_id, "func.arg"):
                continue
            old_det_targets = [
                edge["end"]
                for edge in list(graph["edges"])
                if edge["start"] == det_id and edge["label"] == "as:modification"
            ]
            graph["edges"] = [
                edge
                for edge in graph["edges"]
                if not (edge["start"] == det_id and edge["label"] == "as:modification")
            ]
            for old_target_id in old_det_targets:
                add_edge_once(graph["edges"], hyphen_id, old_target_id, "func.arg")
            retarget_incoming(graph, hyphen_id, det_id)
            add_edge_once(graph["edges"], det_id, hyphen_id, "as:modification")
            repairs += 1
    return repairs


def repair_hyphen_number_percent(sample: dict) -> int:
    words = sample["lingua"]["words"]
    graph = sample["lingua"]["graph"]
    nodes = graph["nodes"]
    repairs = 0
    number_words = {
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    }

    for hyphen_id, node in list(enumerate(nodes)):
        if node.get("type") != "ExpressionFunctor" or node_text(words, node).strip() not in {"-", "–", "—"}:
            continue
        args = func_args(graph, hyphen_id)
        percent_args = [
            arg
            for arg in args
            if nodes[arg].get("type") == "NominalConstant"
            and node_text(words, nodes[arg]).strip().lower() in {"percent", "percentage"}
        ]
        number_args = [
            arg
            for arg in args
            if nodes[arg].get("type") == "NominalConstant"
            and node_text(words, nodes[arg]).strip().lower() in number_words
        ]
        if len(percent_args) != 1 or len(number_args) < 2:
            continue

        percent_id = percent_args[0]
        if not remove_edge_once(graph["edges"], hyphen_id, percent_id, "func.arg"):
            continue
        mod_id = len(nodes)
        nodes.append({"spans": "Modification", "type": "ModificationalFunctor"})
        retarget_incoming(graph, hyphen_id, mod_id)
        add_edge_once(graph["edges"], mod_id, percent_id, "variable")
        add_edge_once(graph["edges"], mod_id, hyphen_id, "body")
        repairs += 1
    return repairs


def repair_dash_endpoint_modifiers(sample: dict) -> int:
    words = sample["lingua"]["words"]
    graph = sample["lingua"]["graph"]
    nodes = graph["nodes"]
    repairs = 0

    for dash_id, node in enumerate(nodes):
        if node.get("type") != "ExpressionFunctor" or node_text(words, node).strip() not in {"-", "–", "—"}:
            continue
        args = func_args(graph, dash_id)
        if len(args) < 3:
            continue

        for mod_id in list(args):
            mod_node = nodes[mod_id]
            mod_type = mod_node.get("type")
            mod_text = node_text(words, mod_node).strip().lower()
            if mod_type == "ModificationalFunctor":
                existing_labels = {edge["label"] for edge in outgoing(graph["edges"]).get(mod_id, [])}
                if "variable" in existing_labels:
                    continue
                candidates = [
                    arg
                    for arg in args
                    if arg != mod_id
                    and structural_first_pos(graph, arg) < structural_first_pos(graph, mod_id)
                    and nodes[arg].get("type") in {"NominalConstant", "ExpressionFunctor"}
                ]
                if not candidates:
                    continue
                endpoint_id = max(candidates, key=lambda arg: structural_first_pos(graph, arg))
                if remove_edge_once(graph["edges"], dash_id, endpoint_id, "func.arg"):
                    add_edge_once(graph["edges"], mod_id, endpoint_id, "variable")
                    repairs += 1
            elif mod_type == "ModificationalConstant" and mod_text in {"deceased", "born"}:
                candidates = [
                    arg
                    for arg in args
                    if arg != mod_id
                    and structural_first_pos(graph, arg) < structural_first_pos(graph, mod_id)
                    and nodes[arg].get("type") in {"NominalConstant", "ExpressionFunctor"}
                ]
                if not candidates:
                    continue
                endpoint_id = max(candidates, key=lambda arg: structural_first_pos(graph, arg))
                if remove_edge_once(graph["edges"], dash_id, endpoint_id, "func.arg"):
                    add_edge_once(graph["edges"], mod_id, endpoint_id, "as:modification")
                    repairs += 1
    return repairs


def repair_duplicate_list_close_wrappers(sample: dict) -> int:
    graph = sample["lingua"]["graph"]
    nodes = graph["nodes"]
    out = outgoing(graph["edges"])
    repairs = 0
    punctuation_targets = {
        (punct_id, edge["end"])
        for punct_id, node in enumerate(nodes)
        if node.get("type") == "PunctuationalConstant"
        for edge in out.get(punct_id, [])
        if edge["label"] == "as:punctuation"
    }
    for list_id, node in enumerate(nodes):
        if node.get("type") != "ListFunctor":
            continue
        args = set(func_args(graph, list_id))
        for punct_id, target_id in sorted(punctuation_targets):
            if punct_id in args and target_id in args:
                if remove_edge_once(graph["edges"], list_id, target_id, "func.arg"):
                    repairs += 1
    return repairs


def main() -> None:
    totals: Counter[str] = Counter()
    for split in ("train", "dev", "test"):
        path = Path("data") / f"{split}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        new_lines: list[str] = []
        for idx, line in enumerate(lines):
            sample = json.loads(line)
            counts = Counter()
            counts["timestamp_ampm"] += repair_timestamp_ampm(sample)
            counts["dash_endpoint_modifier"] += repair_dash_endpoint_modifiers(sample)
            counts["hyphen_with_determiner"] += repair_hyphen_with_determiner(sample)
            counts["hyphen_number_percent"] += repair_hyphen_number_percent(sample)
            counts["duplicate_list_close_wrapper"] += repair_duplicate_list_close_wrappers(sample)
            if sum(counts.values()):
                print(f"{split}:{idx} {dict(counts)}")
            totals.update(counts)
            new_lines.append(json.dumps(sample, ensure_ascii=False, separators=(",", ":")))
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"totals: {dict(totals)}")


if __name__ == "__main__":
    main()
