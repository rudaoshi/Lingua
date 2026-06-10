#!/usr/bin/env python3
"""Repair reviewed same-surface Expression/Modification inconsistencies.

This script is deliberately exact and conservative.  It does not use a global
lexical POS dictionary.  The repair set below contains surfaces reviewed
against docs/standard.md where the internal relation is name/date/unit/format
composition rather than a recoverable head+modifier relation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repair_be_copula_rules import compact_graph, read_jsonl, write_jsonl  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


# Reviewed from reports/hierarchical_surface_conflicts_sharedarg_dc99b63.
# Excluded on purpose:
# - "andaman islands": Islands is a recoverable type/head, so Modification is
#   acceptable and the one Expression variant needs separate manual review.
EXPRESSION_SURFACES = {
    "1 h",
    "2 k",
    "6 : 00 pm",
    "action network",
    "al qaeda",
    "april 20",
    "april 30",
    "aug 23",
    "auto towing",
    "ben taub",
    "beyond control",
    "bin laden",
    "blackburn rovers",
    "cape town",
    "david kemper",
    "golden state warriors",
    "gordon campbell",
    "hidden nook",
    "january 10",
    "june 30",
    "kansas city",
    "ken lay",
    "los angeles",
    "mexico city",
    "mother teresa",
    "mujahideen khalq",
    "nacho libre",
    "new delhi",
    "new orleans",
    "new south wales",
    "new york",
    "no . 2",
    "north america",
    "royal caribbean",
    "san diego",
    "september 7",
    "south africa",
    "south america",
    "st . louis",
    "tampa bay",
    "the hague",
    "united states",
    "us dollars",
    "world war ii",
}

DIRECT_MODIFICATION_SURFACES = {
    # Determiner+noun remains recoverable even inside a field-label fragment;
    # keep the same direct edge style used by ordinary noun phrases.
    "the place",
}

MODIFICATION_SURFACES = {
    # "Islands" is a recoverable type/head; the place name should remain a
    # head+qualifier unit when it further modifies another noun.
    "andaman islands",
}

NAME_PART_TYPES = {
    "DeterminerConstant",
    "ModificationalConstant",
    "OtherConstant",
}


def normalize_surface(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if isinstance(spans, str):
        return []
    indexes: list[int] = []
    for start, end in spans:
        indexes.extend(range(start, end + 1))
    return sorted(indexes)


def spans_for_tokens(tokens: list[int]) -> list[list[int]]:
    tokens = sorted(set(tokens))
    if not tokens:
        return []
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


def descendant_tokens(sample: dict[str, Any], node_index: int) -> list[int]:
    nodes = graph(sample)["nodes"]
    tokens: list[int] = []
    for index in descendant_nodes(sample, node_index):
        tokens.extend(node_token_indexes(nodes[index]))
    return sorted(set(tokens))


def text_for_tokens(sample: dict[str, Any], tokens: list[int]) -> str:
    parts = [sample["lingua"]["words"][index].strip() for index in sorted(set(tokens))]
    return " ".join(part for part in parts if part)


def descendant_text(sample: dict[str, Any], node_index: int) -> str:
    return text_for_tokens(sample, descendant_tokens(sample, node_index))


def own_text(sample: dict[str, Any], node_index: int) -> str:
    return text_for_tokens(sample, node_token_indexes(graph(sample)["nodes"][node_index]))


def first_token(sample: dict[str, Any], node_index: int) -> int:
    tokens = descendant_tokens(sample, node_index)
    return tokens[0] if tokens else 10**9


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return edge["start"], edge["end"], edge["label"]


def add_edge_once(sample: dict[str, Any], start: int, end: int, label: str) -> None:
    new_edge = {"start": start, "end": end, "label": label}
    if edge_key(new_edge) in {edge_key(edge) for edge in graph(sample)["edges"]}:
        return
    graph(sample)["edges"].append(new_edge)


def dedupe_edges(sample: dict[str, Any]) -> None:
    seen: set[tuple[int, int, str]] = set()
    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        key = edge_key(edge)
        if key in seen:
            continue
        seen.add(key)
        new_edges.append(edge)
    graph(sample)["edges"] = new_edges


def remove_node(sample: dict[str, Any], removed_index: int) -> None:
    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        if edge["start"] == removed_index or edge["end"] == removed_index:
            continue
        new_edge = dict(edge)
        if new_edge["start"] > removed_index:
            new_edge["start"] -= 1
        if new_edge["end"] > removed_index:
            new_edge["end"] -= 1
        new_edges.append(new_edge)
    del graph(sample)["nodes"][removed_index]
    graph(sample)["edges"] = new_edges


def set_name_part_types(sample: dict[str, Any], node_indexes: set[int]) -> None:
    nodes = graph(sample)["nodes"]
    for index in node_indexes:
        node = nodes[index]
        if not node_token_indexes(node):
            continue
        if node.get("type") in NAME_PART_TYPES:
            node["type"] = "NominalConstant"


def selected_roots_after_removal(
    sample: dict[str, Any],
    selected: set[int],
    removed_labels: set[str],
    *,
    aux_index: int | None = None,
) -> list[int]:
    incoming = Counter()
    for edge in graph(sample)["edges"]:
        if edge["label"] in removed_labels:
            continue
        if edge["start"] in selected and edge["end"] in selected:
            incoming[edge["end"]] += 1
    roots = [
        index
        for index in selected
        if index != aux_index and node_token_indexes(graph(sample)["nodes"][index]) and incoming[index] == 0
    ]
    if not roots:
        roots = [
            index
            for index in selected
            if index != aux_index and node_token_indexes(graph(sample)["nodes"][index])
        ]
    return sorted(roots, key=lambda index: first_token(sample, index))


def remove_internal_edges(sample: dict[str, Any], selected: set[int], labels: set[str]) -> int:
    kept: list[dict[str, Any]] = []
    removed = 0
    for edge in graph(sample)["edges"]:
        if edge["label"] in labels and edge["start"] in selected and edge["end"] in selected:
            removed += 1
            continue
        kept.append(edge)
    graph(sample)["edges"] = kept
    return removed


def convert_direct_modification_root(sample: dict[str, Any], root: int, surface_key: str) -> str | None:
    selected = descendant_nodes(sample, root)
    if not selected:
        return None
    if normalize_surface(descendant_text(sample, root)) != surface_key:
        return None
    internal_mod_edges = [
        edge
        for edge in graph(sample)["edges"]
        if edge["label"] == "as:modification" and edge["start"] in selected and edge["end"] in selected
    ]
    if not internal_mod_edges:
        return None

    old_text = descendant_text(sample, root)
    expr_index = len(graph(sample)["nodes"])
    graph(sample)["nodes"].append({"spans": "Expression", "type": "ExpressionFunctor"})

    for edge in graph(sample)["edges"]:
        if edge["end"] == root and edge["start"] not in selected:
            edge["end"] = expr_index

    remove_internal_edges(sample, selected, {"as:modification"})
    set_name_part_types(sample, selected)
    for child in selected_roots_after_removal(sample, selected, {"as:modification"}):
        add_edge_once(sample, expr_index, child, "func.arg")
    dedupe_edges(sample)
    return f"direct_modification_to_expression|{surface_key}|{old_text}"


def as_modification_children(sample: dict[str, Any], node_index: int) -> list[int]:
    return [
        edge["end"]
        for edge in graph(sample)["edges"]
        if edge["start"] == node_index and edge["label"] == "as:modification"
    ]


def find_as_modification_path(sample: dict[str, Any], surface_key: str) -> list[int] | None:
    node_count = len(graph(sample)["nodes"])

    def path_text(path: list[int]) -> str:
        tokens: list[int] = []
        for index in path:
            tokens.extend(node_token_indexes(graph(sample)["nodes"][index]))
        return text_for_tokens(sample, tokens)

    def visit(path: list[int]) -> list[int] | None:
        current_key = normalize_surface(path_text(path))
        if current_key == surface_key and len(path) >= 2:
            return path
        if not surface_key.startswith(current_key):
            return None
        if len(path) > len(surface_key.split()) + 2:
            return None
        for child in as_modification_children(sample, path[-1]):
            if child in path:
                continue
            result = visit(path + [child])
            if result is not None:
                return result
        return None

    for start in range(node_count):
        if not node_token_indexes(graph(sample)["nodes"][start]):
            continue
        start_key = normalize_surface(own_text(sample, start))
        if not surface_key.startswith(start_key):
            continue
        result = visit([start])
        if result is not None:
            return result
    return None


def convert_as_modification_path(sample: dict[str, Any], surface_key: str) -> str | None:
    path = find_as_modification_path(sample, surface_key)
    if path is None:
        return None
    selected = set(path)
    old_text = text_for_tokens(
        sample,
        [token for index in path for token in node_token_indexes(graph(sample)["nodes"][index])],
    )
    expr_index = len(graph(sample)["nodes"])
    graph(sample)["nodes"].append({"spans": "Expression", "type": "ExpressionFunctor"})

    first = path[0]
    last = path[-1]
    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        if edge["label"] == "as:modification" and edge["start"] in selected and edge["end"] in selected:
            continue
        new_edge = dict(edge)
        if new_edge["end"] == first and new_edge["start"] not in selected:
            new_edge["end"] = expr_index
        if (
            new_edge["start"] == last
            and new_edge["end"] not in selected
            and new_edge["label"] == "as:modification"
        ):
            new_edge["start"] = expr_index
        new_edges.append(new_edge)
    graph(sample)["edges"] = new_edges

    set_name_part_types(sample, selected)
    for child in sorted(path, key=lambda index: first_token(sample, index)):
        add_edge_once(sample, expr_index, child, "func.arg")
    dedupe_edges(sample)
    return f"path_modification_to_expression|{surface_key}|{old_text}"


def convert_aux_modification(sample: dict[str, Any], aux_index: int, surface_key: str) -> str | None:
    nodes = graph(sample)["nodes"]
    aux = nodes[aux_index]
    if aux.get("spans") != "Modification" or aux.get("type") != "ModificationalFunctor":
        return None
    if normalize_surface(descendant_text(sample, aux_index)) != surface_key:
        return None

    selected = descendant_nodes(sample, aux_index)
    old_text = descendant_text(sample, aux_index)
    aux["spans"] = "Expression"
    aux["type"] = "ExpressionFunctor"

    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        if edge["start"] == aux_index and edge["label"] in {"variable", "body"}:
            continue
        if edge["label"] == "as:modification" and edge["start"] in selected and edge["end"] in selected:
            continue
        new_edges.append(edge)
    graph(sample)["edges"] = new_edges

    set_name_part_types(sample, selected)
    roots = selected_roots_after_removal(sample, selected, {"as:modification", "variable", "body"}, aux_index=aux_index)
    for child in roots:
        add_edge_once(sample, aux_index, child, "func.arg")
    dedupe_edges(sample)
    return f"aux_modification_to_expression|{surface_key}|{old_text}"


def convert_expression_to_modification(sample: dict[str, Any], node_index: int, surface_key: str) -> str | None:
    nodes = graph(sample)["nodes"]
    node = nodes[node_index]
    if node.get("spans") != "Expression" or node.get("type") != "ExpressionFunctor":
        return None
    if normalize_surface(descendant_text(sample, node_index)) != surface_key:
        return None
    args = [
        edge["end"]
        for edge in graph(sample)["edges"]
        if edge["start"] == node_index and edge["label"] == "func.arg"
    ]
    args = sorted(args, key=lambda index: first_token(sample, index))
    if len(args) != 2:
        return None
    old_text = descendant_text(sample, node_index)
    body, variable = args[0], args[1]
    node["spans"] = "Modification"
    node["type"] = "ModificationalFunctor"
    new_edges = [
        edge
        for edge in graph(sample)["edges"]
        if not (edge["start"] == node_index and edge["label"] == "func.arg")
    ]
    graph(sample)["edges"] = new_edges
    add_edge_once(sample, node_index, variable, "variable")
    add_edge_once(sample, node_index, body, "body")
    dedupe_edges(sample)
    return f"expression_to_modification|{surface_key}|{old_text}"


def convert_wrapper_to_direct_modification(sample: dict[str, Any], node_index: int, surface_key: str) -> str | None:
    nodes = graph(sample)["nodes"]
    node = nodes[node_index]
    if normalize_surface(descendant_text(sample, node_index)) != surface_key:
        return None

    modifier: int | None = None
    head: int | None = None
    internal_labels: set[str]
    if node.get("spans") == "Expression" and node.get("type") == "ExpressionFunctor":
        args = [
            edge["end"]
            for edge in graph(sample)["edges"]
            if edge["start"] == node_index and edge["label"] == "func.arg"
        ]
        args = sorted(args, key=lambda index: first_token(sample, index))
        if len(args) != 2:
            return None
        modifier, head = args[0], args[1]
        internal_labels = {"func.arg"}
    elif node.get("spans") == "Modification" and node.get("type") == "ModificationalFunctor":
        variable_args = [
            edge["end"]
            for edge in graph(sample)["edges"]
            if edge["start"] == node_index and edge["label"] == "variable"
        ]
        body_args = [
            edge["end"]
            for edge in graph(sample)["edges"]
            if edge["start"] == node_index and edge["label"] == "body"
        ]
        if len(variable_args) != 1 or len(body_args) != 1:
            return None
        head, modifier = variable_args[0], body_args[0]
        internal_labels = {"variable", "body"}
    else:
        return None

    old_text = descendant_text(sample, node_index)
    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        if edge["start"] == node_index and edge["label"] in internal_labels:
            continue
        new_edge = dict(edge)
        if new_edge["end"] == node_index and new_edge["start"] != node_index:
            new_edge["end"] = head
        new_edges.append(new_edge)
    graph(sample)["edges"] = new_edges
    add_edge_once(sample, modifier, head, "as:modification")
    remove_node(sample, node_index)
    dedupe_edges(sample)
    return f"wrapper_to_direct_modification|{surface_key}|{old_text}"


def repair_sample(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    made_change = True
    while made_change:
        made_change = False
        for node_index in range(len(graph(sample)["nodes"])):
            if node_index >= len(graph(sample)["nodes"]):
                break
            surface_key = normalize_surface(descendant_text(sample, node_index))
            if surface_key in DIRECT_MODIFICATION_SURFACES:
                change = convert_wrapper_to_direct_modification(sample, node_index, surface_key)
                if change is not None:
                    changes.append(change)
                    made_change = True
                    break
            if surface_key in MODIFICATION_SURFACES:
                change = convert_expression_to_modification(sample, node_index, surface_key)
                if change is not None:
                    changes.append(change)
                    made_change = True
                    break
            if surface_key not in EXPRESSION_SURFACES:
                continue
            change = convert_aux_modification(sample, node_index, surface_key)
            if change is None:
                change = convert_direct_modification_root(sample, node_index, surface_key)
            if change is None:
                change = convert_as_modification_path(sample, surface_key)
            if change is None:
                continue
            changes.append(change)
            made_change = True
            break
        if made_change:
            continue
        for surface_key in sorted(EXPRESSION_SURFACES):
            change = convert_as_modification_path(sample, surface_key)
            if change is None:
                continue
            changes.append(change)
            made_change = True
            break
    return changes


def run(apply: bool, data_dir: Path, report_dir: Path) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    case_dir = report_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    changed_records: list[dict[str, Any]] = []

    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        new_rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(rows, 1):
            before = deepcopy(row)
            after = deepcopy(row)
            changes = repair_sample(after)
            if not changes or before == after:
                new_rows.append(row)
                continue
            record = {
                "split": split,
                "line": line_number,
                "sentence": row.get("sentence"),
                "changes": changes,
                "before": compact_graph(before),
                "after": compact_graph(after),
            }
            changed_records.append(record)
            case_path = case_dir / f"{split}_{line_number:05d}.txt"
            case_path.write_text(
                f"{split}:{line_number}\n{row.get('sentence')}\n\n"
                f"CHANGES: {', '.join(changes)}\n\n"
                f"BEFORE\n{record['before']}\n\nAFTER\n{record['after']}\n",
                encoding="utf-8",
            )
            new_rows.append(after)
        if apply:
            write_jsonl(path, new_rows)

    (report_dir / "changes.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in changed_records)
        + ("\n" if changed_records else ""),
        encoding="utf-8",
    )
    by_change = Counter()
    for record in changed_records:
        for change in record["changes"]:
            parts = change.split("|", 2)
            surface = parts[1] if len(parts) == 3 else change
            by_change[surface] += 1
    summary = {
        "apply": apply,
        "changed_samples": len(changed_records),
        "changed_items": sum(len(record["changes"]) for record in changed_records),
        "by_surface": dict(by_change.most_common()),
        "report_dir": str(report_dir),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Hierarchical Surface Conflict Repairs",
        "",
        f"- apply: `{apply}`",
        f"- changed samples: `{summary['changed_samples']}`",
        f"- changed items: `{summary['changed_items']}`",
        "",
        "## By Surface",
    ]
    for surface, count in by_change.most_common():
        lines.append(f"- `{surface}`: {count}")
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORTS / "hierarchical_surface_conflict_repairs",
    )
    args = parser.parse_args()
    return run(args.apply, args.data_dir, args.report_dir)


if __name__ == "__main__":
    raise SystemExit(main())
