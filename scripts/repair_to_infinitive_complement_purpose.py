#!/usr/bin/env python3
"""Repair reviewed `to V` complement/purpose cases.

The input is the audit produced by
`audit_to_infinitive_complement_purpose.py`.  The script applies only the
reviewed categories plus the explicit object-control-as-purpose policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
AUDIT_DIR = REPORTS / "to_infinitive_complement_purpose_audit"
OUT_DIR = REPORTS / "to_infinitive_complement_purpose_repair"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repair_be_copula_rules import (  # noqa: E402
    add_edge,
    add_node,
    compact_graph,
    edge_in,
    edge_out,
    graph,
    node_text,
    node_tokens,
    read_jsonl,
    remove_edges,
    remove_nodes,
    set_node_token_indexes,
    spans_from_token_indexes,
    write_jsonl,
)


FACTUAL = "FactualPredicator"
MOD_FUNCTOR = "ModificationalFunctor"

CURRENT_WRONG_CATEGORIES = {
    "adjectival_selected_complement",
    "degree_complement_scope",
    "nominal_purpose_scope",
    "nominal_selected_modifier_scope",
    "purpose_arg4",
    "purpose_arg4_or_missing_predicate",
}
OBJECT_CONTROL_POLICY_CATEGORIES = {
    "object_control_complement",
    "object_control_vs_purpose",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_jsonl_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def token_indexes(sample: dict[str, Any], node_index: int) -> list[int]:
    return [index for index, _surface, _norm in node_tokens(sample, node_index)]


def first_to_index(sample: dict[str, Any], node_index: int) -> int:
    for index, _surface, norm in node_tokens(sample, node_index):
        if norm == "to":
            return index
    raise ValueError(f"node {node_index} {node_text(sample, node_index)!r} has no `to` token")


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


def add_edge_once(sample: dict[str, Any], start: int, end: int, label: str) -> bool:
    for edge in graph(sample)["edges"]:
        if edge["start"] == start and edge["end"] == end and edge["label"] == label:
            return False
    add_edge(sample, start, end, label)
    return True


def redirect_incoming(
    sample: dict[str, Any],
    old_end: int,
    new_end: int,
    excluded_starts: set[int] | None = None,
) -> None:
    excluded_starts = excluded_starts or set()
    for edge in graph(sample)["edges"]:
        if edge["end"] == old_end and edge["start"] not in excluded_starts:
            edge["end"] = new_end


def arg2_target(sample: dict[str, Any], parent: int) -> int:
    arg2_edges = edge_out(sample, parent, "pred.arg.2")
    if not arg2_edges:
        raise ValueError(f"{node_text(sample, parent)!r} has no pred.arg.2 target")
    return arg2_edges[0]["end"]


def current_arg4_parent(sample: dict[str, Any], to_node: int) -> int:
    parents = [edge["start"] for edge in edge_in(sample, to_node, "pred.arg.4")]
    if len(parents) != 1:
        raise ValueError(f"{node_text(sample, to_node)!r} expected one pred.arg.4 parent, found {parents}")
    return parents[0]


def split_arg4_to_functor(sample: dict[str, Any], to_node: int, variable_mode: str) -> str:
    parent = current_arg4_parent(sample, to_node)
    to_index = first_to_index(sample, to_node)
    old_indexes = token_indexes(sample, to_node)
    body_indexes = [index for index in old_indexes if index != to_index]
    if not body_indexes:
        raise ValueError(f"{node_text(sample, to_node)!r} has no body tokens after splitting `to`")

    variable_node = parent
    if variable_mode == "arg2":
        variable_node = arg2_target(sample, parent)

    old_text = node_text(sample, to_node)
    body_node = add_node(sample, spans_from_token_indexes(body_indexes), FACTUAL)

    for edge in graph(sample)["edges"]:
        if edge["start"] == to_node:
            edge["start"] = body_node

    set_node_token_indexes(sample, to_node, [to_index])
    graph(sample)["nodes"][to_node]["type"] = MOD_FUNCTOR
    remove_edges(sample, lambda edge: edge["start"] == parent and edge["end"] == to_node and edge["label"] == "pred.arg.4")

    if variable_mode == "arg2":
        for edge in graph(sample)["edges"]:
            if edge["start"] == parent and edge["end"] == variable_node and edge["label"] == "pred.arg.2":
                edge["end"] = to_node
        add_edge_once(sample, parent, to_node, "pred.arg.2")
    else:
        redirect_incoming(sample, parent, to_node, excluded_starts={to_node, body_node})

    add_edge_once(sample, to_node, variable_node, "variable")
    add_edge_once(sample, to_node, body_node, "body")
    dedupe_edges(sample)
    return f"{old_text}:split_arg4_to_{variable_mode}_to_functor"


def retarget_existing_to_functor_to_nominal(sample: dict[str, Any], to_node: int) -> str:
    variable_edges = edge_out(sample, to_node, "variable")
    body_edges = edge_out(sample, to_node, "body")
    if len(variable_edges) != 1 or len(body_edges) != 1:
        raise ValueError(f"{node_text(sample, to_node)!r} expected one variable/body")
    parent = variable_edges[0]["end"]
    body = body_edges[0]["end"]
    target = arg2_target(sample, parent)

    redirect_incoming(sample, to_node, parent, excluded_starts={parent})
    for edge in graph(sample)["edges"]:
        if edge["start"] == parent and edge["end"] == target and edge["label"] == "pred.arg.2":
            edge["end"] = to_node
        if edge["start"] == to_node and edge["label"] == "variable":
            edge["end"] = target
    remove_edges(sample, lambda edge: edge["start"] == body and edge["end"] == parent and edge["label"] == "pred.arg.1")
    add_edge_once(sample, parent, to_node, "pred.arg.2")
    dedupe_edges(sample)
    return f"{node_text(sample, to_node)}:retarget_variable_to_nominal_scope"


def parent_arg1(sample: dict[str, Any], parent: int) -> int | None:
    arg1 = edge_out(sample, parent, "pred.arg.1")
    return arg1[0]["end"] if arg1 else None


def merge_to_functor_into_arg4(sample: dict[str, Any], to_node: int) -> str:
    variable_edges = edge_out(sample, to_node, "variable")
    body_edges = edge_out(sample, to_node, "body")
    if len(variable_edges) != 1 or len(body_edges) != 1:
        raise ValueError(f"{node_text(sample, to_node)!r} expected one variable/body")
    parent = variable_edges[0]["end"]
    body = body_edges[0]["end"]
    old_text = f"{node_text(sample, to_node)} {node_text(sample, body)}"

    redirect_incoming(sample, to_node, parent, excluded_starts={parent, body})
    merged_indexes = sorted(set(token_indexes(sample, to_node) + token_indexes(sample, body)))
    set_node_token_indexes(sample, to_node, merged_indexes)
    graph(sample)["nodes"][to_node]["type"] = FACTUAL

    remove_edges(
        sample,
        lambda edge: edge["start"] == to_node and edge["end"] in {parent, body} and edge["label"] in {"variable", "body"},
    )
    for edge in graph(sample)["edges"]:
        if edge["start"] == body:
            edge["start"] = to_node
        if edge["end"] == body:
            edge["end"] = to_node

    inherited_arg1 = parent_arg1(sample, parent)
    for edge in list(graph(sample)["edges"]):
        if edge["start"] == to_node and edge["label"] == "pred.arg.1" and edge["end"] == parent:
            if inherited_arg1 is None:
                remove_edges(sample, lambda item, target=edge: item is target)
            else:
                edge["end"] = inherited_arg1

    add_edge_once(sample, parent, to_node, "pred.arg.4")
    remove_nodes(sample, {body})
    dedupe_edges(sample)
    return f"{old_text}:merge_to_functor_into_arg4"


def load_targets(audit_dir: Path) -> list[dict[str, str]]:
    manual_rows = read_tsv(audit_dir / "manual_flagged_review.tsv")
    case_rows = read_tsv(audit_dir / "cases.tsv")
    manual_by_key = {(row["split"], row["line_no"], row["to_node"]): row for row in manual_rows}

    selected: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in manual_rows:
        category = row["manual_category"]
        decision = row["manual_decision"]
        if (
            decision == "current_wrong"
            or category in OBJECT_CONTROL_POLICY_CATEGORIES
            or (decision == "policy_review" and category == "object_control_vs_purpose")
        ):
            selected[(row["split"], row["line_no"], row["to_node"])] = {
                **row,
                "repair_category": category,
                "repair_source": "manual",
            }

    for row in case_rows:
        key = (row["split"], row["line_no"], row["to_node"])
        if row["reason"] != "object_control_as_purpose_policy" or row["judgment"] != "likely_wrong":
            continue
        if key in selected:
            continue
        selected[key] = {
            **row,
            "manual_decision": "",
            "manual_category": "",
            "manual_note": "",
            "repair_category": "object_control_auto",
            "repair_source": "audit_policy",
        }

    rows = list(selected.values())
    rows.sort(key=lambda row: (row["split"], int(row["line_no"]), int(row["to_node"])))
    return rows


def target_kind(row: dict[str, str]) -> tuple[str, str | None]:
    category = row["repair_category"]
    if category == "adjectival_selected_complement":
        return "merge_to_arg4", None
    if category == "nominal_selected_modifier_scope":
        return "retarget_existing_nominal", None
    if category in {"nominal_purpose_scope", "degree_complement_scope"}:
        return "split_arg4", "arg2"
    if category in {
        "purpose_arg4",
        "purpose_arg4_or_missing_predicate",
        "object_control_auto",
        "object_control_complement",
        "object_control_vs_purpose",
    }:
        return "split_arg4", "parent"
    raise ValueError(f"no repair kind for category {category!r}")


def repair_sample(sample: dict[str, Any], rows: list[dict[str, str]]) -> list[str]:
    changes: list[str] = []
    for row in rows:
        to_node = int(row["to_node"])
        expected = row["to_text"]
        actual = node_text(sample, to_node)
        if actual != expected:
            raise AssertionError(
                f"{row['split']}:{row['line_no']} node {to_node}: expected {expected!r}, got {actual!r}"
            )
        kind, mode = target_kind(row)
        if kind == "split_arg4":
            assert mode is not None
            changes.append(split_arg4_to_functor(sample, to_node, mode))
        elif kind == "retarget_existing_nominal":
            changes.append(retarget_existing_to_functor_to_nominal(sample, to_node))
        elif kind == "merge_to_arg4":
            changes.append(merge_to_functor_into_arg4(sample, to_node))
        else:
            raise ValueError(f"unknown repair kind {kind!r}")
    dedupe_edges(sample)
    return changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--report-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    targets = load_targets(args.audit_dir)
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in targets:
        grouped[(row["split"], int(row["line_no"]))].append(row)

    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["to_node"]))

    args.report_dir.mkdir(parents=True, exist_ok=True)
    changes_report: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    changed_counter: Counter[str] = Counter()
    changed_samples = 0

    for split in ("train", "dev", "test"):
        path = args.data_dir / f"{split}.jsonl"
        samples = read_jsonl(path)
        new_samples = deepcopy(samples)
        for line_no, sample in enumerate(new_samples, 1):
            rows = grouped.get((split, line_no))
            if not rows:
                continue
            before = compact_graph(sample)
            try:
                changes = repair_sample(sample, rows)
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "split": split,
                        "line_no": line_no,
                        "sentence": sample.get("sentence", ""),
                        "error": str(exc),
                    }
                )
                continue
            changed_samples += 1
            for row in rows:
                changed_counter[row["repair_category"]] += 1
            changes_report.append(
                {
                    "split": split,
                    "line_no": line_no,
                    "sentence": sample.get("sentence", ""),
                    "repairs": [
                        {
                            "to_node": row["to_node"],
                            "category": row["repair_category"],
                            "source": row["repair_source"],
                            "to_text": row["to_text"],
                            "parent_text": row.get("parent_text", ""),
                        }
                        for row in rows
                    ],
                    "changes": changes,
                    "before": before,
                    "after": compact_graph(sample),
                }
            )
        if args.apply and not errors:
            write_jsonl(path, new_samples)

    write_jsonl_report(args.report_dir / ("applied_changes.jsonl" if args.apply and not errors else "dryrun_changes.jsonl"), changes_report)
    write_jsonl_report(args.report_dir / "errors.jsonl", errors)
    summary = {
        "apply": args.apply,
        "targets": len(targets),
        "changed_samples": changed_samples,
        "changed_by_category": dict(changed_counter),
        "errors": len(errors),
        "report_dir": str(args.report_dir),
    }
    (args.report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
