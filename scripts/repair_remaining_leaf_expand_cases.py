#!/usr/bin/env python3
"""Repair the remaining reviewed leaf-vs-expand surface conflicts.

After the expression leaf-granularity pass, the only remaining `leaf_vs_expand`
surfaces are:

- `of course`: fixed discourse/modifier phrase; keep it as one leaf.
- `it ' s`, `I m`, `they ' re`: contractions; split into pronoun + factual
  predicate instead of keeping the whole contraction as a nominal/determiner.
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


def text_for_tokens(sample: dict[str, Any], tokens: list[int]) -> str:
    words = sample["lingua"]["words"]
    return " ".join(words[index].strip() for index in sorted(set(tokens)) if words[index].strip())


def own_text(sample: dict[str, Any], node_index: int) -> str:
    return text_for_tokens(sample, node_token_indexes(graph(sample)["nodes"][node_index]))


def spans_for_tokens(tokens: list[int]) -> list[list[int]]:
    tokens = sorted(set(tokens))
    if not tokens:
        return []
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


def add_node(sample: dict[str, Any], spans: Any, node_type: str) -> int:
    index = len(graph(sample)["nodes"])
    graph(sample)["nodes"].append({"spans": spans, "type": node_type})
    return index


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return edge["start"], edge["end"], edge["label"]


def add_edge_once(sample: dict[str, Any], start: int, end: int, label: str) -> None:
    if start == end:
        return
    new_edge = {"start": int(start), "end": int(end), "label": label}
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


def remove_nodes(sample: dict[str, Any], remove: set[int]) -> None:
    if not remove:
        return
    remap: dict[int, int] = {}
    new_nodes: list[dict[str, Any]] = []
    for old_index, node in enumerate(graph(sample)["nodes"]):
        if old_index in remove:
            continue
        remap[old_index] = len(new_nodes)
        new_nodes.append(node)

    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        if edge["start"] in remove or edge["end"] in remove:
            continue
        new_edges.append(
            {
                "start": remap[edge["start"]],
                "end": remap[edge["end"]],
                "label": edge["label"],
            }
        )
    graph(sample)["nodes"] = new_nodes
    graph(sample)["edges"] = new_edges
    dedupe_edges(sample)


def repair_of_course(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    for of_index, node in list(enumerate(graph(sample)["nodes"])):
        if node.get("type") != "ModificationalFunctor" or normalize_surface(own_text(sample, of_index)) != "of":
            continue
        body_edges = [
            edge
            for edge in graph(sample)["edges"]
            if edge["start"] == of_index and edge["label"] == "body"
        ]
        other_out = [
            edge
            for edge in graph(sample)["edges"]
            if edge["start"] == of_index and edge["label"] != "body"
        ]
        if len(body_edges) != 1 or other_out:
            continue
        course_index = body_edges[0]["end"]
        if normalize_surface(own_text(sample, course_index)) != "course":
            continue
        if graph(sample)["nodes"][course_index].get("type") != "NominalConstant":
            continue

        tokens = node_token_indexes(node) + node_token_indexes(graph(sample)["nodes"][course_index])
        graph(sample)["nodes"][of_index] = {
            "spans": spans_for_tokens(tokens),
            "type": "ModificationalConstant",
        }
        remove_nodes(sample, {course_index})
        changes.append("of_course_functor_to_fixed_leaf|of course|of course")
        break
    return changes


def repair_contraction_leaf(
    sample: dict[str, Any],
    surface_key: str,
    allowed_types: set[str],
) -> list[str]:
    changes: list[str] = []
    for node_index, node in list(enumerate(graph(sample)["nodes"])):
        if normalize_surface(own_text(sample, node_index)) != surface_key:
            continue
        if node.get("type") not in allowed_types:
            continue
        tokens = node_token_indexes(node)
        if len(tokens) < 2:
            continue

        old_text = own_text(sample, node_index)
        pronoun_token = tokens[0]
        pred_tokens = tokens[1:]
        graph(sample)["nodes"][node_index] = {
            "spans": [[pronoun_token, pronoun_token]],
            "type": "NominalConstant",
        }
        pred_index = add_node(sample, spans_for_tokens(pred_tokens), "FactualPredicator")

        outgoing_as_mod = [
            edge
            for edge in graph(sample)["edges"]
            if edge["start"] == node_index and edge["label"] == "as:modification"
        ]
        if len(outgoing_as_mod) > 1:
            raise ValueError(f"multiple outgoing as:modification edges for {old_text}")

        replacement = pred_index
        if outgoing_as_mod:
            head = outgoing_as_mod[0]["end"]
            mod_index = add_node(sample, "Modification", "ModificationalFunctor")
            replacement = mod_index
            add_edge_once(sample, mod_index, head, "variable")
            add_edge_once(sample, mod_index, pred_index, "body")

        new_edges: list[dict[str, Any]] = []
        for edge in graph(sample)["edges"]:
            if edge["start"] == node_index and edge["label"] == "as:modification":
                continue
            new_edge = dict(edge)
            if new_edge["end"] == node_index:
                new_edge["end"] = replacement
            new_edges.append(new_edge)
        graph(sample)["edges"] = new_edges
        add_edge_once(sample, pred_index, node_index, "pred.arg.1")
        dedupe_edges(sample)
        changes.append(f"contraction_leaf_to_predicate|{surface_key}|{old_text}")
        break
    return changes


def repair_sample(sample: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    changes.extend(repair_of_course(sample))
    changes.extend(repair_contraction_leaf(sample, "it ' s", {"NominalConstant"}))
    changes.extend(repair_contraction_leaf(sample, "i m", {"NominalConstant"}))
    changes.extend(repair_contraction_leaf(sample, "they ' re", {"DeterminerConstant", "NominalConstant"}))
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
            (case_dir / f"{split}_{line_number:05d}.txt").write_text(
                f"{split}:{line_number}\n{row.get('sentence')}\n\n"
                f"CHANGES: {', '.join(changes)}\n\n"
                f"BEFORE\n{record['before']}\n\nAFTER\n{record['after']}\n",
                encoding="utf-8",
            )
            new_rows.append(after)
        if apply:
            write_jsonl(path, new_rows)

    by_change = Counter()
    for record in changed_records:
        for change in record["changes"]:
            parts = change.split("|", 2)
            by_change[parts[1] if len(parts) == 3 else change] += 1
    summary = {
        "apply": apply,
        "changed_samples": len(changed_records),
        "changed_items": sum(len(record["changes"]) for record in changed_records),
        "by_surface": dict(by_change.most_common()),
        "report_dir": str(report_dir),
    }
    (report_dir / "changes.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in changed_records)
        + ("\n" if changed_records else ""),
        encoding="utf-8",
    )
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Remaining Leaf-vs-Expand Repairs",
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
        default=REPORTS / "remaining_leaf_expand_repair",
    )
    args = parser.parse_args()
    return run(args.apply, args.data_dir, args.report_dir)


if __name__ == "__main__":
    raise SystemExit(main())
