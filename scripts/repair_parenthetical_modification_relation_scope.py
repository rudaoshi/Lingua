#!/usr/bin/env python3
"""Repair parenthetical Modification wrappers with redundant relation edges.

Target shape:

    M/Modification
      variable -> A
      body -> P

    P --as:punctuation--> C
    C -> A
    C -> B

The broken shape has an extra ``M --modification--> C`` edge, while the
parenthesis node still closes only B:

    M --variable--> A
    M --body--> P
    M --modification--> C
    P --as:punctuation--> B

This script keeps M, deletes only the redundant ``modification`` edge, and
retargets the parenthesis close edge from B to C.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_REPORT_DIR = ROOT / "reports" / "parenthetical_modification_relation_scope_repair"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def node_token_ids(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if not isinstance(spans, list):
        return []
    return [index for start, end in spans for index in range(int(start), int(end) + 1)]


def node_text(words: list[str], node: dict[str, Any]) -> str:
    token_ids = node_token_ids(node)
    if token_ids:
        return " ".join(words[index].strip() for index in token_ids if words[index].strip())
    return str(node.get("spans"))


def is_parenthesis_node(words: list[str], node: dict[str, Any]) -> bool:
    if node.get("type") != "PunctuationalConstant":
        return False
    tokens = {words[index].strip() for index in node_token_ids(node)}
    return "(" in tokens and ")" in tokens


def descendants(start: int, edges: list[dict[str, Any]]) -> set[int]:
    outgoing: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        outgoing[int(edge["start"])].append(int(edge["end"]))

    seen: set[int] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(outgoing.get(node, []))
    return seen


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return int(edge["start"]), int(edge["end"]), str(edge["label"])


def dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, str]] = set()
    deduped = []
    for edge in edges:
        key = edge_key(edge)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def graph_form_error(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str | None:
    adjacency = [[] for _ in nodes]
    indegree = [0] * len(nodes)
    for edge in edges:
        start = int(edge["start"])
        end = int(edge["end"])
        if start < 0 or start >= len(nodes) or end < 0 or end >= len(nodes):
            return f"edge_out_of_range:{edge}"
        adjacency[start].append(end)
        indegree[end] += 1

    roots = [index for index, degree in enumerate(indegree) if degree == 0]
    if len(roots) != 1:
        return f"root_count:{roots}"

    reachable = set()
    stack = [roots[0]]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(adjacency[node])
    if len(reachable) != len(nodes):
        return f"unreachable:{len(nodes) - len(reachable)}"

    remaining = list(indegree)
    queue = deque(roots)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in adjacency[node]:
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)
    if visited != len(nodes):
        return "cycle"

    return None


def find_candidates(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    words = sample["lingua"]["words"]
    graph = sample["lingua"]["graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]

    outgoing: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for edge_index, edge in enumerate(edges):
        outgoing[int(edge["start"])].append((edge_index, edge))

    candidates = []
    skips = []
    for node_index, node in enumerate(nodes):
        if node.get("spans") != "Modification" or node.get("type") != "ModificationalFunctor":
            continue

        outgoing_edges = outgoing.get(node_index, [])
        variable_edges = [(index, edge) for index, edge in outgoing_edges if edge.get("label") == "variable"]
        body_edges = [(index, edge) for index, edge in outgoing_edges if edge.get("label") == "body"]
        relation_edges = [
            (index, edge) for index, edge in outgoing_edges if edge.get("label") == "modification"
        ]
        if not relation_edges:
            continue
        if len(variable_edges) != 1 or len(body_edges) != 1 or len(relation_edges) != 1:
            skips.append(
                {
                    "node_index": node_index,
                    "reason": "expected_single_variable_body_modification",
                    "variable_edges": len(variable_edges),
                    "body_edges": len(body_edges),
                    "modification_edges": len(relation_edges),
                }
            )
            continue

        body_index = int(body_edges[0][1]["end"])
        relation_index = int(relation_edges[0][1]["end"])
        if not is_parenthesis_node(words, nodes[body_index]):
            skips.append({"node_index": node_index, "reason": "body_not_parenthesis"})
            continue

        punctuation_edges = [
            (index, edge)
            for index, edge in outgoing.get(body_index, [])
            if edge.get("label") == "as:punctuation"
        ]
        if len(punctuation_edges) != 1:
            skips.append(
                {
                    "node_index": node_index,
                    "reason": "body_expected_single_as_punctuation",
                    "as_punctuation_edges": len(punctuation_edges),
                }
            )
            continue

        old_close_target = int(punctuation_edges[0][1]["end"])
        relation_descendants = descendants(relation_index, edges)
        if old_close_target not in relation_descendants:
            skips.append(
                {
                    "node_index": node_index,
                    "reason": "old_close_target_not_reachable_from_relation",
                    "old_close_target": old_close_target,
                    "relation_index": relation_index,
                }
            )
            continue
        if body_index in relation_descendants:
            skips.append(
                {
                    "node_index": node_index,
                    "reason": "retarget_would_create_cycle",
                    "body_index": body_index,
                    "relation_index": relation_index,
                }
            )
            continue

        candidates.append(
            {
                "node_index": node_index,
                "variable_index": int(variable_edges[0][1]["end"]),
                "body_index": body_index,
                "relation_index": relation_index,
                "old_close_target": old_close_target,
                "modification_edge_index": relation_edges[0][0],
                "punctuation_edge_index": punctuation_edges[0][0],
            }
        )

    return candidates, skips


def repair_sample(sample: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    repaired = copy.deepcopy(sample)
    words = repaired["lingua"]["words"]
    graph = repaired["lingua"]["graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    candidates, skips = find_candidates(repaired)
    if not candidates:
        return repaired, [], skips

    remove_edge_indices = {candidate["modification_edge_index"] for candidate in candidates}
    retarget_edges = {
        candidate["punctuation_edge_index"]: candidate["relation_index"]
        for candidate in candidates
    }
    if len(retarget_edges) != len(candidates):
        skips.append({"reason": "conflicting_punctuation_edge_retargets"})
        return repaired, [], skips

    changes = []
    new_edges = []
    for edge_index, edge in enumerate(edges):
        if edge_index in remove_edge_indices:
            continue
        new_edge = copy.deepcopy(edge)
        if edge_index in retarget_edges:
            new_edge["end"] = retarget_edges[edge_index]
        if int(new_edge["start"]) != int(new_edge["end"]):
            new_edges.append(new_edge)

    graph["edges"] = dedupe_edges(new_edges)
    form_error = graph_form_error(nodes, graph["edges"])
    if form_error is not None:
        skips.append({"reason": "graph_form_after_repair", "error": form_error})
        return sample, [], skips

    for candidate in candidates:
        changes.append(
            {
                **candidate,
                "variable_text": node_text(words, nodes[candidate["variable_index"]]),
                "body_text": node_text(words, nodes[candidate["body_index"]]),
                "relation_text": node_text(words, nodes[candidate["relation_index"]]),
                "relation_aux": nodes[candidate["relation_index"]].get("spans"),
                "relation_type": nodes[candidate["relation_index"]].get("type"),
                "old_close_target_text": node_text(words, nodes[candidate["old_close_target"]]),
            }
        )

    return repaired, changes, skips


def write_reports(report_dir: Path, summary: dict[str, Any], changes: list[dict[str, Any]], skips: list[dict[str, Any]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (report_dir / "changes.jsonl").open("w", encoding="utf-8") as handle:
        for change in changes:
            handle.write(json.dumps(change, ensure_ascii=False) + "\n")
    if skips:
        with (report_dir / "skips.jsonl").open("w", encoding="utf-8") as handle:
            for skip in skips:
                handle.write(json.dumps(skip, ensure_ascii=False) + "\n")

    fields = [
        "split",
        "line",
        "node_index",
        "variable_index",
        "variable_text",
        "body_index",
        "body_text",
        "relation_index",
        "relation_aux",
        "relation_type",
        "old_close_target",
        "old_close_target_text",
        "modification_edge_index",
        "punctuation_edge_index",
        "sentence",
    ]
    with (report_dir / "changes.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(changes)

    lines = [
        "# Parenthetical Modification relation-scope repair",
        "",
        f"- mode: `{summary['mode']}`",
        f"- changed_samples: `{summary['changed_samples']}`",
        f"- repaired_edges: `{summary['repaired_edges']}`",
        f"- skipped_candidates: `{summary['skipped_candidates']}`",
        f"- changes_by_split: `{summary['changes_by_split']}`",
        "",
        "## By relation",
    ]
    for label, count in summary["changes_by_relation"].items():
        lines.append(f"- `{label}`: {count}")
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    all_changes: list[dict[str, Any]] = []
    all_skips: list[dict[str, Any]] = []
    split_summaries = {}

    for split in args.splits:
        path = args.data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        repaired_rows = []
        changed_lines = 0
        for line_number, row in enumerate(rows, 1):
            repaired, changes, skips = repair_sample(row)
            if changes:
                changed_lines += 1
                for change in changes:
                    change.update(
                        {
                            "split": split,
                            "line": line_number,
                            "sample_id": f"{split}.jsonl:{line_number}",
                            "sentence": row.get("sentence", ""),
                        }
                    )
                all_changes.extend(changes)
            for skip in skips:
                skip.update(
                    {
                        "split": split,
                        "line": line_number,
                        "sample_id": f"{split}.jsonl:{line_number}",
                        "sentence": row.get("sentence", ""),
                    }
                )
            all_skips.extend(skips)
            repaired_rows.append(repaired)

        split_summaries[split] = {
            "path": str(path),
            "samples": len(rows),
            "changed_lines": changed_lines,
            "repaired_edges": sum(1 for change in all_changes if change["split"] == split),
        }
        if args.apply and changed_lines:
            write_jsonl(path, repaired_rows)

    relation_counts = Counter(
        f"{change['relation_aux']}|{change['relation_type']}"
        for change in all_changes
    )
    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "split_summaries": split_summaries,
        "changed_samples": len({(change["split"], change["line"]) for change in all_changes}),
        "repaired_edges": len(all_changes),
        "skipped_candidates": len(all_skips),
        "changes_by_split": dict(Counter(change["split"] for change in all_changes)),
        "changes_by_relation": dict(relation_counts.most_common()),
    }
    write_reports(args.report_dir, summary, all_changes, all_skips)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
