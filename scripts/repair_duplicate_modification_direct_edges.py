#!/usr/bin/env python3
"""Remove duplicate Modification aux nodes when an equivalent direct edge exists."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TARGETS = {
    ("train", 2226),
    ("train", 3996),
    ("train", 5878),
    ("train", 9260),
    ("train", 9335),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/duplicate_modification_direct_repair"))
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Repair all strict duplicate triples instead of the known five reviewed cases.",
    )
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    changes: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}

    for split in args.splits:
        path = args.data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        new_rows = []
        changed_lines = 0
        for line_no, row in enumerate(rows, start=1):
            if not args.all and (split, line_no) not in DEFAULT_TARGETS:
                new_rows.append(row)
                continue
            repaired, row_changes = repair_sample(row, split, line_no)
            if row_changes:
                changed_lines += 1
                changes.extend(row_changes)
            new_rows.append(repaired)
        split_summaries[split] = {
            "path": str(path),
            "samples": len(rows),
            "changed_lines": changed_lines,
        }
        if args.apply and changed_lines:
            write_jsonl(path, new_rows)

    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "scope": "all strict duplicate triples" if args.all else "known five reviewed cases",
        "split_summaries": split_summaries,
        "changed_samples": len({(change["split"], change["line"]) for change in changes}),
        "removed_modification_nodes": len(changes),
        "changes_by_split": dict(Counter(change["split"] for change in changes)),
    }
    write_reports(args.report_dir, summary, changes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


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


def repair_sample(row: dict[str, Any], split: str, line_no: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repaired = copy.deepcopy(row)
    graph = repaired["lingua"]["graph"]
    words = repaired["lingua"]["words"]
    changes: list[dict[str, Any]] = []

    while True:
        candidates = find_duplicate_triples(words, graph)
        if not candidates:
            break
        candidate = candidates[0]
        before_graph = compact_graph_view(words, graph, candidate)
        graph = remove_modification_node(graph, candidate)
        repaired["lingua"]["graph"] = graph
        after_graph = compact_graph_view_after(words, graph, candidate)
        changes.append(
            {
                "split": split,
                "line": line_no,
                "sample_id": f"{split}.jsonl:{line_no}",
                "sentence": row.get("sentence", ""),
                "removed_modification_node": candidate["modification_node"],
                "variable_node_A": candidate["variable_node_A"],
                "body_node_B": candidate["body_node_B"],
                "A_text": candidate["A_text"],
                "B_text": candidate["B_text"],
                "incoming_edges_redirected": candidate["incoming_edges"],
                "before": before_graph,
                "after": after_graph,
            }
        )

    return repaired, changes


def find_duplicate_triples(words: list[str], graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = graph["nodes"]
    edges = graph["edges"]
    outgoing: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    incoming: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    direct_as = set()
    for edge_index, edge in enumerate(edges):
        start = int(edge["start"])
        end = int(edge["end"])
        outgoing[start].append((edge_index, edge))
        incoming[end].append((edge_index, edge))
        if edge.get("label") == "as:modification":
            direct_as.add((start, end))

    candidates = []
    for mod_id, node in enumerate(nodes):
        if not is_modification_aux(node):
            continue
        mod_outgoing = outgoing.get(mod_id, [])
        variable_edges = [(index, edge) for index, edge in mod_outgoing if edge.get("label") == "variable"]
        body_edges = [(index, edge) for index, edge in mod_outgoing if edge.get("label") == "body"]
        if len(mod_outgoing) != 2 or len(variable_edges) != 1 or len(body_edges) != 1:
            continue
        variable_edge_index, variable_edge = variable_edges[0]
        body_edge_index, body_edge = body_edges[0]
        variable_node = int(variable_edge["end"])
        body_node = int(body_edge["end"])
        if (body_node, variable_node) not in direct_as:
            continue
        candidates.append(
            {
                "modification_node": mod_id,
                "variable_node_A": variable_node,
                "body_node_B": body_node,
                "variable_edge_index": variable_edge_index,
                "body_edge_index": body_edge_index,
                "incoming_edges": [
                    {
                        "edge_index": edge_index,
                        "start": int(edge["start"]),
                        "end": int(edge["end"]),
                        "label": edge["label"],
                    }
                    for edge_index, edge in incoming.get(mod_id, [])
                ],
                "A_text": descendant_text(words, nodes, outgoing, variable_node),
                "B_text": descendant_text(words, nodes, outgoing, body_node),
            }
        )
    return candidates


def remove_modification_node(graph: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    mod_id = int(candidate["modification_node"])
    body_node = int(candidate["body_node_B"])
    nodes = graph["nodes"]
    edges = graph["edges"]

    def remap(index: int) -> int | None:
        if index == mod_id:
            return None
        return index - 1 if index > mod_id else index

    new_nodes = [copy.deepcopy(node) for index, node in enumerate(nodes) if index != mod_id]
    new_edges: list[dict[str, Any]] = []
    seen_edges = set()
    for edge in edges:
        start = int(edge["start"])
        end = int(edge["end"])
        if start == mod_id:
            continue
        if end == mod_id:
            end = body_node
        new_start = remap(start)
        new_end = remap(end)
        if new_start is None or new_end is None or new_start == new_end:
            continue
        new_edge = copy.deepcopy(edge)
        new_edge["start"] = new_start
        new_edge["end"] = new_end
        edge_key = (
            int(new_edge["start"]),
            int(new_edge["end"]),
            str(new_edge["label"]),
            json.dumps({k: v for k, v in new_edge.items() if k not in {"start", "end", "label"}}, sort_keys=True),
        )
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        new_edges.append(new_edge)
    return {"nodes": new_nodes, "edges": new_edges}


def is_modification_aux(node: dict[str, Any]) -> bool:
    return node.get("spans") == "Modification" and node.get("type") == "ModificationalFunctor"


def node_token_ids(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if isinstance(spans, str):
        return []
    token_ids = []
    for start, end in spans:
        token_ids.extend(range(int(start), int(end) + 1))
    return sorted(set(token_ids))


def descendant_text(
    words: list[str],
    nodes: list[dict[str, Any]],
    outgoing: dict[int, list[tuple[int, dict[str, Any]]]],
    node_id: int,
) -> str:
    seen: set[int] = set()

    def walk(index: int) -> set[int]:
        if index in seen:
            return set()
        seen.add(index)
        token_ids = set(node_token_ids(nodes[index]))
        for _, edge in outgoing.get(index, []):
            token_ids.update(walk(int(edge["end"])))
        return token_ids

    return " ".join(
        words[index].strip()
        for index in sorted(walk(node_id))
        if 0 <= index < len(words) and words[index].strip()
    )


def sample_node_text(words: list[str], node: dict[str, Any]) -> str:
    spans = node.get("spans")
    if isinstance(spans, str):
        return spans
    return " ".join(words[index].strip() for index in node_token_ids(node) if 0 <= index < len(words))


def compact_graph_view(words: list[str], graph: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return compact_graph_view_for_nodes(
        words,
        graph,
        {
            int(candidate["modification_node"]),
            int(candidate["variable_node_A"]),
            int(candidate["body_node_B"]),
            *{int(edge["start"]) for edge in candidate["incoming_edges"]},
        },
    )


def compact_graph_view_after(words: list[str], graph: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    old_mod = int(candidate["modification_node"])

    def remap_old(index: int) -> int | None:
        if index == old_mod:
            return None
        return index - 1 if index > old_mod else index

    focus_nodes = {
        remapped
        for old_index in [
            int(candidate["variable_node_A"]),
            int(candidate["body_node_B"]),
            *[int(edge["start"]) for edge in candidate["incoming_edges"]],
        ]
        for remapped in [remap_old(old_index)]
        if remapped is not None
    }
    return compact_graph_view_for_nodes(words, graph, focus_nodes)


def compact_graph_view_for_nodes(words: list[str], graph: dict[str, Any], focus_nodes: set[int]) -> dict[str, Any]:
    nodes = graph["nodes"]
    edges = graph["edges"]
    expanded_nodes = set(focus_nodes)
    for edge in edges:
        if int(edge["start"]) in focus_nodes or int(edge["end"]) in focus_nodes:
            expanded_nodes.add(int(edge["start"]))
            expanded_nodes.add(int(edge["end"]))
    return {
        "nodes": [
            {
                "id": index,
                "text": sample_node_text(words, nodes[index]),
                "type": nodes[index].get("type"),
                "spans": nodes[index].get("spans"),
            }
            for index in sorted(expanded_nodes)
            if 0 <= index < len(nodes)
        ],
        "edges": [
            {
                "start": int(edge["start"]),
                "end": int(edge["end"]),
                "label": edge["label"],
            }
            for edge in edges
            if int(edge["start"]) in expanded_nodes and int(edge["end"]) in expanded_nodes
        ],
        "roots": root_indices(graph),
    }


def root_indices(graph: dict[str, Any]) -> list[int]:
    node_count = len(graph["nodes"])
    incoming = [0] * node_count
    for edge in graph["edges"]:
        end = int(edge["end"])
        if 0 <= end < node_count:
            incoming[end] += 1
    return [index for index, count in enumerate(incoming) if count == 0]


def write_reports(report_dir: Path, summary: dict[str, Any], changes: list[dict[str, Any]]) -> None:
    (report_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "changes": changes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (report_dir / "changes.jsonl").open("w", encoding="utf-8") as handle:
        for change in changes:
            handle.write(json.dumps(change, ensure_ascii=False) + "\n")

    lines = ["# Duplicate Modification/direct as:modification repair", ""]
    lines.append(f"- mode: `{summary['mode']}`")
    lines.append(f"- scope: `{summary['scope']}`")
    lines.append(f"- changed samples: `{summary['changed_samples']}`")
    lines.append(f"- removed Modification nodes: `{summary['removed_modification_nodes']}`")
    lines.append("")
    for change in changes:
        lines.append(f"## {change['sample_id']}")
        lines.append(change["sentence"])
        lines.append("")
        lines.append(
            f"Remove node `{change['removed_modification_node']}` and keep "
            f"`B --as:modification--> A` for B=`{change['B_text']}`, A=`{change['A_text']}`."
        )
        lines.append("")
        lines.append("Incoming edges redirected to B:")
        for edge in change["incoming_edges_redirected"]:
            lines.append(f"- `{edge['start']} --{edge['label']}--> {edge['end']}`")
        lines.append("")
        lines.append(f"BEFORE roots: `{change['before']['roots']}`")
        lines.append(f"AFTER roots: `{change['after']['roots']}`")
        lines.append("")
        lines.append("BEFORE edges:")
        for edge in change["before"]["edges"]:
            lines.append(f"- `{edge['start']} --{edge['label']}--> {edge['end']}`")
        lines.append("")
        lines.append("AFTER edges:")
        for edge in change["after"]["edges"]:
            lines.append(f"- `{edge['start']} --{edge['label']}--> {edge['end']}`")
        lines.append("")
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
