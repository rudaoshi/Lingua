#!/usr/bin/env python3
"""Place question punctuation between Whether and the inner proposition.

Target shape:

    Whether --func.arg--> Q
    Q --as:punctuation--> inner proposition

where Q is the question-mark punctuation wrapper, or the outermost punctuation
wrapper in a final punctuation chain such as ``...?`` or ``? "``.

This repairs the over-flattened shape:

    Whether --func.arg--> inner proposition
    Whether --punctuation--> Q

which makes the question mark a leaf and loses the close-wrapper structure.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_REPORT_DIR = ROOT / "reports" / "whether_question_punctuation_wrapper_scope_repair"


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


def has_question_mark(words: list[str], node: dict[str, Any]) -> bool:
    return any("?" in words[index] for index in node_token_ids(node))


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


def sort_punctuation_chain(nodes: list[dict[str, Any]], punctuation_edges: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, dict[str, Any]]]:
    """Return punctuation edges from outermost to innermost.

    Sentence-final punctuation closes from right to left: in ``...?`` the
    question mark is outside the ellipsis; in ``? "`` the quote is outside the
    question mark.
    """

    def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        _, edge = item
        token_ids = node_token_ids(nodes[int(edge["end"])])
        if not token_ids:
            return (-1, int(edge["end"]))
        return (max(token_ids), int(edge["end"]))

    return sorted(punctuation_edges, key=sort_key, reverse=True)


def find_repairs(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    words = sample["lingua"]["words"]
    graph = sample["lingua"]["graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]

    outgoing: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for edge_index, edge in enumerate(edges):
        outgoing[int(edge["start"])].append((edge_index, edge))

    repairs = []
    skips = []
    for whether_index, node in enumerate(nodes):
        if node.get("spans") != "Whether" or node.get("type") != "GeneralFunctor":
            continue

        outgoing_edges = outgoing.get(whether_index, [])
        func_arg_edges = [
            (edge_index, edge)
            for edge_index, edge in outgoing_edges
            if edge.get("label") == "func.arg"
        ]
        punctuation_edges = [
            (edge_index, edge)
            for edge_index, edge in outgoing_edges
            if edge.get("label") == "punctuation"
            and nodes[int(edge["end"])].get("type") == "PunctuationalConstant"
        ]
        question_edges = [
            (edge_index, edge)
            for edge_index, edge in punctuation_edges
            if has_question_mark(words, nodes[int(edge["end"])])
        ]
        if not question_edges:
            continue
        if len(func_arg_edges) != 1:
            skips.append(
                {
                    "whether_index": whether_index,
                    "reason": "expected_single_func_arg",
                    "func_arg_edges": len(func_arg_edges),
                    "question_punctuation_edges": len(question_edges),
                }
            )
            continue

        chain_edges = sort_punctuation_chain(nodes, punctuation_edges)
        chain_node_indices = [int(edge["end"]) for _, edge in chain_edges]
        if not any(index in {int(edge["end"]) for _, edge in question_edges} for index in chain_node_indices):
            continue

        bad_punctuation = []
        for punctuation_index in chain_node_indices:
            extra_outgoing = [
                edge
                for edge_index, edge in outgoing.get(punctuation_index, [])
                if edge.get("label") != "as:punctuation"
            ]
            existing_close = [
                edge
                for edge_index, edge in outgoing.get(punctuation_index, [])
                if edge.get("label") == "as:punctuation"
            ]
            if extra_outgoing or existing_close:
                bad_punctuation.append(
                    {
                        "punctuation_index": punctuation_index,
                        "extra_outgoing": extra_outgoing,
                        "existing_close": existing_close,
                    }
                )
        if bad_punctuation:
            skips.append(
                {
                    "whether_index": whether_index,
                    "reason": "punctuation_not_leaf_before_repair",
                    "bad_punctuation": bad_punctuation,
                }
            )
            continue

        repairs.append(
            {
                "whether_index": whether_index,
                "func_arg_edge_index": func_arg_edges[0][0],
                "inner_root_index": int(func_arg_edges[0][1]["end"]),
                "punctuation_edge_indices": [edge_index for edge_index, _ in chain_edges],
                "punctuation_indices": chain_node_indices,
            }
        )

    return repairs, skips


def apply_repairs(sample: dict[str, Any], repairs: list[dict[str, Any]]) -> dict[str, Any]:
    edited = copy.deepcopy(sample)
    graph = edited["lingua"]["graph"]
    edges = graph["edges"]

    remove_indices = set()
    added_edges = []

    for repair in repairs:
        whether_index = int(repair["whether_index"])
        inner_root_index = int(repair["inner_root_index"])
        punctuation_indices = [int(index) for index in repair["punctuation_indices"]]
        punctuation_edge_indices = [int(index) for index in repair["punctuation_edge_indices"]]
        func_arg_edge_index = int(repair["func_arg_edge_index"])

        remove_indices.add(func_arg_edge_index)
        remove_indices.update(punctuation_edge_indices)

        old_func_arg = edges[func_arg_edge_index]
        new_func_arg = copy.deepcopy(old_func_arg)
        new_func_arg["start"] = whether_index
        new_func_arg["end"] = punctuation_indices[0]
        new_func_arg["label"] = "func.arg"
        added_edges.append(new_func_arg)

        for position, punctuation_index in enumerate(punctuation_indices):
            source_edge = edges[punctuation_edge_indices[position]]
            new_close = copy.deepcopy(source_edge)
            new_close["start"] = punctuation_index
            if position + 1 < len(punctuation_indices):
                new_close["end"] = punctuation_indices[position + 1]
            else:
                new_close["end"] = inner_root_index
            new_close["label"] = "as:punctuation"
            added_edges.append(new_close)

    new_edges = [
        copy.deepcopy(edge)
        for edge_index, edge in enumerate(edges)
        if edge_index not in remove_indices
    ]
    new_edges.extend(added_edges)
    graph["edges"] = dedupe_edges(new_edges)
    return edited


def summarize_repair(
    split: str,
    line_number: int,
    sample: dict[str, Any],
    repairs: list[dict[str, Any]],
) -> dict[str, Any]:
    words = sample["lingua"]["words"]
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    return {
        "split": split,
        "line_number": line_number,
        "sentence": sample.get("sentence", " ".join(words)),
        "repairs": [
            {
                "whether_index": repair["whether_index"],
                "old_func_arg": {
                    "start": edges[repair["func_arg_edge_index"]]["start"],
                    "end": edges[repair["func_arg_edge_index"]]["end"],
                    "target_text": node_text(words, nodes[repair["inner_root_index"]]),
                },
                "punctuation_nodes_outer_to_inner": [
                    {
                        "index": punctuation_index,
                        "text": node_text(words, nodes[punctuation_index]),
                    }
                    for punctuation_index in repair["punctuation_indices"]
                ],
            }
            for repair in repairs
        ],
    }


def write_report(report_dir: Path, changes: list[dict[str, Any]], skips: list[dict[str, Any]], apply: bool) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "changes.jsonl").write_text(
        "".join(json.dumps(change, ensure_ascii=False) + "\n" for change in changes),
        encoding="utf-8",
    )
    (report_dir / "skips.jsonl").write_text(
        "".join(json.dumps(skip, ensure_ascii=False) + "\n" for skip in skips),
        encoding="utf-8",
    )

    by_split = Counter(change["split"] for change in changes)
    moved_punctuation_nodes = sum(
        len(repair["punctuation_nodes_outer_to_inner"])
        for change in changes
        for repair in change["repairs"]
    )
    lines = [
        "# Whether Question Punctuation Wrapper Scope Repair",
        "",
        f"- mode: {'apply' if apply else 'dry-run'}",
        f"- changed samples: {len(changes)}",
        f"- punctuation wrappers moved under Whether.func.arg: {moved_punctuation_nodes}",
        f"- skips: {len(skips)}",
        f"- by split: {dict(sorted(by_split.items()))}",
        "",
        "## Examples",
        "",
    ]
    for change in changes[:20]:
        lines.append(
            f"- {change['split']}:{change['line_number']} "
            f"{change['repairs'][0]['punctuation_nodes_outer_to_inner']} :: {change['sentence']}"
        )
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    all_changes: list[dict[str, Any]] = []
    all_skips: list[dict[str, Any]] = []

    for split in ("train", "dev", "test"):
        path = args.data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        edited_rows = []
        changed_rows = 0
        for line_number, sample in enumerate(rows, 1):
            repairs, skips = find_repairs(sample)
            for skip in skips:
                skip.update(
                    {
                        "split": split,
                        "line_number": line_number,
                        "sentence": sample.get("sentence", " ".join(sample["lingua"]["words"])),
                    }
                )
            all_skips.extend(skips)
            if not repairs:
                edited_rows.append(sample)
                continue

            edited = apply_repairs(sample, repairs)
            form_error = graph_form_error(
                edited["lingua"]["graph"]["nodes"],
                edited["lingua"]["graph"]["edges"],
            )
            if form_error:
                all_skips.append(
                    {
                        "split": split,
                        "line_number": line_number,
                        "reason": "edited_graph_form_error",
                        "form_error": form_error,
                        "sentence": sample.get("sentence", " ".join(sample["lingua"]["words"])),
                    }
                )
                edited_rows.append(sample)
                continue

            changed_rows += 1
            all_changes.append(summarize_repair(split, line_number, sample, repairs))
            edited_rows.append(edited if args.apply else sample)

        if args.apply and changed_rows:
            write_jsonl(path, edited_rows)
        print(f"{split}: changed_samples={changed_rows}")

    write_report(args.report_dir, all_changes, all_skips, args.apply)
    print(f"total changed samples: {len(all_changes)}")
    print(f"skips: {len(all_skips)}")
    print(f"report: {args.report_dir}")


if __name__ == "__main__":
    main()
