#!/usr/bin/env python3
"""Repair the reviewed residual same-view parenthetical cases.

These are exact split/line edits reviewed after
``repair_parenthetical_modification_relation_scope.py``.  In each case the
parenthetical B side already contains a lexical/non-symbol relation that points
back to A, so the extra same-token-set aux relation layer is redundant.  The
outer Modification wrapper is kept to express that the parenthetical modifies A.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT = ROOT / "reports" / "remaining_same_view_parenthetical_cases_repair"


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
    if not token_ids:
        return str(node.get("spans"))
    return " ".join(words[index].strip() for index in token_ids if words[index].strip())


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return int(edge["start"]), int(edge["end"]), str(edge["label"])


def make_edge(start: int, end: int, label: str, confidence: int | float | None = 1) -> dict[str, Any]:
    edge: dict[str, Any] = {"start": start, "end": end, "label": label}
    if confidence is not None:
        edge["confidence"] = confidence
    return edge


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


class Editor:
    def __init__(self, sample: dict[str, Any], split: str, line: int, decision: str):
        self.sample = copy.deepcopy(sample)
        self.split = split
        self.line = line
        self.decision = decision
        self.graph = self.sample["lingua"]["graph"]
        self.words = self.sample["lingua"]["words"]
        self.remove_nodes: set[int] = set()
        self.remove_edges: set[tuple[int, int, str]] = set()
        self.add_edges: list[dict[str, Any]] = []
        self.notes: list[str] = []

    @property
    def nodes(self) -> list[dict[str, Any]]:
        return self.graph["nodes"]

    @property
    def edges(self) -> list[dict[str, Any]]:
        return self.graph["edges"]

    def node_matches(self, index: int, spans: Any, node_type: str, text: str | None = None) -> bool:
        if index < 0 or index >= len(self.nodes):
            return False
        node = self.nodes[index]
        if node.get("spans") != spans or node.get("type") != node_type:
            return False
        if text is not None and node_text(self.words, node) != text:
            return False
        return True

    def edge_exists(self, start: int, end: int, label: str) -> bool:
        return any(edge_key(edge) == (start, end, label) for edge in self.edges)

    def require_node(self, index: int, spans: Any, node_type: str, text: str | None = None) -> bool:
        if self.node_matches(index, spans, node_type, text):
            return True
        self.notes.append(f"skip:node_mismatch:{index}")
        return False

    def remove_node(self, index: int, note: str) -> None:
        self.remove_nodes.add(index)
        self.notes.append(note)

    def remove_edge(self, start: int, end: int, label: str, note: str) -> None:
        if self.edge_exists(start, end, label):
            self.remove_edges.add((start, end, label))
            self.notes.append(note)

    def add_edge(self, start: int, end: int, label: str, note: str) -> None:
        if not self.edge_exists(start, end, label) and edge_key(make_edge(start, end, label)) not in {
            edge_key(edge) for edge in self.add_edges
        }:
            self.add_edges.append(make_edge(start, end, label))
            self.notes.append(note)

    def apply(self) -> tuple[dict[str, Any], list[str]]:
        if not self.remove_nodes and not self.remove_edges and not self.add_edges:
            return self.sample, []

        new_edges = []
        seen_edges: set[tuple[int, int, str]] = set()
        for edge in self.edges:
            key = edge_key(edge)
            if key in self.remove_edges:
                continue
            if int(edge["start"]) in self.remove_nodes or int(edge["end"]) in self.remove_nodes:
                continue
            if key in seen_edges:
                continue
            seen_edges.add(key)
            new_edges.append(copy.deepcopy(edge))

        for edge in self.add_edges:
            if int(edge["start"]) in self.remove_nodes or int(edge["end"]) in self.remove_nodes:
                continue
            key = edge_key(edge)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            new_edges.append(copy.deepcopy(edge))

        if self.remove_nodes:
            index_map: dict[int, int] = {}
            new_nodes = []
            for old_index, node in enumerate(self.nodes):
                if old_index in self.remove_nodes:
                    continue
                index_map[old_index] = len(new_nodes)
                new_nodes.append(node)
            for edge in new_edges:
                edge["start"] = index_map[int(edge["start"])]
                edge["end"] = index_map[int(edge["end"])]
            self.graph["nodes"] = new_nodes
        self.graph["edges"] = new_edges

        form_error = graph_form_error(self.graph["nodes"], self.graph["edges"])
        if form_error is not None:
            raise ValueError(f"{self.split}:{self.line} invalid graph after repair: {form_error}")
        return self.sample, self.notes


def repair_train_6479(sample: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    editor = Editor(sample, "train", 6479, "remove redundant Attribute; parenthesis closes completed predicate")
    if not editor.require_node(11, "Attribute", "AttributePredicator"):
        return sample, []
    editor.add_edge(9, 5, "as:punctuation", "retarget parenthesis close to completed")
    editor.remove_node(11, "remove Attribute aux: completed already pred.arg.1 -> Greenland")
    return editor.apply()


def repair_train_7397(sample: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    editor = Editor(sample, "train", 7397, "remove redundant Attribute/Modification; born takes date directly")
    if not (
        editor.require_node(20, "Modification", "ModificationalFunctor")
        and editor.require_node(22, "Attribute", "AttributePredicator")
    ):
        return sample, []
    editor.add_edge(2, 23, "pred.arg.2", "add born -> date expression")
    editor.add_edge(17, 2, "as:punctuation", "retarget parenthesis close to born")
    editor.remove_node(20, "remove Modification aux between born and date")
    editor.remove_node(22, "remove Attribute aux: born already pred.arg.1 -> name")
    return editor.apply()


def repair_train_7987(sample: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    editor = Editor(sample, "train", 7987, "remove redundant Attribute; parenthesis closes lexical born predicate")
    if not editor.require_node(26, "Attribute", "AttributePredicator"):
        return sample, []
    editor.add_edge(20, 3, "as:punctuation", "retarget parenthesis close to born")
    editor.remove_node(26, "remove Attribute aux: born already pred.arg.1 -> name")
    return editor.apply()


def repair_train_9109(sample: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    editor = Editor(sample, "train", 9109, "remove redundant Parataxis; parenthesis closes lexical or")
    if not editor.require_node(14, "Parataxis", "ConjunctionalFunctor"):
        return sample, []
    editor.add_edge(13, 4, "as:punctuation", "retarget parenthesis close to or")
    editor.remove_node(14, "remove Parataxis aux: or already connects Write/call")
    return editor.apply()


def repair_train_10068(sample: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    editor = Editor(sample, "train", 10068, "opened 2003 is a predicate with a date argument, not Expression")
    if not (
        editor.require_node(4, "Attribute", "AttributePredicator")
        and editor.require_node(5, "Expression", "ExpressionFunctor")
    ):
        return sample, []
    editor.add_edge(1, 2, "pred.arg.2", "add opened -> 2003")
    editor.add_edge(3, 1, "as:punctuation", "retarget parenthesis close to opened")
    editor.remove_node(4, "remove Attribute aux: opened already pred.arg.1 -> Marshalls")
    editor.remove_node(5, "remove wrong Expression(opened, 2003)")
    return editor.apply()


def repair_train_10818(sample: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    editor = Editor(sample, "train", 10818, "remove redundant Attribute; parenthesis closes originally-in relation")
    if not editor.require_node(23, "Attribute", "AttributePredicator"):
        return sample, []
    editor.add_edge(21, 4, "as:punctuation", "retarget parenthesis close to originally")
    editor.remove_node(23, "remove Attribute aux: in/originally already points to Ben Harper")
    return editor.apply()


def repair_dev_416(sample: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    editor = Editor(sample, "dev", 416, "remove redundant Attribute; born takes year directly")
    if not editor.require_node(17, "Attribute", "AttributePredicator"):
        return sample, []
    editor.remove_edge(10, 9, "as:modification", "remove reversed 1976 -> born modifier edge")
    editor.add_edge(9, 10, "pred.arg.2", "add born -> 1976")
    editor.add_edge(15, 9, "as:punctuation", "retarget parenthesis close to born")
    editor.remove_node(17, "remove Attribute aux: born already pred.arg.1 -> name")
    return editor.apply()


def repair_test_113(sample: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    editor = Editor(sample, "test", 113, "remove redundant Apposition/Modification around appositive conjunction")
    if not (
        editor.require_node(36, "Apposition", "AppositionalPredicator")
        and editor.require_node(37, "Modification", "ModificationalFunctor")
    ):
        return sample, []
    editor.add_edge(31, 5, "as:punctuation", "retarget closing comma to appositive conjunction")
    editor.add_edge(5, 6, "func.arg", "connect conjunction to by/thought branch directly")
    editor.remove_node(36, "remove Apposition aux: conjunction already contains the appositive relation")
    editor.remove_node(37, "remove Modification aux: thought already pred.arg.1 -> Zacarias Moussaoui")
    return editor.apply()


def repair_test_502(sample: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    editor = Editor(sample, "test", 502, "remove redundant Parataxis; parenthesis closes representing predicate")
    if not editor.require_node(21, "Parataxis", "ConjunctionalFunctor"):
        return sample, []
    editor.add_edge(18, 2, "as:punctuation", "retarget parenthesis close to representing")
    editor.remove_node(21, "remove Parataxis aux: representing already pred.arg.1 -> a negative")
    return editor.apply()


def repair_test_598(sample: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    editor = Editor(sample, "test", 598, "born 1953 is a predicate with a year argument, not Expression")
    if not (
        editor.require_node(5, "Attribute", "AttributePredicator")
        and editor.require_node(6, "Expression", "ExpressionFunctor")
    ):
        return sample, []
    editor.add_edge(1, 2, "pred.arg.2", "add born -> 1953")
    editor.add_edge(4, 1, "as:punctuation", "retarget parenthesis close to born")
    editor.remove_node(5, "remove Attribute aux: born already pred.arg.1 -> name")
    editor.remove_node(6, "remove wrong Expression(born, 1953)")
    return editor.apply()


REPAIRS: dict[tuple[str, int], Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]] = {
    ("train", 6479): repair_train_6479,
    ("train", 7397): repair_train_7397,
    ("train", 7987): repair_train_7987,
    ("train", 9109): repair_train_9109,
    ("train", 10068): repair_train_10068,
    ("train", 10818): repair_train_10818,
    ("dev", 416): repair_dev_416,
    ("test", 113): repair_test_113,
    ("test", 502): repair_test_502,
    ("test", 598): repair_test_598,
}


def write_report(report_dir: Path, summary: dict[str, Any], changes: list[dict[str, Any]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (report_dir / "changes.jsonl").open("w", encoding="utf-8") as handle:
        for change in changes:
            handle.write(json.dumps(change, ensure_ascii=False) + "\n")
    with (report_dir / "changes.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["split", "line", "changed", "sentence", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(changes)

    lines = [
        "# Remaining Same-View Parenthetical Cases Repair",
        "",
        f"- mode: `{summary['mode']}`",
        f"- reviewed_cases: `{summary['reviewed_cases']}`",
        f"- changed_cases: `{summary['changed_cases']}`",
        f"- unchanged_cases: `{summary['unchanged_cases']}`",
        f"- changes_by_split: `{summary['changes_by_split']}`",
        "",
        "## Decisions",
        "",
    ]
    for change in changes:
        status = "changed" if change["changed"] else "unchanged"
        lines.append(f"- `{change['split']}:{change['line']}` {status}: {change['notes']}")
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    changes: list[dict[str, Any]] = []
    rows_by_split: dict[str, list[dict[str, Any]]] = {}

    for split in ["train", "dev", "test"]:
        path = args.data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        rows_by_split[split] = rows
        split_changed = False
        for line_number, row in enumerate(rows, 1):
            repair = REPAIRS.get((split, line_number))
            if repair is None:
                continue
            repaired, notes = repair(row)
            changed = bool(notes)
            changes.append(
                {
                    "split": split,
                    "line": line_number,
                    "changed": changed,
                    "sentence": row.get("sentence", ""),
                    "notes": "; ".join(notes) if notes else "already repaired or shape mismatch",
                }
            )
            if changed:
                rows[line_number - 1] = repaired
                split_changed = True
        if args.apply and split_changed:
            write_jsonl(path, rows)

    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "reviewed_cases": len(changes),
        "changed_cases": sum(1 for change in changes if change["changed"]),
        "unchanged_cases": sum(1 for change in changes if not change["changed"]),
        "changes_by_split": dict(Counter(change["split"] for change in changes if change["changed"])),
    }
    write_report(args.report_dir, summary, changes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
