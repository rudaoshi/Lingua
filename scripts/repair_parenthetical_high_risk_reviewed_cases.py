#!/usr/bin/env python3
"""Apply the reviewed high-risk A(B) parenthetical repairs.

This is intentionally target-driven.  The 51 skipped parentheticals from
``repair_parenthetical_aux_modification_standard.py`` were manually reviewed;
32 clear A(B) cases are repaired here, while the remaining 19 stay skipped.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "train": ROOT / "data" / "train.jsonl",
    "dev": ROOT / "data" / "dev.jsonl",
    "test": ROOT / "data" / "test.jsonl",
}
REPORT_DEFAULT = ROOT / "reports" / "parenthetical_high_risk_reviewed_apply"

RELATION_TYPES = {
    "Attribute": "AttributePredicator",
    "Apposition": "AppositionalPredicator",
    "Discourse": "ModificationalFunctor",
    "Modification": "ModificationalFunctor",
    "Parataxis": "ConjunctionalFunctor",
    "Ref": "ReferentialPredicator",
}


@dataclass(frozen=True)
class ReplaceEdge:
    start: int
    label: str
    old_end: int
    new: str | int = "C"
    occurrence: int | None = None


@dataclass(frozen=True)
class RemoveEdge:
    start: int
    label: str
    end: int


@dataclass(frozen=True)
class AddEdgeSpec:
    start: str | int
    label: str
    end: str | int


@dataclass(frozen=True)
class Target:
    split: str
    line: int
    a: int
    d: int
    b: int
    relation: str
    c_existing: int | None = None
    existing_relation: int | None = None
    replace: tuple[ReplaceEdge, ...] = ()
    remove: tuple[RemoveEdge, ...] = ()
    add: tuple[AddEdgeSpec, ...] = ()
    relation_arg_overrides: tuple[ReplaceEdge, ...] = ()
    normalize_existing_relation_node: bool = False
    note: str = ""


TARGETS: tuple[Target, ...] = (
    Target(
        "train",
        96,
        2,
        17,
        4,
        "Attribute",
        c_existing=23,
        replace=(ReplaceEdge(20, "variable", 23, 0), ReplaceEdge(20, "body", 2, "C")),
    ),
    Target(
        "train",
        112,
        12,
        10,
        1,
        "Modification",
        existing_relation=1,
        replace=(ReplaceEdge(9, "as:punctuation", 10, 0), ReplaceEdge(11, "pred.arg.2", 12, "C")),
    ),
    Target("train", 558, 13, 22, 12, "Apposition", existing_relation=25, replace=(ReplaceEdge(8, "body", 22),)),
    Target(
        "train",
        582,
        9,
        13,
        8,
        "Attribute",
        existing_relation=14,
        replace=(ReplaceEdge(5, "body", 9, "C"),),
        remove=(RemoveEdge(9, "as:modification", 14),),
        add=(AddEdgeSpec("C", "as:modification", 6),),
    ),
    Target(
        "train",
        593,
        8,
        19,
        10,
        "Attribute",
        c_existing=21,
        replace=(ReplaceEdge(20, "variable", 21, 5), ReplaceEdge(20, "body", 8, "C")),
    ),
    Target(
        "train",
        655,
        12,
        23,
        9,
        "Apposition",
        existing_relation=24,
        replace=(ReplaceEdge(18, "as:punctuation", 23), ReplaceEdge(4, "pred.arg.2", 24, 18)),
    ),
    Target(
        "train",
        661,
        17,
        37,
        18,
        "Apposition",
        existing_relation=42,
        replace=(ReplaceEdge(19, "as:modification", 17, "C"),),
        remove=(
            RemoveEdge(17, "as:modification", 42),
            RemoveEdge(42, "variable", 16),
            RemoveEdge(42, "body", 37),
        ),
        add=(AddEdgeSpec("C", "as:modification", 16),),
        normalize_existing_relation_node=True,
    ),
    Target(
        "train",
        1387,
        1,
        18,
        0,
        "Parataxis",
        existing_relation=20,
        replace=(ReplaceEdge(17, "as:punctuation", 20), ReplaceEdge(20, "func.arg", 18, "A")),
        remove=(RemoveEdge(0, "func.arg", 1), RemoveEdge(20, "func.arg", 21)),
    ),
    Target(
        "train",
        1748,
        8,
        20,
        23,
        "Attribute",
        existing_relation=22,
        replace=(ReplaceEdge(21, "variable", 22, 3), ReplaceEdge(21, "body", 8, "C")),
    ),
    Target("train", 2319, 22, 24, 2, "Discourse", replace=(ReplaceEdge(30, "func.arg", 24),)),
    Target(
        "train",
        2497,
        5,
        17,
        6,
        "Attribute",
        c_existing=20,
        replace=(ReplaceEdge(19, "variable", 20, 3), ReplaceEdge(19, "body", 5, "C")),
    ),
    Target(
        "train",
        2527,
        29,
        24,
        20,
        "Attribute",
        replace=(
            ReplaceEdge(26, "body", 24),
            ReplaceEdge(25, "variable", 26, 12),
            ReplaceEdge(25, "body", 29, 26),
        ),
    ),
    Target(
        "train",
        2697,
        15,
        29,
        17,
        "Attribute",
        c_existing=33,
        replace=(ReplaceEdge(31, "variable", 33, 13), ReplaceEdge(31, "body", 15, "C")),
    ),
    Target("train", 2762, 11, 26, 12, "Attribute", replace=(ReplaceEdge(31, "func.arg", 26),)),
    Target("train", 3189, 2, 19, 1, "Modification", existing_relation=1, replace=(ReplaceEdge(0, "body", 19),)),
    Target(
        "train",
        5418,
        22,
        18,
        9,
        "Apposition",
        existing_relation=25,
        replace=(ReplaceEdge(16, "pred.arg.2", 25),),
    ),
    Target("train", 5932, 3, 34, 40, "Discourse", replace=(ReplaceEdge(41, "func.arg", 34),)),
    Target(
        "train",
        6326,
        3,
        24,
        5,
        "Apposition",
        replace=(ReplaceEdge(25, "body", 24), ReplaceEdge(6, "variable", 3, 25)),
        remove=(RemoveEdge(3, "as:modification", 25),),
    ),
    Target(
        "train",
        6475,
        13,
        23,
        15,
        "Apposition",
        replace=(ReplaceEdge(24, "body", 23), ReplaceEdge(12, "as:modification", 13, 24)),
        remove=(RemoveEdge(13, "as:modification", 24),),
    ),
    Target("train", 6479, 4, 9, 5, "Attribute"),
    Target(
        "train",
        6956,
        12,
        16,
        11,
        "Attribute",
        replace=(ReplaceEdge(9, "func.arg", 12),),
        remove=(RemoveEdge(12, "as:modification", 16),),
    ),
    Target(
        "train",
        6988,
        15,
        16,
        8,
        "Apposition",
        c_existing=18,
        replace=(ReplaceEdge(17, "variable", 18, 5), ReplaceEdge(17, "body", 15, "C")),
    ),
    Target(
        "train",
        7299,
        8,
        22,
        7,
        "Attribute",
        replace=(ReplaceEdge(23, "variable", 8),),
        remove=(RemoveEdge(8, "as:modification", 22),),
    ),
    Target(
        "train",
        7434,
        6,
        19,
        7,
        "Apposition",
        replace=(ReplaceEdge(22, "body", 19), ReplaceEdge(5, "as:modification", 6, 22)),
        remove=(RemoveEdge(6, "as:modification", 22),),
    ),
    Target(
        "train",
        9023,
        12,
        18,
        11,
        "Attribute",
        replace=(ReplaceEdge(20, "body", 18), ReplaceEdge(12, "pred.arg.1", 20, 10)),
    ),
    Target("train", 9109, 1, 13, 4, "Parataxis", existing_relation=14, replace=(ReplaceEdge(3, "as:modification", 14), ReplaceEdge(14, "func.arg", 13, "B"))),
    Target(
        "train",
        9853,
        8,
        11,
        6,
        "Attribute",
        c_existing=13,
        replace=(ReplaceEdge(12, "variable", 13, 5), ReplaceEdge(12, "body", 8, "C")),
    ),
    Target(
        "train",
        10447,
        6,
        8,
        7,
        "Attribute",
        replace=(ReplaceEdge(20, "body", 8), ReplaceEdge(10, "pred.arg.1", 6, 20)),
        remove=(RemoveEdge(6, "as:modification", 20),),
    ),
    Target(
        "dev",
        574,
        1,
        22,
        10,
        "Attribute",
        replace=(ReplaceEdge(15, "as:punctuation", 10),),
        remove=(RemoveEdge(10, "as:modification", 22),),
    ),
    Target("dev", 674, 12, 10, 7, "Discourse", existing_relation=11, replace=(ReplaceEdge(9, "as:punctuation", 11),)),
    Target(
        "test",
        326,
        26,
        23,
        27,
        "Ref",
        existing_relation=25,
        replace=(ReplaceEdge(22, "pred.arg.2", 25),),
        relation_arg_overrides=(ReplaceEdge(27, "body", 23, 6),),
    ),
    Target(
        "test",
        341,
        8,
        20,
        7,
        "Apposition",
        existing_relation=22,
        replace=(ReplaceEdge(19, "pred.arg.1", 22, 10), ReplaceEdge(9, "as:modification", 8)),
    ),
)


REVIEWED_NOT_CHANGED = (
    ("train", 1006, "no left-side A"),
    ("train", 1027, "compact chemical expression"),
    ("train", 1333, "compact numeric expression"),
    ("train", 2074, "interrupted predicate, not clear A(B)"),
    ("train", 2360, "compact chemical expression"),
    ("train", 4393, "compact symbol expression"),
    ("train", 4563, "B modifies right-side head"),
    ("train", 5274, "B modifies right-side head"),
    ("train", 6821, "parenthetical argument position, no left-side A"),
    ("train", 8187, "compact repeat expression"),
    ("train", 9048, "unclear colloquial supplement"),
    ("train", 9745, "unintelligible placeholder"),
    ("train", 10102, "Ro(SSA) compact alias"),
    ("train", 10102, "La(SSB) compact alias"),
    ("train", 10215, "B modifies right-side head"),
    ("dev", 454, "B modifies right-side head"),
    ("dev", 731, "B modifies right-side head"),
    ("test", 287, "compact protein alias"),
    ("test", 406, "B comments on expert/he, not account"),
)


def node_text(words: list[str], node: dict[str, Any]) -> str:
    spans = node.get("spans")
    if not isinstance(spans, list):
        return str(spans)
    return "".join(words[index] for start, end in spans for index in range(start, end + 1)).strip()


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return edge["start"], edge["end"], edge["label"]


def add_edge(edges: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    key = (start, end, label)
    if key not in {edge_key(edge) for edge in edges}:
        edges.append({"start": start, "end": end, "label": label})


def remove_edge(edges: list[dict[str, Any]], spec: RemoveEdge) -> bool:
    for index, edge in enumerate(edges):
        if edge["start"] == spec.start and edge["end"] == spec.end and edge["label"] == spec.label:
            del edges[index]
            return True
    return False


def resolve(value: str | int, c_node: int, target: Target) -> int:
    if value == "C":
        return c_node
    if value == "A":
        return target.a
    if value == "B":
        return target.b
    return int(value)


def replace_edge(edges: list[dict[str, Any]], spec: ReplaceEdge, c_node: int, target: Target) -> int:
    matches = [
        index
        for index, edge in enumerate(edges)
        if edge["start"] == spec.start and edge["label"] == spec.label and edge["end"] == spec.old_end
    ]
    if spec.occurrence is not None:
        matches = matches[spec.occurrence : spec.occurrence + 1]
    if not matches:
        raise ValueError(f"missing edge {spec}")
    new_end = resolve(spec.new, c_node, target)
    for index in matches:
        edges[index]["end"] = new_end
    return len(matches)


def set_relation_edges(edges: list[dict[str, Any]], relation: int, kind: str, a: int, b: int) -> None:
    if kind in {"Attribute", "Apposition", "Ref"}:
        set_single_edge(edges, relation, "pred.arg.1", a)
        set_single_edge(edges, relation, "pred.arg.2", b)
    elif kind in {"Modification", "Discourse"}:
        set_single_edge(edges, relation, "variable", a)
        set_single_edge(edges, relation, "body", b)
    elif kind == "Parataxis":
        remove_outgoing_label(edges, relation, "func.arg")
        add_edge(edges, relation, a, "func.arg")
        add_edge(edges, relation, b, "func.arg")
    else:
        raise ValueError(f"unsupported relation kind {kind}")


def remove_outgoing_label(edges: list[dict[str, Any]], start: int, label: str) -> None:
    edges[:] = [edge for edge in edges if not (edge["start"] == start and edge["label"] == label)]


def set_single_edge(edges: list[dict[str, Any]], start: int, label: str, end: int) -> None:
    remove_outgoing_label(edges, start, label)
    add_edge(edges, start, end, label)


def graph_snapshot(sample: dict[str, Any], focus: set[int]) -> list[str]:
    words = sample["lingua"]["words"]
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    related = set(focus)
    for edge in edges:
        if edge["start"] in focus or edge["end"] in focus:
            related.add(edge["start"])
            related.add(edge["end"])
    lines = []
    for index in sorted(related):
        node = nodes[index]
        lines.append(f"n{index}: {node.get('type')} spans={node.get('spans')} text={node_text(words, node)!r}")
    for edge in edges:
        if edge["start"] in related or edge["end"] in related:
            lines.append(f"e n{edge['start']} -{edge['label']}-> n{edge['end']}")
    return lines


def assert_acyclic(edges: list[dict[str, Any]], node_count: int) -> None:
    adjacency = [[] for _ in range(node_count)]
    for edge in edges:
        adjacency[edge["start"]].append(edge["end"])
    color = [0] * node_count

    def visit(node: int) -> bool:
        color[node] = 1
        for child in adjacency[node]:
            if color[child] == 1:
                return False
            if color[child] == 0 and not visit(child):
                return False
        color[node] = 2
        return True

    for node in range(node_count):
        if color[node] == 0 and not visit(node):
            raise ValueError("cycle detected")


def root_count(edges: list[dict[str, Any]], node_count: int) -> int:
    indegree = [0] * node_count
    for edge in edges:
        indegree[edge["end"]] += 1
    return sum(1 for degree in indegree if degree == 0)


def repair_target(sample: dict[str, Any], target: Target) -> dict[str, Any]:
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    for index in (target.a, target.d, target.b):
        if not (0 <= index < len(nodes)):
            raise ValueError(f"{target.split}:{target.line} invalid node index {index}")

    before = graph_snapshot(sample, {target.a, target.d, target.b})
    if target.c_existing is None:
        c_node = len(nodes)
        nodes.append({"spans": "Modification", "type": "ModificationalFunctor"})
        add_edge(edges, c_node, target.a, "variable")
        add_edge(edges, c_node, target.d, "body")
    else:
        c_node = target.c_existing
        nodes[c_node]["spans"] = "Modification"
        nodes[c_node]["type"] = "ModificationalFunctor"
        set_single_edge(edges, c_node, "variable", target.a)
        set_single_edge(edges, c_node, "body", target.d)

    for spec in target.replace:
        replace_edge(edges, spec, c_node, target)
    for spec in target.remove:
        if not remove_edge(edges, spec):
            raise ValueError(f"{target.split}:{target.line} missing removable edge {spec}")
    for spec in target.add:
        add_edge(edges, resolve(spec.start, c_node, target), resolve(spec.end, c_node, target), spec.label)

    # Ensure the parenthesis wrapper closes B.
    set_single_edge(edges, target.d, "as:punctuation", target.b)

    if target.existing_relation is None:
        relation = len(nodes)
        nodes.append({"spans": target.relation, "type": RELATION_TYPES[target.relation]})
        set_relation_edges(edges, relation, target.relation, target.a, target.b)
    else:
        relation = target.existing_relation
        if target.normalize_existing_relation_node:
            nodes[relation]["spans"] = target.relation
            nodes[relation]["type"] = RELATION_TYPES[target.relation]
        if relation == target.b and target.relation in {"Modification", "Discourse"}:
            # The parenthesized B is already the semantic relation node, e.g.
            # "excellent (for the most part)". Keep its inner body and point
            # its variable at A.
            set_single_edge(edges, relation, "variable", target.a)
        else:
            set_relation_edges(edges, relation, target.relation, target.a, target.b)

    for spec in target.relation_arg_overrides:
        replace_edge(edges, spec, c_node, target)

    add_edge(edges, c_node, relation, "modification")
    assert_acyclic(edges, len(nodes))
    after_roots = root_count(edges, len(nodes))
    return {
        "split": target.split,
        "line": target.line,
        "sentence": sample["sentence"],
        "a": target.a,
        "a_text": node_text(sample["lingua"]["words"], nodes[target.a]),
        "b": target.b,
        "b_text": node_text(sample["lingua"]["words"], nodes[target.b]),
        "d": target.d,
        "c": c_node,
        "relation": relation,
        "relation_kind": target.relation,
        "root_count_after": after_roots,
        "note": target.note,
        "before": before,
        "after": graph_snapshot(sample, {target.a, target.b, target.d, c_node, relation}),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args()

    rows_by_split = {
        split: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for split, path in DATA.items()
    }
    touched: list[dict[str, Any]] = []

    for target in TARGETS:
        original = rows_by_split[target.split][target.line - 1]
        sample = copy.deepcopy(original)
        result = repair_target(sample, target)
        rows_by_split[target.split][target.line - 1] = sample
        touched.append(result)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.report_dir / "changes.jsonl", touched)
    write_jsonl(
        args.report_dir / "reviewed_not_changed.jsonl",
        [
            {"split": split, "line": line, "reason": reason}
            for split, line, reason in REVIEWED_NOT_CHANGED
        ],
    )
    by_split = Counter(item["split"] for item in touched)
    by_relation = Counter(item["relation_kind"] for item in touched)
    root_counts = Counter(item["root_count_after"] for item in touched)
    summary = [
        "# Reviewed High-Risk Parenthetical Repairs",
        "",
        f"changed samples: {len(touched)}",
        f"reviewed not changed: {len(REVIEWED_NOT_CHANGED)}",
        "",
        "## Changed by split",
        "",
    ]
    for split in ("train", "dev", "test"):
        summary.append(f"- {split}: {by_split[split]}")
    summary.extend(["", "## Changed by relation", ""])
    for relation, count in by_relation.most_common():
        summary.append(f"- {relation}: {count}")
    summary.extend(["", "## Root Count After Local Repair", ""])
    for count, n in sorted(root_counts.items()):
        summary.append(f"- {count}: {n}")
    summary.extend(["", "Detailed rows are in `changes.jsonl`."])
    (args.report_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    if args.apply:
        for split, rows in rows_by_split.items():
            DATA[split].write_text(
                "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )

    mode = "applied" if args.apply else "dry-run"
    print(f"{mode}: changed_samples={len(touched)} reviewed_not_changed={len(REVIEWED_NOT_CHANGED)}")
    print(f"report: {args.report_dir}")


if __name__ == "__main__":
    main()
