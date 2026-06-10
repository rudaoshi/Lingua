#!/usr/bin/env python3
"""Normalize clear A(B) parentheticals to a separate punctuation wrapper.

For a reviewed parenthetical A(B), the target shape is:

    C: Modification / ModificationalFunctor
    D: existing parenthesis PunctuationalConstant

    C --variable--> A
    C --body--> D
    D --as:punctuation--> B

The semantic relation between A and B is preserved separately.  For example,
an existing ``Attribute --pred.arg.2--> D`` is rewritten to
``Attribute --pred.arg.2--> B`` while incoming parents of the old semantic
relation are moved to C.

This script intentionally skips cases where the old graph points the
parenthetical at a post-parenthesis head, e.g. ``A (B) head``.  Those require
sentence-level review because the surface parenthetical wrapper belongs around
the left phrase A, while the existing graph often attaches D to the following
head.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = [
    ROOT / "data" / "train.jsonl",
    ROOT / "data" / "dev.jsonl",
    ROOT / "data" / "test.jsonl",
]

PRED_ARG2_RELATIONS = {"Attribute", "Apposition", "Ref"}
FUNC_ARG_RELATIONS = {"Parataxis", "Expression", "List"}
INNER_RELATIONS = {"Attribute", "Apposition", "Ref", "Modification"}


def node_tokens(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if not isinstance(spans, list):
        return []
    return [index for start, end in spans for index in range(start, end + 1)]


def node_text(words: list[str], node: dict[str, Any]) -> str:
    indexes = node_tokens(node)
    if indexes:
        return "".join(words[index] for index in indexes).strip()
    return str(node.get("spans"))


def relation_name(node: dict[str, Any]) -> str | None:
    spans = node.get("spans")
    return spans if isinstance(spans, str) else None


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return int(edge["start"]), int(edge["end"]), str(edge["label"])


def add_edge(edges: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    key = (start, end, label)
    if key not in {edge_key(edge) for edge in edges}:
        edges.append({"start": start, "end": end, "label": label})


def outgoing_edges(edges: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    outgoing: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        outgoing[edge["start"]].append(edge)
    return outgoing


def incoming_edges(edges: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    incoming: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        incoming[edge["end"]].append(edge)
    return incoming


def descendant_tokens(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], start: int) -> list[int]:
    outgoing = outgoing_edges(edges)
    seen: set[int] = set()
    indexes: list[int] = []
    stack = [start]
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        indexes.extend(node_tokens(nodes[index]))
        stack.extend(edge["end"] for edge in outgoing.get(index, []))
    return sorted(set(indexes))


def is_before_open(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], node: int, open_index: int) -> bool:
    indexes = descendant_tokens(nodes, edges, node)
    return bool(indexes) and max(indexes) < open_index


def token_pairs(words: list[str]) -> list[tuple[int, int]]:
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for index, word in enumerate(words):
        stripped = word.strip()
        if stripped == "(":
            stack.append(index)
        elif stripped == ")" and stack:
            pairs.append((stack.pop(), index))
    return pairs


def parenthesis_nodes(nodes: list[dict[str, Any]], open_index: int, close_index: int) -> list[int]:
    out: list[int] = []
    required = {open_index, close_index}
    for index, node in enumerate(nodes):
        if node.get("type") != "PunctuationalConstant":
            continue
        if required.issubset(set(node_tokens(node))):
            out.append(index)
    return out


def single_child(edges: list[dict[str, Any]], source: int, label: str) -> int | None:
    children = [edge["end"] for edge in edges if edge["start"] == source and edge["label"] == label]
    return children[0] if len(children) == 1 else None


def choose_func_arg_a(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    source: int,
    d_node: int,
    open_index: int,
) -> int | None:
    before: list[tuple[int, int]] = []
    for edge in edges:
        if edge["start"] != source or edge["label"] != "func.arg" or edge["end"] == d_node:
            continue
        indexes = descendant_tokens(nodes, edges, edge["end"])
        if indexes and max(indexes) < open_index:
            before.append((max(indexes), edge["end"]))
    return sorted(before)[-1][1] if before else None


def semantic_a_for_d_edge(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    source: int,
    label: str,
    d_node: int,
    open_index: int,
) -> tuple[int | None, str]:
    rel = relation_name(nodes[source])
    if label == "pred.arg.2" and rel in PRED_ARG2_RELATIONS:
        return single_child(edges, source, "pred.arg.1"), "pred.arg.2"
    if label in {"body", "variable"} and rel == "Modification":
        sibling = "variable" if label == "body" else "body"
        return single_child(edges, source, sibling), label
    if label == "func.arg" and rel in FUNC_ARG_RELATIONS:
        return choose_func_arg_a(nodes, edges, source, d_node, open_index), "func.arg"
    return None, "unsupported_incoming"


def inner_relation_ab(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], relation: int
) -> tuple[int | None, int | None, str | None]:
    rel = relation_name(nodes[relation])
    if rel in PRED_ARG2_RELATIONS:
        return (
            single_child(edges, relation, "pred.arg.1"),
            single_child(edges, relation, "pred.arg.2"),
            "pred.arg.2",
        )
    if rel == "Modification":
        return (
            single_child(edges, relation, "variable"),
            single_child(edges, relation, "body"),
            "body",
        )
    return None, None, None


def children_by_label(edges: list[dict[str, Any]], source: int, label: str) -> list[int]:
    return [edge["end"] for edge in edges if edge["start"] == source and edge["label"] == label]


def has_separate_relation_between(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    a_node: int,
    b_node: int,
    *,
    exclude: int,
) -> bool:
    pair = {a_node, b_node}
    for index, node in enumerate(nodes):
        if index == exclude:
            continue
        rel = relation_name(node)
        if rel in PRED_ARG2_RELATIONS:
            if {
                *children_by_label(edges, index, "pred.arg.1"),
                *children_by_label(edges, index, "pred.arg.2"),
            } >= pair:
                return True
        if rel == "Modification":
            if {
                *children_by_label(edges, index, "variable"),
                *children_by_label(edges, index, "body"),
            } >= pair:
                return True
        if rel in FUNC_ARG_RELATIONS:
            if set(children_by_label(edges, index, "func.arg")) >= pair:
                return True
    return False


def find_or_add_wrapper(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    a_node: int,
    d_node: int,
    *,
    exclude: set[int] | None = None,
) -> tuple[int, bool]:
    exclude = exclude or set()
    for index, node in enumerate(nodes):
        if index in exclude:
            continue
        if node.get("spans") != "Modification" or node.get("type") != "ModificationalFunctor":
            continue
        has_variable = any(
            edge["start"] == index and edge["end"] == a_node and edge["label"] == "variable"
            for edge in edges
        )
        has_body = any(
            edge["start"] == index and edge["end"] == d_node and edge["label"] == "body"
            for edge in edges
        )
        if has_variable and has_body:
            return index, False
    wrapper = len(nodes)
    nodes.append({"spans": "Modification", "type": "ModificationalFunctor"})
    add_edge(edges, wrapper, a_node, "variable")
    add_edge(edges, wrapper, d_node, "body")
    return wrapper, True


def retarget_external_incoming(
    edges: list[dict[str, Any]],
    old_target: int,
    new_target: int,
    *,
    skip_sources: set[int] | None = None,
) -> int:
    skip_sources = skip_sources or set()
    changed = 0
    for edge in edges:
        if edge["end"] != old_target:
            continue
        if edge["start"] in skip_sources:
            continue
        edge["end"] = new_target
        changed += 1
    return changed


def as_punctuation_target(edges: list[dict[str, Any]], d_node: int) -> int | None:
    targets = [edge["end"] for edge in edges if edge["start"] == d_node and edge["label"] == "as:punctuation"]
    return targets[0] if len(targets) == 1 else None


def graph_snapshot(sample: dict[str, Any], focus: set[int]) -> list[str]:
    words = sample["lingua"]["words"]
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    related = set(focus)
    for edge in edges:
        if edge["start"] in focus or edge["end"] in focus:
            related.add(edge["start"])
            related.add(edge["end"])
    lines: list[str] = []
    for index in sorted(related):
        node = nodes[index]
        lines.append(
            f"n{index}: {node.get('type')} spans={node.get('spans')} text={node_text(words, node)!r}"
        )
    for edge in edges:
        if edge["start"] in related or edge["end"] in related:
            lines.append(f"e n{edge['start']} -{edge['label']}-> n{edge['end']}")
    return lines


def record_change(
    changes: list[dict[str, Any]],
    sample: dict[str, Any],
    split: str,
    line: int,
    mode: str,
    open_index: int,
    close_index: int,
    d_node: int,
    a_node: int,
    b_node: int,
    relation_node: int,
    relation_role: str,
    wrapper: int,
    wrapper_added: bool,
    parent_retargets: int,
    before: list[str],
) -> None:
    nodes = sample["lingua"]["graph"]["nodes"]
    words = sample["lingua"]["words"]
    changes.append(
        {
            "split": split,
            "line": line,
            "mode": mode,
            "sentence": sample["sentence"],
            "open_token": open_index,
            "close_token": close_index,
            "a_node": a_node,
            "a_text": node_text(words, nodes[a_node]),
            "b_node": b_node,
            "b_text": node_text(words, nodes[b_node]),
            "d_node": d_node,
            "relation_node": relation_node,
            "relation": relation_name(nodes[relation_node]) or nodes[relation_node].get("type"),
            "relation_role": relation_role,
            "wrapper_node": wrapper,
            "wrapper_added": wrapper_added,
            "parent_retargets": parent_retargets,
            "before": before,
            "after": graph_snapshot(sample, {a_node, b_node, d_node, relation_node, wrapper}),
        }
    )


def record_skip(
    skips: list[dict[str, Any]],
    sample: dict[str, Any],
    split: str,
    line: int,
    reason: str,
    open_index: int,
    close_index: int,
    d_node: int | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    skips.append(
        {
            "split": split,
            "line": line,
            "reason": reason,
            "sentence": sample["sentence"],
            "open_token": open_index,
            "close_token": close_index,
            "d_node": d_node,
            "detail": detail or {},
        }
    )


def repair_sample(sample: dict[str, Any], split: str, line: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = sample["lingua"]["graph"]["nodes"]
    edges = sample["lingua"]["graph"]["edges"]
    words = sample["lingua"]["words"]
    changes: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []

    for open_index, close_index in token_pairs(words):
        if close_index <= open_index + 1:
            continue
        if not "".join(words[:open_index]).strip():
            continue
        p_nodes = parenthesis_nodes(nodes, open_index, close_index)
        if not p_nodes:
            record_skip(skips, sample, split, line, "no_parenthesis_punctuation_node", open_index, close_index)
            continue

        for d_node in p_nodes:
            b_target = as_punctuation_target(edges, d_node)
            if b_target is None:
                record_skip(skips, sample, split, line, "no_single_as_punctuation_target", open_index, close_index, d_node)
                continue

            incoming = incoming_edges(edges)
            incoming_relation_edges: list[tuple[dict[str, Any], int, str]] = []
            for edge in list(incoming.get(d_node, [])):
                a_node, role = semantic_a_for_d_edge(
                    nodes, edges, edge["start"], edge["label"], d_node, open_index
                )
                if a_node is None:
                    continue
                if (
                    relation_name(nodes[edge["start"]]) == "Modification"
                    and has_separate_relation_between(
                        nodes,
                        edges,
                        a_node,
                        b_target,
                        exclude=edge["start"],
                    )
                ):
                    continue
                incoming_relation_edges.append((edge, a_node, role))

            if incoming_relation_edges:
                for edge, a_node, role in incoming_relation_edges:
                    if not is_before_open(nodes, edges, a_node, open_index):
                        record_skip(
                            skips,
                            sample,
                            split,
                            line,
                            "a_not_before_parenthesis",
                            open_index,
                            close_index,
                            d_node,
                            {
                                "a_node": a_node,
                                "a_text": node_text(words, nodes[a_node]),
                                "relation_node": edge["start"],
                                "relation": relation_name(nodes[edge["start"]])
                                or nodes[edge["start"]].get("type"),
                            },
                        )
                        continue

                    relation_node = edge["start"]
                    before = graph_snapshot(sample, {a_node, b_target, d_node, relation_node})
                    wrapper, wrapper_added = find_or_add_wrapper(
                        nodes,
                        edges,
                        a_node,
                        d_node,
                        exclude={relation_node},
                    )
                    edge["end"] = b_target
                    add_edge(edges, wrapper, relation_node, "modification")
                    parent_retargets = retarget_external_incoming(
                        edges,
                        relation_node,
                        wrapper,
                        skip_sources={wrapper},
                    )
                    record_change(
                        changes,
                        sample,
                        split,
                        line,
                        "incoming_relation_to_parenthesis",
                        open_index,
                        close_index,
                        d_node,
                        a_node,
                        b_target,
                        relation_node,
                        role,
                        wrapper,
                        wrapper_added,
                        parent_retargets,
                        before,
                    )
                continue

            inner_relation = b_target
            if relation_name(nodes[inner_relation]) in INNER_RELATIONS:
                a_node, b_node, relation_role = inner_relation_ab(nodes, edges, inner_relation)
                if a_node is None or b_node is None or relation_role is None:
                    record_skip(
                        skips,
                        sample,
                        split,
                        line,
                        "inner_relation_without_binary_args",
                        open_index,
                        close_index,
                        d_node,
                        {"relation_node": inner_relation},
                    )
                    continue
                if not is_before_open(nodes, edges, a_node, open_index):
                    record_skip(
                        skips,
                        sample,
                        split,
                        line,
                        "inner_relation_a_not_before_parenthesis",
                        open_index,
                        close_index,
                        d_node,
                        {
                            "a_node": a_node,
                            "a_text": node_text(words, nodes[a_node]),
                            "relation_node": inner_relation,
                            "relation": relation_name(nodes[inner_relation]),
                        },
                    )
                    continue

                before = graph_snapshot(sample, {a_node, b_node, d_node, inner_relation})
                wrapper, wrapper_added = find_or_add_wrapper(
                    nodes,
                    edges,
                    a_node,
                    d_node,
                    exclude={inner_relation},
                )
                for edge in edges:
                    if edge["start"] == d_node and edge["end"] == inner_relation and edge["label"] == "as:punctuation":
                        edge["end"] = b_node
                        break
                add_edge(edges, wrapper, inner_relation, "modification")
                parent_retargets = retarget_external_incoming(
                    edges,
                    d_node,
                    wrapper,
                    skip_sources={wrapper},
                )
                record_change(
                    changes,
                    sample,
                    split,
                    line,
                    "parenthesis_to_inner_relation",
                    open_index,
                    close_index,
                    d_node,
                    a_node,
                    b_node,
                    inner_relation,
                    relation_role,
                    wrapper,
                    wrapper_added,
                    parent_retargets,
                    before,
                )
                continue

            record_skip(
                skips,
                sample,
                split,
                line,
                "no_supported_parenthetical_relation",
                open_index,
                close_index,
                d_node,
                {"b_node": b_target, "b_text": node_text(words, nodes[b_target])},
            )

    return changes, skips


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_summary(report_dir: Path, changes: list[dict[str, Any]], skips: list[dict[str, Any]]) -> None:
    by_split = Counter(change["split"] for change in changes)
    by_relation = Counter(change["relation"] for change in changes)
    by_skip = Counter(skip["reason"] for skip in skips)
    lines = [
        "# Parenthetical A(B) Modification Wrapper Repair",
        "",
        f"changed pairs: {len(changes)}",
        f"changed samples: {len({(c['split'], c['line']) for c in changes})}",
        f"skipped pairs: {len(skips)}",
        "",
        "## Changed by split",
        "",
    ]
    for split in ["train", "dev", "test"]:
        lines.append(f"- {split}: {by_split[split]}")
    lines.extend(["", "## Changed by relation", ""])
    for relation, count in by_relation.most_common():
        lines.append(f"- {relation}: {count}")
    lines.extend(["", "## Skips", ""])
    for reason, count in by_skip.most_common():
        lines.append(f"- {reason}: {count}")
    lines.extend(
        [
            "",
            "Detailed rows are in `changes.jsonl` and `skips.jsonl`.",
            "",
        ]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "reports" / "parenthetical_aux_modification_standard_dryrun",
    )
    args = parser.parse_args()

    all_changes: list[dict[str, Any]] = []
    all_skips: list[dict[str, Any]] = []
    repaired_by_path: dict[Path, list[dict[str, Any]]] = {}

    for path in DATA_FILES:
        split = path.stem
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        out_rows: list[dict[str, Any]] = []
        for line, row in enumerate(rows, start=1):
            sample = copy.deepcopy(row)
            changes, skips = repair_sample(sample, split, line)
            all_changes.extend(changes)
            all_skips.extend(skips)
            out_rows.append(sample if changes else row)
        repaired_by_path[path] = out_rows

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(report_dir / "changes.jsonl", all_changes)
    write_jsonl(report_dir / "skips.jsonl", all_skips)
    write_summary(report_dir, all_changes, all_skips)

    if args.apply:
        for path, rows in repaired_by_path.items():
            path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )

    mode = "applied" if args.apply else "dry-run"
    print(f"{mode}: changed_pairs={len(all_changes)} changed_samples={len({(c['split'], c['line']) for c in all_changes})}")
    print(f"{mode}: skipped_pairs={len(all_skips)}")
    print(f"report: {report_dir}")


if __name__ == "__main__":
    main()
