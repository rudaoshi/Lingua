#!/usr/bin/env python3
"""Apply UD-backed second-pass fixes for reviewed multiword leaf splits.

This pass only touches repairs created by
``repair_multiword_nonpredicate_leaf_nodes.py`` and only for two review
classes:

- ``ud_suggests_head_modifier_not_expression``: remove the auxiliary
  Expression wrapper and restore the token-level modifier chain.
- ``ud_suggests_flat_expression_not_modifier_chain``: replace the token-level
  modifier chain with a flat Expression wrapper.

Lingua's adjacent multi-modifier chain convention is intentionally left alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict, deque
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
DEFAULT_CHANGES = REPORTS / "multiword_nonpredicate_leaf_repair_apply" / "changes.jsonl"
DEFAULT_REVIEW = REPORTS / "multiword_nonpredicate_leaf_ud_validation_20260611" / "review_queue.tsv"

HEAD_MODIFIER_CLASS = "ud_suggests_head_modifier_not_expression"
FLAT_EXPRESSION_CLASS = "ud_suggests_flat_expression_not_modifier_chain"


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return int(edge["start"]), int(edge["end"]), str(edge["label"])


def add_edge_once(edges: list[dict[str, Any]], start: int, end: int, label: str) -> None:
    if start == end:
        return
    new_edge = {"start": int(start), "end": int(end), "label": label}
    if edge_key(new_edge) in {edge_key(edge) for edge in edges}:
        return
    edges.append(new_edge)


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if isinstance(spans, str):
        return []
    indexes: list[int] = []
    for span in spans:
        if (
            isinstance(span, list)
            and len(span) == 2
            and isinstance(span[0], int)
            and isinstance(span[1], int)
        ):
            indexes.extend(range(span[0], span[1] + 1))
    return sorted(set(indexes))


def token_node_map(sample: dict[str, Any]) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = defaultdict(list)
    for index, node in enumerate(graph(sample)["nodes"]):
        tokens = node_token_indexes(node)
        if len(tokens) == 1 and node.get("spans") == [[tokens[0], tokens[0]]]:
            mapping[tokens[0]].append(index)
    return mapping


def node_text(sample: dict[str, Any], node_index: int) -> str:
    node = graph(sample)["nodes"][node_index]
    spans = node.get("spans")
    if isinstance(spans, str):
        return spans
    words = sample["lingua"]["words"]
    pieces: list[str] = []
    for token in node_token_indexes(node):
        if 0 <= token < len(words):
            pieces.append(words[token].strip())
    return " ".join(piece for piece in pieces if piece)


def find_token_nodes(sample: dict[str, Any], tokens: list[int]) -> tuple[list[int] | None, str | None]:
    mapping = token_node_map(sample)
    nodes: list[int] = []
    for token in tokens:
        candidates = mapping.get(token, [])
        if len(candidates) != 1:
            return None, f"token_{token}_node_count_{len(candidates)}"
        nodes.append(candidates[0])
    if len(set(nodes)) != len(nodes):
        return None, "duplicate_token_nodes"
    return nodes, None


def outgoing_edges(sample: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    outgoing: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph(sample).get("edges", []):
        outgoing[int(edge["start"])].append(edge)
    return outgoing


def incoming_edges(sample: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    incoming: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph(sample).get("edges", []):
        incoming[int(edge["end"])].append(edge)
    return incoming


def find_expression_root(sample: dict[str, Any], token_nodes: list[int]) -> int | None:
    target = set(token_nodes)
    outgoing = outgoing_edges(sample)
    candidates: list[int] = []
    for index, node in enumerate(graph(sample)["nodes"]):
        if node.get("type") != "ExpressionFunctor" or node.get("spans") != "Expression":
            continue
        children = [
            int(edge["end"])
            for edge in outgoing.get(index, [])
            if edge.get("label") == "func.arg"
        ]
        if set(children) == target and len(children) == len(target):
            candidates.append(index)
    if len(candidates) == 1:
        return candidates[0]
    return None


def has_expression_root(sample: dict[str, Any], token_nodes: list[int]) -> bool:
    return find_expression_root(sample, token_nodes) is not None


def has_consecutive_chain(sample: dict[str, Any], token_nodes: list[int]) -> bool:
    edge_keys = {edge_key(edge) for edge in graph(sample).get("edges", [])}
    return all(
        (left, right, "as:modification") in edge_keys
        for left, right in zip(token_nodes, token_nodes[1:])
    )


def remove_node_reindex(nodes: list[dict[str, Any]], remove_index: int) -> tuple[list[dict[str, Any]], dict[int, int]]:
    new_nodes: list[dict[str, Any]] = []
    reindex: dict[int, int] = {}
    for index, node in enumerate(nodes):
        if index == remove_index:
            continue
        reindex[index] = len(new_nodes)
        new_nodes.append(node)
    return new_nodes, reindex


def expression_to_chain(sample: dict[str, Any], target: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    token_nodes, reason = find_token_nodes(sample, target["tokens"])
    if token_nodes is None:
        return sample, f"skip:{reason}", {}

    expr_root = find_expression_root(sample, token_nodes)
    if expr_root is None:
        if has_consecutive_chain(sample, token_nodes):
            return sample, "noop:already_chain", {"token_nodes": token_nodes}
        return sample, "skip:expression_root_not_found", {"token_nodes": token_nodes}

    target_set = set(token_nodes)
    outgoing = outgoing_edges(sample)
    unexpected_outgoing = [
        edge
        for edge in outgoing.get(expr_root, [])
        if not (edge.get("label") == "func.arg" and int(edge["end"]) in target_set)
    ]
    if unexpected_outgoing:
        return sample, "skip:expression_root_has_external_outgoing", {
            "expression_root": expr_root,
            "unexpected_outgoing": unexpected_outgoing,
            "token_nodes": token_nodes,
        }

    old_nodes = graph(sample)["nodes"]
    new_nodes, reindex = remove_node_reindex([dict(node) for node in old_nodes], expr_root)
    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample).get("edges", []):
        start = int(edge["start"])
        end = int(edge["end"])
        label = str(edge["label"])
        if start == expr_root:
            continue
        if end == expr_root:
            end = token_nodes[0]
        if start == expr_root or end == expr_root:
            continue
        add_edge_once(new_edges, reindex[start], reindex[end], label)
    for left, right in zip(token_nodes, token_nodes[1:]):
        add_edge_once(new_edges, reindex[left], reindex[right], "as:modification")

    repaired = deepcopy(sample)
    repaired["lingua"] = deepcopy(sample["lingua"])
    repaired["lingua"]["graph"] = {"nodes": new_nodes, "edges": new_edges}
    return repaired, "changed:expression_to_chain", {
        "expression_root": expr_root,
        "old_token_nodes": token_nodes,
        "new_token_nodes": [reindex[node] for node in token_nodes],
    }


def chain_to_expression(sample: dict[str, Any], target: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    token_nodes, reason = find_token_nodes(sample, target["tokens"])
    if token_nodes is None:
        return sample, f"skip:{reason}", {}

    if has_expression_root(sample, token_nodes) and not has_consecutive_chain(sample, token_nodes):
        return sample, "noop:already_expression", {"token_nodes": token_nodes}
    if not has_consecutive_chain(sample, token_nodes):
        return sample, "skip:chain_not_found", {"token_nodes": token_nodes}

    target_set = set(token_nodes)
    external_outgoing = [
        edge
        for edge in graph(sample).get("edges", [])
        if int(edge["start"]) in target_set and int(edge["end"]) not in target_set
    ]
    if external_outgoing:
        return sample, "skip:target_has_external_outgoing", {
            "token_nodes": token_nodes,
            "external_outgoing": external_outgoing,
        }

    expr_root = len(graph(sample)["nodes"])
    new_nodes = [dict(node) for node in graph(sample)["nodes"]]
    new_nodes.append({"spans": "Expression", "type": "ExpressionFunctor"})

    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample).get("edges", []):
        start = int(edge["start"])
        end = int(edge["end"])
        label = str(edge["label"])
        if label == "as:modification" and start in target_set and end in target_set:
            continue
        if end in target_set and start not in target_set:
            end = expr_root
        add_edge_once(new_edges, start, end, label)
    for node in token_nodes:
        add_edge_once(new_edges, expr_root, node, "func.arg")

    repaired = deepcopy(sample)
    repaired["lingua"] = deepcopy(sample["lingua"])
    repaired["lingua"]["graph"] = {"nodes": new_nodes, "edges": new_edges}
    return repaired, "changed:chain_to_expression", {
        "expression_root": expr_root,
        "token_nodes": token_nodes,
    }


def validate_graph_shape(sample: dict[str, Any]) -> None:
    nodes = graph(sample)["nodes"]
    adjacency = [[] for _ in nodes]
    for edge in graph(sample).get("edges", []):
        start = int(edge["start"])
        end = int(edge["end"])
        if not (0 <= start < len(nodes) and 0 <= end < len(nodes)):
            raise ValueError(f"dangling edge: {edge}")
        adjacency[start].append(end)

    color = [0] * len(nodes)

    def visit(node: int) -> None:
        if color[node] == 1:
            raise ValueError("cycle detected")
        if color[node] == 2:
            return
        color[node] = 1
        for child in adjacency[node]:
            visit(child)
        color[node] = 2

    for index in range(len(nodes)):
        visit(index)


def load_review_targets(changes_path: Path, review_path: Path) -> list[dict[str, Any]]:
    review_rows = [
        row
        for row in csv.DictReader(review_path.open(encoding="utf-8"))
        if row.get("classification") in {HEAD_MODIFIER_CLASS, FLAT_EXPRESSION_CLASS}
    ]
    wanted: dict[tuple[str, int, str, str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in review_rows:
        key = (
            row["split"],
            int(row["sample_id"]),
            row["old_type"],
            row["action"],
            row["text"],
        )
        wanted[key].append(row)

    targets: list[dict[str, Any]] = []
    for line in changes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for repair in record["repairs"]:
            key = (
                record["split"],
                int(record["sample_id"]),
                str(repair["old_type"]),
                str(repair["action"]),
                str(repair["text"]),
            )
            if not wanted.get(key):
                continue
            review = wanted[key].popleft()
            classification = review["classification"]
            transform = (
                "expression_to_chain"
                if classification == HEAD_MODIFIER_CLASS
                else "chain_to_expression"
            )
            targets.append(
                {
                    "split": record["split"],
                    "line": int(record["line"]),
                    "sample_id": int(record["sample_id"]),
                    "sentence": record["sentence"],
                    "old_type": repair["old_type"],
                    "action": repair["action"],
                    "text": repair["text"],
                    "tokens": repair["tokens"],
                    "classification": classification,
                    "transform": transform,
                    "ud_deprels": review.get("ud_deprels"),
                    "ud_roots": review.get("ud_roots"),
                    "ud_internal_edges": review.get("ud_internal_edges"),
                }
            )

    unmatched = sum(len(rows) for rows in wanted.values())
    if unmatched:
        raise RuntimeError(f"{unmatched} review rows were not matched to repair changes")
    return sorted(targets, key=lambda item: (item["split"], item["sample_id"], item["tokens"]))


def target_status(sample: dict[str, Any], target: dict[str, Any]) -> str:
    token_nodes, reason = find_token_nodes(sample, target["tokens"])
    if token_nodes is None:
        return f"unknown:{reason}"
    expression = has_expression_root(sample, token_nodes)
    chain = has_consecutive_chain(sample, token_nodes)
    if target["transform"] == "expression_to_chain":
        if chain and not expression:
            return "fixed"
        if expression:
            return "bad:still_expression"
    else:
        if expression and not chain:
            return "fixed"
        if chain:
            return "bad:still_chain"
    return "unknown:neither_expected_shape"


def run(
    *,
    apply: bool,
    data_dir: Path,
    report_dir: Path,
    changes_path: Path,
    review_path: Path,
) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    case_dir = report_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    targets = load_review_targets(changes_path, review_path)
    targets_by_split_sample: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        targets_by_split_sample[(target["split"], target["sample_id"])].append(target)

    changed_records: list[dict[str, Any]] = []
    item_records: list[dict[str, Any]] = []
    by_result = Counter()
    by_transform = Counter()
    by_classification = Counter()
    residual_status = Counter()

    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        new_rows: list[dict[str, Any]] = []
        for sample_id, row in enumerate(rows):
            sample_targets = targets_by_split_sample.get((split, sample_id), [])
            if not sample_targets:
                new_rows.append(row)
                continue

            before = deepcopy(row)
            after = deepcopy(row)
            applied_items: list[dict[str, Any]] = []
            for target in sample_targets:
                if target["transform"] == "expression_to_chain":
                    after, result, details = expression_to_chain(after, target)
                else:
                    after, result, details = chain_to_expression(after, target)
                validate_graph_shape(after)
                by_result[result] += 1
                by_transform[target["transform"]] += 1
                by_classification[target["classification"]] += 1
                status = target_status(after, target)
                residual_status[status] += 1
                item = {
                    **target,
                    "result": result,
                    "post_status": status,
                    "details": details,
                }
                item_records.append(item)
                applied_items.append(item)

            if before != after:
                record = {
                    "split": split,
                    "line": sample_id + 1,
                    "sample_id": sample_id,
                    "sentence": row.get("sentence"),
                    "items": applied_items,
                    "before": compact_graph(before),
                    "after": compact_graph(after),
                }
                changed_records.append(record)
                (case_dir / f"{split}_{sample_id + 1:05d}.txt").write_text(
                    f"{split}:{sample_id}\n{row.get('sentence')}\n\n"
                    f"ITEMS: {json.dumps(applied_items, ensure_ascii=False)}\n\n"
                    f"BEFORE\n{record['before']}\n\nAFTER\n{record['after']}\n",
                    encoding="utf-8",
                )
            new_rows.append(after)
        if apply:
            write_jsonl(path, new_rows)

    summary = {
        "apply": apply,
        "target_items": len(targets),
        "changed_samples": len(changed_records),
        "changed_items": sum(1 for item in item_records if item["result"].startswith("changed:")),
        "by_result": dict(by_result.most_common()),
        "by_transform": dict(by_transform.most_common()),
        "by_classification": dict(by_classification.most_common()),
        "post_status": dict(residual_status.most_common()),
        "report_dir": str(report_dir),
    }
    (report_dir / "changes.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in changed_records)
        + ("\n" if changed_records else ""),
        encoding="utf-8",
    )
    (report_dir / "items.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in item_records)
        + ("\n" if item_records else ""),
        encoding="utf-8",
    )
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Multiword UD-Reviewed Structure Repair",
        "",
        f"- apply: `{apply}`",
        f"- target items: `{summary['target_items']}`",
        f"- changed samples: `{summary['changed_samples']}`",
        f"- changed items: `{summary['changed_items']}`",
        "",
        "## By Result",
    ]
    for key, count in by_result.most_common():
        lines.append(f"- `{key}`: {count}")
    lines.append("")
    lines.append("## Post Status")
    for key, count in residual_status.most_common():
        lines.append(f"- `{key}`: {count}")
    lines.append("")
    lines.append("## By Classification")
    for key, count in by_classification.most_common():
        lines.append(f"- `{key}`: {count}")
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--changes", type=Path, default=DEFAULT_CHANGES)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    return run(
        apply=args.apply,
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        changes_path=args.changes,
        review_path=args.review,
    )


if __name__ == "__main__":
    raise SystemExit(main())
