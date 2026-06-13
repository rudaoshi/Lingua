#!/usr/bin/env python3
"""Repair high-confidence to-infinitive ARG2/ARG4/Modification roles.

This follows docs/standard.md:

* matrix-selected ``verb + to VP`` content events are ``pred.arg.2``;
* object/recipient-controlled ``verb + ARG2/ARG3 + to VP`` complements are
  ``pred.arg.4``;
* purpose ``to VP`` modifiers stay as Modification and are not touched here.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = [
    ROOT / "data" / "train.jsonl",
    ROOT / "data" / "dev.jsonl",
    ROOT / "data" / "test.jsonl",
]
REPORT_DIR = ROOT / "reports" / "to_infinitive_arg_role_repair"

CONTENT_VERBS = {
    "agree",
    "agreed",
    "agrees",
    "appear",
    "appeared",
    "appears",
    "attempt",
    "attempted",
    "attempting",
    "attempts",
    "begin",
    "beginning",
    "begins",
    "began",
    "begun",
    "choose",
    "chooses",
    "chose",
    "chosen",
    "continue",
    "continued",
    "continues",
    "continuing",
    "decide",
    "decided",
    "decides",
    "deciding",
    "expect",
    "expected",
    "expects",
    "fail",
    "failed",
    "fails",
    "forget",
    "forgot",
    "forgotten",
    "forgot",
    "hope",
    "hoped",
    "hopes",
    "intend",
    "intended",
    "intends",
    "learn",
    "learned",
    "learns",
    "learning",
    "manage",
    "managed",
    "manages",
    "mean",
    "means",
    "meant",
    "need",
    "needed",
    "needs",
    "plan",
    "planned",
    "plans",
    "prefer",
    "preferred",
    "prefers",
    "promise",
    "promised",
    "promises",
    "refuse",
    "refused",
    "refuses",
    "remember",
    "remembered",
    "remembers",
    "seem",
    "seemed",
    "seems",
    "start",
    "started",
    "starting",
    "starts",
    "tend",
    "tended",
    "tends",
    "try",
    "tried",
    "tries",
    "trying",
    "want",
    "wanted",
    "wants",
}

CONTROL_VERBS = {
    "allow",
    "allowed",
    "allows",
    "ask",
    "asked",
    "asks",
    "assign",
    "assigned",
    "assigns",
    "cause",
    "caused",
    "causes",
    "enable",
    "enabled",
    "enables",
    "encourage",
    "encouraged",
    "encourages",
    "force",
    "forced",
    "forces",
    "get",
    "gets",
    "got",
    "help",
    "helped",
    "helps",
    "order",
    "ordered",
    "orders",
    "permit",
    "permitted",
    "permits",
    "persuade",
    "persuaded",
    "persuades",
    "require",
    "required",
    "requires",
    "teach",
    "taught",
    "teaches",
    "tell",
    "told",
    "tells",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def words(sample: dict[str, Any]) -> list[str]:
    return [str(word).strip() for word in sample["lingua"]["words"]]


def node_tokens(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if not isinstance(spans, list):
        return []
    tokens: list[int] = []
    for start, end in spans:
        tokens.extend(range(int(start), int(end) + 1))
    return sorted(set(tokens))


def spans_for_tokens(tokens: list[int]) -> list[list[int]]:
    tokens = sorted(set(tokens))
    if not tokens:
        return []
    spans: list[list[int]] = []
    start = previous = tokens[0]
    for token in tokens[1:]:
        if token == previous + 1:
            previous = token
            continue
        spans.append([start, previous])
        start = previous = token
    spans.append([start, previous])
    return spans


def own_words(sample: dict[str, Any], node_index: int) -> list[str]:
    ws = words(sample)
    return [ws[token].lower() for token in node_tokens(graph(sample)["nodes"][node_index])]


def text_for_tokens(sample: dict[str, Any], tokens: set[int] | list[int]) -> str:
    ws = words(sample)
    return " ".join(ws[token] for token in sorted(set(tokens)) if 0 <= token < len(ws))


def build_outgoing(sample: dict[str, Any]) -> list[list[dict[str, Any]]]:
    nodes = graph(sample)["nodes"]
    outgoing: list[list[dict[str, Any]]] = [[] for _ in nodes]
    for edge in graph(sample)["edges"]:
        outgoing[int(edge["start"])].append(edge)
    return outgoing


def descendant_nodes(sample: dict[str, Any], node_index: int) -> set[int]:
    outgoing = build_outgoing(sample)
    seen: set[int] = set()
    stack = [node_index]
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        stack.extend(int(edge["end"]) for edge in outgoing[index])
    return seen


def descendant_tokens(sample: dict[str, Any], node_index: int) -> set[int]:
    nodes = graph(sample)["nodes"]
    tokens: set[int] = set()
    for index in descendant_nodes(sample, node_index):
        tokens.update(node_tokens(nodes[index]))
    return tokens


def descendant_text(sample: dict[str, Any], node_index: int) -> str:
    return text_for_tokens(sample, descendant_tokens(sample, node_index))


def child_edges(sample: dict[str, Any], node_index: int, label: str) -> list[dict[str, Any]]:
    return [
        edge
        for edge in graph(sample)["edges"]
        if int(edge["start"]) == node_index and edge["label"] == label
    ]


def targets(sample: dict[str, Any], node_index: int, label: str) -> list[int]:
    return [int(edge["end"]) for edge in child_edges(sample, node_index, label)]


def equivalent_desc(sample: dict[str, Any], left: int, right: int) -> bool:
    return bool(descendant_tokens(sample, left)) and descendant_tokens(sample, left) == descendant_tokens(sample, right)


def node_has_word(sample: dict[str, Any], node_index: int, choices: set[str]) -> bool:
    return any(word in choices for word in own_words(sample, node_index))


def to_vp_predicates(sample: dict[str, Any], root: int) -> list[int]:
    nodes = graph(sample)["nodes"]
    out: list[int] = []
    for index in descendant_nodes(sample, root):
        node = nodes[index]
        if node.get("type") != "FactualPredicator":
            continue
        if "to" in own_words(sample, index):
            out.append(index)
    return out


def same_subject_to_vp(sample: dict[str, Any], parent: int, child: int) -> bool:
    parent_arg1 = targets(sample, parent, "pred.arg.1")
    if not parent_arg1:
        return not targets(sample, parent, "pred.arg.2") and not targets(sample, parent, "pred.arg.3")
    for predicate in to_vp_predicates(sample, child):
        for child_arg1 in targets(sample, predicate, "pred.arg.1"):
            if any(equivalent_desc(sample, child_arg1, parent_subject) for parent_subject in parent_arg1):
                return True
    return False


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return int(edge["start"]), int(edge["end"]), str(edge["label"])


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


def add_edge_once(sample: dict[str, Any], start: int, end: int, label: str) -> None:
    edge = {"start": start, "end": end, "label": label}
    if edge_key(edge) not in {edge_key(existing) for existing in graph(sample)["edges"]}:
        graph(sample)["edges"].append(edge)


def delete_node(sample: dict[str, Any], deleted: int) -> None:
    new_edges: list[dict[str, Any]] = []
    for edge in graph(sample)["edges"]:
        start = int(edge["start"])
        end = int(edge["end"])
        if start == deleted or end == deleted:
            continue
        new_edge = dict(edge)
        if start > deleted:
            new_edge["start"] = start - 1
        if end > deleted:
            new_edge["end"] = end - 1
        new_edges.append(new_edge)
    del graph(sample)["nodes"][deleted]
    graph(sample)["edges"] = new_edges


def merge_own_spans_into(sample: dict[str, Any], source: int, target: int) -> None:
    nodes = graph(sample)["nodes"]
    source_tokens = node_tokens(nodes[source])
    target_tokens = node_tokens(nodes[target])
    nodes[target]["spans"] = spans_for_tokens(source_tokens + target_tokens)


def body_arg1_controlled_by(sample: dict[str, Any], body: int, controllers: list[int]) -> bool:
    if graph(sample)["nodes"][body].get("type") != "FactualPredicator":
        return False
    for arg1 in targets(sample, body, "pred.arg.1"):
        if any(equivalent_desc(sample, arg1, controller) for controller in controllers):
            return True
    return False


def own_text(sample: dict[str, Any], node_index: int) -> str:
    return text_for_tokens(sample, node_tokens(graph(sample)["nodes"][node_index]))


def sentence(sample: dict[str, Any]) -> str:
    return sample.get("sentence") or " ".join(words(sample))


def repair_arg4_to_arg2(sample: dict[str, Any], split: str, line_no: int) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for edge in list(graph(sample)["edges"]):
        if edge["label"] != "pred.arg.4":
            continue
        parent = int(edge["start"])
        child = int(edge["end"])
        if graph(sample)["nodes"][parent].get("type") != "FactualPredicator":
            continue
        if not node_has_word(sample, parent, CONTENT_VERBS):
            continue
        if not to_vp_predicates(sample, child):
            continue
        if targets(sample, parent, "pred.arg.2") or targets(sample, parent, "pred.arg.3"):
            continue
        if not same_subject_to_vp(sample, parent, child):
            continue
        before_label = edge["label"]
        edge["label"] = "pred.arg.2"
        changes.append(
            {
                "split": split,
                "line_no": line_no,
                "action": "arg4_to_arg2",
                "parent": own_text(sample, parent),
                "child": descendant_text(sample, child),
                "before": before_label,
                "after": "pred.arg.2",
                "sentence": sentence(sample),
            }
        )
    if changes:
        dedupe_edges(sample)
    return changes


def find_modification_to_arg4_candidate(sample: dict[str, Any]) -> tuple[int, int, int] | None:
    nodes = graph(sample)["nodes"]
    for to_node, node in enumerate(nodes):
        if node.get("type") != "ModificationalFunctor":
            continue
        if own_words(sample, to_node) != ["to"]:
            continue
        variable_edges = child_edges(sample, to_node, "variable")
        body_edges = child_edges(sample, to_node, "body")
        if len(variable_edges) != 1 or len(body_edges) != 1:
            continue
        parent = int(variable_edges[0]["end"])
        body = int(body_edges[0]["end"])
        if nodes[parent].get("type") != "FactualPredicator":
            continue
        if not node_has_word(sample, parent, CONTROL_VERBS):
            continue
        controllers = targets(sample, parent, "pred.arg.2") + targets(sample, parent, "pred.arg.3")
        if not controllers:
            continue
        if not body_arg1_controlled_by(sample, body, controllers):
            continue
        if any(int(edge["start"]) == parent and int(edge["end"]) == body and edge["label"] == "pred.arg.4" for edge in graph(sample)["edges"]):
            continue
        return to_node, parent, body
    return None


def repair_modification_to_arg4(sample: dict[str, Any], split: str, line_no: int) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    while True:
        candidate = find_modification_to_arg4_candidate(sample)
        if candidate is None:
            break
        to_node, parent, body = candidate
        changes.append(
            {
                "split": split,
                "line_no": line_no,
                "action": "modification_to_arg4",
                "parent": own_text(sample, parent),
                "body": descendant_text(sample, body),
                "removed_wrapper": own_text(sample, to_node),
                "sentence": sentence(sample),
            }
        )
        for edge in graph(sample)["edges"]:
            if int(edge["end"]) == to_node:
                edge["end"] = parent
        merge_own_spans_into(sample, to_node, body)
        add_edge_once(sample, parent, body, "pred.arg.4")
        delete_node(sample, to_node)
        dedupe_edges(sample)
    return changes


def repair_sample(sample: dict[str, Any], split: str, line_no: int) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    changes.extend(repair_arg4_to_arg2(sample, split, line_no))
    changes.extend(repair_modification_to_arg4(sample, split, line_no))
    return changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    all_changes: list[dict[str, Any]] = []
    updated_by_file: dict[Path, list[dict[str, Any]]] = {}
    for path in DATA_FILES:
        split = path.stem
        rows = read_jsonl(path)
        new_rows = copy.deepcopy(rows)
        for index, row in enumerate(new_rows, start=1):
            all_changes.extend(repair_sample(row, split, index))
        updated_by_file[path] = new_rows

    summary = Counter(change["action"] for change in all_changes)
    split_summary = Counter((change["split"], change["action"]) for change in all_changes)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / ("changes_apply.jsonl" if args.apply else "changes_dryrun.jsonl")).open("w", encoding="utf-8") as handle:
        for change in all_changes:
            handle.write(json.dumps(change, ensure_ascii=False) + "\n")
    with (REPORT_DIR / ("summary_apply.json" if args.apply else "summary_dryrun.json")).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "applied": args.apply,
                "total_changes": len(all_changes),
                "by_action": dict(summary),
                "by_split_action": {f"{split}:{action}": count for (split, action), count in split_summary.items()},
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    if args.apply:
        for path, rows in updated_by_file.items():
            write_jsonl(path, rows)

    print(json.dumps({"applied": args.apply, "total_changes": len(all_changes), "by_action": dict(summary)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
