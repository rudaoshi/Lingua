#!/usr/bin/env python3
"""Apply manually reviewed be-copula fixes for the remaining candidates.

This is intentionally not a new global rule pass.  The candidate list was
reviewed item by item; the key sets below are the explicit manual decisions.
Candidates not listed in LEAVE_KEYS or SPECIAL_ACTIONS are manually accepted as
plain `be + complement` merges.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "be_copula_manual_review"
CANDIDATES = ROOT / "reports" / "be_copula_manual_review_candidates.tsv"
REPAIR_SCRIPT = ROOT / "scripts" / "repair_be_copula_rules.py"


spec = importlib.util.spec_from_file_location("repair_be_copula_rules", REPAIR_SCRIPT)
repair = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(repair)


Key = tuple[str, int, int, int]


LEAVE_KEYS: dict[Key, str] = {
    ("train", 1033, 9, 11): "quoted song title / nominal title expression, not a simple predicative adjective",
    ("train", 2330, 0, 5): "nominal expression `her first night back`, not simple be+adjective",
    ("train", 3037, 1, 5): "classification NP `important stories`; `all/also/important` belong inside the nominal expression",
    ("train", 3091, 0, 8): "identity NP `its direct descendant`, not a simple state complement",
    ("train", 3266, 14, 7): "existential `there be`; complement is the existent NP modifier chain",
    ("train", 3576, 0, 10): "classification NP `both good politicians`, not a simple state complement",
    ("train", 5011, 0, 3): "ill-formed nominal/classification fragment `new our master`; not safely recoverable as simple copula",
    ("train", 6647, 0, 14): "idiom `it's just so her to...`; `just so her` is not a simple adjectival complement",
    ("dev", 342, 0, 6): "existential `there be`; complement is the existent NP modifier chain",
    ("test", 103, 7, 11): "degree-marked NP `that big a deal`, not a simple predicative adjective",
}


SPECIAL_ACTIONS: dict[Key, tuple[str, Any]] = {
    ("train", 264, 8, 22): ("split_comp", {"head": "last", "modifier": "prefix"}),
    ("train", 316, 0, 35): ("split_comp", {"head": "first", "modifier": "suffix"}),
    ("train", 847, 3, 4): ("merge_chain_head", {}),
    ("train", 1646, 0, 5): ("merge_chain_head", {}),
    ("train", 2765, 5, 7): ("merge_chain_head", {}),
    ("train", 4644, 13, 20): ("split_comp", {"head": "last", "modifier": "prefix"}),
    ("train", 5083, 3, 4): ("merge_chain_head", {}),
    ("train", 5104, 1, 14): ("split_comp", {"head": "last", "modifier": "prefix"}),
    ("train", 5348, 3, 4): ("merge_chain_head", {}),
    ("train", 6647, 8, 10): ("merge_leaf", {}),
    ("train", 7433, 10, 21): ("split_comp", {"head": "first", "modifier": "suffix"}),
    ("train", 7816, 0, 4): ("merge_chain_head", {}),
    ("train", 9955, 4, 5): ("merge_chain_head", {}),
    ("train", 10899, 3, 4): ("merge_chain_head", {}),
    ("test", 201, 6, 22): ("split_comp", {"head": "last", "modifier": "prefix"}),
}


def key_from_row(row: dict[str, str]) -> Key:
    return (row["split"], int(row["line"]), int(row["pred_node"]), int(row["comp_node"]))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def replace_index(edges: list[dict[str, Any]], old: int, new: int, start: bool = True, end: bool = True) -> None:
    for edge in edges:
        if start and edge["start"] == old:
            edge["start"] = new
        if end and edge["end"] == old:
            edge["end"] = new


def remove_nodes(sample: dict[str, Any], removed: set[int]) -> dict[int, int]:
    return repair.remove_nodes(sample, removed)


def remove_pred_arg2(sample: dict[str, Any], pred: int, comp: int) -> None:
    repair.remove_edges(
        sample,
        lambda edge: edge["start"] == pred and edge["end"] == comp and edge["label"] == "pred.arg.2",
    )


def transfer_incoming(sample: dict[str, Any], old: int, new: int, excluded_starts: set[int] | None = None) -> None:
    repair.transfer_incoming_to_new_node(sample, old, new, excluded_starts or set())


def token_indexes(sample: dict[str, Any], node: int) -> list[int]:
    return [index for index, _, _ in repair.node_tokens(sample, node)]


def set_tokens(sample: dict[str, Any], node: int, indexes: list[int]) -> None:
    repair.set_node_token_indexes(sample, node, indexes)


def merge_leaf(sample: dict[str, Any], pred: int, comp: int) -> str:
    set_tokens(sample, pred, token_indexes(sample, pred) + token_indexes(sample, comp))
    remove_pred_arg2(sample, pred, comp)
    replace_index(repair.graph(sample)["edges"], comp, pred, start=False, end=True)
    if repair.edge_out(sample, comp):
        raise ValueError(f"leaf complement has outgoing edges: {repair.node_text(sample, comp)}")
    remove_nodes(sample, {comp})
    return "merge_leaf"


def collect_chain(sample: dict[str, Any], comp: int) -> list[int]:
    chain = [comp]
    seen = {comp}
    current = comp
    while True:
        outgoing = repair.edge_out(sample, current, "as:modification")
        if not outgoing:
            break
        if len(outgoing) != 1:
            raise ValueError(f"expected single modifier chain from {repair.node_text(sample, current)}")
        current = outgoing[0]["end"]
        if current in seen:
            raise ValueError("cycle in modifier chain")
        seen.add(current)
        chain.append(current)
    return chain


def merge_chain_head(sample: dict[str, Any], pred: int, comp: int) -> str:
    chain = collect_chain(sample, comp)
    if len(chain) < 2:
        return merge_leaf(sample, pred, comp)

    head = chain[-1]
    previous = chain[-2]
    set_tokens(sample, pred, token_indexes(sample, pred) + token_indexes(sample, head))
    remove_pred_arg2(sample, pred, comp)
    for edge in repair.graph(sample)["edges"]:
        if edge["start"] == previous and edge["end"] == head and edge["label"] == "as:modification":
            edge["end"] = pred
    transfer_incoming(sample, pred, comp, excluded_starts=set(chain))
    if repair.edge_out(sample, head):
        raise ValueError(f"chain head has outgoing edges after rewiring: {repair.node_text(sample, head)}")
    remove_nodes(sample, {head})
    return "merge_chain_head"


def split_comp(sample: dict[str, Any], pred: int, comp: int, head: str, modifier: str) -> str:
    tokens = repair.node_tokens(sample, comp)
    if len(tokens) < 2:
        return merge_leaf(sample, pred, comp)

    if head == "last":
        head_tokens = [tokens[-1][0]]
        modifier_tokens = [index for index, _, _ in tokens[:-1]]
    elif head == "first":
        head_tokens = [tokens[0][0]]
        modifier_tokens = [index for index, _, _ in tokens[1:]]
    else:
        raise ValueError(f"unknown head mode {head}")

    set_tokens(sample, pred, token_indexes(sample, pred) + head_tokens)
    remove_pred_arg2(sample, pred, comp)
    replace_index(repair.graph(sample)["edges"], comp, pred, start=False, end=True)
    if repair.edge_out(sample, comp):
        raise ValueError(f"split complement has outgoing edges: {repair.node_text(sample, comp)}")

    modifier_node = repair.add_node(
        sample,
        repair.spans_from_token_indexes(modifier_tokens),
        "ModificationalConstant",
    )
    repair.add_edge(sample, modifier_node, pred, "as:modification")
    transfer_incoming(sample, pred, modifier_node, excluded_starts={modifier_node})
    remove_nodes(sample, {comp})
    return f"split_comp_{modifier}_{head}"


def apply_action(sample: dict[str, Any], key: Key, action: str, params: dict[str, Any]) -> str:
    _, _, pred, comp = key
    if action == "merge_leaf":
        return merge_leaf(sample, pred, comp)
    if action == "merge_chain_head":
        return merge_chain_head(sample, pred, comp)
    if action == "split_comp":
        return split_comp(sample, pred, comp, params["head"], params["modifier"])
    raise ValueError(f"unknown action {action}")


def compact(sample: dict[str, Any]) -> str:
    return repair.compact_graph(sample)


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with CANDIDATES.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    decisions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decision_rows: list[dict[str, Any]] = []
    for row in rows:
        key = key_from_row(row)
        if key in LEAVE_KEYS:
            action = "leave"
            params: dict[str, Any] = {"reason": LEAVE_KEYS[key]}
        else:
            action, params = SPECIAL_ACTIONS.get(key, ("merge_leaf", {}))
        decisions[key[0]].append({"key": key, "action": action, "params": params, "row": row})
        decision_rows.append({**row, "action": action, "reason": params.get("reason", "")})

    with (REPORT_DIR / "manual_decisions.tsv").open("w", newline="") as handle:
        fieldnames = ["split", "line", "pred_node", "comp_node", "pred", "comp", "sentence", "action", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(decision_rows)

    changes: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for split in ("train", "dev", "test"):
        samples = read_jsonl(DATA / f"{split}.jsonl")
        by_line: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for decision in decisions.get(split, []):
            by_line[decision["key"][1]].append(decision)

        for line_number, line_decisions in by_line.items():
            sample = samples[line_number - 1]
            # Removing nodes shifts later indexes; process higher indexes first.
            line_decisions.sort(key=lambda item: max(item["key"][2], item["key"][3]), reverse=True)
            before = compact(sample)
            applied: list[dict[str, Any]] = []
            for decision in line_decisions:
                if decision["action"] == "leave":
                    counts["leave"] += 1
                    applied.append(
                        {
                            "key": decision["key"],
                            "action": "leave",
                            "reason": decision["params"]["reason"],
                        }
                    )
                    continue
                result = apply_action(sample, decision["key"], decision["action"], decision["params"])
                counts[result] += 1
                applied.append({"key": decision["key"], "action": result})
            if any(item["action"] != "leave" for item in applied):
                changes.append(
                    {
                        "split": split,
                        "line": line_number,
                        "sentence": sample["sentence"],
                        "applied": applied,
                        "before": before,
                        "after": compact(sample),
                    }
                )
        write_jsonl(DATA / f"{split}.jsonl", samples)

    (REPORT_DIR / "changes.jsonl").write_text(
        "\n".join(json.dumps(change, ensure_ascii=False) for change in changes) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Manual Be-Copula Review",
        "",
        f"- candidates reviewed: {len(rows)}",
        f"- changed samples: {len(changes)}",
        "",
        "| action | count |",
        "|---|---:|",
    ]
    for action, count in sorted(counts.items()):
        lines.append(f"| {action} | {count} |")
    (REPORT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"reviewed": len(rows), "changed_samples": len(changes), "counts": counts}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
