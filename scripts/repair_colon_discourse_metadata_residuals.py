#!/usr/bin/env python3
"""Repair manually reviewed colon metadata residuals.

Speaker/source/header prefixes such as `GOODWYN:` and `EDIT:` are discourse
metadata when they introduce an independent utterance.  Pure key-value fields
such as `DATE:` are intentionally left as Attribute structures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def make_explicit_discourse(
    sample: dict[str, Any],
    *,
    aux: int,
    body_target: int,
    variable_target: int,
    pred_arg1_label: str = "pred.arg.1",
    pred_arg2_label: str = "pred.arg.2",
) -> None:
    g = graph(sample)
    nodes = g["nodes"]
    edges = g["edges"]
    if (
        nodes[aux] == {"spans": "Discourse", "type": "ModificationalFunctor"}
        and {"start": aux, "end": body_target, "label": "body"} in edges
        and {"start": aux, "end": variable_target, "label": "variable"} in edges
    ):
        return
    nodes[aux] = {"spans": "Discourse", "type": "ModificationalFunctor"}
    changed = 0
    for item in edges:
        if item == {"start": aux, "end": body_target, "label": pred_arg1_label}:
            item["label"] = "body"
            changed += 1
        elif item == {"start": aux, "end": variable_target, "label": pred_arg2_label}:
            item["label"] = "variable"
            changed += 1
    if changed != 2:
        raise ValueError(f"failed to rewrite explicit discourse aux {aux}")


def make_compact_discourse(sample: dict[str, Any], *, source: int, target: int) -> None:
    g = graph(sample)
    nodes = g["nodes"]
    edges = g["edges"]
    if nodes[source]["type"] == "NominalConstant" and {"start": source, "end": target, "label": "as:discourse"} in edges:
        return
    nodes[source]["type"] = "NominalConstant"
    changed = 0
    for item in edges:
        if item == {"start": source, "end": target, "label": "pred.arg.2"}:
            item["label"] = "as:discourse"
            changed += 1
    if changed != 1:
        raise ValueError(f"failed to rewrite compact discourse source {source}")


def fix_train_543(sample: dict[str, Any]) -> None:
    make_explicit_discourse(sample, aux=6, body_target=1, variable_target=2)


def fix_train_5725(sample: dict[str, Any]) -> None:
    make_compact_discourse(sample, source=3, target=1)


def fix_train_5067(sample: dict[str, Any]) -> None:
    make_compact_discourse(sample, source=6, target=1)


def fix_train_5853(sample: dict[str, Any]) -> None:
    make_compact_discourse(sample, source=3, target=1)


def fix_train_8739(sample: dict[str, Any]) -> None:
    make_explicit_discourse(sample, aux=15, body_target=0, variable_target=14)


def fix_train_9315(sample: dict[str, Any]) -> None:
    make_explicit_discourse(sample, aux=6, body_target=3, variable_target=5)


def remove_edges(edges: list[dict[str, Any]], *, start: int | None = None, end: int | None = None, label: str | None = None) -> None:
    edges[:] = [
        item
        for item in edges
        if not (
            (start is None or item["start"] == start)
            and (end is None or item["end"] == end)
            and (label is None or item["label"] == label)
        )
    ]


def add_edge_once(edges: list[dict[str, Any]], item: dict[str, Any]) -> None:
    if item not in edges:
        edges.append(item)


def fix_dev_493(sample: dict[str, Any]) -> None:
    g = graph(sample)
    nodes = g["nodes"]
    edges = g["edges"]
    if nodes[1]["type"] == "AttributePredicator" and {"start": 1, "end": 2, "label": "pred.arg.2"} in edges:
        return
    nodes[1]["type"] = "AttributePredicator"
    remove_edges(edges, start=8, end=0, label="pred.arg.1")
    remove_edges(edges, start=8, end=2, label="pred.arg.2")
    nodes.pop()
    add_edge_once(edges, {"start": 1, "end": 2, "label": "pred.arg.2"})


def fix_test_8(sample: dict[str, Any]) -> None:
    g = graph(sample)
    nodes = g["nodes"]
    edges = g["edges"]
    if nodes[2]["type"] == "AttributePredicator" and {"start": 2, "end": 1, "label": "pred.arg.2"} in edges:
        return
    nodes[2]["type"] = "AttributePredicator"
    remove_edges(edges, start=7, end=0, label="pred.arg.1")
    remove_edges(edges, start=7, end=1, label="pred.arg.2")
    for item in edges:
        if item == {"start": 6, "end": 7, "label": "as:punctuation"}:
            item["end"] = 0
    nodes.pop()
    add_edge_once(edges, {"start": 2, "end": 1, "label": "pred.arg.2"})


FIXES = {
    "train": {
        543: fix_train_543,
        5067: fix_train_5067,
        5725: fix_train_5725,
        5853: fix_train_5853,
        8739: fix_train_8739,
        9315: fix_train_9315,
    },
    "dev": {
        493: fix_dev_493,
    },
    "test": {
        8: fix_test_8,
    },
}


def main() -> None:
    changed = []
    for split, fixes in FIXES.items():
        path = DATA / f"{split}.jsonl"
        rows = read_jsonl(path)
        for idx, fix in fixes.items():
            before = json.dumps(rows[idx]["lingua"]["graph"], ensure_ascii=False, sort_keys=True)
            fix(rows[idx])
            after = json.dumps(rows[idx]["lingua"]["graph"], ensure_ascii=False, sort_keys=True)
            if before != after:
                changed.append(f"{split}:{idx}")
        write_jsonl(path, rows)
    print("changed", len(changed), ", ".join(changed))


if __name__ == "__main__":
    main()
