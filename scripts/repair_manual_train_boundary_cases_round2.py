#!/usr/bin/env python3
"""Hand repairs for the second train-only variable/body audit batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def graph(row: dict[str, Any]) -> dict[str, Any]:
    return row["lingua"]["graph"]


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return int(edge["start"]), int(edge["end"]), str(edge["label"])


def assert_sentence(row: dict[str, Any], needle: str) -> None:
    if needle not in row["sentence"]:
        raise AssertionError((needle, row["sentence"]))


def remove_edge(row: dict[str, Any], start: int, end: int, label: str) -> None:
    edges = graph(row)["edges"]
    before = len(edges)
    edges[:] = [edge for edge in edges if edge_key(edge) != (start, end, label)]
    if len(edges) != before - 1:
        raise AssertionError(("remove_edge", start, end, label, before, len(edges)))


def relabel_edge(row: dict[str, Any], start: int, end: int, old_label: str, new_label: str) -> None:
    matches = [edge for edge in graph(row)["edges"] if edge_key(edge) == (start, end, old_label)]
    if len(matches) != 1:
        raise AssertionError(("relabel_edge", start, end, old_label, len(matches)))
    matches[0]["label"] = new_label


def retarget_edge(row: dict[str, Any], start: int, old_end: int, label: str, new_end: int) -> None:
    matches = [edge for edge in graph(row)["edges"] if edge_key(edge) == (start, old_end, label)]
    if len(matches) != 1:
        raise AssertionError(("retarget_edge", start, old_end, label, len(matches)))
    matches[0]["end"] = new_end


def delete_nodes(row: dict[str, Any], deleted: set[int]) -> None:
    nodes = graph(row)["nodes"]
    mapping: dict[int, int] = {}
    new_nodes: list[dict[str, Any]] = []
    for old_index, node in enumerate(nodes):
        if old_index in deleted:
            continue
        mapping[old_index] = len(new_nodes)
        new_nodes.append(node)

    new_edges: list[dict[str, Any]] = []
    for edge in graph(row)["edges"]:
        start = int(edge["start"])
        end = int(edge["end"])
        if start in deleted or end in deleted:
            continue
        new_edge = dict(edge)
        new_edge["start"] = mapping[start]
        new_edge["end"] = mapping[end]
        new_edges.append(new_edge)
    graph(row)["nodes"] = new_nodes
    graph(row)["edges"] = new_edges


def repair_train_130(row: dict[str, Any]) -> str:
    assert_sentence(row, "stating that they were personal attacks")
    # The comma before `stating` was incorrectly used as the body of the final
    # `against her husband`; close the participial clause with that comma.
    retarget_edge(row, 3, 5, "body", 19)
    retarget_edge(row, 17, 20, "body", 5)
    retarget_edge(row, 20, 19, "as:punctuation", 6)
    retarget_edge(row, 24, 6, "body", 20)
    return "train:130 retarget participial comma and final against-body"


def repair_train_1806(row: dict[str, Any]) -> str:
    assert_sentence(row, "Besides eating good foods")
    # `Besides eating good foods,` modifies the main question; `what else` is
    # the object of `do`, not the variable of `Besides`.
    retarget_edge(row, 9, 0, "pred.arg.2", 11)
    retarget_edge(row, 0, 11, "variable", 2)
    remove_edge(row, 1, 3, "pred.arg.1")
    retarget_edge(row, 8, 12, "as:punctuation", 0)
    delete_nodes(row, {12})
    return "train:1806 repair Besides preposed modifier boundary"


def repair_train_1923(row: dict[str, Any]) -> str:
    assert_sentence(row, "five years, four months and five days")
    # The `of his original ... commitment` phrase modifies the whole duration
    # list; the extra Modification around the comma-closed first item is wrong.
    retarget_edge(row, 2, 23, "pred.arg.2", 4)
    retarget_edge(row, 5, 10, "func.arg", 20)
    delete_nodes(row, {23})
    return "train:1923 attach serving object to duration-of relation"


def repair_train_2810(row: dict[str, Any]) -> str:
    assert_sentence(row, "especially judging by the reviews")
    # `judging by ...` is an evidential modifier; it should not share the matrix
    # subject `I`, which polluted the modifier body's token boundary.
    remove_edge(row, 11, 4, "pred.arg.1")
    return "train:2810 remove spurious subject from judging-by modifier"


def repair_train_4393(row: dict[str, Any]) -> str:
    assert_sentence(row, "gamma(c) cytoplasmic domain")
    nodes = graph(row)["nodes"]
    nodes[28]["spans"] = "Expression"
    nodes[28]["type"] = "ExpressionFunctor"
    relabel_edge(row, 28, 6, "variable", "func.arg")
    relabel_edge(row, 28, 23, "body", "func.arg")
    retarget_edge(row, 11, 25, "as:modification", 13)
    delete_nodes(row, {25})
    return "train:4393 convert gamma(c) to Expression and remove redundant receptor Modification"


def main() -> None:
    train = read_jsonl(TRAIN)
    changes = [
        repair_train_130(train[129]),
        repair_train_1806(train[1805]),
        repair_train_1923(train[1922]),
        repair_train_2810(train[2809]),
        repair_train_4393(train[4392]),
    ]
    write_jsonl(TRAIN, train)
    for change in changes:
        print(change)


if __name__ == "__main__":
    main()
