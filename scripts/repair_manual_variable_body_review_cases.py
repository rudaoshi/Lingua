#!/usr/bin/env python3
"""Apply hand-reviewed fixes from the variable/body manual-review queue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def graph(row: dict[str, Any]) -> dict[str, Any]:
    return row["lingua"]["graph"]


def add_node(row: dict[str, Any], spans: str, node_type: str) -> int:
    nodes = graph(row)["nodes"]
    nodes.append({"spans": spans, "type": node_type})
    return len(nodes) - 1


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return int(edge["start"]), int(edge["end"]), str(edge["label"])


def add_edge(row: dict[str, Any], start: int, end: int, label: str) -> None:
    edges = graph(row)["edges"]
    key = (start, end, label)
    if key not in {edge_key(edge) for edge in edges}:
        edges.append({"start": start, "end": end, "label": label})


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


def assert_sentence(row: dict[str, Any], needle: str) -> None:
    if needle not in row["sentence"]:
        raise AssertionError((needle, row["sentence"]))


def repair_train_1747(row: dict[str, Any]) -> str:
    """Use Attribute for the parenthetical GMT conversion."""
    assert_sentence(row, "2.00 p.m. ( 0400 GMT )")
    nodes = graph(row)["nodes"]
    if nodes[20] != {"spans": "Expression", "type": "ExpressionFunctor"}:
        raise AssertionError(nodes[20])
    nodes[20] = {"spans": "Attribute", "type": "AttributePredicator"}
    relabel_edge(row, 20, 18, "func.arg", "pred.arg.1")
    relabel_edge(row, 20, 11, "func.arg", "pred.arg.2")
    return "train:1747 convert time-zone parenthetical C from Expression to Attribute"


def repair_train_3053(row: dict[str, Any]) -> str:
    """Make the original square-mile expression the shared A in A(B)."""
    assert_sentence(row, '9.0 " sq " mi ( 23.3 " km ² )')
    nodes = graph(row)["nodes"]
    if nodes[27] != {"spans": "Expression", "type": "ExpressionFunctor"}:
        raise AssertionError(nodes[27])
    original = add_node(row, "Expression", "ExpressionFunctor")
    add_edge(row, original, 10, "func.arg")
    add_edge(row, original, 18, "func.arg")

    nodes[27] = {"spans": "Attribute", "type": "AttributePredicator"}
    remove_edge(row, 27, 10, "func.arg")
    remove_edge(row, 27, 18, "func.arg")
    remove_edge(row, 27, 28, "func.arg")
    add_edge(row, 27, original, "pred.arg.1")
    add_edge(row, 27, 28, "pred.arg.2")
    retarget_edge(row, 35, 18, "variable", original)
    return "train:3053 repair measurement parenthetical A/B boundary"


def repair_train_6976(row: dict[str, Any]) -> str:
    """Represent `11 September` as a date Expression before it modifies `hijackers`."""
    assert_sentence(row, "the 11 September hijackers")
    date_expr = add_node(row, "Expression", "ExpressionFunctor")
    date_mod = add_node(row, "Modification", "ModificationalFunctor")
    remove_edge(row, 9, 10, "as:modification")
    remove_edge(row, 10, 8, "as:modification")
    add_edge(row, date_expr, 9, "func.arg")
    add_edge(row, date_expr, 10, "func.arg")
    add_edge(row, date_mod, 8, "variable")
    add_edge(row, date_mod, date_expr, "body")
    retarget_edge(row, 11, 9, "as:modification", date_mod)
    return "train:6976 split date expression from NP modifier"


def repair_train_7299(row: dict[str, Any]) -> str:
    """Make `130 cm` the shared A for the `(52 inches)` conversion."""
    assert_sentence(row, '130 " cm ( 52 inches )')
    retarget_edge(row, 23, 19, "body", 22)
    retarget_edge(row, 24, 22, "body", 19)
    retarget_edge(row, 25, 8, "pred.arg.1", 24)
    return "train:7299 repair centimetre/inches parenthetical A/B boundary"


def repair_test_130(row: dict[str, Any]) -> str:
    """Represent `6 August` as a date Expression before it modifies the briefing title."""
    assert_sentence(row, "a 6 August Presidential Daily Briefing")
    date_expr = add_node(row, "Expression", "ExpressionFunctor")
    date_mod = add_node(row, "Modification", "ModificationalFunctor")
    remove_edge(row, 18, 19, "as:modification")
    remove_edge(row, 19, 20, "as:modification")
    add_edge(row, date_expr, 18, "func.arg")
    add_edge(row, date_expr, 19, "func.arg")
    add_edge(row, date_mod, 20, "variable")
    add_edge(row, date_mod, date_expr, "body")
    retarget_edge(row, 22, 18, "as:modification", date_mod)
    return "test:130 split date expression from briefing-title modifier"


def main() -> None:
    train = read_jsonl(DATA / "train.jsonl")
    dev = read_jsonl(DATA / "dev.jsonl")
    test = read_jsonl(DATA / "test.jsonl")

    changes = [
        repair_train_1747(train[1746]),
        repair_train_3053(train[3052]),
        repair_train_6976(train[6975]),
        repair_train_7299(train[7298]),
        repair_test_130(test[129]),
    ]

    write_jsonl(DATA / "train.jsonl", train)
    write_jsonl(DATA / "dev.jsonl", dev)
    write_jsonl(DATA / "test.jsonl", test)
    for change in changes:
        print(change)


if __name__ == "__main__":
    main()
