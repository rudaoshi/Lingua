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


def replace_word(row: dict[str, Any], index: int, old: str, new: str) -> None:
    words = row["lingua"]["words"]
    if words[index] == old:
        words[index] = new
        return
    if words[index] != new:
        raise AssertionError((index, words[index], old, new))


def replace_sentence(row: dict[str, Any], old: str, new: str) -> None:
    sentence = row["sentence"]
    if old in sentence:
        row["sentence"] = sentence.replace(old, new)
    elif new not in sentence:
        raise AssertionError((old, new, sentence))


def edge(row: dict[str, Any], start: int, label: str) -> list[dict[str, Any]]:
    return [
        item
        for item in row["lingua"]["graph"]["edges"]
        if int(item["start"]) == start and item["label"] == label
    ]


def repair_dev_326(row: dict[str, Any]) -> None:
    replace_sentence(row, "must by comforting", "must be comforting")
    replace_word(row, 13, " by", " be")

    graph = row["lingua"]["graph"]
    nodes = graph["nodes"]
    edges = graph["edges"]

    # The typo introduced a bogus ModificationalConstant `by` node. Fold `be`
    # back into the factual predicator "must be comforting".
    nodes[23]["spans"] = [[12, 14]]

    for item in edges:
        if item["start"] == 22 and item["end"] == 31 and item["label"] == "as:punctuation":
            item["end"] = 23

    graph["edges"] = [
        item
        for item in edges
        if not (item["start"] == 31 and item["end"] == 23 and item["label"] == "as:modification")
    ]
    if len(nodes) > 31 and nodes[31].get("spans") == [[13, 13]]:
        del nodes[31]


def repair_test_2(row: dict[str, Any]) -> None:
    graph = row["lingua"]["graph"]
    discourse = next(
        index
        for index, node in enumerate(graph["nodes"])
        if node["spans"] == "Discourse" and node["type"] == "ModificationalFunctor"
    )
    variable = edge(row, discourse, "variable")
    body = edge(row, discourse, "body")
    if len(variable) != 1 or len(body) != 1:
        raise AssertionError((variable, body))
    variable[0]["end"] = 7
    body[0]["end"] = 3


def repair_test_755(row: dict[str, Any]) -> None:
    changed = False
    for item in row["lingua"]["graph"]["edges"]:
        if item["start"] == 1 and item["end"] == 0 and item["label"] == "as:discourse":
            item["label"] = "as:punctuation"
            changed = True
    if not changed and not any(
        item["start"] == 1 and item["end"] == 0 and item["label"] == "as:punctuation"
        for item in row["lingua"]["graph"]["edges"]
    ):
        raise AssertionError("test:755 comma edge not found")


def main() -> None:
    train = read_jsonl(DATA / "train.jsonl")
    dev = read_jsonl(DATA / "dev.jsonl")
    test = read_jsonl(DATA / "test.jsonl")

    replace_sentence(train[4498], "truely", "truly")
    replace_word(train[4498], 2, " truely", " truly")

    replace_sentence(dev[257], "truely", "truly")
    replace_word(dev[257], 10, " truely", " truly")

    repair_dev_326(dev[325])

    replace_sentence(dev[875], "you steep into", "you step into")
    replace_word(dev[875], 13, " steep", " step")

    repair_test_2(test[1])

    replace_sentence(test[166], "mircles", "miracles")
    replace_word(test[166], 19, " mircles", " miracles")

    repair_test_755(test[754])

    write_jsonl(DATA / "train.jsonl", train)
    write_jsonl(DATA / "dev.jsonl", dev)
    write_jsonl(DATA / "test.jsonl", test)


if __name__ == "__main__":
    main()
