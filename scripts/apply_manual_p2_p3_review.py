#!/usr/bin/env python3
"""Apply conservative sample-specific fixes from the P2/P3 review queue."""

from __future__ import annotations

import csv
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "manual_p2_p3_review"
QUEUE_PATH = (
    ROOT.parent
    / "experiments"
    / "lingua"
    / "samples"
    / "confidence_3fold_v4"
    / "repair_review_queue.tsv"
)
REPAIR_SCRIPT = ROOT / "scripts" / "repair_be_copula_rules.py"

spec = importlib.util.spec_from_file_location("repair_be_copula_rules", REPAIR_SCRIPT)
repair = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(repair)

Fix = Callable[[dict[str, Any]], str]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def set_graph(sample: dict[str, Any], nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    sample["lingua"]["graph"] = {"nodes": nodes, "edges": edges}


def node(spans: Any, node_type: str) -> dict[str, Any]:
    return {"spans": spans, "type": node_type}


def edge(start: int, end: int, label: str) -> dict[str, Any]:
    return {"start": start, "end": end, "label": label}


def compact(sample: dict[str, Any]) -> str:
    return repair.compact_graph(sample)


def fix_train_9802(sample: dict[str, Any]) -> str:
    graph(sample)["nodes"][0]["type"] = "InterjectionConstant"
    return "classify standalone congratulations as an interjection"


def fix_train_1710(sample: dict[str, Any]) -> str:
    set_graph(
        sample,
        [
            node([[0, 0]], "NominalConstant"),
            node([[2, 2]], "NominalConstant"),
            node([[4, 4]], "PunctuationalConstant"),
            node([[1, 1], [3, 3]], "FactualPredicator"),
        ],
        [edge(3, 0, "pred.arg.1"), edge(3, 1, "pred.arg.2"), edge(2, 3, "as:punctuation")],
    )
    return "merge phrasal predicate `picked up`"


def fix_train_1680(sample: dict[str, Any]) -> str:
    set_graph(
        sample,
        [
            node([[1, 1]], "NominalConstant"),
            node([[0, 0]], "ModificationalConstant"),
            node([[3, 3]], "NominalConstant"),
            node([[5, 5]], "PunctuationalConstant"),
            node([[2, 2], [4, 4]], "FactualPredicator"),
        ],
        [
            edge(1, 0, "as:modification"),
            edge(4, 1, "pred.arg.1"),
            edge(4, 2, "pred.arg.2"),
            edge(3, 4, "as:punctuation"),
        ],
    )
    return "merge phrasal predicate `backed up`"


def fix_train_3777(sample: dict[str, Any]) -> str:
    for item in graph(sample)["edges"]:
        if item == {"start": 3, "end": 1, "label": "pred.arg.2"}:
            item["label"] = "pred.arg.4"
            return "treat `old` as result/state complement of `getting`"
    raise ValueError("expected getting -> old pred.arg.2 edge not found")


def fix_train_11127(sample: dict[str, Any]) -> str:
    for item in graph(sample)["edges"]:
        if item == {"start": 7, "end": 4, "label": "pred.arg.2"}:
            item["label"] = "pred.arg.4"
            return "treat `negative` as result/state complement of `came back`"
    raise ValueError("expected came back -> negative pred.arg.2 edge not found")


def fix_train_11051(sample: dict[str, Any]) -> str:
    set_graph(
        sample,
        [
            node([[0, 0]], "NominalConstant"),
            node([[8, 8]], "NominalConstant"),
            node([[6, 6]], "NominalConstant"),
            node([[9, 9]], "ModificationalFunctor"),
            node([[10, 10]], "NominalConstant"),
            node([[11, 11]], "PunctuationalConstant"),
            node([[7, 7]], "PunctuationalConstant"),
            node([[1, 3]], "FactualPredicator"),
            node([[4, 5]], "FactualPredicator"),
        ],
        [
            edge(2, 1, "as:modification"),
            edge(2, 6, "punctuation"),
            edge(3, 4, "body"),
            edge(3, 2, "variable"),
            edge(5, 7, "as:punctuation"),
            edge(7, 0, "pred.arg.1"),
            edge(7, 8, "pred.arg.4"),
            edge(8, 3, "pred.arg.2"),
            edge(8, 0, "pred.arg.1"),
        ],
    )
    return "attach `to Betsy` to the goodbye object and make `to say` the going complement"


def fix_train_9784(sample: dict[str, Any]) -> str:
    graph(sample)["edges"] = [
        edge(2, 1, "as:modification"),
        edge(4, 5, "pred.arg.1"),
        edge(4, 7, "pred.arg.2"),
        edge(7, 8, "as:modification"),
        edge(8, 6, "variable"),
        edge(8, 10, "body"),
        edge(10, 9, "as:modification"),
        edge(17, 11, "punctuation"),
        edge(12, 13, "as:punctuation"),
        edge(13, 17, "pred.arg.1"),
        edge(13, 4, "pred.arg.2"),
        edge(14, 3, "pred.arg.2"),
        edge(14, 16, "pred.arg.1"),
        edge(15, 16, "pred.arg.1"),
        edge(16, 0, "variable"),
        edge(16, 2, "body"),
        edge(17, 15, "variable"),
        edge(17, 14, "body"),
    ]
    return "make `firing off flares` a modifier of the flying event with the helicopter subject"


def fix_train_8996(sample: dict[str, Any]) -> str:
    graph(sample)["nodes"][0]["type"] = "PunctuationalConstant"
    graph(sample)["edges"] = [
        edge(2, 1, "as:modification"),
        edge(4, 3, "as:modification"),
        edge(6, 7, "as:modification"),
        edge(7, 5, "as:modification"),
        edge(9, 14, "variable"),
        edge(9, 11, "body"),
        edge(9, 0, "punctuation"),
        edge(11, 12, "as:modification"),
        edge(12, 10, "as:modification"),
        edge(13, 9, "as:punctuation"),
        edge(14, 8, "pred.arg.2"),
        edge(14, 15, "pred.arg.1"),
        edge(15, 2, "func.arg"),
        edge(15, 4, "func.arg"),
        edge(15, 6, "func.arg"),
    ]
    return "attach the fronted `In the second stage` modifier to the completed-event"


def fix_train_10533(sample: dict[str, Any]) -> str:
    graph(sample)["nodes"].append(node("Missing", "NominalConstant"))
    missing = len(graph(sample)["nodes"]) - 1
    graph(sample)["edges"] = [
        edge(0, 1, "pred.arg.1"),
        edge(0, 5, "pred.arg.2"),
        edge(3, 2, "pred.arg.1"),
        edge(4, 6, "as:modification"),
        edge(5, 4, "as:modification"),
        edge(5, missing, "pred.arg.1"),
        edge(6, 2, "variable"),
        edge(6, 3, "body"),
    ]
    return "make the starting-place phrase the copular complement and mark `to look` subject implicit"


def fix_train_4499(sample: dict[str, Any]) -> str:
    graph(sample)["nodes"][5]["spans"] = [[1, 1]]
    graph(sample)["nodes"].append(node([[2, 2]], "ModificationalConstant"))
    truely = len(graph(sample)["nodes"]) - 1
    graph(sample)["edges"] = [
        edge(2, 1, "as:modification"),
        edge(3, 2, "as:modification"),
        edge(4, truely, "as:punctuation"),
        edge(5, 0, "pred.arg.1"),
        edge(5, 3, "pred.arg.2"),
        edge(truely, 5, "as:modification"),
    ]
    return "split adverb `truely` out of the copular predicate"


def fix_train_3726(sample: dict[str, Any]) -> str:
    graph(sample)["edges"] = [
        edge(0, 2, "pred.arg.2"),
        edge(1, 0, "variable"),
        edge(1, 4, "body"),
        edge(4, 5, "as:modification"),
        edge(5, 3, "as:modification"),
        edge(5, 8, "punctuation"),
        edge(6, 1, "as:punctuation"),
        edge(2, 7, "punctuation"),
        edge(0, 9, "pred.arg.1"),
    ]
    return "make quoted `paste` the object of `select` rather than the quote marks"


def fix_train_5271(sample: dict[str, Any]) -> str:
    set_graph(
        sample,
        [
            node([[0, 0]], "NominalConstant"),
            node([[1, 1]], "NominalConstant"),
            node([[2, 2]], "PunctuationalConstant"),
            node([[3, 3]], "NominalConstant"),
            node("Expression", "ExpressionFunctor"),
            node("Attribute", "AttributePredicator"),
        ],
        [
            edge(4, 0, "func.arg"),
            edge(4, 1, "func.arg"),
            edge(5, 4, "pred.arg.1"),
            edge(5, 3, "pred.arg.2"),
            edge(5, 2, "punctuation"),
        ],
    )
    return "convert question-id key-value record to Attribute"


def fix_dev_381(sample: dict[str, Any]) -> str:
    set_graph(
        sample,
        [
            node([[0, 0]], "NominalConstant"),
            node([[1, 1]], "PunctuationalConstant"),
            node([[2, 2]], "NominalConstant"),
            node([[3, 3]], "NominalConstant"),
            node("Expression", "ExpressionFunctor"),
            node("Attribute", "AttributePredicator"),
        ],
        [
            edge(4, 2, "func.arg"),
            edge(4, 3, "func.arg"),
            edge(5, 0, "pred.arg.1"),
            edge(5, 4, "pred.arg.2"),
            edge(5, 1, "punctuation"),
        ],
    )
    return "convert attendees key-value record to Attribute"


def fix_test_100(sample: dict[str, Any]) -> str:
    for item in graph(sample)["edges"]:
        if item == {"start": 1, "end": 4, "label": "pred.arg.2"}:
            item["label"] = "pred.arg.4"
            return "treat `better` as result/state complement of `got`"
    raise ValueError("expected got -> better pred.arg.2 edge not found")


def fix_train_10551(sample: dict[str, Any]) -> str:
    graph(sample)["nodes"][0]["type"] = "InterjectionConstant"
    graph(sample)["edges"] = [
        edge(1, 0, "as:vocative"),
        edge(0, 2, "punctuation"),
        edge(3, 1, "as:punctuation"),
    ]
    return "represent `Hello, Bill!` as interjection plus vocative"


def fix_train_512(sample: dict[str, Any]) -> str:
    for item in graph(sample)["edges"]:
        if item == {"start": 0, "end": 15, "label": "pred.arg.1"}:
            item["end"] = 17
            for punct in graph(sample)["edges"]:
                if punct == {"start": 15, "end": 17, "label": "as:punctuation"}:
                    punct["start"] = 17
                    punct["end"] = 15
                    punct["label"] = "punctuation"
                    break
            return "make the coordinated name list the subject of `appeared`, not the comma"
    raise ValueError("expected appeared -> comma pred.arg.1 edge not found")


FIXES: dict[str, Fix] = {
    "train:9802": fix_train_9802,
    "train:1710": fix_train_1710,
    "train:1680": fix_train_1680,
    "train:3777": fix_train_3777,
    "train:11127": fix_train_11127,
    "train:11051": fix_train_11051,
    "train:9784": fix_train_9784,
    "train:8996": fix_train_8996,
    "train:10533": fix_train_10533,
    "train:4499": fix_train_4499,
    "train:3726": fix_train_3726,
    "train:5271": fix_train_5271,
    "dev:381": fix_dev_381,
    "test:100": fix_test_100,
    "train:10551": fix_train_10551,
    "train:512": fix_train_512,
}


def load_queue() -> list[dict[str, str]]:
    with QUEUE_PATH.open(encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle, delimiter="\t") if row["priority"] in {"P2", "P3"}]


def run() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    queue_rows = load_queue()
    queue_by_id = {row["sample_id"]: row for row in queue_rows}
    changes: list[dict[str, Any]] = []

    for split in ("train", "dev", "test"):
        path = DATA / f"{split}.jsonl"
        rows = read_jsonl(path)
        for line_number, sample in enumerate(rows, 1):
            sample_id = f"{split}:{line_number}"
            fix = FIXES.get(sample_id)
            if not fix:
                continue
            before = deepcopy(sample)
            note = fix(sample)
            changes.append(
                {
                    "sample_id": sample_id,
                    "priority": queue_by_id.get(sample_id, {}).get("priority", ""),
                    "category": queue_by_id.get(sample_id, {}).get("category", ""),
                    "sentence": sample.get("sentence", ""),
                    "note": note,
                    "before": compact(before),
                    "after": compact(sample),
                    "sample": sample,
                }
            )
        write_jsonl(path, rows)

    decisions_path = REPORT_DIR / "decisions.tsv"
    with decisions_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["priority", "sample_id", "category", "decision", "note", "sentence"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in queue_rows:
            sample_id = row["sample_id"]
            change = next((item for item in changes if item["sample_id"] == sample_id), None)
            if change:
                decision = "fixed"
                note = change["note"]
            else:
                decision = "leave"
                note = "reviewed; no safe data correction beyond model-coverage or acceptable annotation variation"
            writer.writerow(
                {
                    "priority": row["priority"],
                    "sample_id": sample_id,
                    "category": row["category"],
                    "decision": decision,
                    "note": note,
                    "sentence": row["sentence"],
                }
            )

    queue_path = REPORT_DIR / "queue.tsv"
    with queue_path.open("w", encoding="utf-8", newline="") as handle:
        if queue_rows:
            writer = csv.DictWriter(handle, fieldnames=list(queue_rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(queue_rows)

    changes_path = REPORT_DIR / "changes.jsonl"
    changes_path.write_text(
        "\n".join(json.dumps(change, ensure_ascii=False) for change in changes) + ("\n" if changes else ""),
        encoding="utf-8",
    )

    fixed_by_priority: dict[str, int] = {"P2": 0, "P3": 0}
    total_by_priority: dict[str, int] = {"P2": 0, "P3": 0}
    for row in queue_rows:
        total_by_priority[row["priority"]] += 1
    for change in changes:
        if change["priority"] in fixed_by_priority:
            fixed_by_priority[change["priority"]] += 1

    summary_lines = [
        "# P2/P3 Manual Review",
        "",
        f"- Reviewed queue: {len(queue_rows)}",
        f"- Fixed: {len(changes)}",
        f"- Left unchanged: {len(queue_rows) - len(changes)}",
        f"- P2 fixed/total: {fixed_by_priority['P2']}/{total_by_priority['P2']}",
        f"- P3 fixed/total: {fixed_by_priority['P3']}/{total_by_priority['P3']}",
        "",
        "## Fixed",
        "",
    ]
    for change in changes:
        summary_lines.append(f"- `{change['sample_id']}`: {change['note']} — {change['sentence']}")
    (REPORT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"reviewed={len(queue_rows)} fixed={len(changes)} report={REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
