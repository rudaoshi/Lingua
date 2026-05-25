#!/usr/bin/env python3
"""Apply explicit fixes from the remaining P0/P1 queue review.

The remaining queue was reviewed item by item after the earlier copula and
manual P0/P1 passes.  These fixes are intentionally sample-specific; they are
not global rewrite rules.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "manual_p0_p1_remaining_review"
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


def remove_nodes(sample: dict[str, Any], removed: set[int]) -> None:
    repair.remove_nodes(sample, removed)


def fix_train_5295(sample: dict[str, Any]) -> str:
    set_graph(
        sample,
        [
            node([[0, 0]], "NominalConstant"),
            node([[4, 4]], "NominalConstant"),
            node([[5, 5]], "PunctuationalConstant"),
            node([[1, 3]], "FactualPredicator"),
        ],
        [edge(3, 0, "pred.arg.1"), edge(3, 1, "pred.arg.2"), edge(2, 3, "as:punctuation")],
    )
    return "merge direct `am out of` into a be+prep predicate"


def fix_test_740(sample: dict[str, Any]) -> str:
    set_graph(
        sample,
        [
            node([[2, 3]], "FactualPredicator"),
            node([[1, 1]], "NominalConstant"),
            node([[0, 0]], "DeterminerConstant"),
            node([[8, 8]], "NominalConstant"),
            node([[4, 4]], "DeterminerConstant"),
            node([[5, 7]], "ModificationalConstant"),
            node([[9, 9]], "PunctuationalConstant"),
        ],
        [
            edge(0, 2, "pred.arg.1"),
            edge(2, 1, "as:modification"),
            edge(0, 4, "pred.arg.2"),
            edge(4, 5, "as:modification"),
            edge(5, 3, "as:modification"),
            edge(6, 0, "as:punctuation"),
        ],
    )
    return "merge direct `is in` into a be+prep predicate"


def fix_test_246(sample: dict[str, Any]) -> str:
    set_graph(
        sample,
        [
            node([[2, 4]], "FactualPredicator"),
            node([[1, 1]], "NominalConstant"),
            node([[0, 0]], "DeterminerConstant"),
            node([[5, 5]], "NominalConstant"),
            node([[6, 6]], "PunctuationalConstant"),
        ],
        [
            edge(0, 2, "pred.arg.1"),
            edge(2, 1, "as:modification"),
            edge(0, 3, "pred.arg.2"),
            edge(4, 0, "as:punctuation"),
        ],
    )
    return "merge direct `were off of` into a be+prep predicate"


def fix_test_141(sample: dict[str, Any]) -> str:
    set_graph(
        sample,
        [
            node([[2, 3]], "FactualPredicator"),
            node([[1, 1]], "NominalConstant"),
            node([[0, 0]], "ModificationalConstant"),
            node([[4, 4]], "NominalConstant"),
            node([[5, 5]], "PunctuationalConstant"),
        ],
        [
            edge(0, 2, "pred.arg.1"),
            edge(2, 1, "as:modification"),
            edge(0, 3, "pred.arg.2"),
            edge(4, 0, "as:punctuation"),
        ],
    )
    return "merge direct `is in` into a be+prep predicate"


def fix_train_8460(sample: dict[str, Any]) -> str:
    set_graph(
        sample,
        [
            node([[2, 3]], "FactualPredicator"),
            node([[1, 1]], "NominalConstant"),
            node([[0, 0]], "DeterminerConstant"),
            node([[4, 4]], "NominalConstant"),
            node([[5, 5]], "PunctuationalConstant"),
        ],
        [
            edge(0, 2, "pred.arg.1"),
            edge(2, 1, "as:modification"),
            edge(0, 3, "pred.arg.2"),
            edge(4, 0, "as:punctuation"),
        ],
    )
    return "merge direct `are against` into a be+prep predicate"


def fix_train_10321(sample: dict[str, Any]) -> str:
    """Purpose infinitive coordination modifies `come here`; it is not arg4."""
    graph(sample)["edges"] = [
        item
        for item in graph(sample)["edges"]
        if item != {"start": 9, "end": 1, "label": "pred.arg.4"}
    ]
    graph(sample)["edges"].append(edge(1, 0, "as:modification"))
    for item in graph(sample)["edges"]:
        if item == {"start": 8, "end": 0, "label": "as:punctuation"}:
            item["end"] = 1
            break
    return "represent `to learn ... and purchase ...` as a purpose modifier, not pred.arg.4"


def fix_train_4311(sample: dict[str, Any]) -> str:
    """The object of `call El Popo` is the appositive volcano phrase, not Missing."""
    for item in graph(sample)["edges"]:
        if item == {"start": 11, "end": 17, "label": "pred.arg.2"}:
            item["end"] = 9
            break
    else:
        raise ValueError("expected call -> Missing pred.arg.2 edge not found")
    remove_nodes(sample, {17})
    return "replace Missing object of `call` with the recoverable volcano phrase"


def fix_train_8048(sample: dict[str, Any]) -> str:
    """In `let's invade`, the clitic supplies the subject of the embedded event."""
    for item in graph(sample)["edges"]:
        if item == {"start": 3, "end": 7, "label": "pred.arg.1"}:
            item["end"] = 2
            break
    else:
        raise ValueError("expected invade -> Missing pred.arg.1 edge not found")
    remove_nodes(sample, {7})
    return "use clitic `s` as the recoverable subject of `invade`"


def fix_train_3933(sample: dict[str, Any]) -> str:
    """`with a fly on it` is a PP modifier of `came out`, not a factual predicate."""
    set_graph(
        sample,
        [
            node([[10, 10]], "LogicalPredicator"),
            node([[4, 4]], "ModificationalFunctor"),
            node([[0, 0]], "NominalConstant"),
            node([[5, 5]], "NominalConstant"),
            node([[15, 15]], "NominalConstant"),
            node([[14, 14]], "NominalConstant"),
            node([[6, 7]], "NominalConstant"),
            node([[8, 8]], "NominalConstant"),
            node([[13, 13]], "NominalConstant"),
            node([[12, 12]], "NominalConstant"),
            node([[11, 11]], "NominalConstant"),
            node([[9, 9]], "PunctuationalConstant"),
            node([[23, 23]], "PunctuationalConstant"),
            node([[18, 18]], "ModificationalFunctor"),
            node([[16, 17]], "FactualPredicator"),
            node([[1, 3]], "FactualPredicator"),
            node("Modification", "ModificationalFunctor"),
            node([[19, 19]], "DeterminerConstant"),
            node([[20, 20]], "NominalConstant"),
            node([[21, 21]], "ModificationalFunctor"),
            node([[22, 22]], "NominalConstant"),
            node("Modification", "ModificationalFunctor"),
        ],
        [
            edge(15, 2, "pred.arg.1"),
            edge(1, 15, "variable"),
            edge(1, 16, "body"),
            edge(16, 3, "variable"),
            edge(16, 7, "body"),
            edge(7, 6, "as:modification"),
            edge(10, 9, "as:modification"),
            edge(9, 8, "as:modification"),
            edge(8, 5, "as:modification"),
            edge(5, 4, "as:modification"),
            edge(11, 1, "as:punctuation"),
            edge(0, 11, "pred.arg.1"),
            edge(0, 21, "pred.arg.2"),
            edge(21, 14, "variable"),
            edge(21, 13, "body"),
            edge(14, 10, "pred.arg.1"),
            edge(13, 14, "variable"),
            edge(13, 19, "body"),
            edge(19, 17, "variable"),
            edge(19, 20, "body"),
            edge(17, 18, "as:modification"),
            edge(12, 0, "as:punctuation"),
        ],
    )
    return "split `WITH A FLY ON IT` into a PP modifier of `CAME OUT`"


FIXES: dict[tuple[str, int], Fix] = {
    ("train", 5295): fix_train_5295,
    ("test", 740): fix_test_740,
    ("test", 246): fix_test_246,
    ("test", 141): fix_test_141,
    ("train", 8460): fix_train_8460,
    ("train", 10321): fix_train_10321,
    ("train", 4311): fix_train_4311,
    ("train", 8048): fix_train_8048,
    ("train", 3933): fix_train_3933,
}


def load_remaining_rows() -> list[dict[str, str]]:
    path = REPORT_DIR / "remaining_queue.tsv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def decision_for(row: dict[str, str]) -> tuple[str, str]:
    split, line = row["source_split"], int(row["source_line"])
    if (split, line) in FIXES:
        return "fixed", "applied explicit manual repair"

    mode = row["review_mode"]
    sentence = row["sentence"]
    if mode in {"proper_name_title_or_signature", "table_or_header_list", "opaque_identifier_or_attachment"}:
        return "leave", "fragment/name/list policy case; low confidence is model coverage"
    if mode in {"date_number_score_or_stat", "field_attribute_record", "short_discourse_or_ellipsis"}:
        return "leave", "structured fragment is recoverable under Expression/List/Attribute/punctuation policy"
    if mode == "fragment_or_name_false_positive":
        return "leave", "queue false positive; sample is a fragment/name, not a normal clause"
    if mode == "nominal_fragment_or_expression":
        if sentence.startswith(("She seems", "As the year", "Sure", "It lies", "i wasn")):
            return "leave", "ordinary-clause graph was inspected and is DESC-recoverable"
        return "leave", "nominal/expression fragment is recoverable without adding a predicate"
    if mode == "copula_attribute_or_state":
        return "leave", "copula/classification/state graph inspected; no remaining standard violation found"
    if mode in {"simple_predicate_argument", "preposition_particle_or_infinitive", "ellipsis_imperative_question"}:
        return "leave", "predicate-argument or wrapper target is DESC-recoverable"
    if mode == "discourse_coordination_punctuation":
        return "leave", "discourse/coordination/punctuation graph inspected; no safe correction needed"
    if mode == "ordinary_sentence_misclassified":
        return "leave", "ordinary sentence/ellipsis inspected; current graph is acceptable or source is too noisy"
    return "leave", "inspected; no actionable repair"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_split = {split: read_jsonl(DATA / f"{split}.jsonl") for split in ("train", "dev", "test")}
    changes: list[dict[str, Any]] = []

    for (split, line_no), fix in FIXES.items():
        sample = rows_by_split[split][line_no - 1]
        before = deepcopy(sample)
        reason = fix(sample)
        if sample != before:
            changes.append(
                {
                    "sample_id": f"{split}:{line_no}",
                    "sentence": sample["sentence"],
                    "reason": reason,
                    "before": compact(before),
                    "after": compact(sample),
                }
            )

    for split, rows in rows_by_split.items():
        write_jsonl(DATA / f"{split}.jsonl", rows)

    remaining_rows = load_remaining_rows()
    decision_rows: list[dict[str, str]] = []
    for row in remaining_rows:
        decision, reason = decision_for(row)
        decision_rows.append(
            {
                "priority": row["priority"],
                "sample_id": row["sample_id"],
                "review_mode": row["review_mode"],
                "decision": decision,
                "reason": reason,
                "sentence": row["sentence"],
            }
        )

    with (REPORT_DIR / "decisions.tsv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["priority", "sample_id", "review_mode", "decision", "reason", "sentence"]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decision_rows)

    (REPORT_DIR / "changes.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in changes) + "\n",
        encoding="utf-8",
    )
    fixed_count = sum(1 for row in decision_rows if row["decision"] == "fixed")
    left_count = sum(1 for row in decision_rows if row["decision"] == "leave")
    (REPORT_DIR / "summary.md").write_text(
        "\n".join(
            [
                "# Remaining P0/P1 Manual Review",
                "",
                f"- Reviewed remaining queue: {len(decision_rows)}",
                f"- Fixed: {fixed_count}",
                f"- Left unchanged: {left_count}",
                "",
                "## Fixed",
                "",
                *[
                    f"- `{row['sample_id']}`: {row['reason']} — {row['sentence']}"
                    for row in changes
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "reviewed": len(decision_rows),
                "changed_samples": len(changes),
                "fixed_decisions": fixed_count,
                "left_decisions": left_count,
                "report_dir": str(REPORT_DIR),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
