#!/usr/bin/env python3
"""Repair high-confidence rare shared-role annotation cases.

The target cases come from the hierarchical oracle composite-role audit for:

- ARG2+ARG3
- ARG3+ARG4
- ARG2+FUNC_ARG
- ARG1+ARG3

This script only applies sample-specific repairs where the underlying Lingua
edge labels are clearly inconsistent with the current standard.  Legitimate
control/shared-subject cases are intentionally left unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repair_be_copula_rules import compact_graph, read_jsonl, write_jsonl  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
TARGET_COMPOSITES = {"ARG2+ARG3", "ARG3+ARG4", "ARG2+FUNC_ARG", "ARG1+ARG3"}


@dataclass(frozen=True)
class Op:
    kind: str
    start: int
    end: int
    label: str
    new_label: str | None = None
    note: str = ""


REPAIRS: dict[tuple[str, int], tuple[Op, ...]] = {
    # object-control / result-complement: complement is ARG4, object remains ARG2
    ("train", 229): (Op("relabel", 9, 2, "pred.arg.3", "pred.arg.4", "make it settle: complement is ARG4"),),
    ("train", 4957): (Op("relabel", 0, 1, "pred.arg.3", "pred.arg.4", "let me know: selected complement is ARG4"),),
    ("train", 5289): (Op("relabel", 7, 25, "pred.arg.3", "pred.arg.4", "find themselves spending: object complement is ARG4"),),
    ("train", 5381): (Op("relabel", 10, 1, "pred.arg.3", "pred.arg.4", "get motor replaced: result complement is ARG4"),),
    ("train", 5579): (Op("relabel", 10, 0, "pred.arg.3", "pred.arg.4", "help them solve: complement is ARG4"),),
    ("train", 5967): (Op("relabel", 17, 4, "pred.arg.3", "pred.arg.4", "make it feel: complement is ARG4"),),
    ("train", 6001): (Op("relabel", 4, 1, "pred.arg.3", "pred.arg.4", "make you Lose Control: complement is ARG4"),),
    ("train", 6249): (Op("relabel", 8, 1, "pred.arg.3", "pred.arg.4", "help her win: complement is ARG4"),),
    ("train", 7301): (Op("relabel", 0, 12, "pred.arg.3", "pred.arg.4", "found hotel to be clean: complement is ARG4"),),
    ("train", 7318): (Op("relabel", 0, 1, "pred.arg.3", "pred.arg.4", "let me say: selected complement is ARG4"),),
    ("train", 7394): (Op("relabel", 1, 2, "pred.arg.3", "pred.arg.4", "helped Mr. Bush understand: complement is ARG4"),),
    ("train", 7489): (Op("relabel", 23, 3, "pred.arg.3", "pred.arg.4", "have Alito on the Court: object complement is ARG4"),),
    ("train", 7645): (Op("relabel", 17, 3, "pred.arg.3", "pred.arg.4", "see you visiting: complement is ARG4"),),
    ("train", 8845): (Op("relabel", 0, 1, "pred.arg.3", "pred.arg.4", "Let s be clear: selected complement is ARG4"),),
    # direct object in object-control verbs was mislabeled as ARG3
    ("train", 7311): (Op("relabel", 8, 10, "pred.arg.3", "pred.arg.2", "cause him to walk: him is direct object ARG2"),),
    ("train", 7654): (Op("relabel", 0, 3, "pred.arg.3", "pred.arg.2", "ordered him to pay: him is direct object ARG2"),),
    ("train", 10971): (Op("relabel", 0, 18, "pred.arg.3", "pred.arg.2", "allows us to characterize: us is direct object ARG2"),),
    # recipient/content or recipient/theme roles were swapped.
    ("train", 1563): (
        Op("relabel", 0, 4, "pred.arg.2", "pred.arg.3", "give them the chance: recipient is ARG3"),
        Op("relabel", 0, 18, "pred.arg.3", "pred.arg.2", "give them the chance: theme is ARG2"),
    ),
    ("train", 3080): (
        Op("relabel", 0, 7, "pred.arg.2", "pred.arg.3", "told the bus driver: addressee is ARG3"),
        Op("relabel", 0, 1, "pred.arg.3", "pred.arg.2", "told where to drop: content is ARG2"),
    ),
    ("train", 3312): (
        Op("relabel", 0, 3, "pred.arg.2", "pred.arg.3", "told Clinton: addressee is ARG3"),
        Op("relabel", 0, 18, "pred.arg.3", "pred.arg.2", "told clause: content is ARG2"),
    ),
    ("train", 4117): (
        Op("relabel", 2, 6, "pred.arg.2", "pred.arg.3", "denying individuals: recipient/affected party is ARG3"),
        Op("relabel", 2, 26, "pred.arg.3", "pred.arg.2", "denying right: withheld theme is ARG2"),
    ),
    ("train", 4375): (
        Op("relabel", 0, 7, "pred.arg.2", "pred.arg.3", "give them time: recipient is ARG3"),
        Op("relabel", 0, 24, "pred.arg.3", "pred.arg.2", "give them time: theme is ARG2"),
    ),
    ("train", 5239): (
        Op("relabel", 0, 6, "pred.arg.2", "pred.arg.3", "tell them: addressee punctuation wrapper is ARG3"),
        Op("relabel", 0, 8, "pred.arg.3", "pred.arg.2", "tell quote/content: content is ARG2"),
    ),
    ("train", 5869): (
        Op("relabel", 0, 4, "pred.arg.2", "pred.arg.3", "told this poor guy: addressee is ARG3"),
        Op("relabel", 0, 18, "pred.arg.3", "pred.arg.4", "told to give: selected command complement is ARG4"),
    ),
    ("train", 5983): (
        Op("relabel", 0, 4, "pred.arg.2", "pred.arg.3", "told me: addressee is ARG3"),
        Op("relabel", 0, 1, "pred.arg.3", "pred.arg.4", "told to bring: selected command complement is ARG4"),
    ),
    ("train", 5998): (
        Op("relabel", 0, 4, "pred.arg.2", "pred.arg.3", "teach them: learner/addressee is ARG3"),
        Op("relabel", 0, 14, "pred.arg.3", "pred.arg.4", "teach not to use: selected complement is ARG4"),
    ),
    # fix a let-parse where embedded subject was incorrectly promoted to parent ARG1.
    ("train", 8020): (
        Op("remove", 0, 6, "pred.arg.1", note="remove wrong parent ARG1 from embedded if-clause subject"),
        Op("add", 0, 5, "pred.arg.2", note="let me: me is direct object"),
        Op("relabel", 0, 2, "pred.arg.3", "pred.arg.4", "let me know: selected complement is ARG4"),
    ),
    # room to progress: attach the infinitival modifier to room instead of as matrix ARG4.
    ("train", 8386): (
        Op("remove", 10, 3, "pred.arg.2", note="replace bare room ARG2 with full room-to-progress argument"),
        Op("relabel", 10, 0, "pred.arg.4", "pred.arg.2", "full room-to-progress argument is ARG2"),
        Op("add", 0, 3, "as:modification", note="to progress in group modifies room"),
    ),
    # participial clause should modify the preceding coordination, not be a third logical child.
    ("test", 967): (
        Op("remove", 17, 10, "func.arg", note="suggesting-clause is not a parallel logical child"),
        Op("remove", 16, 17, "as:punctuation", note="outer punctuation now closes suggesting modifier"),
        Op("add", 10, 17, "as:modification", note="suggesting modifies the preceding but-clause"),
        Op("add", 16, 10, "as:punctuation", note="period closes suggesting modifier"),
    ),
}

INTENTIONAL_SKIPS: dict[tuple[str, int], str] = {
    ("train", 509): "tell me not to write: addressee ARG3 plus selected complement ARG4 is acceptable",
    ("train", 2445): "the Russians is both recipient of gave and subject of having two riders",
    ("train", 4136): "guaranteed refugees the right to return: recipient ARG3 controls return",
    ("train", 4719): "give me places to look: recipient controls infinitival modifier",
    ("train", 10615): "gives you a chance to get: recipient controls infinitival modifier",
}


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def edge_key(edge: dict[str, Any]) -> tuple[int, int, str]:
    return int(edge["start"]), int(edge["end"]), str(edge["label"])


def add_edge_once(edges: list[dict[str, Any]], start: int, end: int, label: str) -> bool:
    key = (int(start), int(end), str(label))
    if any(edge_key(edge) == key for edge in edges):
        return False
    edges.append({"start": int(start), "end": int(end), "label": str(label)})
    return True


def apply_ops(sample: dict[str, Any], ops: tuple[Op, ...]) -> tuple[dict[str, Any], list[str]]:
    repaired = deepcopy(sample)
    edges = graph(repaired)["edges"]
    notes: list[str] = []
    for op in ops:
        if op.kind == "relabel":
            matches = [
                edge
                for edge in edges
                if edge_key(edge) == (op.start, op.end, op.label)
            ]
            if len(matches) == 0 and op.new_label is not None:
                already = any(edge_key(edge) == (op.start, op.end, op.new_label) for edge in edges)
                if already:
                    notes.append(f"already relabeled {op.start}->{op.end} {op.new_label}: {op.note}")
                    continue
            if len(matches) != 1:
                raise ValueError(f"expected one edge to relabel, found {len(matches)}: {op}")
            matches[0]["label"] = str(op.new_label)
            notes.append(f"relabel {op.start}->{op.end} {op.label} to {op.new_label}: {op.note}")
        elif op.kind == "remove":
            before = len(edges)
            edges[:] = [
                edge
                for edge in edges
                if edge_key(edge) != (op.start, op.end, op.label)
            ]
            if len(edges) == before:
                notes.append(f"already removed {op.start}->{op.end} {op.label}: {op.note}")
                continue
            if len(edges) != before - 1:
                raise ValueError(f"expected one edge to remove: {op}")
            notes.append(f"remove {op.start}->{op.end} {op.label}: {op.note}")
        elif op.kind == "add":
            if add_edge_once(edges, op.start, op.end, op.label):
                notes.append(f"add {op.start}->{op.end} {op.label}: {op.note}")
            else:
                notes.append(f"already had {op.start}->{op.end} {op.label}: {op.note}")
        else:
            raise ValueError(f"unknown op kind: {op.kind}")
    validate_graph_shape(repaired)
    return repaired, notes


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


def _role_set_from_bio_label(label: str) -> str:
    if label == "O":
        return "O"
    return label.split("-", 1)[1] if "-" in label else label


def rare_composite_counts(data_dir: Path) -> Counter[str]:
    from lingua.learn.hierarchical.dataset import (  # noqa: PLC0415
        build_hierarchical_examples,
    )

    counts: Counter[str] = Counter()
    for split in ("train", "dev", "test"):
        for example in build_hierarchical_examples(data_dir / f"{split}.jsonl"):
            labels_by_token = {
                token_id: tuple(labels)
                for token_id, labels in example.role_labels_by_token
            }
            for token_id in example.input_token_ids:
                labels = labels_by_token.get(token_id, ("O",))
                key = _role_set_from_bio_label(labels[0] if labels else "O")
                if key in TARGET_COMPOSITES:
                    counts[key] += 1
    return counts


def run(*, apply: bool, data_dir: Path, report_dir: Path) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    case_dir = report_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    changed_records: list[dict[str, Any]] = []
    by_split = Counter()
    by_op = Counter()

    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        new_rows: list[dict[str, Any]] = []
        split_changed = False
        for sample_id, row in enumerate(rows):
            ops = REPAIRS.get((split, sample_id))
            if not ops:
                new_rows.append(row)
                continue
            before = deepcopy(row)
            after, notes = apply_ops(row, ops)
            if before != after:
                split_changed = True
                by_split[split] += 1
                by_op.update(op.kind for op in ops)
                record = {
                    "split": split,
                    "line": sample_id + 1,
                    "sample_id": sample_id,
                    "sentence": row.get("sentence"),
                    "notes": notes,
                    "before": compact_graph(before),
                    "after": compact_graph(after),
                }
                changed_records.append(record)
                (case_dir / f"{split}_{sample_id + 1:05d}.txt").write_text(
                    f"{split}:{sample_id}\n{row.get('sentence')}\n\n"
                    f"NOTES\n" + "\n".join(f"- {note}" for note in notes) + "\n\n"
                    f"BEFORE\n{record['before']}\n\nAFTER\n{record['after']}\n",
                    encoding="utf-8",
                )
            new_rows.append(after)
        if apply and split_changed:
            write_jsonl(path, new_rows)

    summary = {
        "apply": apply,
        "changed_samples": len(changed_records),
        "changed_by_split": dict(by_split),
        "op_counts": dict(by_op),
        "intentional_skips": {f"{split}:{sample_id}": reason for (split, sample_id), reason in INTENTIONAL_SKIPS.items()},
        "report_dir": str(report_dir),
    }
    (report_dir / "changes.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in changed_records)
        + ("\n" if changed_records else ""),
        encoding="utf-8",
    )
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Rare Shared Role Repair",
        "",
        f"- apply: `{apply}`",
        f"- changed samples: `{len(changed_records)}`",
        "",
        "## Changed By Split",
    ]
    for key, value in by_split.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Intentional Skips")
    for (split, sample_id), reason in sorted(INTENTIONAL_SKIPS.items()):
        lines.append(f"- `{split}:{sample_id}`: {reason}")
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    return run(apply=args.apply, data_dir=args.data_dir, report_dir=args.report_dir)


if __name__ == "__main__":
    raise SystemExit(main())
