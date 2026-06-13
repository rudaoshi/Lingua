#!/usr/bin/env python3
"""Repair reviewed VBG/VBN nominal nodes that are adjectival attributes."""

from __future__ import annotations

import argparse
import json
import sys
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
DEFAULT_REPORT_DIR = REPORTS / "reviewed_nominal_participle_attribute_repair"


# split, 1-based line number, token index, expected token, review decision
REPAIRS = [
    ("train", 724, 4, "born", "adjectival_result_state"),
    ("train", 3318, 5, "blacklined", "adjectival_document_state"),
    ("train", 4364, 13, "run", "hyphenated_adjectival_modifier"),
    ("train", 5119, 5, "Dated", "dated_brent_attribute"),
    ("train", 7604, 8, "run", "hyphenated_adjectival_modifier"),
    ("train", 9871, 1, "dried", "freeze_dried_attribute"),
    ("train", 10147, 6, "blacklined", "adjectival_document_state"),
    ("train", 2079, 10, "finned", "ray_finned_attribute"),
]


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if isinstance(spans, str):
        return []
    indexes: list[int] = []
    for span in spans or []:
        if isinstance(span, list) and len(span) == 2:
            indexes.extend(range(int(span[0]), int(span[1]) + 1))
    return sorted(set(indexes))


def find_single_token_node(sample: dict[str, Any], token_index: int) -> int:
    matches = [
        node_index
        for node_index, node in enumerate(sample["lingua"]["graph"]["nodes"])
        if node_token_indexes(node) == [token_index]
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one node for token {token_index}, found {matches}")
    return matches[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[tuple[int, int, str, str]]] = {}
    for split, line_number, token_index, token, reason in REPAIRS:
        by_split.setdefault(split, []).append((line_number, token_index, token, reason))

    changes: list[dict[str, Any]] = []
    for split, repairs in by_split.items():
        path = data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        new_rows = list(rows)
        for line_number, token_index, expected_token, reason in repairs:
            sample = deepcopy(new_rows[line_number - 1])
            words = sample["lingua"]["words"]
            actual_token = words[token_index].strip()
            if actual_token != expected_token:
                raise ValueError(
                    f"{split}:{line_number}:{token_index} expected {expected_token!r}, got {actual_token!r}"
                )
            node_index = find_single_token_node(sample, token_index)
            node = sample["lingua"]["graph"]["nodes"][node_index]
            before_type = node.get("type")
            if before_type != "NominalConstant":
                raise ValueError(
                    f"{split}:{line_number}:{token_index} expected NominalConstant, got {before_type}"
                )
            before = deepcopy(sample)
            node["type"] = "ModificationalConstant"
            changes.append(
                {
                    "split": split,
                    "line_number": line_number,
                    "token_index": token_index,
                    "token": actual_token,
                    "node_index": node_index,
                    "reason": reason,
                    "before_type": before_type,
                    "after_type": "ModificationalConstant",
                    "sentence": sample.get("sentence", ""),
                    "before": compact_graph(before),
                    "after": compact_graph(sample),
                }
            )
            new_rows[line_number - 1] = sample
        if args.apply:
            write_jsonl(path, new_rows)

    summary = {
        "data_dir": str(data_dir),
        "report_dir": str(report_dir),
        "apply": bool(args.apply),
        "changed_nodes": len(changes),
    }
    (report_dir / "changes.jsonl").write_text(
        "\n".join(json.dumps(change, ensure_ascii=False) for change in changes)
        + ("\n" if changes else ""),
        encoding="utf-8",
    )
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
