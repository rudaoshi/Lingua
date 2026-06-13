#!/usr/bin/env python3
"""Repair VBG/VBN nominal nodes that are actually adjectival attributes.

The selector is intentionally pattern-based but conservative:

* exact one-token Lingua node
* UD tag is VBG/VBN
* current Lingua type is NominalConstant
* the token is used as an attributive modifier (`amod` with modifier edges), a
  hyphenated attributive participle, or an adjective fragment like
  `over priced`

Gerund/event nominals and obvious proper-title/name false positives are
reported but left unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
DEFAULT_REPORT_DIR = REPORTS / "nominal_participle_attribute_pattern_repair"

FALSE_PARTICIPLE_NOUNS = {
    "crap",
    "patent",
    "saw",
    "thought",
    "united",
}

PROPER_TITLE_PARTS = {
    "hacking",
    "kicking",
    "reading",
    "sprinkling",
}

PARTICLE_TARGETS = {
    "up",
}


def node_token_indexes(node: dict[str, Any]) -> list[int]:
    spans = node.get("spans")
    if isinstance(spans, str):
        return []
    indexes: list[int] = []
    for span in spans or []:
        if isinstance(span, list) and len(span) == 2:
            indexes.extend(range(int(span[0]), int(span[1]) + 1))
    return sorted(set(indexes))


def token_text(sample: dict[str, Any], token_index: int) -> str:
    return sample["lingua"]["words"][token_index].strip()


def graph(sample: dict[str, Any]) -> dict[str, Any]:
    return sample["lingua"]["graph"]


def node_text(sample: dict[str, Any], node_index: int) -> str:
    indexes = node_token_indexes(graph(sample)["nodes"][node_index])
    if indexes:
        return " ".join(token_text(sample, index) for index in indexes)
    return str(graph(sample)["nodes"][node_index].get("spans"))


def outgoing(sample: dict[str, Any], node_index: int, label: str | None = None) -> list[dict[str, Any]]:
    return [
        edge
        for edge in graph(sample).get("edges", [])
        if edge["start"] == node_index and (label is None or edge["label"] == label)
    ]


def incoming(sample: dict[str, Any], node_index: int, label: str | None = None) -> list[dict[str, Any]]:
    return [
        edge
        for edge in graph(sample).get("edges", [])
        if edge["end"] == node_index and (label is None or edge["label"] == label)
    ]


def exact_node_for_token(sample: dict[str, Any], token_index: int) -> int | None:
    matches = [
        node_index
        for node_index, node in enumerate(graph(sample)["nodes"])
        if node_token_indexes(node) == [token_index]
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def adjacent_hyphen(sample: dict[str, Any], token_index: int) -> bool:
    words = sample["lingua"]["words"]
    return (
        token_index > 0
        and words[token_index - 1].strip() == "-"
        or token_index + 1 < len(words)
        and words[token_index + 1].strip() == "-"
    )


def has_modifier_out(sample: dict[str, Any], node_index: int) -> bool:
    return bool(outgoing(sample, node_index, "as:modification"))


def has_nominal_modifier_head(sample: dict[str, Any], node_index: int) -> bool:
    nodes = graph(sample)["nodes"]
    for edge in outgoing(sample, node_index, "as:modification"):
        target = nodes[edge["end"]]
        if target.get("type") == "NominalConstant" and node_text(sample, edge["end"]).lower() not in PARTICLE_TARGETS:
            return True
    return False


def has_non_determiner_modifier_in(sample: dict[str, Any], node_index: int) -> bool:
    nodes = graph(sample)["nodes"]
    for edge in incoming(sample, node_index, "as:modification"):
        source = nodes[edge["start"]]
        if source.get("type") != "DeterminerConstant":
            return True
    return False


def classify_candidate(
    sample: dict[str, Any],
    ud_word: dict[str, Any],
    token_index: int,
    node_index: int,
) -> tuple[str, str]:
    text = token_text(sample, token_index)
    lower = text.lower()
    xpos = str(ud_word.get("xpos") or "")
    deprel = str(ud_word.get("deprel") or "")

    if lower in FALSE_PARTICIPLE_NOUNS:
        return "review_false_participle_or_nominal_name", "surface_false_participle"
    if lower in PROPER_TITLE_PARTS and token_index == 0:
        return "review_proper_title_or_name_part", "proper_title_part"

    if deprel == "amod" and has_modifier_out(sample, node_index):
        return "repair", "ud_amod_attribute_modifier_source"
    if deprel == "amod" and adjacent_hyphen(sample, token_index):
        return "repair", "hyphenated_ud_amod_attribute"
    if deprel in {"csubj", "nsubj"} and has_nominal_modifier_head(sample, node_index):
        return "repair", "subject_phrase_attribute_modifier_root"
    if xpos == "VBN" and deprel in {"root", "ccomp"} and has_non_determiner_modifier_in(sample, node_index):
        return "repair", "vbn_adjective_fragment_with_degree_modifier"

    return "review_nominal_or_eventive_participle", "not_attribute_pattern"


def ud_words(ud_row: dict[str, Any]) -> list[dict[str, Any]]:
    sentences = ud_row.get("sentences") or []
    if not sentences:
        return []
    return list(sentences[0].get("words") or [])


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    changes: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for split in ("train", "dev", "test"):
        data_path = data_dir / f"{split}.jsonl"
        ud_path = data_dir / f"{split}_ud.jsonl"
        rows = read_jsonl(data_path)
        ud_rows = read_jsonl(ud_path)
        if len(rows) != len(ud_rows):
            raise ValueError(f"{split}: data/UD row count mismatch {len(rows)} != {len(ud_rows)}")

        new_rows: list[dict[str, Any]] = []
        for line_number, (row, ud_row) in enumerate(zip(rows, ud_rows), 1):
            words = row["lingua"]["words"]
            words_ud = ud_words(ud_row)
            if len(words) != len(words_ud):
                new_rows.append(row)
                counts[f"{split}:skipped_ud_alignment"] += 1
                continue

            after = deepcopy(row)
            row_changes: list[dict[str, Any]] = []
            row_reviews: list[dict[str, Any]] = []

            for token_index, ud_word in enumerate(words_ud):
                if str(ud_word.get("xpos") or "") not in {"VBG", "VBN"}:
                    continue
                node_index = exact_node_for_token(after, token_index)
                if node_index is None:
                    continue
                node = graph(after)["nodes"][node_index]
                if node.get("type") != "NominalConstant":
                    continue

                action, reason = classify_candidate(after, ud_word, token_index, node_index)
                record = {
                    "split": split,
                    "line_number": line_number,
                    "token_index": token_index,
                    "token": token_text(after, token_index),
                    "node_index": node_index,
                    "reason": reason,
                    "ud": {
                        "text": ud_word.get("text"),
                        "upos": ud_word.get("upos"),
                        "xpos": ud_word.get("xpos"),
                        "deprel": ud_word.get("deprel"),
                        "head": ud_word.get("head"),
                    },
                    "sentence": row.get("sentence", ""),
                    "incoming": [
                        {
                            "start": edge["start"],
                            "label": edge["label"],
                            "source": node_text(after, edge["start"]),
                        }
                        for edge in incoming(after, node_index)
                    ],
                    "outgoing": [
                        {
                            "label": edge["label"],
                            "end": edge["end"],
                            "target": node_text(after, edge["end"]),
                        }
                        for edge in outgoing(after, node_index)
                    ],
                }
                if action == "repair":
                    before = deepcopy(after)
                    node["type"] = "ModificationalConstant"
                    row_changes.append(
                        {
                            **record,
                            "before_type": "NominalConstant",
                            "after_type": "ModificationalConstant",
                            "before": compact_graph(before),
                            "after": compact_graph(after),
                        }
                    )
                else:
                    row_reviews.append({**record, "kind": action})

            if row_changes:
                for change in row_changes:
                    counts[change["reason"]] += 1
                    counts[f"{split}:{change['reason']}"] += 1
                changes.extend(row_changes)
                new_rows.append(after)
            else:
                new_rows.append(row)

            if row_reviews:
                for review in row_reviews:
                    counts[review["kind"]] += 1
                    counts[f"{split}:{review['kind']}"] += 1
                reviews.extend(row_reviews)

        if args.apply:
            write_jsonl(data_path, new_rows)

    (report_dir / "changes.jsonl").write_text(
        "\n".join(json.dumps(change, ensure_ascii=False) for change in changes)
        + ("\n" if changes else ""),
        encoding="utf-8",
    )
    (report_dir / "reviews.jsonl").write_text(
        "\n".join(json.dumps(review, ensure_ascii=False) for review in reviews)
        + ("\n" if reviews else ""),
        encoding="utf-8",
    )
    summary = {
        "data_dir": str(data_dir),
        "report_dir": str(report_dir),
        "apply": bool(args.apply),
        "changed_nodes": len(changes),
        "review_nodes": len(reviews),
        "counts": dict(counts),
    }
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
