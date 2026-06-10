#!/usr/bin/env python3
"""Repair reviewed lexical POS context drift cases.

This pass targets two same-surface leaf POS clusters:

* DeterminerConstant / ModificationalConstant / NominalConstant
* DeterminerConstant / NominalConstant

The rules are intentionally context-sensitive.  They do not make a surface
dictionary; pronoun, determiner, quantifier, adverbial, numeric, and name-part
uses are kept distinct.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

REVIEW_SURFACES_A = {
    "all",
    "any",
    "both",
    "each",
    "either",
    "few",
    "many",
    "most",
    "multiple",
    "one",
    "other",
    "several",
    "six",
    "some",
    "such",
    "the",
    "their",
    "these",
    "three",
    "two",
    "various",
    "you",
}

REVIEW_SURFACES_B = {
    "an",
    "another",
    "el",
    "four",
    "her",
    "his",
    "il",
    "la",
    "le",
    "my",
    "our",
    "te",
    "this",
    "those",
}

CARDINAL_SURFACES = {"one", "two", "three", "four", "six"}
PRONOUN_SURFACES = {"i", "you", "we", "they", "he", "she", "it"}
DETERMINER_SURFACES = {"the", "their", "your", "my", "his", "her", "our", "this", "that", "these", "those"}
POS_TYPES = {"DeterminerConstant", "ModificationalConstant", "NominalConstant"}

# Exact keeps are contexts where the surface-level conflict is real:
# misspellings whose function is different from the written form, name/URL
# parts, gerund subjects, or foreign-language determiners in fragments.
THE_KEEP_NOMINAL = {
    ("test", 79, (23,)),  # typo for "they": "the don't do any damage"
    ("test", 404, (34,)),  # URL path part
    ("train", 5352, (10,)),  # typo for "they"
    ("train", 6465, (0,)),  # THE HAGUE name part
    ("train", 8790, (6,)),  # The Hague name part
}

FOREIGN_NAME_TO_NOMINAL = {
    ("train", 2973, (10,)),  # Palma il Giovane
    ("train", 4150, (9,)),  # El Chaparro
    ("train", 399, (8,)),  # La Hulpe
    ("train", 3399, (0,)),  # La Citta
    ("train", 10128, (11,)),  # La Besseyre-Saint-Mary
    ("train", 8970, (7,)),  # Le Havre
}

EXACT_TO_DETERMINER = {
    ("train", 4656, (0,)): "these before coordinated nominal head",
    ("train", 4151, (7,)): "another before nominal head",
    ("train", 8120, (3,)): "our before appositive nominal head",
}

EXACT_TO_NOMINAL = {
    ("train", 1433, (5,)): "standalone quantifier too many",
}

EXACT_TO_MODIFICATIONAL = {
    ("train", 9618, (13,)): "other is adjectival modifier",
    ("train", 8565, (9,)): "multiple is adjectival modifier",
    ("train", 8997, (18,)): "various is adjectival modifier",
    ("train", 3911, (8,)): "multiple is adjectival modifier",
}

KEEP_EXACT = {
    ("train", 7086, (3,)): "typo for locative there: keep adverbial modifier",
    ("train", 3387, (12,)): "genitive subject of gerund clause",
    ("train", 1134, (7,)): "genitive subject of gerund clause",
    ("train", 5966, (14,)): "genitive subject of gerund clause",
    ("train", 9859, (5,)): "genitive subject of gerund clause",
    ("train", 9368, (3,)): "idiom worst of all",
    ("train", 11127, (4,)): "floating all shares predicate and subject scope",
    ("dev", 945, (3,)): "foreign-language determiner in fragment",
    ("train", 3098, (10,)): "abbreviated title Most Rev.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def normalize_surface(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def node_token_indexes(node: dict[str, Any]) -> tuple[int, ...]:
    spans = node.get("spans")
    if isinstance(spans, str):
        return ()
    indexes: list[int] = []
    for start, end in spans:
        indexes.extend(range(start, end + 1))
    return tuple(indexes)


def node_text(words: list[str], node: dict[str, Any]) -> str:
    return " ".join(words[index].strip() for index in node_token_indexes(node) if words[index].strip())


def previous_next(words: list[str], tokens: tuple[int, ...]) -> tuple[str, str]:
    if not tokens:
        return "", ""
    previous = normalize_surface(words[tokens[0] - 1]) if tokens[0] > 0 else ""
    next_token = normalize_surface(words[tokens[-1] + 1]) if tokens[-1] + 1 < len(words) else ""
    return previous, next_token


def outgoing_children(
    words: list[str],
    graph: dict[str, Any],
    node_index: int,
) -> list[tuple[dict[str, Any], str, str]]:
    nodes = graph["nodes"]
    return [
        (edge, node_text(words, nodes[edge["end"]]), nodes[edge["end"]].get("type", ""))
        for edge in graph["edges"]
        if edge["start"] == node_index
    ]


def incoming_parents(
    words: list[str],
    graph: dict[str, Any],
    node_index: int,
) -> list[tuple[dict[str, Any], str, str]]:
    nodes = graph["nodes"]
    return [
        (edge, node_text(words, nodes[edge["start"]]), nodes[edge["start"]].get("type", ""))
        for edge in graph["edges"]
        if edge["end"] == node_index
    ]


def target_type(
    split: str,
    line_number: int,
    sample: dict[str, Any],
    node_index: int,
) -> tuple[str, str] | None:
    graph = sample["lingua"]["graph"]
    words = sample["lingua"]["words"]
    node = graph["nodes"][node_index]
    current_type = node.get("type")
    if current_type not in POS_TYPES:
        return None

    surface = normalize_surface(node_text(words, node))
    if surface not in REVIEW_SURFACES_A and surface not in REVIEW_SURFACES_B:
        return None

    tokens = node_token_indexes(node)
    exact_key = (split, line_number, tokens)
    if exact_key in KEEP_EXACT:
        return None
    if exact_key in FOREIGN_NAME_TO_NOMINAL and current_type != "NominalConstant":
        return "NominalConstant", "foreign article/name part"
    if exact_key in EXACT_TO_DETERMINER and current_type != "DeterminerConstant":
        return "DeterminerConstant", EXACT_TO_DETERMINER[exact_key]
    if exact_key in EXACT_TO_NOMINAL and current_type != "NominalConstant":
        return "NominalConstant", EXACT_TO_NOMINAL[exact_key]
    if exact_key in EXACT_TO_MODIFICATIONAL and current_type != "ModificationalConstant":
        return "ModificationalConstant", EXACT_TO_MODIFICATIONAL[exact_key]

    children = outgoing_children(words, graph, node_index)
    parents = incoming_parents(words, graph, node_index)
    previous, next_token = previous_next(words, tokens)

    if surface in CARDINAL_SURFACES and current_type != "NominalConstant":
        return "NominalConstant", "cardinal numeral"

    if surface == "the":
        if exact_key in THE_KEEP_NOMINAL:
            return None
        if current_type != "DeterminerConstant":
            return "DeterminerConstant", "article the"

    if surface == "you" and current_type == "ModificationalConstant":
        return "DeterminerConstant", "typo for possessive your before noun"

    if surface == "these" and current_type == "ModificationalConstant":
        return "DeterminerConstant", "demonstrative determiner before nominal"

    if surface == "some" and current_type == "ModificationalConstant":
        if any(child_type in {"NominalConstant", "ConjunctionalFunctor"} for _edge, _text, child_type in children):
            return "DeterminerConstant", "some before nominal/conjunction"

    if surface == "any" and current_type == "ModificationalConstant":
        if not any(child_text == "better" for _edge, child_text, _child_type in children):
            return "DeterminerConstant", "any before nominal/conjunction"

    if surface == "many" and current_type == "ModificationalConstant":
        if any(child_text == "many" for _edge, child_text, _child_type in children):
            return None
        return "DeterminerConstant", "many as quantificational determiner"

    if surface == "several" and current_type == "ModificationalConstant":
        return "DeterminerConstant", "several as quantificational determiner"

    if surface == "few" and current_type != "DeterminerConstant":
        if next_token == "of":
            return None
        if current_type == "NominalConstant" and next_token in {"is", "are", "was", "were", "be"}:
            return None
        return "DeterminerConstant", "few as quantificational determiner"

    if surface == "such" and current_type != "DeterminerConstant":
        if current_type == "NominalConstant":
            return None
        if previous in {"no", "one", "many"} or any(
            normalize_surface(parent_text) in {"no", "one", "many"}
            for _edge, parent_text, _parent_type in parents
        ):
            return None
        if any(
            child_type
            in {
                "DeterminerConstant",
                "NominalConstant",
                "ModificationalConstant",
                "FactualPredicator",
                "ModificationalFunctor",
            }
            for _edge, _child_text, child_type in children
        ):
            return "DeterminerConstant", "such before nominal phrase"

    if surface == "most" and current_type == "ModificationalConstant":
        if previous == "the":
            return None
        if any(
            child_type in {"NominalConstant", "ConjunctionalFunctor"} or child_text == "other"
            for _edge, child_text, child_type in children
        ):
            return "DeterminerConstant", "most before nominal"

    if surface == "both" and current_type != "DeterminerConstant":
        if previous in PRONOUN_SURFACES:
            if current_type != "ModificationalConstant":
                return "ModificationalConstant", "floating/distributive both"
            return None
        if any(
            child_type in {"NominalConstant", "ConjunctionalFunctor", "DeterminerConstant"}
            for _edge, _child_text, child_type in children
        ):
            return "DeterminerConstant", "both before nominal/conjunction"

    if surface == "all" and current_type != "DeterminerConstant":
        if next_token in DETERMINER_SURFACES or any(
            child_type == "DeterminerConstant" or child_text in {"these", "those", "this", "that"}
            for _edge, child_text, child_type in children
        ):
            return "DeterminerConstant", "all before determiner/pronoun nominal"

    return None


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(apply: bool, data_dir: Path, report_dir: Path) -> int:
    report_dir.mkdir(parents=True, exist_ok=True)
    changes: list[dict[str, Any]] = []
    kept = Counter()
    changed_by_surface = Counter()
    changed_by_transition = Counter()

    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        new_rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(rows, 1):
            after = deepcopy(row)
            graph = after["lingua"]["graph"]
            words = after["lingua"]["words"]
            for node_index, node in enumerate(graph["nodes"]):
                surface = normalize_surface(node_text(words, node))
                current_type = node.get("type")
                if current_type in POS_TYPES and (
                    surface in REVIEW_SURFACES_A or surface in REVIEW_SURFACES_B
                ):
                    exact_key = (split, line_number, node_token_indexes(node))
                    if exact_key in KEEP_EXACT or exact_key in THE_KEEP_NOMINAL:
                        kept[(surface, current_type, KEEP_EXACT.get(exact_key, "reviewed keep"))] += 1
                target = target_type(split, line_number, after, node_index)
                if target is None:
                    continue
                new_type, reason = target
                old_type = node.get("type")
                if old_type == new_type:
                    continue
                node["type"] = new_type
                record = {
                    "split": split,
                    "line": line_number,
                    "node_index": node_index,
                    "token_ids": json.dumps(list(node_token_indexes(node))),
                    "surface": node_text(words, node),
                    "old_type": old_type,
                    "new_type": new_type,
                    "reason": reason,
                    "sentence": row.get("sentence", ""),
                }
                changes.append(record)
                changed_by_surface[surface] += 1
                changed_by_transition[(old_type, new_type, reason)] += 1
            new_rows.append(after)
        if apply:
            write_jsonl(path, new_rows)

    write_tsv(
        report_dir / "changes.tsv",
        changes,
        ["split", "line", "node_index", "token_ids", "surface", "old_type", "new_type", "reason", "sentence"],
    )
    (report_dir / "changes.jsonl").write_text(
        "".join(json.dumps(change, ensure_ascii=False, sort_keys=True) + "\n" for change in changes),
        encoding="utf-8",
    )
    summary = {
        "apply": apply,
        "changed_nodes": len(changes),
        "changed_by_surface": dict(changed_by_surface.most_common()),
        "changed_by_transition": {
            f"{old}->{new}|{reason}": count
            for (old, new, reason), count in changed_by_transition.most_common()
        },
        "reviewed_keep_counts": {
            f"{surface}|{node_type}|{reason}": count
            for (surface, node_type, reason), count in kept.most_common()
        },
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Lexical POS Context Repair",
        "",
        f"- apply: {apply}",
        f"- changed_nodes: {len(changes)}",
        "",
        "## Changed By Surface",
    ]
    for surface, count in changed_by_surface.most_common():
        lines.append(f"- `{surface}`: {count}")
    lines.extend(["", "## Changed By Transition"])
    for (old_type, new_type, reason), count in changed_by_transition.most_common():
        lines.append(f"- `{old_type}` -> `{new_type}` ({reason}): {count}")
    lines.extend(["", "## Reviewed Keeps"])
    for (surface, node_type, reason), count in kept.most_common():
        lines.append(f"- `{surface}` `{node_type}` ({reason}): {count}")
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORTS / "lexical_pos_context_repair",
    )
    args = parser.parse_args()
    return run(args.apply, args.data_dir, args.report_dir)


if __name__ == "__main__":
    raise SystemExit(main())
