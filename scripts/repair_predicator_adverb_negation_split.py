#!/usr/bin/env python3
"""Split adverbs and negation out of FactualPredicator nodes.

Standards applied by this repair:

1. FactualPredicator nodes keep verbal/copular predicate material.
2. Internal adverbs and negation (`not`, `n't`, `nt`, split `n ' t`) are
   represented as `ModificationalConstant` nodes with `as:modification` scope.
3. Existing predicate arguments stay attached to the predicate head. Incoming
   scope edges to the old merged phrase move to the outermost new modifier.
4. Phrasal/directional particles such as `look forward`, `go back`, `pick up`
   are reviewed rather than automatically split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from repair_be_copula_rules import (  # noqa: E402
    add_modifier_chain,
    compact_graph,
    graph,
    node_text,
    node_tokens,
    read_jsonl,
    set_node_token_indexes,
    spans_from_token_indexes,
    transfer_incoming_to_new_node,
    write_jsonl,
)


ADV_TAGS = {"RB", "RBR", "RBS", "WRB"}
NEGATION_SINGLE = {"not", "n't", "nt"}
CONTRACTION_NEGATION_PARTS = {"n", "'", "t"}

# These are often lexical particles or locative predicate material in this
# dataset. They are left for review unless they are explicit negation.
PARTICLE_OR_LOCATIVE_ADVERBS = {
    "about",
    "above",
    "across",
    "ahead",
    "along",
    "apart",
    "around",
    "aside",
    "away",
    "back",
    "down",
    "forward",
    "here",
    "in",
    "indoors",
    "inside",
    "off",
    "on",
    "out",
    "outdoors",
    "outside",
    "over",
    "there",
    "through",
    "together",
    "under",
    "up",
    "upstairs",
}

WH_ADVERBS = {"how", "when", "where", "why"}
SAFE_ANYWHERE_ADVERBS = {
    "actually",
    "already",
    "also",
    "always",
    "basically",
    "certainly",
    "clearly",
    "completely",
    "currently",
    "definitely",
    "even",
    "eventually",
    "ever",
    "finally",
    "fully",
    "generally",
    "hardly",
    "highly",
    "initially",
    "just",
    "largely",
    "mainly",
    "mostly",
    "nearly",
    "never",
    "normally",
    "often",
    "only",
    "particularly",
    "previously",
    "probably",
    "quite",
    "rarely",
    "really",
    "recently",
    "repeatedly",
    "scarcely",
    "simply",
    "so",
    "sometimes",
    "still",
    "usually",
    "very",
}
SAFE_POSTVERBAL_ADVERBS = {
    "early",
    "fast",
    "hard",
    "late",
    "quick",
    "quickly",
    "significantly",
    "slowly",
    "substantially",
    "well",
}


def norm(text: str) -> str:
    return text.strip().lower().replace("’", "'")


def make_doc(nlp: Language, sample: dict[str, Any]) -> Doc:
    words = [word.strip() or word for word in sample["lingua"]["words"]]
    spaces = [True] * max(0, len(words) - 1) + ([False] if words else [])
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    for _name, pipe in nlp.pipeline:
        doc = pipe(doc)
    return doc


def token_infos(sample: dict[str, Any], doc: Doc, indexes: list[int]) -> list[dict[str, Any]]:
    words = sample["lingua"]["words"]
    return [
        {
            "index": index,
            "text": words[index].strip(),
            "norm": norm(words[index]),
            "lemma": doc[index].lemma_,
            "pos": doc[index].pos_,
            "tag": doc[index].tag_,
            "dep": doc[index].dep_,
        }
        for index in indexes
        if 0 <= index < len(doc)
    ]


def token_indexes_for_node(sample: dict[str, Any], node_index: int) -> list[int]:
    return [index for index, _surface, _normalized in node_tokens(sample, node_index)]


def is_factual_predicator(sample: dict[str, Any], node_index: int) -> bool:
    return graph(sample)["nodes"][node_index].get("type") == "FactualPredicator"


def has_split_negation(indexes: set[int], norms: dict[int, str], index: int) -> bool:
    return (
        index in indexes
        and index + 1 in indexes
        and index + 2 in indexes
        and norms.get(index) == "n"
        and norms.get(index + 1) == "'"
        and norms.get(index + 2) == "t"
    )


def collect_candidate_groups(
    sample: dict[str, Any],
    doc: Doc,
    node_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexes = token_indexes_for_node(sample, node_index)
    index_set = set(indexes)
    norms = {index: norm(sample["lingua"]["words"][index]) for index in indexes}
    groups: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    used: set[int] = set()

    for index in indexes:
        if index in used:
            continue
        if has_split_negation(index_set, norms, index):
            group_indexes = [index, index + 1, index + 2]
            groups.append(
                {
                    "indexes": group_indexes,
                    "reason": "split_negation_contraction",
                    "auto": True,
                }
            )
            used.update(group_indexes)

    for index in indexes:
        if index in used:
            continue
        normalized = norms[index]
        if normalized in NEGATION_SINGLE:
            groups.append(
                {
                    "indexes": [index],
                    "reason": f"negation:{normalized}",
                    "auto": True,
                }
            )
            used.add(index)

    for index in indexes:
        if index in used:
            continue
        normalized = norms[index]
        token = doc[index]
        is_adverb = token.pos_ == "ADV" or token.tag_ in ADV_TAGS
        if not is_adverb:
            continue
        if normalized in CONTRACTION_NEGATION_PARTS:
            reviews.append(review_group(sample, doc, node_index, [index], "loose_negation_piece"))
            continue
        auto, reason = should_auto_split_adverb(sample, doc, indexes, norms, index)
        group = {
            "indexes": [index],
            "reason": reason,
            "auto": auto,
        }
        if auto:
            groups.append(group)
            used.add(index)
        else:
            reviews.append(review_group(sample, doc, node_index, [index], reason))

    groups = sorted(groups, key=lambda item: item["indexes"][0])
    return groups, reviews


def content_indexes_after_removal(indexes: list[int], removed: set[int]) -> list[int]:
    return [index for index in indexes if index not in removed]


def has_hyphen_neighbor(sample: dict[str, Any], indexes: set[int], index: int) -> bool:
    words = sample["lingua"]["words"]
    return (
        index - 1 in indexes
        and norm(words[index - 1]) == "-"
        or index + 1 in indexes
        and norm(words[index + 1]) == "-"
    )


def should_auto_split_adverb(
    sample: dict[str, Any],
    doc: Doc,
    indexes: list[int],
    norms: dict[int, str],
    index: int,
) -> tuple[bool, str]:
    normalized = norms[index]
    index_set = set(indexes)
    remaining = content_indexes_after_removal(indexes, {index})
    if not remaining:
        return False, "empty_after_split"
    if normalized in WH_ADVERBS:
        return False, f"wh_adverb_review:{normalized}"
    if has_hyphen_neighbor(sample, index_set, index):
        return False, "hyphenated_predicate_piece"
    if normalized in PARTICLE_OR_LOCATIVE_ADVERBS:
        return False, f"particle_or_locative_review:{normalized}"

    first_remaining = min(remaining)
    last_remaining = max(remaining)
    if index < last_remaining:
        return True, "pre_or_interior_adverb"
    if normalized in SAFE_ANYWHERE_ADVERBS:
        return True, f"safe_anywhere_adverb:{normalized}"
    if normalized in SAFE_POSTVERBAL_ADVERBS or normalized.endswith("ly"):
        return True, f"safe_postverbal_adverb:{normalized}"
    if index < first_remaining:
        return True, "pre_adverb"
    return False, f"postverbal_adverb_review:{normalized}"


def review_group(
    sample: dict[str, Any],
    doc: Doc,
    node_index: int,
    indexes: list[int],
    reason: str,
) -> dict[str, Any]:
    return {
        "kind": "predicator_modifier_review",
        "node_index": node_index,
        "predicate_text": node_text(sample, node_index),
        "reason": reason,
        "tokens": token_infos(sample, doc, indexes),
    }


def split_groups(
    sample: dict[str, Any],
    doc: Doc,
    node_index: int,
    groups: list[dict[str, Any]],
) -> dict[str, Any] | None:
    removed = {index for group in groups for index in group["indexes"]}
    original_indexes = token_indexes_for_node(sample, node_index)
    predicate_indexes = content_indexes_after_removal(original_indexes, removed)
    if not predicate_indexes:
        return None

    before_text = node_text(sample, node_index)
    set_node_token_indexes(sample, node_index, predicate_indexes)

    modifier_spans = [spans_from_token_indexes(group["indexes"])[0] for group in groups]
    old_node_count = len(graph(sample)["nodes"])
    modifier_root = add_modifier_chain(sample, modifier_spans, node_index)
    new_nodes = set(range(old_node_count, len(graph(sample)["nodes"])))
    transfer_incoming_to_new_node(sample, node_index, modifier_root, excluded_starts=new_nodes)

    return {
        "kind": "predicator_adverb_negation_split",
        "node_index": node_index,
        "before_text": before_text,
        "predicate_text": node_text(sample, node_index),
        "modifier_root": modifier_root,
        "modifiers": [
            {
                "reason": group["reason"],
                "text": " ".join(sample["lingua"]["words"][index].strip() for index in group["indexes"]),
                "spans": spans_from_token_indexes(group["indexes"]),
                "tokens": token_infos(sample, doc, group["indexes"]),
            }
            for group in groups
        ],
    }


def repair_sample(sample: dict[str, Any], doc: Doc) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    changes: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    original_node_count = len(graph(sample)["nodes"])
    for node_index in range(original_node_count):
        if not is_factual_predicator(sample, node_index):
            continue
        groups, node_reviews = collect_candidate_groups(sample, doc, node_index)
        reviews.extend(node_reviews)
        if not groups:
            continue
        change = split_groups(sample, doc, node_index, groups)
        if change is None:
            for group in groups:
                reviews.append(
                    review_group(sample, doc, node_index, group["indexes"], "empty_after_split")
                )
            continue
        changes.append(change)
    return changes, reviews


def write_report(
    report_dir: Path,
    args: argparse.Namespace,
    all_changes: list[dict[str, Any]],
    all_reviews: list[dict[str, Any]],
    counts: Counter[str],
) -> None:
    changes_path = report_dir / "changes.jsonl"
    changes_path.write_text(
        "\n".join(json.dumps(change, ensure_ascii=False) for change in all_changes)
        + ("\n" if all_changes else ""),
        encoding="utf-8",
    )
    reviews_path = report_dir / "reviews.jsonl"
    reviews_path.write_text(
        "\n".join(json.dumps(review, ensure_ascii=False) for review in all_reviews)
        + ("\n" if all_reviews else ""),
        encoding="utf-8",
    )

    standard_lines = [
        "# Predicator Modifier Split Standard",
        "",
        "## FactualPredicator Adverbs and Negation",
        "",
        "- Predicate nodes keep verbal/copular predicate material.",
        "- `not`, `n't`, `nt`, and split `n ' t` are `ModificationalConstant` nodes.",
        "- Ordinary adverbs inside predicate nodes are `ModificationalConstant` nodes.",
        "- Modifier nodes attach to the predicate by `as:modification`; multiple modifiers form a surface-order scope chain.",
        "- Phrasal/directional particles such as `look forward`, `go back`, `pick up`, and locatives such as `here/there` are review-only, because they can be lexical predicate material rather than removable modifiers.",
        "",
        "## Be + Adjectival Complement",
        "",
        "- Copular predicates keep the be/modal/aux material in `FactualPredicator`.",
        "- The adjectival complement is a `ModificationalConstant` attached by `pred.arg.2`.",
        "- Composite adjectival complements stay together as one complement node: `is 6 years old` becomes `is ->pred.arg.2-> 6 years old`.",
        "- Degree/negation modifiers scope outside the copula predicate: `is not very old` becomes `not -> very -> is`, and `is ->pred.arg.2-> old`.",
        "- Pure passive/progressive participles are not split by the be+adj rule unless spaCy marks the complement adjectival.",
        "",
    ]
    (report_dir / "standard.md").write_text("\n".join(standard_lines), encoding="utf-8")

    summary_lines = [
        "# Predicator Adverb/Negation Split Summary",
        "",
        f"- input_dir: `{args.input_dir}`",
        f"- output_dir: `{args.output_dir}`",
        f"- report_dir: `{args.report_dir}`",
        f"- spacy_model: `{args.spacy_model}`",
        f"- changed samples: {len(all_changes)}",
        f"- review samples: {len(all_reviews)}",
        "",
        "## Counts",
        "",
        "| key | count |",
        "|---|---:|",
    ]
    for key, count in sorted(counts.items()):
        summary_lines.append(f"| {key} | {count} |")

    summary_lines.extend(["", "## First 50 Changes", ""])
    for change in all_changes[:50]:
        summary_lines.extend(
            [
                f"### {change['split']}:{change['line_number']} "
                f"`{', '.join(item['kind'] for item in change['changes'])}`",
                "",
                change["sentence"],
                "",
                "```text",
                change["before"],
                "",
                "--- after ---",
                change["after"],
                "```",
                "",
            ]
        )
    summary_lines.extend(["", "## First 50 Reviews", ""])
    for review in all_reviews[:50]:
        summary_lines.extend(
            [
                f"### {review['split']}:{review['line_number']} "
                f"`{', '.join(item['reason'] for item in review['reviews'])}`",
                "",
                review["sentence"],
                "",
                "```text",
                review["graph"],
                "```",
                "",
            ]
        )
    (report_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (report_dir / "summary.json").write_text(
        json.dumps(
            {
                "input_dir": str(args.input_dir),
                "output_dir": str(args.output_dir),
                "report_dir": str(args.report_dir),
                "spacy_model": args.spacy_model,
                "changed_samples": len(all_changes),
                "review_samples": len(all_reviews),
                "counts": dict(counts),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    nlp = spacy.load(args.spacy_model)
    all_changes: list[dict[str, Any]] = []
    all_reviews: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for split in ("train", "dev", "test"):
        rows = read_jsonl(input_dir / f"{split}.jsonl")
        new_rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(rows, 1):
            before = deepcopy(row)
            after = deepcopy(row)
            doc = make_doc(nlp, after)
            changes, reviews = repair_sample(after, doc)
            if changes:
                for change in changes:
                    counts[change["kind"]] += 1
                    counts[f"{split}:{change['kind']}"] += 1
                    for modifier in change["modifiers"]:
                        counts[f"modifier_reason:{modifier['reason']}"] += 1
                all_changes.append(
                    {
                        "split": split,
                        "line_number": line_number,
                        "sentence": row.get("sentence", ""),
                        "changes": changes,
                        "before": compact_graph(before),
                        "after": compact_graph(after),
                        "sample": after if args.include_samples_in_report else None,
                    }
                )
                new_rows.append(after)
            else:
                new_rows.append(row)
            if reviews:
                for review in reviews:
                    counts["predicator_modifier_review"] += 1
                    counts[f"{split}:predicator_modifier_review"] += 1
                    counts[f"review_reason:{review['reason']}"] += 1
                all_reviews.append(
                    {
                        "split": split,
                        "line_number": line_number,
                        "sentence": row.get("sentence", ""),
                        "reviews": reviews,
                        "graph": compact_graph(after),
                    }
                )
        write_jsonl(output_dir / f"{split}.jsonl", new_rows)

    write_report(report_dir, args, all_changes, all_reviews, counts)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "spacy_model": args.spacy_model,
        "changed_samples": len(all_changes),
        "review_samples": len(all_reviews),
        "counts": dict(counts),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--include-samples-in-report", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
