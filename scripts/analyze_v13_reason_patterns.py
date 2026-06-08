#!/usr/bin/env python3
"""Aggregate V13 non-exact analysis outputs into reason-pattern reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def top_counter(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(limit)]


def edge_label(edge: dict[str, Any]) -> str:
    return str(edge.get("label"))


def counter_items(value: Any) -> list[tuple[str, int]]:
    if isinstance(value, dict):
        return [(str(key), int(count)) for key, count in value.items()]
    if isinstance(value, list):
        return [(str(row["key"]), int(row["count"])) for row in value]
    return []


def load_split(base: Path, split: str) -> dict[str, Any]:
    return {
        "summary": json.loads((base / split / "nonexact_summary.json").read_text(encoding="utf-8")),
        "cases": read_jsonl(base / split / "nonexact_cases_with_confidence.jsonl"),
        "lingua_only": read_jsonl(base / split / "word_exact_lingua_nonexact.jsonl"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--top_k", type=int, default=80)
    args = parser.parse_args()

    base = args.base
    by_split = {split: load_split(base, split) for split in SPLITS}

    all_cases: list[dict[str, Any]] = []
    for split in SPLITS:
        for case in by_split[split]["cases"]:
            case["_split"] = split
            all_cases.append(case)

    aggregate_counters: dict[str, Counter[str]] = defaultdict(Counter)
    for split in SPLITS:
        summary = by_split[split]["summary"]
        for key in (
            "category_counts",
            "word_missing_label_counts",
            "word_extra_label_counts",
            "word_missing_type_counts",
            "word_extra_type_counts",
            "word_same_endpoint_label_confusions",
            "word_same_label_scope_attachment",
            "node_type_confusion_counts",
            "node_span_mismatch_counts",
            "low_pred_unmatched_label_counts",
            "low_edge_family_counts",
            "low_edge_status_counts",
            "root_mismatch_counts",
            "lingua_missing_label_counts",
            "lingua_extra_label_counts",
        ):
            for key_name, count in counter_items(summary.get(key)):
                aggregate_counters[key][key_name] += count

    case_pattern: Counter[str] = Counter()
    combo_counter: Counter[tuple[str, ...]] = Counter()
    case_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    label_conf_case: Counter[str] = Counter()
    scope_case: Counter[str] = Counter()
    node_type_case: Counter[str] = Counter()
    low_extra_case: Counter[str] = Counter()
    missing_case: Counter[str] = Counter()
    extra_case: Counter[str] = Counter()

    for case in all_cases:
        split = case["_split"]
        tags: list[str] = []
        categories = set(case.get("categories") or [])
        for category in sorted(categories):
            if category in {"word_edge_mismatch", "lingua_mismatch"}:
                continue
            tags.append(f"family:{category}")

        if case.get("node_type_confusions"):
            tags.append("node_type_confusion")
            for pair in case["node_type_confusions"]:
                key = f"{pair['gold'].get('type')} -> {pair['pred'].get('type')}"
                node_type_case[key] += 1
        if case.get("node_span_mismatches"):
            tags.append("node_span_mismatch")
        if case.get("same_endpoint_label_confusions"):
            tags.append("same_endpoint_label_confusion")
            for pair in case["same_endpoint_label_confusions"]:
                if "gold" in pair and "pred" in pair:
                    key = f"{pair['gold'].get('label')} -> {pair['pred'].get('label')}"
                else:
                    key = f"{pair.get('gold_label')} -> {pair.get('pred_label')}"
                label_conf_case[key] += 1
        if case.get("same_label_scope_attachment"):
            tags.append("same_label_scope_attachment")
            for pair in case["same_label_scope_attachment"]:
                scope_case[str(pair.get("label"))] += 1
        if case.get("pred_roots") != case.get("gold_roots"):
            tags.append("root_mismatch")
        if case.get("low_unmatched_edge_count", 0):
            tags.append("low_confidence_unmatched_edge")
            for edge in case.get("lowest_unmatched_edges", []):
                if edge.get("graph") == "pred":
                    low_extra_case[edge_label(edge)] += 1
        for edge in case.get("unmatched_true_edges", []):
            missing_case[edge_label(edge)] += 1
        for edge in case.get("unmatched_pred_edges", []):
            extra_case[edge_label(edge)] += 1

        if not tags:
            tags = ["edge_mismatch_only"]
        combo_counter[tuple(sorted(tags))] += 1
        for tag in tags:
            case_pattern[tag] += 1
            if len(case_examples[tag]) < 8:
                case_examples[tag].append(
                    {
                        "split": split,
                        "idx": case.get("idx"),
                        "line": case.get("line"),
                        "sentence": case.get("sentence"),
                        "categories": case.get("categories"),
                        "lowest_unmatched_edges": case.get("lowest_unmatched_edges", [])[:2],
                        "label_confusions": case.get("same_endpoint_label_confusions", [])[:2],
                        "scope": case.get("same_label_scope_attachment", [])[:2],
                        "node_type_confusions": case.get("node_type_confusions", [])[:2],
                    }
                )

    lingua_only_counter: Counter[str] = Counter()
    lingua_only_examples: list[dict[str, Any]] = []
    for split in SPLITS:
        for case in by_split[split]["lingua_only"]:
            labels: list[str] = []
            labels.extend(f"missing:{edge_label(edge)}" for edge in case.get("unmatched_true_edges", []))
            labels.extend(f"extra:{edge_label(edge)}" for edge in case.get("unmatched_pred_edges", []))
            if case.get("unmatched_true_nodes") or case.get("unmatched_pred_nodes"):
                labels.append("lingua_node_mismatch")
            if not labels:
                labels.append("lingua_structure_only")
            for label in labels:
                lingua_only_counter[label] += 1
            if len(lingua_only_examples) < 12:
                lingua_only_examples.append(
                    {
                        "split": split,
                        "idx": case.get("idx"),
                        "line": case.get("line"),
                        "sentence": case.get("sentence"),
                        "unmatched_pred_edges": case.get("unmatched_pred_edges", [])[:3],
                        "unmatched_true_edges": case.get("unmatched_true_edges", [])[:3],
                        "unmatched_pred_nodes": case.get("unmatched_pred_nodes", [])[:2],
                        "unmatched_true_nodes": case.get("unmatched_true_nodes", [])[:2],
                    }
                )

    def category_counts(category: str) -> dict[str, int]:
        return {
            split: int(by_split[split]["summary"]["category_counts"].get(category, 0))
            for split in SPLITS
        }

    reason_families = [
        {
            "family": "Modification scope/layer",
            "category": "modification",
            "counts": category_counts("modification"),
            "description": "Modification variable/body/@modification_of/as:modification choice or attachment scope.",
        },
        {
            "family": "Predicate argument role/scope",
            "category": "predicate_args",
            "counts": category_counts("predicate_args"),
            "description": "pred.arg.N role confusion or same role attached to the wrong predicate/argument.",
        },
        {
            "family": "Expression/list/formula structure",
            "category": "expression_list",
            "counts": category_counts("expression_list"),
            "description": "ExpressionFunctor/ListFunctor/func.arg/@expression/@list structure errors.",
        },
        {
            "family": "Punctuation attachment",
            "category": "punctuation",
            "counts": category_counts("punctuation"),
            "description": "PunctuationalConstant/as:punctuation/punctuation-as-structure attachment errors.",
        },
        {
            "family": "Node type classification",
            "category": "word_node_mismatch",
            "counts": category_counts("word_node_mismatch"),
            "description": "Correct span but wrong node type, plus occasional node split/span issue.",
        },
        {
            "family": "Attribute/apposition/ref",
            "category": "attribute_appos_ref",
            "counts": category_counts("attribute_appos_ref"),
            "description": "@attribute_of/@appositive_of/@ref_of/as:ref confusion.",
        },
        {
            "family": "Lower/upper lifting",
            "category": "lower_upper",
            "counts": category_counts("lower_upper"),
            "description": "@lower/@upper and compound lifted-argument confusion.",
        },
        {
            "family": "Linear next_word",
            "category": "linearization",
            "counts": category_counts("linearization"),
            "description": "next_word/linear-chain used or omitted where semantic edge expected.",
        },
        {
            "family": "Parataxis",
            "category": "parataxis",
            "counts": category_counts("parataxis"),
            "description": "@parataxis clausal/list relation errors.",
        },
        {
            "family": "Discourse",
            "category": "discourse",
            "counts": category_counts("discourse"),
            "description": "as:discourse/@discourse_of errors.",
        },
        {
            "family": "Copula",
            "category": "copula",
            "counts": category_counts("copula"),
            "description": "Small number of @copula errors.",
        },
        {
            "family": "Other",
            "category": "other",
            "counts": category_counts("other"),
            "description": "Labels outside the main families.",
        },
    ]

    output = {
        "output_base": str(base),
        "model": "best-combined/root checkpoint",
        "decode_config": (
            "ILP, edge_factor_decode_weight=0.2, candidates=16, labels=4, "
            "time_limit=20, lazy_constraints=true, max_lazy_rounds=4"
        ),
        "split_totals": {split: by_split[split]["summary"]["totals"] for split in SPLITS},
        "word_lingua_nonexact_total": sum(len(by_split[split]["cases"]) for split in SPLITS),
        "lingua_only_nonexact_total": sum(len(by_split[split]["lingua_only"]) for split in SPLITS),
        "reason_families": reason_families,
        "case_pattern_counts": top_counter(case_pattern, args.top_k),
        "top_pattern_combinations": [
            {"patterns": " + ".join(patterns), "count": count}
            for patterns, count in combo_counter.most_common(args.top_k)
        ],
        "aggregate_counters": {
            key: top_counter(counter, args.top_k)
            for key, counter in sorted(aggregate_counters.items())
        },
        "case_level_label_confusions": top_counter(label_conf_case, args.top_k),
        "case_level_scope_labels": top_counter(scope_case, args.top_k),
        "case_level_node_type_confusions": top_counter(node_type_case, args.top_k),
        "case_level_low_extra_labels": top_counter(low_extra_case, args.top_k),
        "case_level_missing_labels": top_counter(missing_case, args.top_k),
        "case_level_extra_labels": top_counter(extra_case, args.top_k),
        "examples_by_case_pattern": case_examples,
        "lingua_only_patterns": top_counter(lingua_only_counter, args.top_k),
        "lingua_only_examples": lingua_only_examples,
    }

    (base / "reason_patterns_summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(base, output, by_split)
    print(
        json.dumps(
            {
                "summary": str(base / "reason_patterns_summary.json"),
                "report": str(base / "reason_patterns_report.md"),
                "word_lingua_nonexact_total": output["word_lingua_nonexact_total"],
                "lingua_only_nonexact_total": output["lingua_only_nonexact_total"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def write_markdown(base: Path, output: dict[str, Any], by_split: dict[str, Any]) -> None:
    lines: list[str] = [
        "# V13 non-exact reason patterns",
        "",
        f"Output base: `{base}`",
        "",
        (
            "Decode: ILP, edge_factor_decode_weight=0.2, candidates=16, labels=4, "
            "time_limit=20, lazy=true/max_lazy_rounds=4."
        ),
        "",
        "## Split totals",
        "| split | total | Word exact | Word non-exact | Lingua exact | Lingua non-exact | Word-exact/Lingua-nonexact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        totals = by_split[split]["summary"]["totals"]
        lingua_only = len(by_split[split]["lingua_only"])
        lines.append(
            f"| {split} | {totals.get('total', 0)} | {totals.get('word_exact', 0)} | "
            f"{totals.get('word_nonexact', 0)} | {totals.get('lingua_exact', 0)} | "
            f"{totals.get('lingua_nonexact', 0)} | {lingua_only} |"
        )

    lines.extend(
        [
            "",
            "## Non-mutually-exclusive reason families",
            "| family | train | dev | test | note |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for family in output["reason_families"]:
        counts = family["counts"]
        lines.append(
            f"| {family['family']} | {counts.get('train', 0)} | {counts.get('dev', 0)} | "
            f"{counts.get('test', 0)} | {family['description']} |"
        )

    counter_sections = [
        ("Case pattern tags", "case_pattern_counts"),
        ("Label confusion on same endpoint", "case_level_label_confusions"),
        ("Same-label wrong attachment/scope", "case_level_scope_labels"),
        ("Node type confusions", "case_level_node_type_confusions"),
        ("Low-confidence unmatched predicted labels", "case_level_low_extra_labels"),
        ("Lingua-only patterns", "lingua_only_patterns"),
    ]
    lines.extend(["", "## Top case-level mechanisms"])
    for title, key in counter_sections:
        lines.extend(["", f"### {title}"])
        for row in output[key][:20]:
            lines.append(f"- `{row['key']}`: {row['count']}")

    lines.extend(["", "## Representative examples"])
    for tag in (
        "family:modification",
        "same_label_scope_attachment",
        "same_endpoint_label_confusion",
        "node_type_confusion",
        "low_confidence_unmatched_edge",
        "root_mismatch",
        "family:expression_list",
        "family:predicate_args",
        "family:attribute_appos_ref",
        "family:lower_upper",
        "family:linearization",
    ):
        examples = output["examples_by_case_pattern"].get(tag, [])[:3]
        if not examples:
            continue
        lines.extend(["", f"### {tag}"])
        for example in examples:
            lines.append(f"- {example['split']}:{example['line']} `{example['sentence']}`")
            if example.get("label_confusions"):
                lines.append(
                    "  - label_confusion: "
                    + json.dumps(example["label_confusions"][0], ensure_ascii=False)
                )
            if example.get("scope"):
                lines.append("  - scope: " + json.dumps(example["scope"][0], ensure_ascii=False))
            if example.get("node_type_confusions"):
                lines.append(
                    "  - node_type: "
                    + json.dumps(example["node_type_confusions"][0], ensure_ascii=False)
                )
            if example.get("lowest_unmatched_edges"):
                edge = example["lowest_unmatched_edges"][0]
                lines.append(
                    f"  - low/unmatched: {edge.get('graph')} {edge.get('head_word')} "
                    f"--{edge.get('label')}--> {edge.get('dep_word')} conf={edge.get('edge_confidence')}"
                )

    lines.extend(["", "### word exact but Lingua non-exact"])
    for example in output["lingua_only_examples"][:8]:
        lines.append(f"- {example['split']}:{example['line']} `{example['sentence']}`")
        if example.get("unmatched_true_edges"):
            lines.append(
                "  - missing: "
                + json.dumps(example["unmatched_true_edges"][0], ensure_ascii=False)
            )
        if example.get("unmatched_pred_edges"):
            lines.append(
                "  - extra: "
                + json.dumps(example["unmatched_pred_edges"][0], ensure_ascii=False)
            )

    (base / "reason_patterns_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
