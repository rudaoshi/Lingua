#!/usr/bin/env python3
"""List same-surface hierarchical units that map to multiple structures.

The report is intentionally evidence-heavy: surface text alone is not a safe
repair key because function words, punctuation, names, and modifiers can be
legitimately context-dependent.  This script therefore writes both grouped
conflicts and every occurrence that belongs to a conflict group.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lingua.learn.hierarchical import DEFAULT_VIEW, NON_AUX_KIND, extract_hierarchical_oracle  # noqa: E402


DATA = ROOT / "data"
REPORTS = ROOT / "reports"
WH_WORDS = {"what", "who", "which", "whom", "whose", "where", "when", "why", "how"}


STANDARD_REFERENCES = {
    "content_attach_dual_view": "hierarchical oracle default/content view policy",
    "expression_vs_modification": "docs/standard.md 6.6, proper names section",
    "explicit_aux_vs_direct_modification_functor": "docs/standard.md 7.2.1 redundant Modification",
    "factual_predicator_vs_modification": "docs/standard.md 7.2.1 redundant Modification",
    "nominal_vs_modificational_constant": "docs/standard.md entity/noun phrase + POS policy",
    "wh_pos_context": "docs/standard.md wh-word POS policy",
    "whether_vs_other": "docs/standard.md complement/ref/predicate policy",
    "punctuation_or_symbol": "docs/standard.md punctuation/expression policy",
    "leaf_vs_expand": "docs/standard.md recoverability by DESC(root)",
    "other": "docs/standard.md full graph review required",
}


STANDARD_REVIEW_NOTES = {
    "content_attach_dual_view": (
        "Intentional dual-view supervision: default view predicts the upper wrapper, "
        "while content view parses the same-token-set child referenced by that wrapper."
    ),
    "expression_vs_modification": (
        "Audit whether the internal relation is format/name composition or head+qualifier. "
        "Human names, opaque official names, codes, dates, URLs, and pure format strings should be Expression; "
        "recoverable type/head+scope structures should be Modification."
    ),
    "explicit_aux_vs_direct_modification_functor": (
        "Audit whether an explicit Modification aux adds head/scope information. "
        "If the same modifier relation is already expressed by a body functor, remove the redundant wrapper; "
        "if the aux is the only head+body unit, keep it."
    ),
    "factual_predicator_vs_modification": (
        "Audit whether the surface is a real predicate with arguments or merely a modifier unit. "
        "Do not wrap a factual/complement relation in Modification when the predicate already recovers the structure."
    ),
    "nominal_vs_modificational_constant": (
        "Usually context-dependent.  Use NominalConstant for entities/arguments, ModificationalConstant for adjectival "
        "or modifier-only uses.  Do not global-repair by lexical surface."
    ),
    "wh_pos_context": (
        "Review lexical wh-word POS/use in context.  Wh question and relative markers are GeneralFunctor, "
        "but acronyms, titles, quoted strings, and determiner-like uses can legitimately differ."
    ),
    "whether_vs_other": (
        "Usually context-dependent.  Whether is an outer yes/no or whether-clause scope wrapper; "
        "lexical complementizers and referential uses must be reviewed in the full graph."
    ),
    "punctuation_or_symbol": (
        "Punctuation and symbol surfaces are often context-dependent.  Check whether the token closes punctuation, "
        "acts as an expression operator, or was mistyped as Other/Nominal."
    ),
    "leaf_vs_expand": (
        "Audit whether the expanded structure adds recoverable internal semantics.  If it is an opaque entity/name, "
        "a leaf or Expression may be correct; if head/scope is recoverable, expansion may be required."
    ),
    "other": "Requires full-sentence graph review before repair.",
}


def iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            yield line_number, json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def normalize_surface(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def coarse_label(unit: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        unit.get("view_kind", DEFAULT_VIEW),
        unit["unit_pos"],
        unit["aux_kind"],
        unit["decision"],
    )


def label_text(label: tuple[str, ...]) -> str:
    return "|".join(label)


def child_shape(child: dict[str, Any]) -> str:
    role = child["role"]
    unit_pos = child["unit_pos"]
    aux_kind = child["aux_kind"]
    edge_label = child.get("edge_label", "")
    shared = "+shared" if child.get("is_shared") else ""
    shared_arg = "+shared_arg" if child.get("has_shared_arg") else ""
    return f"{role}/{edge_label}:{unit_pos}/{aux_kind}{shared}{shared_arg}"


def role_shape(unit: dict[str, Any]) -> str:
    if unit["decision"] == "leaf":
        return "leaf"
    return " ".join(sorted(child_shape(child) for child in unit.get("children", [])))


def structure_signature(unit: dict[str, Any]) -> str:
    return f"{label_text(coarse_label(unit))} :: {role_shape(unit)}"


def example_from_unit(
    split: str,
    line_number: int,
    sample: dict[str, Any],
    unit_index: int,
    unit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "split": split,
        "line": line_number,
        "sample_id": sample.get("id") or f"{split}.jsonl:{line_number}",
        "unit_index": unit_index,
        "surface": unit["input_text"],
        "surface_key": normalize_surface(unit["input_text"]),
        "sentence": sample.get("sentence") or unit.get("sentence"),
        "label": label_text(coarse_label(unit)),
        "role_shape": role_shape(unit),
        "structure": structure_signature(unit),
        "token_ids": unit["input_token_ids"],
        "view_kind": unit.get("view_kind", DEFAULT_VIEW),
        "unit_pos": unit["unit_pos"],
        "aux_kind": unit["aux_kind"],
        "decision": unit["decision"],
        "is_root": unit["is_root"],
        "depth": unit["depth"],
        "has_shared_child": unit.get("has_shared_child", False),
        "has_shared_arg": unit.get("has_shared_arg", False),
        "has_external_child": unit.get("has_external_child", False),
        "children": [
            {
                "role": child["role"],
                "edge_label": child["edge_label"],
                "text": child["text"],
                "label_text": child.get("label_text", child["text"]),
                "unit_pos": child["unit_pos"],
                "aux_kind": child["aux_kind"],
                "token_ids": child["token_ids"],
                "label_token_ids": child.get("label_token_ids", child["token_ids"]),
                "is_shared": child["is_shared"],
                "has_shared_arg": child.get("has_shared_arg", False),
            }
            for child in unit.get("children", [])
        ],
    }


def surface_category(labels: set[tuple[str, str, str, str]], surface_key: str) -> str:
    views = {label[0] for label in labels}
    unit_pos = {label[1] for label in labels}
    aux_kind = {label[2] for label in labels}
    decisions = {label[3] for label in labels}

    if is_content_attach_dual_view(labels):
        return "content_attach_dual_view"
    if "Expression" in aux_kind and "Modification" in aux_kind:
        return "expression_vs_modification"
    if "Modification" in aux_kind and "ModificationalFunctor" in unit_pos:
        return "explicit_aux_vs_direct_modification_functor"
    if "Modification" in aux_kind and "FactualPredicator" in unit_pos:
        return "factual_predicator_vs_modification"
    if {"NominalConstant", "ModificationalConstant"} <= unit_pos:
        return "nominal_vs_modificational_constant"
    if surface_key in WH_WORDS and "GeneralFunctor" in unit_pos:
        return "wh_pos_context"
    if "Whether" in aux_kind or "GeneralFunctor" in unit_pos:
        return "whether_vs_other"
    if (
        "PunctuationalConstant" in unit_pos
        or "SymbolConstant" in unit_pos
        or surface_key in {".", ",", ":", ";", "-", "/", "?", "!", "(", ")", "[", "]"}
    ):
        return "punctuation_or_symbol"
    if len(decisions) > 1:
        return "leaf_vs_expand"
    return "other"


def category_priority(category: str) -> int:
    return {
        "expression_vs_modification": 0,
        "explicit_aux_vs_direct_modification_functor": 1,
        "factual_predicator_vs_modification": 2,
        "leaf_vs_expand": 3,
        "nominal_vs_modificational_constant": 4,
        "wh_pos_context": 5,
        "punctuation_or_symbol": 5,
        "whether_vs_other": 6,
        "other": 7,
        "content_attach_dual_view": 8,
    }.get(category, 9)


def is_content_attach_dual_view(labels: set[tuple[str, str, str, str]]) -> bool:
    views = {label[0] for label in labels}
    if views != {"default", "content"}:
        return False

    default_labels = [label for label in labels if label[0] == "default"]
    content_labels = [label for label in labels if label[0] == "content"]
    if not default_labels or not content_labels:
        return False

    return (
        all(label[1] == "ModificationalFunctor" and label[2] == "Modification" for label in default_labels)
        and all(label[1] == "ModificationalFunctor" for label in content_labels)
    )


def occurrence_key(row: dict[str, Any]) -> tuple[str, int, tuple[int, ...]]:
    return (row["split"], int(row["line"]), tuple(row["token_ids"]))


def is_intentional_whether_parent_child(row: dict[str, Any], group: list[dict[str, Any]]) -> bool:
    if row["unit_pos"] != "GeneralFunctor" or row["aux_kind"] != "Whether":
        return False
    children = row.get("children", [])
    if len(children) != 1:
        return False
    child = children[0]
    if child.get("edge_label") != "func.arg" or child.get("role") not in {"FUNC_ARG", "FUNC_ARG_REF"}:
        return False
    if tuple(child.get("token_ids", [])) != tuple(row["token_ids"]):
        return False

    key = occurrence_key(row)
    for other in group:
        if other is row or (other["unit_pos"] == "GeneralFunctor" and other["aux_kind"] == "Whether"):
            continue
        if occurrence_key(other) != key:
            continue
        if other["unit_pos"] == child.get("unit_pos") and other["aux_kind"] == child.get("aux_kind"):
            return True
    return False


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def build_reports(args: argparse.Namespace) -> dict[str, Any]:
    occurrences: list[dict[str, Any]] = []
    by_surface: dict[str, list[dict[str, Any]]] = defaultdict(list)
    samples_by_split = Counter()
    units_by_split = Counter()
    failed_samples: list[dict[str, Any]] = []

    for split in args.splits:
        path = args.data_dir / f"{split}.jsonl"
        for line_number, sample in iter_jsonl(path):
            samples_by_split[split] += 1
            try:
                oracle = extract_hierarchical_oracle(sample).to_dict()
            except Exception as exc:  # pragma: no cover - audit output path
                failed_samples.append(
                    {
                        "split": split,
                        "line": line_number,
                        "sample_id": sample.get("id") or f"{split}.jsonl:{line_number}",
                        "error": str(exc),
                    }
                )
                continue
            for unit_index, unit in enumerate(oracle["units"]):
                units_by_split[split] += 1
                surface_key = normalize_surface(unit["input_text"])
                if not surface_key:
                    continue
                occurrence = example_from_unit(split, line_number, sample, unit_index, unit)
                occurrences.append(occurrence)
                by_surface[surface_key].append(occurrence)

    conflict_groups: list[dict[str, Any]] = []
    conflict_occurrences: list[dict[str, Any]] = []
    excluded_dual_view_surface_count = 0
    excluded_dual_view_occurrence_count = 0
    excluded_whether_parent_child_surface_count = 0
    excluded_whether_parent_child_occurrence_count = 0
    resolved_by_whether_parent_child_pruning_surface_count = 0
    for surface_key, group in by_surface.items():
        removed_whether_parent_child = []
        if not args.include_whether_parent_child_conflicts:
            kept_group = []
            for row in group:
                if is_intentional_whether_parent_child(row, group):
                    removed_whether_parent_child.append(row)
                else:
                    kept_group.append(row)
            group = kept_group
            if removed_whether_parent_child:
                excluded_whether_parent_child_surface_count += 1
                excluded_whether_parent_child_occurrence_count += len(removed_whether_parent_child)

        label_counts = Counter(row["label"] for row in group)
        if len(label_counts) <= 1:
            if removed_whether_parent_child:
                resolved_by_whether_parent_child_pruning_surface_count += 1
            continue
        shape_counts = Counter(row["structure"] for row in group)
        exact_counts = Counter(row["surface"] for row in group)
        split_counts = Counter(row["split"] for row in group)
        labels = {tuple(label.split("|")) for label in label_counts}
        category = surface_category(labels, surface_key)
        if category == "content_attach_dual_view" and not args.include_dual_view_conflicts:
            excluded_dual_view_surface_count += 1
            excluded_dual_view_occurrence_count += len(group)
            continue

        examples_by_label: dict[str, list[dict[str, Any]]] = {}
        for label, _count in label_counts.most_common():
            label_rows = [row for row in group if row["label"] == label]
            examples_by_label[label] = label_rows[: args.max_examples_per_label]

        row = {
            "surface_key": surface_key,
            "surface_variants": dict(exact_counts.most_common()),
            "total": len(group),
            "label_count": len(label_counts),
            "structure_count": len(shape_counts),
            "labels": dict(label_counts.most_common()),
            "structures": dict(shape_counts.most_common(args.max_structures_per_surface)),
            "splits": dict(split_counts),
            "category": category,
            "priority": category_priority(category),
            "label_entropy": round(entropy(label_counts), 4),
            "standard_reference": STANDARD_REFERENCES[category],
            "standard_review_note": STANDARD_REVIEW_NOTES[category],
            "examples_by_label": examples_by_label,
        }
        conflict_groups.append(row)
        conflict_occurrences.extend(group)

    conflict_groups.sort(
        key=lambda row: (
            row["priority"],
            -int(row["total"]),
            -float(row["label_entropy"]),
            row["surface_key"],
        )
    )
    conflict_occurrences.sort(
        key=lambda row: (row["surface_key"], row["split"], int(row["line"]), int(row["unit_index"]))
    )

    summary = {
        "data_dir": str(args.data_dir),
        "splits": list(args.splits),
        "samples_by_split": dict(samples_by_split),
        "units_by_split": dict(units_by_split),
        "all_surface_count": len(by_surface),
        "all_occurrence_count": len(occurrences),
        "conflict_surface_count": len(conflict_groups),
        "conflict_occurrence_count": len(conflict_occurrences),
        "excluded_dual_view_surface_count": excluded_dual_view_surface_count,
        "excluded_dual_view_occurrence_count": excluded_dual_view_occurrence_count,
        "excluded_whether_parent_child_surface_count": excluded_whether_parent_child_surface_count,
        "excluded_whether_parent_child_occurrence_count": excluded_whether_parent_child_occurrence_count,
        "resolved_by_whether_parent_child_pruning_surface_count": resolved_by_whether_parent_child_pruning_surface_count,
        "failed_sample_count": len(failed_samples),
        "category_counts": dict(Counter(row["category"] for row in conflict_groups).most_common()),
        "category_occurrences": dict(
            Counter({}).most_common()
        ),
    }
    category_occurrences = Counter()
    for row in conflict_groups:
        category_occurrences[row["category"]] += int(row["total"])
    summary["category_occurrences"] = dict(category_occurrences.most_common())

    return {
        "summary": summary,
        "groups": conflict_groups,
        "occurrences": conflict_occurrences,
        "failed_samples": failed_samples,
    }


def write_reports(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    groups = report["groups"]
    occurrences = report["occurrences"]
    failed_samples = report["failed_samples"]

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "surface_conflicts.json", groups)
    if failed_samples:
        write_json(output_dir / "failed_samples.json", failed_samples)

    group_rows = []
    for row in groups:
        group_rows.append(
            {
                "surface_key": row["surface_key"],
                "total": row["total"],
                "label_count": row["label_count"],
                "structure_count": row["structure_count"],
                "category": row["category"],
                "priority": row["priority"],
                "label_entropy": row["label_entropy"],
                "splits": compact_json(row["splits"]),
                "surface_variants": compact_json(row["surface_variants"]),
                "labels": compact_json(row["labels"]),
                "structures": compact_json(row["structures"]),
                "standard_reference": row["standard_reference"],
                "standard_review_note": row["standard_review_note"],
                "examples_by_label": compact_json(row["examples_by_label"]),
            }
        )
    fields = [
        "surface_key",
        "total",
        "label_count",
        "structure_count",
        "category",
        "priority",
        "label_entropy",
        "splits",
        "surface_variants",
        "labels",
        "structures",
        "standard_reference",
        "standard_review_note",
        "examples_by_label",
    ]
    write_tsv(output_dir / "surface_conflicts.tsv", group_rows, fields)
    write_tsv(output_dir / "review_queue.tsv", group_rows, fields)
    wh_pos_group_rows = [row for row in group_rows if row["category"] == "wh_pos_context"]
    write_tsv(output_dir / "wh_pos_review_queue.tsv", wh_pos_group_rows, fields)

    occurrence_fields = [
        "surface_key",
        "surface",
        "split",
        "line",
        "sample_id",
        "unit_index",
        "label",
        "role_shape",
        "structure",
        "token_ids",
        "view_kind",
        "unit_pos",
        "aux_kind",
        "decision",
        "is_root",
        "depth",
        "has_shared_child",
        "has_shared_arg",
        "has_external_child",
        "sentence",
        "children",
    ]
    occurrence_rows = []
    for row in occurrences:
        out = dict(row)
        out["token_ids"] = compact_json(out["token_ids"])
        out["children"] = compact_json(out["children"])
        occurrence_rows.append(out)
    write_tsv(output_dir / "surface_conflict_occurrences.tsv", occurrence_rows, occurrence_fields)
    wh_pos_surface_keys = {row["surface_key"] for row in groups if row["category"] == "wh_pos_context"}
    wh_pos_occurrence_rows = [row for row in occurrence_rows if row["surface_key"] in wh_pos_surface_keys]
    write_tsv(output_dir / "wh_pos_review_occurrences.tsv", wh_pos_occurrence_rows, occurrence_fields)

    with (output_dir / "surface_conflict_occurrences.jsonl").open("w", encoding="utf-8") as handle:
        for row in occurrences:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (output_dir / "wh_pos_review_occurrences.jsonl").open("w", encoding="utf-8") as handle:
        for row in occurrences:
            if row["surface_key"] in wh_pos_surface_keys:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    markdown = [
        "# Hierarchical Surface Conflict Audit",
        "",
        f"- data_dir: `{summary['data_dir']}`",
        f"- splits: `{', '.join(summary['splits'])}`",
        f"- samples: `{compact_json(summary['samples_by_split'])}`",
        f"- units: `{compact_json(summary['units_by_split'])}`",
        f"- conflict surfaces: `{summary['conflict_surface_count']}` / `{summary['all_surface_count']}`",
        f"- conflict occurrences: `{summary['conflict_occurrence_count']}` / `{summary['all_occurrence_count']}`",
        f"- excluded default/content dual-view surfaces: `{summary['excluded_dual_view_surface_count']}`",
        f"- excluded default/content dual-view occurrences: `{summary['excluded_dual_view_occurrence_count']}`",
        f"- excluded intentional Whether parent/child surfaces: `{summary['excluded_whether_parent_child_surface_count']}`",
        f"- excluded intentional Whether parent/child occurrences: `{summary['excluded_whether_parent_child_occurrence_count']}`",
        f"- surfaces resolved by Whether parent/child pruning: `{summary['resolved_by_whether_parent_child_pruning_surface_count']}`",
        f"- failed samples: `{summary['failed_sample_count']}`",
        "",
        "## Categories",
        "",
    ]
    for category, count in summary["category_counts"].items():
        markdown.append(
            f"- `{category}`: surfaces `{count}`, occurrences `{summary['category_occurrences'][category]}`"
        )
    markdown.extend(
        [
            "",
            "## Files",
            "",
            "- `surface_conflicts.tsv`: all same-surface conflict groups.",
            "- `surface_conflict_occurrences.tsv`: every unit occurrence belonging to those groups.",
            "- `surface_conflicts.json`: grouped report with sampled evidence by label.",
            "- `review_queue.tsv`: same rows as `surface_conflicts.tsv`, sorted by standard-review priority.",
            "- `wh_pos_review_queue.tsv`: wh-word POS/context conflicts only.",
            "- `wh_pos_review_occurrences.tsv`: every occurrence for the wh-word POS/context queue.",
            "",
            "## Standard Review Policy",
            "",
            "- Do not repair globally by lexical surface.  The same text can be legitimately different under different roles.",
            "- Repair only exact split/line/node cases after whole-graph review.",
            "- Expression vs Modification follows the standard's format/name-composition vs recoverable head+qualifier distinction.",
            "- Redundant Modification wrappers should be removed only when the body already recovers the relation and no scope is lost.",
            "- Pure default/content dual-view surfaces are excluded by default because they encode two intentional questions over the same token set.",
            "- Same-span `Whether -> func.arg` parent/child layers are excluded by default because they encode the outer yes/no scope and the inner proposition separately.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    wh_pos_markdown = [
        "# WH/POS Review Queue",
        "",
        f"- surfaces: `{len(wh_pos_group_rows)}`",
        f"- occurrences: `{len(wh_pos_occurrence_rows)}`",
        "",
        "## Policy",
        "",
        "- Direct wh-question and relative wh markers should usually be lexical `GeneralFunctor`.",
        "- Acronyms, quoted titles, determiner-like uses, and ordinary nominal mentions may legitimately be non-`GeneralFunctor`.",
        "- Review by split/line occurrence; do not global-repair a wh surface by string alone.",
        "",
        "## Files",
        "",
        "- `wh_pos_review_queue.tsv`",
        "- `wh_pos_review_occurrences.tsv`",
        "- `wh_pos_review_occurrences.jsonl`",
    ]
    if wh_pos_group_rows:
        wh_pos_markdown.extend(["", "## Surfaces", ""])
        for row in wh_pos_group_rows[:50]:
            wh_pos_markdown.append(
                f"- `{row['surface_key']}`: total `{row['total']}`, labels `{row['labels']}`"
            )
    (output_dir / "wh_pos_review.md").write_text("\n".join(wh_pos_markdown) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=REPORTS / "hierarchical_surface_conflicts")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--max-examples-per-label", type=int, default=5)
    parser.add_argument("--max-structures-per-surface", type=int, default=20)
    parser.add_argument(
        "--include-dual-view-conflicts",
        action="store_true",
        help="Keep pure default/content dual-view surfaces in the conflict report.",
    )
    parser.add_argument(
        "--include-whether-parent-child-conflicts",
        action="store_true",
        help="Keep same-span Whether->func.arg parent/child layers in the conflict report.",
    )
    args = parser.parse_args()

    report = build_reports(args)
    write_reports(report, args.output_dir)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
