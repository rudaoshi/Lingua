#!/usr/bin/env python3
"""Generate review queues for same-label scope attachment errors."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def span_key(node: dict[str, Any]) -> str:
    return json.dumps(node.get("spans"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def endpoint_relation(gold: dict[str, Any], pred: dict[str, Any]) -> str:
    gold_text = gold.get("text")
    pred_text = pred.get("text")
    gold_type = gold.get("type")
    pred_type = pred.get("type")
    gold_span = span_key(gold)
    pred_span = span_key(pred)
    if gold_text == pred_text and gold_type == pred_type and gold_span == pred_span:
        return "same"
    if gold_text == pred_text and gold_type == pred_type and gold_span != pred_span:
        return "same_text_type_diff_span"
    if gold_text == pred_text and gold_span == pred_span and gold_type != pred_type:
        return "same_text_span_diff_type"
    if gold_text == pred_text and gold_type != pred_type and gold_span != pred_span:
        return "same_text_diff_type_span"
    if gold_span == pred_span and gold_type == pred_type and gold_text != pred_text:
        return "same_span_type_diff_text"
    if gold_type == pred_type and gold_text != pred_text:
        return "same_type_diff_text"
    if gold_text == pred_text and gold_type != pred_type:
        return "same_text_diff_type"
    if gold_span == pred_span and gold_type != pred_type:
        return "same_span_diff_text_type"
    if gold.get("spans") is None or pred.get("spans") is None:
        return "abstract_or_wrapper"
    return "different_text_type_span"


def pair_category(pair: dict[str, Any]) -> str:
    relations = [
        endpoint_relation(pair["gold_source"], pair["pred_source"]),
        endpoint_relation(pair["gold_target"], pair["pred_target"]),
    ]
    changed = {relation for relation in relations if relation != "same"}
    if not changed:
        return "same"
    if changed <= {"same_text_type_diff_span"}:
        return "duplicate_surface_span_swap"
    if changed <= {"same_text_span_diff_type"}:
        return "node_type_only_induced"
    if changed <= {"same_text_type_diff_span", "same_text_span_diff_type", "same_text_diff_type_span"}:
        return "same_surface_span_type_mixed"
    return "real_attachment_choice"


def endpoint_diff(pair: dict[str, Any]) -> str:
    source_same = endpoint_relation(pair["gold_source"], pair["pred_source"]) == "same"
    target_same = endpoint_relation(pair["gold_target"], pair["pred_target"]) == "same"
    if source_same and not target_same:
        return "target_only"
    if not source_same and target_same:
        return "source_only"
    if not source_same and not target_same:
        return "both_endpoints"
    return "none"


def label_group(label: str) -> str:
    parts = set(label.split("+"))
    if parts & {"as:modification", "@modification_of", "body", "variable"}:
        return "modification"
    if any(part.startswith("pred.arg.") for part in parts):
        return "predicate_args"
    if parts & {"as:punctuation", "punctuation"}:
        return "punctuation"
    if parts & {"func.arg", "@expression", "@list"}:
        return "expression_list"
    if parts & {"next_word"}:
        return "linearization"
    if parts & {"@ref_of", "@appositive_of", "@attribute_of", "as:ref", "as:appositive", "as:attribute"}:
        return "attribute_appos_ref"
    if parts & {"@lower", "@upper", "@lower+variable", "@upper+variable"}:
        return "lower_upper"
    if parts & {"as:discourse", "@discourse_of"}:
        return "discourse"
    if any(part.startswith("@parataxis") for part in parts):
        return "parataxis"
    if parts & {"as:vocative", "@vocative_of"}:
        return "vocative"
    return "other"


def node_type_group(gold_type: str | None, pred_type: str | None) -> str:
    pair = {gold_type, pred_type}
    if pair <= {"ModificationalConstant", "NominalConstant"}:
        return "modifier_constant_vs_nominal"
    if ("FactualPredicator" in pair or "LogicalPredicator" in pair) and (
        "ModificationalConstant" in pair or "ModificationalFunctor" in pair
    ):
        return "predicator_vs_modifier"
    if ("FactualPredicator" in pair or "LogicalPredicator" in pair) and "NominalConstant" in pair:
        return "predicator_vs_nominal"
    if any(t in pair for t in {"PunctuationalConstant", "ConjunctionalFunctor", "ExpressionFunctor", "ListFunctor"}) and any(
        t in pair for t in {"PunctuationalConstant", "ConjunctionalFunctor", "ExpressionFunctor", "ListFunctor", "LogicalPredicator"}
    ):
        return "punct_coord_expression_boundary"
    if "DeterminerConstant" in pair:
        return "determiner_boundary"
    if any(t in pair for t in {"AttributePredicator", "AppositionalPredicator", "GeneralFunctor"}):
        return "special_wrapper_type"
    if any(t in pair for t in {"InterjectionConstant", "OtherConstant"}):
        return "interjection_other"
    return "other"


def queue_name(category: str, group: str) -> str:
    if category == "duplicate_surface_span_swap":
        return "duplicate_span_swap"
    if category in {"node_type_only_induced", "same_surface_span_type_mixed"}:
        return "node_type_induced_scope"
    if category != "real_attachment_choice":
        return "mixed_other_scope"
    if group == "modification":
        return "real_scope_modification"
    if group == "predicate_args":
        return "real_scope_pred_args"
    if group in {"punctuation", "expression_list", "linearization"}:
        return "real_scope_punctuation_func"
    return "real_scope_other"


def node_value(node: dict[str, Any], field: str) -> str:
    value = node.get(field)
    if field == "spans":
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def compact_pairs(pairs: list[dict[str, Any]], limit: int = 4) -> str:
    items = []
    for pair in pairs[:limit]:
        gold = pair.get("gold") or {}
        pred = pair.get("pred") or {}
        items.append(
            f"{gold.get('text')}:{gold.get('type')}->{pred.get('type')}"
        )
    return " | ".join(items)


FIELDNAMES = [
    "queue",
    "split",
    "line",
    "idx",
    "pair_index",
    "label",
    "label_group",
    "pair_category",
    "endpoint_diff",
    "source_relation",
    "target_relation",
    "case_pair_count",
    "case_scope_categories",
    "case_node_type_count",
    "case_node_type_summary",
    "audit_priority",
    "recommended_action",
    "gold_source_text",
    "gold_source_type",
    "gold_source_spans",
    "gold_target_text",
    "gold_target_type",
    "gold_target_spans",
    "pred_source_text",
    "pred_source_type",
    "pred_source_spans",
    "pred_target_text",
    "pred_target_type",
    "pred_target_spans",
    "gold_edge_score",
    "pred_edge_score",
    "gold_edge_rank",
    "pred_edge_rank",
    "score_note",
    "sentence",
]


def recommended_action(category: str, group: str, label: str) -> tuple[str, str]:
    if category == "duplicate_surface_span_swap":
        return "P2", "duplicate-token tie-break; inspect only if repeated spans change semantics"
    if category in {"node_type_only_induced", "same_surface_span_type_mixed"}:
        return "P1", "fix node type disambiguation first; edge mismatch is mostly induced"
    if group == "modification":
        if label in {"body", "variable"}:
            return "P0", "review Modification body/variable endpoint; consider structural constraint"
        return "P0", "review modifier-head scope; decide direct modifier vs wrapper attachment"
    if group == "predicate_args":
        return "P0", "review predicate argument endpoint; candidate is same role but wrong predicate/argument"
    if group in {"punctuation", "expression_list", "linearization"}:
        return "P1", "review punctuation/list/coord carrier and local span ordering"
    return "P2", "lower-volume real scope queue; review after P0/P1"


def row_for_pair(case: dict[str, Any], pair: dict[str, Any], pair_index: int, layer: str) -> dict[str, str]:
    label = str(pair["label"])
    group = label_group(label)
    category = pair_category(pair)
    queue = queue_name(category, group)
    priority, action = recommended_action(category, group, label)
    node_type_pairs = (case.get(layer) or {}).get("node_type_confusions") or []
    source_relation = endpoint_relation(pair["gold_source"], pair["pred_source"])
    target_relation = endpoint_relation(pair["gold_target"], pair["pred_target"])
    scope_pairs = (case.get(layer) or {}).get("same_label_scope_attachment") or []
    scope_categories = Counter(pair_category(item) for item in scope_pairs)
    row = {
        "queue": queue,
        "split": str(case.get("split")),
        "line": str(case.get("line")),
        "idx": str(case.get("idx")),
        "pair_index": str(pair_index),
        "label": label,
        "label_group": group,
        "pair_category": category,
        "endpoint_diff": endpoint_diff(pair),
        "source_relation": source_relation,
        "target_relation": target_relation,
        "case_pair_count": str(len(scope_pairs)),
        "case_scope_categories": json.dumps(dict(scope_categories), ensure_ascii=False, sort_keys=True),
        "case_node_type_count": str(len(node_type_pairs)),
        "case_node_type_summary": compact_pairs(node_type_pairs),
        "audit_priority": priority,
        "recommended_action": action,
        "gold_edge_score": "",
        "pred_edge_score": "",
        "gold_edge_rank": "",
        "pred_edge_rank": "",
        "score_note": "not_available_in_prediction_dump",
        "sentence": str(case.get("sentence")),
    }
    for prefix, node in (
        ("gold_source", pair["gold_source"]),
        ("gold_target", pair["gold_target"]),
        ("pred_source", pair["pred_source"]),
        ("pred_target", pair["pred_target"]),
    ):
        row[f"{prefix}_text"] = node_value(node, "text")
        row[f"{prefix}_type"] = node_value(node, "type")
        row[f"{prefix}_spans"] = node_value(node, "spans")
    return row


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_queue(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "by_split": dict(Counter(row["split"] for row in rows).most_common()),
        "by_label": dict(Counter(row["label"] for row in rows).most_common(30)),
        "by_endpoint_diff": dict(Counter(row["endpoint_diff"] for row in rows).most_common()),
        "by_pair_category": dict(Counter(row["pair_category"] for row in rows).most_common()),
        "by_priority": dict(Counter(row["audit_priority"] for row in rows).most_common()),
    }


def write_markdown(path: Path, queues: dict[str, list[dict[str, str]]], summary: dict[str, Any]) -> None:
    lines = [
        "# Scope Error Review Queues",
        "",
        f"这些 TSV 来自当前 V13 prediction dump 的 {summary['layer']} non-exact cases。",
        "",
        "注意：当前 prediction dump 没保存逐候选边 score/rank，所以 score/rank 列先留空；需要时要补跑 targeted forward 或扩展 dump。",
        "",
        "## Queue Counts",
        "",
        "| queue | rows | train | dev | test | top labels |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, rows in sorted(queues.items()):
        stats = summary["queues"][name]
        labels = ", ".join(f"{label}:{count}" for label, count in list(stats["by_label"].items())[:6])
        lines.append(
            f"| `{name}` | {stats['rows']} | {stats['by_split'].get('train', 0)} | "
            f"{stats['by_split'].get('dev', 0)} | {stats['by_split'].get('test', 0)} | {labels} |"
        )

    lines.extend(
        [
            "",
            "## Suggested Order",
            "",
            "1. `real_scope_modification.tsv`: P0，先看 `variable/body/as:modification/@modification_of` 的真实范围错。",
            "2. `real_scope_pred_args.tsv`: P0，看 `pred.arg.1/2` 是否是谓词候选排序问题或标注标准问题。",
            "3. `real_scope_punctuation_func.tsv`: P1，看 punctuation/list/coord carrier，尤其逗号、引号、URL/公式。",
            "4. `node_type_induced_scope.tsv`: P1，先修 node type，再重新评估 scope。",
            "5. `duplicate_span_swap.tsv`: P2，重复同形 token 的 tie-break/metric 敏感问题。",
            "",
            "## Files",
            "",
        ]
    )
    for name in sorted(queues):
        lines.append(f"- `{name}.tsv`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = list(read_jsonl(args.cases))
    queues: dict[str, list[dict[str, str]]] = defaultdict(list)
    all_rows: list[dict[str, str]] = []
    for case in cases:
        pairs = (case.get(args.layer) or {}).get("same_label_scope_attachment") or []
        for pair_index, pair in enumerate(pairs):
            row = row_for_pair(case, pair, pair_index, args.layer)
            queues[row["queue"]].append(row)
            all_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "all_scope_attachment_pairs.tsv", all_rows)
    for name, rows in sorted(queues.items()):
        write_tsv(args.output_dir / f"{name}.tsv", rows)

    summary = {
        "input_cases": str(args.cases),
        "layer": args.layer,
        "output_dir": str(args.output_dir),
        "total_scope_pairs": len(all_rows),
        "total_scope_cases": len({(row["split"], row["line"]) for row in all_rows}),
        "queues": {name: summarize_queue(rows) for name, rows in sorted(queues.items())},
    }
    (args.output_dir / "scope_error_queues_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "scope_error_queues_report.md", queues, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("reports/v13_overwrite_latest_nonexact_patterns/nonexact_cases.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/v13_overwrite_latest_scope_error_queues"),
    )
    parser.add_argument("--layer", choices=["word_lingua", "lingua"], default="word_lingua")
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "output_dir": summary["output_dir"],
                "total_scope_cases": summary["total_scope_cases"],
                "total_scope_pairs": summary["total_scope_pairs"],
                "queues": {name: value["rows"] for name, value in summary["queues"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
