#!/usr/bin/env python3
"""Analyze non-exact case patterns from saved prediction dumps."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")
LAYERS = (
    ("word_lingua", "word_lingua_metric", "gold_word_lingua_graph", "pred_word_lingua_graph"),
    ("lingua", "lingua_metric", "gold_lingua_graph", "pred_lingua_graph"),
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_spans(spans: Any) -> str:
    return json.dumps(spans, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def node_text_from_spans(words: list[str], spans: Any) -> str:
    if not isinstance(spans, list):
        return ""
    parts: list[str] = []
    for span in spans:
        if not isinstance(span, list) or len(span) != 2:
            continue
        start, end = span
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end >= len(words) or start > end:
            continue
        parts.append("".join(words[start : end + 1]))
    return "".join(parts)


def node_info(node: dict[str, Any] | None, words: list[str] | None = None) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {"text": "", "type": None, "spans": None}
    spans = node.get("spans")
    text = node.get("text")
    if text is None and words is not None:
        text = node_text_from_spans(words, spans)
    return {
        "text": "" if text is None else str(text),
        "type": None if node.get("type") is None else str(node.get("type")),
        "spans": spans,
    }


def node_key(node: dict[str, Any], words: list[str] | None = None) -> tuple[str, str | None, str]:
    info = node_info(node, words)
    return (info["text"], info["type"], normalize_spans(info["spans"]))


def node_surface_key(node: dict[str, Any], words: list[str] | None = None) -> tuple[str, str]:
    info = node_info(node, words)
    return (info["text"], normalize_spans(info["spans"]))


def node_type_span_key(node: dict[str, Any], words: list[str] | None = None) -> tuple[str | None, str]:
    info = node_info(node, words)
    return (info["type"], normalize_spans(info["spans"]))


def node_text_type_key(node: dict[str, Any], words: list[str] | None = None) -> tuple[str, str | None]:
    info = node_info(node, words)
    return (info["text"], info["type"])


def edge_info(edge: Any, words: list[str] | None = None) -> dict[str, Any]:
    if isinstance(edge, dict):
        source = node_info(edge.get("source"), words)
        target = node_info(edge.get("target"), words)
        label = str(edge.get("label"))
    else:
        source = node_info(edge[0], words)
        label = str(edge[1])
        target = node_info(edge[2], words)
    return {"source": source, "label": label, "target": target}


def edge_endpoint_key(edge: Any, words: list[str] | None = None) -> tuple[Any, Any]:
    item = edge_info(edge, words)
    return (
        (item["source"]["text"], item["source"]["type"], normalize_spans(item["source"]["spans"])),
        (item["target"]["text"], item["target"]["type"], normalize_spans(item["target"]["spans"])),
    )


def edge_label(edge: Any) -> str:
    if isinstance(edge, dict):
        return str(edge.get("label"))
    return str(edge[1])


def edge_family(label: str, source_type: str | None = None, target_type: str | None = None) -> set[str]:
    parts = set(label.split("+"))
    node_types = {source_type or "", target_type or ""}
    families: set[str] = set()
    if any("punctuation" in part for part in parts) or any("PunctuationalConstant" in t for t in node_types):
        families.add("punctuation")
    if parts & {"@expression", "@list", "func.arg"} or any(
        t in {"ExpressionFunctor", "ListFunctor", "ExpressionOperator", "ListOperator"} for t in node_types
    ):
        families.add("expression_list")
    if parts & {"@modification_of", "as:modification", "variable", "body"} or any(
        "Modificational" in t or t == "Modification" for t in node_types
    ):
        families.add("modification")
    if parts & {"@attribute_of", "@appositive_of", "@ref_of", "as:attribute", "as:appositive", "as:ref"} or any(
        t in {"Ref", "Apposition", "AttributePredicator", "Attribute"} for t in node_types
    ):
        families.add("attribute_appos_ref")
    if any(part.startswith("pred.arg.") for part in parts) or any("Predicator" in t for t in node_types):
        families.add("predicate_args")
    if parts & {"@lower", "@upper", "@lower+variable", "@upper+variable"}:
        families.add("lower_upper")
    if parts & {"next_word"}:
        families.add("linearization")
    if parts & {"as:discourse", "@discourse_of"}:
        families.add("discourse")
    if any(part.startswith("@parataxis") for part in parts):
        families.add("parataxis")
    if parts & {"@copula"}:
        families.add("copula")
    if parts & {"as:vocative", "@vocative_of"}:
        families.add("vocative")
    if parts & {"repeat"}:
        families.add("repeat")
    if not families:
        families.add("other")
    return families


def graph_roots(graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not graph:
        return []
    words = graph.get("words") or []
    payload = graph.get("graph") or {}
    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []
    incoming = {int(edge["end"]) for edge in edges if isinstance(edge, dict) and "end" in edge}
    roots = []
    for idx, node in enumerate(nodes):
        if idx not in incoming:
            roots.append(node_info(node, words))
    return roots


def roots_key(roots: list[dict[str, Any]]) -> list[tuple[str, str | None, str]]:
    return sorted((r["text"], r["type"], normalize_spans(r["spans"])) for r in roots)


def pair_label_confusions(pred_edges: list[Any], true_edges: list[Any]) -> list[dict[str, Any]]:
    pred_by_endpoint: dict[Any, list[Any]] = defaultdict(list)
    true_by_endpoint: dict[Any, list[Any]] = defaultdict(list)
    for edge in pred_edges:
        pred_by_endpoint[edge_endpoint_key(edge)].append(edge)
    for edge in true_edges:
        true_by_endpoint[edge_endpoint_key(edge)].append(edge)

    pairs = []
    for key in sorted(set(pred_by_endpoint) & set(true_by_endpoint), key=str):
        for true_edge, pred_edge in zip(true_by_endpoint[key], pred_by_endpoint[key]):
            gold_label = edge_label(true_edge)
            pred_label = edge_label(pred_edge)
            if gold_label == pred_label:
                continue
            true_info = edge_info(true_edge)
            pairs.append(
                {
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "source": true_info["source"],
                    "target": true_info["target"],
                }
            )
    return pairs


def pair_same_label_scope(pred_edges: list[Any], true_edges: list[Any]) -> list[dict[str, Any]]:
    pred_by_label: dict[str, list[Any]] = defaultdict(list)
    true_by_label: dict[str, list[Any]] = defaultdict(list)
    for edge in pred_edges:
        pred_by_label[edge_label(edge)].append(edge)
    for edge in true_edges:
        true_by_label[edge_label(edge)].append(edge)

    pairs = []
    for label in sorted(set(pred_by_label) & set(true_by_label)):
        for true_edge, pred_edge in zip(true_by_label[label], pred_by_label[label]):
            if edge_endpoint_key(true_edge) == edge_endpoint_key(pred_edge):
                continue
            true_info = edge_info(true_edge)
            pred_info = edge_info(pred_edge)
            pairs.append(
                {
                    "label": label,
                    "gold_source": true_info["source"],
                    "gold_target": true_info["target"],
                    "pred_source": pred_info["source"],
                    "pred_target": pred_info["target"],
                }
            )
    return pairs


def pair_node_type_confusions(pred_nodes: list[dict[str, Any]], true_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pred_by_surface: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    true_by_surface: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for node in pred_nodes:
        pred_by_surface[node_surface_key(node)].append(node)
    for node in true_nodes:
        true_by_surface[node_surface_key(node)].append(node)

    pairs = []
    for key in sorted(set(pred_by_surface) & set(true_by_surface), key=str):
        for true_node, pred_node in zip(true_by_surface[key], pred_by_surface[key]):
            gold = node_info(true_node)
            pred = node_info(pred_node)
            if gold["type"] == pred["type"]:
                continue
            pairs.append({"gold": gold, "pred": pred})
    return pairs


def pair_node_span_mismatches(pred_nodes: list[dict[str, Any]], true_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pred_by_text_type: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    true_by_text_type: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for node in pred_nodes:
        pred_by_text_type[node_text_type_key(node)].append(node)
    for node in true_nodes:
        true_by_text_type[node_text_type_key(node)].append(node)

    pairs = []
    for key in sorted(set(pred_by_text_type) & set(true_by_text_type), key=str):
        for true_node, pred_node in zip(true_by_text_type[key], pred_by_text_type[key]):
            gold = node_info(true_node)
            pred = node_info(pred_node)
            if normalize_spans(gold["spans"]) == normalize_spans(pred["spans"]):
                continue
            pairs.append({"gold": gold, "pred": pred})
    return pairs


def classify_metric(
    metric: dict[str, Any],
    gold_graph: dict[str, Any] | None,
    pred_graph: dict[str, Any] | None,
) -> dict[str, Any]:
    pred_edges = metric.get("unmatched_pred_edges") or []
    true_edges = metric.get("unmatched_true_edges") or []
    pred_nodes = metric.get("unmatched_pred_nodes") or []
    true_nodes = metric.get("unmatched_true_nodes") or []

    label_confusions = pair_label_confusions(pred_edges, true_edges)
    scope_pairs = pair_same_label_scope(pred_edges, true_edges)
    node_type_confusions = pair_node_type_confusions(pred_nodes, true_nodes)
    node_span_mismatches = pair_node_span_mismatches(pred_nodes, true_nodes)

    gold_roots = graph_roots(gold_graph)
    pred_roots = graph_roots(pred_graph)
    root_mismatch = roots_key(gold_roots) != roots_key(pred_roots)

    patterns: set[str] = set()
    families: set[str] = set()
    if pred_nodes or true_nodes:
        patterns.add("node_mismatch")
    if pred_edges or true_edges:
        patterns.add("edge_mismatch")
    if label_confusions:
        patterns.add("same_endpoint_label_confusion")
    if scope_pairs:
        patterns.add("same_label_scope_attachment")
    if node_type_confusions:
        patterns.add("node_type_confusion")
    if node_span_mismatches:
        patterns.add("node_span_mismatch")
    if root_mismatch:
        patterns.add("root_mismatch")

    for edge in pred_edges + true_edges:
        item = edge_info(edge)
        families.update(edge_family(item["label"], item["source"]["type"], item["target"]["type"]))
    for node in pred_nodes + true_nodes:
        info = node_info(node)
        families.update(edge_family("", info["type"], info["type"]))
    if not patterns:
        patterns.add("unknown")
    if not families:
        families.add("other")

    return {
        "patterns": sorted(patterns),
        "families": sorted(families),
        "same_endpoint_label_confusions": label_confusions,
        "same_label_scope_attachment": scope_pairs,
        "node_type_confusions": node_type_confusions,
        "node_span_mismatches": node_span_mismatches,
        "gold_roots": gold_roots,
        "pred_roots": pred_roots,
        "root_mismatch": root_mismatch,
        "unmatched_pred_edges": [edge_info(edge) for edge in pred_edges],
        "unmatched_true_edges": [edge_info(edge) for edge in true_edges],
        "unmatched_pred_nodes": [node_info(node) for node in pred_nodes],
        "unmatched_true_nodes": [node_info(node) for node in true_nodes],
    }


def compact_counter(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(limit)]


def add_example(bucket: dict[str, list[dict[str, Any]]], key: str, example: dict[str, Any], limit: int) -> None:
    if len(bucket[key]) < limit:
        bucket[key].append(example)


def summarize_layer(
    split: str,
    layer: str,
    metric: dict[str, Any],
    case: dict[str, Any],
    counters: dict[str, Counter[str]],
    examples: dict[str, list[dict[str, Any]]],
    example_limit: int,
) -> None:
    analysis = case[layer]
    prefix = f"{split}.{layer}"
    all_prefix = f"all.{layer}"
    for key in (prefix, all_prefix):
        counters[f"{key}.pattern"].update(analysis["patterns"])
        counters[f"{key}.family"].update(analysis["families"])
        counters[f"{key}.missing_label"].update(edge["label"] for edge in analysis["unmatched_true_edges"])
        counters[f"{key}.extra_label"].update(edge["label"] for edge in analysis["unmatched_pred_edges"])
        counters[f"{key}.missing_node_type"].update(str(node["type"]) for node in analysis["unmatched_true_nodes"])
        counters[f"{key}.extra_node_type"].update(str(node["type"]) for node in analysis["unmatched_pred_nodes"])
        for pair in analysis["same_endpoint_label_confusions"]:
            counters[f"{key}.label_confusion"][f"{pair['gold_label']} -> {pair['pred_label']}"] += 1
        for pair in analysis["same_label_scope_attachment"]:
            counters[f"{key}.scope_label"][pair["label"]] += 1
        for pair in analysis["node_type_confusions"]:
            counters[f"{key}.node_type_confusion"][
                f"{pair['gold']['type']} -> {pair['pred']['type']}"
            ] += 1
        for pair in analysis["node_span_mismatches"]:
            counters[f"{key}.node_span_mismatch"][
                f"{pair['gold']['type']}: {pair['gold']['text']} -> {pair['pred']['text']}"
            ] += 1

    combo = "+".join(sorted(analysis["patterns"] + [f"family:{f}" for f in analysis["families"]]))
    counters[f"{prefix}.combo"][combo] += 1
    counters[f"{all_prefix}.combo"][combo] += 1

    example = {
        "split": split,
        "line": case["line"],
        "idx": case["idx"],
        "sentence": case["sentence"],
        "edge_f1": metric.get("edge_f1"),
        "node_f1": metric.get("node_f1"),
        "patterns": analysis["patterns"],
        "families": analysis["families"],
        "same_endpoint_label_confusions": analysis["same_endpoint_label_confusions"][:2],
        "same_label_scope_attachment": analysis["same_label_scope_attachment"][:2],
        "node_type_confusions": analysis["node_type_confusions"][:2],
        "node_span_mismatches": analysis["node_span_mismatches"][:2],
        "unmatched_true_edges": analysis["unmatched_true_edges"][:2],
        "unmatched_pred_edges": analysis["unmatched_pred_edges"][:2],
        "unmatched_true_nodes": analysis["unmatched_true_nodes"][:2],
        "unmatched_pred_nodes": analysis["unmatched_pred_nodes"][:2],
        "gold_roots": analysis["gold_roots"][:2],
        "pred_roots": analysis["pred_roots"][:2],
    }
    for pattern in analysis["patterns"]:
        add_example(examples, f"{layer}.pattern.{pattern}", example, example_limit)
    for family in analysis["families"]:
        add_example(examples, f"{layer}.family.{family}", example, example_limit)
    for pair in analysis["same_endpoint_label_confusions"]:
        add_example(examples, f"{layer}.label_confusion.{pair['gold_label']} -> {pair['pred_label']}", example, example_limit)
    for pair in analysis["same_label_scope_attachment"]:
        add_example(examples, f"{layer}.scope_label.{pair['label']}", example, example_limit)
    for pair in analysis["node_type_confusions"]:
        add_example(
            examples,
            f"{layer}.node_type_confusion.{pair['gold']['type']} -> {pair['pred']['type']}",
            example,
            example_limit,
        )


def metric_totals_from_summary(path: Path, total: int, exact_counts: dict[str, int]) -> dict[str, Any]:
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        return {
            "word_lingua": summary.get("word_lingua_metrics", {}),
            "lingua": summary.get("lingua_metrics", {}),
        }
    return {
        "word_lingua": {"exact_match_count": exact_counts["word_lingua"], "exact_match_rate": exact_counts["word_lingua"] / total if total else 0},
        "lingua": {"exact_match_count": exact_counts["lingua"], "exact_match_rate": exact_counts["lingua"] / total if total else 0},
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    base = args.base
    output_dir = args.output_dir or base / "nonexact_patterns"
    output_dir.mkdir(parents=True, exist_ok=True)

    counters: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_summaries: dict[str, Any] = {}
    all_case_path = output_dir / "nonexact_cases.jsonl"

    with all_case_path.open("w", encoding="utf-8") as all_out:
        for split in SPLITS:
            predictions_path = base / split / "predictions.jsonl"
            split_case_path = output_dir / split / "nonexact_cases.jsonl"
            split_case_path.parent.mkdir(parents=True, exist_ok=True)
            totals = Counter()
            exact_counts = Counter()
            split_nonexact = Counter()
            with split_case_path.open("w", encoding="utf-8") as split_out:
                for row in read_jsonl(predictions_path):
                    totals["total"] += 1
                    case: dict[str, Any] = {
                        "split": split,
                        "idx": row.get("idx"),
                        "line": row.get("line"),
                        "sentence": row.get("sentence"),
                        "word_lingua_exact": None,
                        "lingua_exact": None,
                    }

                    has_nonexact = False
                    for layer, metric_key, gold_key, pred_key in LAYERS:
                        metric = row.get(metric_key)
                        exact = bool(metric and metric.get("exact_same"))
                        case[f"{layer}_exact"] = exact if metric is not None else None
                        if exact:
                            exact_counts[layer] += 1
                            continue
                        has_nonexact = True
                        split_nonexact[layer] += 1
                        case[layer] = classify_metric(metric or {}, row.get(gold_key), row.get(pred_key))
                        summarize_layer(split, layer, metric or {}, case, counters, examples, args.example_limit)

                    if not has_nonexact:
                        continue

                    word_exact = case["word_lingua_exact"]
                    lingua_exact = case["lingua_exact"]
                    if word_exact and not lingua_exact:
                        split_nonexact["word_exact_lingua_nonexact"] += 1
                    if not word_exact and lingua_exact:
                        split_nonexact["word_nonexact_lingua_exact"] += 1
                    if not word_exact and not lingua_exact:
                        split_nonexact["both_nonexact"] += 1

                    line = json.dumps(case, ensure_ascii=False, separators=(",", ":"))
                    split_out.write(line + "\n")
                    all_out.write(line + "\n")

            metric_totals = metric_totals_from_summary(base / split / "summary.json", totals["total"], exact_counts)
            split_summaries[split] = {
                "total": totals["total"],
                "case_jsonl": str(split_case_path),
                "word_lingua": metric_totals["word_lingua"],
                "lingua": metric_totals["lingua"],
                "nonexact": dict(split_nonexact),
            }

    summary = {
        "input_base": str(base),
        "output_dir": str(output_dir),
        "all_cases_jsonl": str(all_case_path),
        "splits": split_summaries,
        "counters": {key: compact_counter(counter, args.top_k) for key, counter in sorted(counters.items())},
        "examples": examples,
    }
    write_json(output_dir / "nonexact_patterns_summary.json", summary)
    write_markdown(output_dir / "nonexact_patterns_report.md", summary, args.top_k)
    return summary


def pct(numer: int, denom: int) -> str:
    if not denom:
        return "0.0%"
    return f"{numer / denom * 100:.1f}%"


def counter_map(summary: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return summary["counters"].get(key, [])


def row_count(rows: list[dict[str, Any]], key: str) -> int:
    for row in rows:
        if row["key"] == key:
            return int(row["count"])
    return 0


def zh_family_name(name: str) -> str:
    return {
        "modification": "Modification 修饰/包装层",
        "predicate_args": "谓词论元角色/范围",
        "expression_list": "Expression/List/公式结构",
        "punctuation": "标点附着",
        "attribute_appos_ref": "Attribute/Apposition/Ref",
        "lower_upper": "Lower/Upper 提升",
        "linearization": "next_word 线性边",
        "discourse": "Discourse",
        "parataxis": "Parataxis",
        "copula": "Copula",
        "vocative": "Vocative",
        "repeat": "Repeat",
        "other": "其他",
    }.get(name, name)


def write_markdown(path: Path, summary: dict[str, Any], top_k: int) -> None:
    lines = [
        "# 当前 V13 train/dev/test non-exact case 形态",
        "",
        f"输入预测目录：`{summary['input_base']}`",
        f"输出目录：`{summary['output_dir']}`",
        "",
        "## 1. 分 split 指标",
        "",
        "| split | total | word exact | word non-exact | lingua exact | lingua non-exact | both non-exact | word exact / lingua non-exact | word non-exact / lingua exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = summary["splits"][split]
        total = item["total"]
        word_exact = int(item["word_lingua"].get("exact_match_count", 0))
        lingua_exact = int(item["lingua"].get("exact_match_count", 0))
        non = item["nonexact"]
        lines.append(
            f"| {split} | {total} | {word_exact} ({pct(word_exact, total)}) | "
            f"{total - word_exact} ({pct(total - word_exact, total)}) | "
            f"{lingua_exact} ({pct(lingua_exact, total)}) | {total - lingua_exact} ({pct(total - lingua_exact, total)}) | "
            f"{non.get('both_nonexact', 0)} | {non.get('word_exact_lingua_nonexact', 0)} | "
            f"{non.get('word_nonexact_lingua_exact', 0)} |"
        )

    for layer in ("word_lingua", "lingua"):
        title = "word-lingua 层" if layer == "word_lingua" else "lingua 层"
        lines.extend(["", f"## 2. {title} 的主形态", ""])
        lines.append("| family | train | dev | test | all |")
        lines.append("|---|---:|---:|---:|---:|")
        all_rows = counter_map(summary, f"all.{layer}.family")
        for row in all_rows[:top_k]:
            family = row["key"]
            counts = [
                row_count(counter_map(summary, f"{split}.{layer}.family"), family)
                for split in SPLITS
            ]
            lines.append(f"| {zh_family_name(family)} | {counts[0]} | {counts[1]} | {counts[2]} | {row['count']} |")

        lines.extend(["", "### 机制标签", ""])
        for row in counter_map(summary, f"all.{layer}.pattern")[:20]:
            lines.append(f"- `{row['key']}`: {row['count']}")

        lines.extend(["", "### 高频缺失/多余边标签", ""])
        lines.append("缺失 gold 边：")
        for row in counter_map(summary, f"all.{layer}.missing_label")[:15]:
            lines.append(f"- `{row['key']}`: {row['count']}")
        lines.append("")
        lines.append("预测多出的边：")
        for row in counter_map(summary, f"all.{layer}.extra_label")[:15]:
            lines.append(f"- `{row['key']}`: {row['count']}")

        sections = [
            ("同端点标签混淆", "label_confusion"),
            ("同标签范围/附着错误", "scope_label"),
            ("节点类型混淆", "node_type_confusion"),
        ]
        for section_title, suffix in sections:
            rows = counter_map(summary, f"all.{layer}.{suffix}")[:15]
            if not rows:
                continue
            lines.extend(["", f"### {section_title}", ""])
            for row in rows:
                lines.append(f"- `{row['key']}`: {row['count']}")

    lines.extend(["", "## 3. 代表例子", ""])
    example_keys = [
        "word_lingua.family.modification",
        "word_lingua.family.predicate_args",
        "word_lingua.family.expression_list",
        "word_lingua.pattern.same_endpoint_label_confusion",
        "word_lingua.pattern.same_label_scope_attachment",
        "word_lingua.pattern.node_type_confusion",
        "word_lingua.pattern.root_mismatch",
        "lingua.family.modification",
        "lingua.family.attribute_appos_ref",
        "lingua.pattern.same_label_scope_attachment",
    ]
    for key in example_keys:
        examples = summary["examples"].get(key) or []
        if not examples:
            continue
        lines.extend(["", f"### `{key}`"])
        for example in examples[:3]:
            lines.append(f"- {example['split']}:{example['line']} `{example['sentence']}`")
            if example.get("same_endpoint_label_confusions"):
                lines.append(
                    "  - label: "
                    + json.dumps(example["same_endpoint_label_confusions"][0], ensure_ascii=False)
                )
            if example.get("same_label_scope_attachment"):
                lines.append(
                    "  - scope: "
                    + json.dumps(example["same_label_scope_attachment"][0], ensure_ascii=False)
                )
            if example.get("node_type_confusions"):
                lines.append(
                    "  - node_type: "
                    + json.dumps(example["node_type_confusions"][0], ensure_ascii=False)
                )
            if example.get("unmatched_true_edges"):
                lines.append("  - missing: " + json.dumps(example["unmatched_true_edges"][0], ensure_ascii=False))
            if example.get("unmatched_pred_edges"):
                lines.append("  - extra: " + json.dumps(example["unmatched_pred_edges"][0], ensure_ascii=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Prediction dump base with train/dev/test subdirectories.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=80)
    parser.add_argument("--example_limit", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    summary = analyze(parse_args())
    print(
        json.dumps(
            {
                "summary": str(Path(summary["output_dir"]) / "nonexact_patterns_summary.json"),
                "report": str(Path(summary["output_dir"]) / "nonexact_patterns_report.md"),
                "all_cases": summary["all_cases_jsonl"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
