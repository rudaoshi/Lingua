#!/usr/bin/env python3
"""Compare residual modifier wrapper queue against V13 gold/pred diffs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pairs(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            pairs.add((normalize_text(row["left"]), normalize_text(row["right"])))
    return pairs


def node_text(words: list[str], node: dict[str, Any]) -> str:
    spans = node.get("spans")
    if spans == "Modification":
        return "Modification"
    if spans is None:
        return str(node.get("spans"))
    if not isinstance(spans, list):
        return str(spans)
    tokens: list[str] = []
    for span in spans:
        if isinstance(span, list) and len(span) == 2:
            start, end = int(span[0]), int(span[1])
            tokens.extend(words[start : end + 1])
    return " ".join(tokens).strip()


def edge_text(edge: dict[str, Any]) -> str:
    source = edge.get("source") or {}
    target = edge.get("target") or {}
    return (
        f"{source.get('text')}[{source.get('type')}] "
        f"--{edge.get('label')}--> "
        f"{target.get('text')}[{target.get('type')}]"
    )


def edge_relevant(edge: dict[str, Any], terms: set[str]) -> bool:
    source = normalize_text(str((edge.get("source") or {}).get("text") or ""))
    target = normalize_text(str((edge.get("target") or {}).get("text") or ""))
    return source in terms or target in terms


def node_relevant(node: dict[str, Any], terms: set[str]) -> bool:
    return normalize_text(str(node.get("text") or "")) in terms


def format_current_edge(words: list[str], nodes: list[dict[str, Any]], edge: dict[str, Any]) -> str:
    start = int(edge["start"])
    end = int(edge["end"])
    return (
        f"{node_text(words, nodes[start])}[{nodes[start].get('type')}] "
        f"--{edge.get('label')}--> "
        f"{node_text(words, nodes[end])}[{nodes[end].get('type')}]"
    )


def collect_residual_instances(data_dir: Path, pairs: set[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        for idx, item in enumerate(read_jsonl(data_dir / f"{split}.jsonl")):
            words = item["lingua"]["words"]
            nodes = item["lingua"]["graph"]["nodes"]
            edges = item["lingua"]["graph"]["edges"]
            outgoing: dict[int, list[dict[str, Any]]] = defaultdict(list)
            incoming: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for edge in edges:
                outgoing[int(edge["start"])].append(edge)
                incoming[int(edge["end"])].append(edge)

            for mod_idx, node in enumerate(nodes):
                if node.get("spans") != "Modification" or node.get("type") != "ModificationalFunctor":
                    continue
                mod_edges = outgoing.get(mod_idx, [])
                variable_edges = [edge for edge in mod_edges if edge.get("label") == "variable"]
                body_edges = [edge for edge in mod_edges if edge.get("label") == "body"]
                if not variable_edges or not body_edges:
                    continue
                variable_idx = int(variable_edges[0]["end"])
                body_idx = int(body_edges[0]["end"])
                body_text = node_text(words, nodes[body_idx])
                variable_text = node_text(words, nodes[variable_idx])
                pair = (normalize_text(body_text), normalize_text(variable_text))
                if pair not in pairs:
                    continue
                body_outgoing = outgoing.get(body_idx, [])
                if not body_outgoing:
                    continue
                body_out_terms = {
                    normalize_text(node_text(words, nodes[int(edge["end"])]))
                    for edge in body_outgoing
                }
                rows.append(
                    {
                        "split": split,
                        "line": idx + 1,
                        "idx": idx,
                        "pair": f"{pair[0]} {pair[1]}",
                        "body": body_text,
                        "variable": variable_text,
                        "body_type": nodes[body_idx].get("type"),
                        "variable_type": nodes[variable_idx].get("type"),
                        "body_outgoing": "; ".join(format_current_edge(words, nodes, edge) for edge in body_outgoing),
                        "incoming_to_modification": "; ".join(format_current_edge(words, nodes, edge) for edge in incoming.get(mod_idx, [])),
                        "terms": sorted({pair[0], pair[1], *body_out_terms}),
                        "sentence": item["sentence"],
                    }
                )
    return rows


def load_v13_cases(base: Path) -> dict[tuple[str, int], dict[str, Any]]:
    cases: dict[tuple[str, int], dict[str, Any]] = {}
    for split in SPLITS:
        for source_name in ("nonexact_cases_with_confidence.jsonl", "word_exact_lingua_nonexact.jsonl"):
            for case in read_jsonl(base / split / source_name):
                case["_case_source"] = source_name
                cases[(split, int(case["line"]))] = case
    return cases


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    args.report_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(args.queue_tsv)
    instances = collect_residual_instances(args.data_dir, pairs)
    cases = load_v13_cases(args.v13_nonexact_dir)

    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    relevant_missing_label_counts: Counter[str] = Counter()
    relevant_extra_label_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()

    for instance in instances:
        key = (instance["split"], int(instance["line"]))
        case = cases.get(key)
        terms = set(instance["terms"])
        if case is None:
            status = "v13_exact_no_gold_pred_diff"
            missing_edges: list[dict[str, Any]] = []
            extra_edges: list[dict[str, Any]] = []
            missing_nodes: list[dict[str, Any]] = []
            extra_nodes: list[dict[str, Any]] = []
            categories: list[str] = []
            node_confusions: list[dict[str, Any]] = []
            scope: list[dict[str, Any]] = []
            label_confusions: list[dict[str, Any]] = []
        else:
            categories = [str(category) for category in case.get("categories", [])]
            missing_edges = [
                edge for edge in case.get("unmatched_true_edges", [])
                if edge_relevant(edge, terms)
            ]
            extra_edges = [
                edge for edge in case.get("unmatched_pred_edges", [])
                if edge_relevant(edge, terms)
            ]
            missing_nodes = [
                node for node in case.get("unmatched_true_nodes", [])
                if node_relevant(node, terms)
            ]
            extra_nodes = [
                node for node in case.get("unmatched_pred_nodes", [])
                if node_relevant(node, terms)
            ]
            node_confusions = [
                row for row in case.get("node_type_confusions", [])
                if node_relevant(row.get("gold") or {}, terms) or node_relevant(row.get("pred") or {}, terms)
            ]
            scope = [
                row for row in case.get("same_label_scope_attachment", [])
                if edge_relevant(row.get("gold") or {}, terms) or edge_relevant(row.get("pred") or {}, terms)
            ]
            label_confusions = [
                row for row in case.get("same_endpoint_label_confusions", [])
                if edge_relevant(row.get("gold") or {}, terms) or edge_relevant(row.get("pred") or {}, terms)
            ]
            if missing_edges or extra_edges or missing_nodes or extra_nodes or node_confusions or scope or label_confusions:
                status = "v13_nonexact_relevant_diff"
            else:
                status = "v13_nonexact_elsewhere"

        status_counts[status] += 1
        pair_counts[str(instance["pair"])] += 1
        for edge in missing_edges:
            relevant_missing_label_counts[str(edge.get("label"))] += 1
        for edge in extra_edges:
            relevant_extra_label_counts[str(edge.get("label"))] += 1

        rows.append(
            {
                "split": instance["split"],
                "line": instance["line"],
                "pair": instance["pair"],
                "status": status,
                "categories": ",".join(categories),
                "body": instance["body"],
                "variable": instance["variable"],
                "terms": ",".join(instance["terms"]),
                "gold_current_wrapper": (
                    f"Modification(body={instance['body']}[{instance['body_type']}], "
                    f"variable={instance['variable']}[{instance['variable_type']}])"
                ),
                "gold_body_outgoing": instance["body_outgoing"],
                "gold_incoming_to_wrapper": instance["incoming_to_modification"],
                "missing_gold_edges": " | ".join(edge_text(edge) for edge in missing_edges),
                "extra_pred_edges": " | ".join(edge_text(edge) for edge in extra_edges),
                "missing_gold_nodes": json.dumps(missing_nodes, ensure_ascii=False),
                "extra_pred_nodes": json.dumps(extra_nodes, ensure_ascii=False),
                "node_type_confusions": json.dumps(node_confusions, ensure_ascii=False),
                "scope_attachment_diffs": json.dumps(scope[:6], ensure_ascii=False),
                "same_endpoint_label_confusions": json.dumps(label_confusions[:6], ensure_ascii=False),
                "sentence": instance["sentence"],
            }
        )

    tsv_path = args.report_dir / "lexical_simple_modifier_wrapper_gold_pred_diff.tsv"
    fields = [
        "split",
        "line",
        "pair",
        "status",
        "categories",
        "body",
        "variable",
        "terms",
        "gold_current_wrapper",
        "gold_body_outgoing",
        "gold_incoming_to_wrapper",
        "missing_gold_edges",
        "extra_pred_edges",
        "missing_gold_nodes",
        "extra_pred_nodes",
        "node_type_confusions",
        "scope_attachment_diffs",
        "same_endpoint_label_confusions",
        "sentence",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "queue_tsv": str(args.queue_tsv),
        "v13_nonexact_dir": str(args.v13_nonexact_dir),
        "instance_count": len(rows),
        "pair_count": len({row["pair"] for row in rows}),
        "status_counts": dict(status_counts),
        "pair_counts": dict(pair_counts),
        "relevant_missing_label_counts": dict(relevant_missing_label_counts),
        "relevant_extra_label_counts": dict(relevant_extra_label_counts),
        "outputs": {
            "tsv": str(tsv_path),
            "markdown": str(args.report_dir / "lexical_simple_modifier_wrapper_gold_pred_diff_zh.md"),
        },
    }
    (args.report_dir / "lexical_simple_modifier_wrapper_gold_pred_diff_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.report_dir / "lexical_simple_modifier_wrapper_gold_pred_diff_zh.md", summary, rows)
    return summary


def sample_rows(rows: list[dict[str, Any]], status: str, limit: int = 8) -> list[dict[str, Any]]:
    return [row for row in rows if row["status"] == status][:limit]


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 词汇修饰包装节点残留队列的 gold / predicted 差异复查",
        "",
        "## 口径",
        "",
        "这份报告补充的是模型误差视角：把当前归一化后的残留队列，与 V13 评估产出的 `nonexact_cases_with_confidence.jsonl` / `word_exact_lingua_nonexact.jsonl` 对齐。",
        "",
        "- `missing_gold_edges`：gold 有、predicted 没有的边。",
        "- `extra_pred_edges`：predicted 多出来、gold 没有的边。",
        "- `v13_exact_no_gold_pred_diff`：该样本在 V13 诊断文件里不是 non-exact，说明这个残留 gold wrapper 没有造成当时的 gold/pred mismatch。",
        "- `v13_nonexact_elsewhere`：样本是 non-exact，但差异不在这个残留 pair 附近。",
        "- `v13_nonexact_relevant_diff`：样本是 non-exact，且差异命中该 pair、其 `body`、`variable` 或 `body` 的内部子边目标。",
        "",
        "注意：V13 诊断文件来自远端评估目录，评估数据是 V13 训练/评估时的数据快照；当前 corpus 已经额外做过 simple-leaf 归一化。因此这里用于判断模型差异，不把它当作当前 gold 图的完整重评估结果。",
        "",
        "## 总览",
        "",
        f"- 残留 wrapper 实例：`{summary['instance_count']}`",
        f"- 覆盖 pair：`{summary['pair_count']}`",
        "",
        "| 状态 | 数量 |",
        "|---|---:|",
    ]
    for status, count in sorted(summary["status_counts"].items()):
        lines.append(f"| `{status}` | {count} |")
    lines.extend(
        [
            "",
            "相关 missing gold 边 label：",
            "",
            "| label | count |",
            "|---|---:|",
        ]
    )
    for label, count in sorted(summary["relevant_missing_label_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{label}` | {count} |")
    lines.extend(["", "相关 extra predicted 边 label：", "", "| label | count |", "|---|---:|"])
    for label, count in sorted(summary["relevant_extra_label_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{label}` | {count} |")

    for title, status in (
        ("命中 pair 附近的 gold/pred 差异", "v13_nonexact_relevant_diff"),
        ("样本 non-exact 但差异在别处", "v13_nonexact_elsewhere"),
        ("V13 下该样本 exact", "v13_exact_no_gold_pred_diff"),
    ):
        lines.extend(["", f"## {title}", ""])
        selected = sample_rows(rows, status, limit=12)
        if not selected:
            lines.append("无。")
            continue
        lines.append("| sample | pair | gold 当前结构 | gold body 内部边 | missing gold | extra predicted | sentence |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in selected:
            sample = f"{row['split']}:{row['line']}"
            missing = row["missing_gold_edges"] or "-"
            extra = row["extra_pred_edges"] or "-"
            sentence = row["sentence"].replace("|", " ")
            lines.append(
                "| "
                + " | ".join(
                    [
                        sample,
                        f"`{row['pair']}`",
                        f"`{row['gold_current_wrapper']}`",
                        f"`{row['gold_body_outgoing']}`",
                        f"`{missing}`",
                        f"`{extra}`",
                        sentence,
                    ]
                )
                + " |"
            )

    lines.extend(["", "## 输出", "", f"- TSV: `{summary['outputs']['tsv']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--queue_tsv", type=Path, required=True)
    parser.add_argument("--v13_nonexact_dir", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    summary = analyze(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
