#!/usr/bin/env python3
"""Evaluate prediction dumps after simple leaf Modification normalization."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TextIO


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lingua.structure.gpgraph import GPGraph  # noqa: E402
from lingua.utils.metric import graph_match_metric  # noqa: E402
from lingua.utils.normalization import normalize_simple_leaf_modifications  # noqa: E402


METRIC_KEYS = (
    "pred_node_num",
    "true_node_num",
    "matched_node_num",
    "pred_edge_num",
    "true_edge_num",
    "matched_edge_num",
)


def read_jsonl(path: str) -> Iterable[dict[str, Any]]:
    if path == "-":
        yield from _read_jsonl_stream(sys.stdin)
        return

    with Path(path).open("r", encoding="utf-8") as f:
        yield from _read_jsonl_stream(f)


def _read_jsonl_stream(stream: TextIO) -> Iterable[dict[str, Any]]:
    for line in stream:
        if line.strip():
            yield json.loads(line)


def add_metric(prefix: str, totals: Counter[str], metric: dict[str, Any] | None) -> None:
    if metric is None:
        totals[f"{prefix}_metric_errors"] += 1
        return
    totals[f"{prefix}_exact"] += int(bool(metric["exact_same"]))
    for key in METRIC_KEYS:
        totals[f"{prefix}_{key}"] += int(metric[key])


def prf(matched: int, pred: int, true: int) -> dict[str, float]:
    precision = matched / pred if pred else 1.0
    recall = matched / true if true else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize(prefix: str, totals: Counter[str], total: int) -> dict[str, Any]:
    node = prf(
        totals[f"{prefix}_matched_node_num"],
        totals[f"{prefix}_pred_node_num"],
        totals[f"{prefix}_true_node_num"],
    )
    edge = prf(
        totals[f"{prefix}_matched_edge_num"],
        totals[f"{prefix}_pred_edge_num"],
        totals[f"{prefix}_true_edge_num"],
    )
    return {
        "exact_match_count": totals[f"{prefix}_exact"],
        "exact_match_rate": totals[f"{prefix}_exact"] / total if total else 0.0,
        "metric_errors": totals[f"{prefix}_metric_errors"],
        "node_precision": node["precision"],
        "node_recall": node["recall"],
        "node_f1": node["f1"],
        "edge_precision": edge["precision"],
        "edge_recall": edge["recall"],
        "edge_f1": edge["f1"],
        "pred_nodes": totals[f"{prefix}_pred_node_num"],
        "true_nodes": totals[f"{prefix}_true_node_num"],
        "matched_nodes": totals[f"{prefix}_matched_node_num"],
        "pred_edges": totals[f"{prefix}_pred_edge_num"],
        "true_edges": totals[f"{prefix}_true_edge_num"],
        "matched_edges": totals[f"{prefix}_matched_edge_num"],
    }


def safe_metric(pred_data: dict[str, Any] | None, gold_data: dict[str, Any] | None, *, with_type: bool) -> dict[str, Any] | None:
    if pred_data is None or gold_data is None:
        return None
    pred_graph = GPGraph.parse(pred_data)
    gold_graph = GPGraph.parse(gold_data)
    return graph_match_metric(pred_graph, gold_graph, edge_match="exact", with_type=with_type, with_spans=True)


def graph_fields(graph_kind: str) -> tuple[str, str]:
    if graph_kind == "lingua":
        return "gold_lingua_graph", "pred_lingua_graph"
    if graph_kind == "word_lingua":
        return "gold_word_lingua_graph", "pred_word_lingua_graph"
    raise ValueError(f"unsupported graph kind: {graph_kind}")


def evaluate_stream(split: str, prediction_jsonl: str, graph_kind: str) -> dict[str, Any]:
    gold_field, pred_field = graph_fields(graph_kind)
    totals: Counter[str] = Counter()

    recovered_exact_typed: list[int] = []
    recovered_exact_no_type: list[int] = []
    lost_exact_typed: list[int] = []
    lost_exact_no_type: list[int] = []

    for record in read_jsonl(prediction_jsonl):
        totals["total"] += 1
        line = int(record.get("line") or record.get("idx", totals["total"] - 1) + 1)
        gold = record.get(gold_field)
        pred = record.get(pred_field)

        original_typed = safe_metric(pred, gold, with_type=True)
        original_no_type = safe_metric(pred, gold, with_type=False)
        add_metric("original_typed", totals, original_typed)
        add_metric("original_no_type", totals, original_no_type)

        normalized_gold = normalized_pred = None
        if gold is not None:
            normalized_gold, gold_stats = normalize_simple_leaf_modifications(gold)
            totals["gold_flattened_simple_leaf_modifications"] += gold_stats.get(
                "flattened_simple_leaf_modifications", 0
            )
        if pred is not None:
            normalized_pred, pred_stats = normalize_simple_leaf_modifications(pred)
            totals["pred_flattened_simple_leaf_modifications"] += pred_stats.get(
                "flattened_simple_leaf_modifications", 0
            )

        normalized_typed = safe_metric(normalized_pred, normalized_gold, with_type=True)
        normalized_no_type = safe_metric(normalized_pred, normalized_gold, with_type=False)
        add_metric("normalized_typed", totals, normalized_typed)
        add_metric("normalized_no_type", totals, normalized_no_type)

        if original_typed and normalized_typed:
            if not original_typed["exact_same"] and normalized_typed["exact_same"]:
                recovered_exact_typed.append(line)
            elif original_typed["exact_same"] and not normalized_typed["exact_same"]:
                lost_exact_typed.append(line)
        if original_no_type and normalized_no_type:
            if not original_no_type["exact_same"] and normalized_no_type["exact_same"]:
                recovered_exact_no_type.append(line)
            elif original_no_type["exact_same"] and not normalized_no_type["exact_same"]:
                lost_exact_no_type.append(line)

    total = totals["total"]
    return {
        "split": split,
        "prediction_jsonl": prediction_jsonl,
        "graph_kind": graph_kind,
        "total_examples": total,
        "normalization": {
            "rule": "flatten Modification(variable=H, body=B) to B --as:modification--> H only when B is a single leaf node",
            "gold_flattened_simple_leaf_modifications": totals["gold_flattened_simple_leaf_modifications"],
            "pred_flattened_simple_leaf_modifications": totals["pred_flattened_simple_leaf_modifications"],
        },
        "typed": {
            "original": summarize("original_typed", totals, total),
            "normalized": summarize("normalized_typed", totals, total),
            "exact_recovered_count": len(recovered_exact_typed),
            "exact_lost_count": len(lost_exact_typed),
            "exact_recovered_lines": recovered_exact_typed[:200],
            "exact_lost_lines": lost_exact_typed[:200],
        },
        "ignore_node_type": {
            "original": summarize("original_no_type", totals, total),
            "normalized": summarize("normalized_no_type", totals, total),
            "exact_recovered_count": len(recovered_exact_no_type),
            "exact_lost_count": len(lost_exact_no_type),
            "exact_recovered_lines": recovered_exact_no_type[:200],
            "exact_lost_lines": lost_exact_no_type[:200],
        },
    }


def parse_split_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SPLIT=PATH")
    split, path = value.split("=", 1)
    if not split:
        raise argparse.ArgumentTypeError("split name is empty")
    if not path:
        raise argparse.ArgumentTypeError("path is empty")
    return split, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction_jsonl",
        action="append",
        type=parse_split_path,
        required=True,
        help="Prediction dump as SPLIT=PATH. Use SPLIT=- to read JSONL from stdin.",
    )
    parser.add_argument("--graph_kind", choices=["lingua", "word_lingua"], default="lingua")
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [
        evaluate_stream(split, path, args.graph_kind)
        for split, path in args.prediction_jsonl
    ]
    output = {
        "graph_kind": args.graph_kind,
        "splits": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
