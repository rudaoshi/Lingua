#!/usr/bin/env python3
"""Dump full word-lingua and Lingua prediction graphs for an evaluation split."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_DIR = REPO_ROOT / "src"
for path in (str(SCRIPT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from lingua.learn.wordgraph.decoding.inference import InferenceConfig, WordLinguaInference  # noqa: E402
from lingua.learn.wordgraph.modeler.gp2word import lingua2wordlingua  # noqa: E402
from lingua.structure.gpgraph import GPGraph  # noqa: E402
from lingua.utils.metric import graph_match_metric  # noqa: E402


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def graph_data(graph: GPGraph | None) -> dict[str, Any] | None:
    return graph.data() if graph is not None else None


def safe_metric(pred: GPGraph | None, gold: GPGraph | None, edge_match: str) -> dict[str, Any] | None:
    if pred is None or gold is None:
        return None
    try:
        metric = graph_match_metric(pred, gold, edge_match=edge_match, with_spans=True)
        return to_jsonable(metric)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"error": str(exc)}


def load_samples(path: Path, split: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for idx, item in enumerate(read_jsonl(path)):
        gold_lingua_graph = GPGraph.parse(item["lingua"])
        gold_word_lingua_graph = None
        gold_word_lingua_error = None
        try:
            gold_word_lingua_graph, _, _ = lingua2wordlingua(copy.deepcopy(gold_lingua_graph))
        except Exception as exc:
            gold_word_lingua_error = str(exc)

        samples.append(
            {
                "idx": idx,
                "line": idx + 1,
                "split": split,
                "sentence": item["sentence"],
                "gold_lingua_graph": gold_lingua_graph,
                "gold_word_lingua_graph": gold_word_lingua_graph,
                "gold_word_lingua_error": gold_word_lingua_error,
            }
        )
    return samples


def add_metric_counts(prefix: str, totals: Counter[str], metric: dict[str, Any] | None) -> None:
    if not metric or "error" in metric:
        totals[f"{prefix}_metric_errors"] += 1
        return
    totals[f"{prefix}_exact"] += int(bool(metric["exact_same"]))
    totals[f"{prefix}_pred_nodes"] += int(metric["pred_node_num"])
    totals[f"{prefix}_true_nodes"] += int(metric["true_node_num"])
    totals[f"{prefix}_matched_nodes"] += int(metric["matched_node_num"])
    totals[f"{prefix}_pred_edges"] += int(metric["pred_edge_num"])
    totals[f"{prefix}_true_edges"] += int(metric["true_edge_num"])
    totals[f"{prefix}_matched_edges"] += int(metric["matched_edge_num"])


def prf(matched: int, pred: int, true: int) -> dict[str, float]:
    precision = matched / pred if pred else 1.0
    recall = matched / true if true else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def metric_summary(prefix: str, totals: Counter[str]) -> dict[str, Any]:
    total = int(totals["total"])
    node = prf(
        totals[f"{prefix}_matched_nodes"],
        totals[f"{prefix}_pred_nodes"],
        totals[f"{prefix}_true_nodes"],
    )
    edge = prf(
        totals[f"{prefix}_matched_edges"],
        totals[f"{prefix}_pred_edges"],
        totals[f"{prefix}_true_edges"],
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
        "pred_nodes": totals[f"{prefix}_pred_nodes"],
        "true_nodes": totals[f"{prefix}_true_nodes"],
        "matched_nodes": totals[f"{prefix}_matched_nodes"],
        "pred_edges": totals[f"{prefix}_pred_edges"],
        "true_edges": totals[f"{prefix}_true_edges"],
        "matched_edges": totals[f"{prefix}_matched_edges"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = InferenceConfig(
        max_length=args.max_length,
        arc_threshold=args.arc_threshold,
        lam_parent_prob=args.lam_parent_prob,
        lam_sibling_prob=args.lam_sibling_prob,
        lam_pos_prob=args.lam_pos_prob,
        lam_parent_given_child_prob=args.lam_parent_given_child_prob,
        edge_factor_decode_weight=args.edge_factor_decode_weight,
        decoder_mode=args.decoder_mode,
        ilp_max_candidates_per_child=args.ilp_max_candidates_per_child,
        ilp_max_labels_per_pair=args.ilp_max_labels_per_pair,
        ilp_time_limit=args.ilp_time_limit,
        ilp_mip_rel_gap=args.ilp_mip_rel_gap,
        ilp_root_score_weight=args.ilp_root_score_weight,
        ilp_lazy_constraints=args.ilp_lazy_constraints,
        ilp_max_lazy_rounds=args.ilp_max_lazy_rounds,
        hybrid_fallback_policy=args.hybrid_fallback_policy,
        enforce_connected=args.enforce_connected,
        enable_phase4_rewrites=args.enable_phase4_rewrites,
        enforce_single_root=args.enforce_single_root,
        debug=args.debug,
        debug_dir=str(args.debug_dir) if args.debug_dir else None,
    )

    inference = WordLinguaInference.from_pretrained(
        model_dir=str(args.model_dir),
        device=device,
        statistics_path=str(args.statistics) if args.statistics else None,
        config=config,
    )

    samples = load_samples(args.data_path, args.split)
    if args.max_samples:
        samples = samples[: args.max_samples]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    totals: Counter[str] = Counter()

    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for batch_start in tqdm(range(0, len(samples), args.batch_size), desc=f"Dumping {args.split}"):
            batch = samples[batch_start : batch_start + args.batch_size]
            results = inference.parse_batch([sample["sentence"] for sample in batch], debug=args.debug)
            for sample, result in zip(batch, results):
                word_metric = safe_metric(
                    result.word_lingua_graph,
                    sample["gold_word_lingua_graph"],
                    args.edge_match,
                )
                lingua_metric = safe_metric(result.lingua_graph, sample["gold_lingua_graph"], args.edge_match)

                add_metric_counts("word_lingua", totals, word_metric)
                add_metric_counts("lingua", totals, lingua_metric)
                totals["total"] += 1
                totals["lingua_conversion_errors"] += int(result.lingua_graph is None)

                record = {
                    "idx": sample["idx"],
                    "line": sample["line"],
                    "split": sample["split"],
                    "sentence": sample["sentence"],
                    "words": result.words,
                    "gold_word_lingua_error": sample["gold_word_lingua_error"],
                    "lingua_conversion_error": result.lingua_conversion_error,
                    "word_lingua_metric": word_metric,
                    "lingua_metric": lingua_metric,
                    "gold_word_lingua_graph": graph_data(sample["gold_word_lingua_graph"]),
                    "pred_word_lingua_graph": graph_data(result.word_lingua_graph),
                    "gold_lingua_graph": graph_data(sample["gold_lingua_graph"]),
                    "pred_lingua_graph": graph_data(result.lingua_graph),
                    "edges": [
                        {"head": int(head), "dep": int(dep), "label": label}
                        for head, dep, label in result.edges
                    ],
                    "node_types": [str(node_type) for node_type in result.node_types],
                    "child_of_whether": [bool(value) for value in result.child_of_whether],
                    "decoding_info": to_jsonable(result.decoding_info) if args.include_decoding_info else None,
                }
                out.write(json.dumps(to_jsonable(record), ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "split": args.split,
        "total_examples": totals["total"],
        "output_jsonl": str(args.output_jsonl),
        "word_lingua_metrics": metric_summary("word_lingua", totals),
        "lingua_metrics": metric_summary("lingua", totals),
        "lingua_conversion_errors": totals["lingua_conversion_errors"],
        "config": {
            "model_dir": str(args.model_dir),
            "data_path": str(args.data_path),
            "device": device,
            "max_length": args.max_length,
            "arc_threshold": args.arc_threshold,
            "edge_factor_decode_weight": args.edge_factor_decode_weight,
            "decoder_mode": args.decoder_mode,
            "ilp_max_candidates_per_child": args.ilp_max_candidates_per_child,
            "ilp_max_labels_per_pair": args.ilp_max_labels_per_pair,
            "ilp_time_limit": args.ilp_time_limit,
            "ilp_mip_rel_gap": args.ilp_mip_rel_gap,
            "ilp_root_score_weight": args.ilp_root_score_weight,
            "ilp_lazy_constraints": args.ilp_lazy_constraints,
            "ilp_max_lazy_rounds": args.ilp_max_lazy_rounds,
            "hybrid_fallback_policy": args.hybrid_fallback_policy,
            "enforce_connected": args.enforce_connected,
            "enable_phase4_rewrites": args.enable_phase4_rewrites,
            "enforce_single_root": args.enforce_single_root,
            "edge_match": args.edge_match,
            "include_decoding_info": args.include_decoding_info,
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--statistics", type=Path, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--edge_match", choices=["exact", "loose"], default="exact")
    parser.add_argument("--include_decoding_info", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_dir", type=Path, default=None)

    parser.add_argument("--arc_threshold", type=float, default=0.5)
    parser.add_argument("--lam_parent_prob", type=float, default=1.0)
    parser.add_argument("--lam_sibling_prob", type=float, default=1.0)
    parser.add_argument("--lam_pos_prob", type=float, default=1.0)
    parser.add_argument("--lam_parent_given_child_prob", type=float, default=1.0)
    parser.add_argument("--edge_factor_decode_weight", type=float, default=0.0)
    parser.add_argument("--decoder_mode", choices=["hybrid", "ilp", "sdp_graph"], default="ilp")
    parser.add_argument("--ilp_max_candidates_per_child", type=int, default=16)
    parser.add_argument("--ilp_max_labels_per_pair", type=int, default=4)
    parser.add_argument("--ilp_time_limit", type=float, default=30.0)
    parser.add_argument("--ilp_mip_rel_gap", type=float, default=0.001)
    parser.add_argument("--ilp_root_score_weight", type=float, default=1.0)
    parser.add_argument("--ilp_lazy_constraints", dest="ilp_lazy_constraints", action="store_true")
    parser.add_argument("--no_ilp_lazy_constraints", dest="ilp_lazy_constraints", action="store_false")
    parser.set_defaults(ilp_lazy_constraints=True)
    parser.add_argument("--ilp_max_lazy_rounds", type=int, default=4)
    parser.add_argument("--hybrid_fallback_policy", choices=["conversion", "issues"], default="conversion")
    parser.add_argument("--enforce_connected", action="store_true")
    parser.add_argument("--enable_phase4_rewrites", action="store_true")
    parser.add_argument("--enforce_single_root", dest="enforce_single_root", action="store_true")
    parser.add_argument("--no_enforce_single_root", dest="enforce_single_root", action="store_false")
    parser.set_defaults(enforce_single_root=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
