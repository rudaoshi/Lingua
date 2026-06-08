#!/usr/bin/env python3
"""Evaluate word-lingua and Lingua graph metrics while ignoring node types."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for idx, item in enumerate(read_jsonl(path)):
        lingua_graph = GPGraph.parse(item["lingua"])
        word_graph, _, _ = lingua2wordlingua(copy.deepcopy(lingua_graph))
        samples.append(
            {
                "idx": idx,
                "sentence": item["sentence"],
                "gold_lingua_graph": lingua_graph,
                "gold_word_graph": word_graph,
            }
        )
    return samples


def add_metric_totals(prefix: str, totals: Counter[str], metric: dict[str, Any]) -> None:
    totals[f"{prefix}_exact"] += int(metric["exact_same"])
    totals[f"{prefix}_pred_nodes"] += int(metric["pred_node_num"])
    totals[f"{prefix}_true_nodes"] += int(metric["true_node_num"])
    totals[f"{prefix}_matched_nodes"] += int(metric["matched_node_num"])
    totals[f"{prefix}_pred_edges"] += int(metric["pred_edge_num"])
    totals[f"{prefix}_true_edges"] += int(metric["true_edge_num"])
    totals[f"{prefix}_matched_edges"] += int(metric["matched_edge_num"])


def precision_recall_f1(matched: int, pred: int, true: int) -> dict[str, float]:
    precision = matched / pred if pred else 1.0
    recall = matched / true if true else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def build_metrics(prefix: str, totals: Counter[str]) -> dict[str, Any]:
    node = precision_recall_f1(
        totals[f"{prefix}_matched_nodes"],
        totals[f"{prefix}_pred_nodes"],
        totals[f"{prefix}_true_nodes"],
    )
    edge = precision_recall_f1(
        totals[f"{prefix}_matched_edges"],
        totals[f"{prefix}_pred_edges"],
        totals[f"{prefix}_true_edges"],
    )
    total = totals["total"]
    return {
        "exact_match_count": totals[f"{prefix}_exact"],
        "exact_match_rate": totals[f"{prefix}_exact"] / total if total else 0.0,
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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = InferenceConfig(
        max_length=args.max_length,
        arc_threshold=args.arc_threshold,
        edge_factor_decode_weight=args.edge_factor_decode_weight,
        decoder_mode=args.decoder_mode,
        ilp_max_candidates_per_child=args.ilp_max_candidates_per_child,
        ilp_max_labels_per_pair=args.ilp_max_labels_per_pair,
        ilp_time_limit=args.ilp_time_limit,
        ilp_mip_rel_gap=args.ilp_mip_rel_gap,
        ilp_root_score_weight=args.ilp_root_score_weight,
        enforce_connected=args.enforce_connected,
        enable_phase4_rewrites=args.enable_phase4_rewrites,
        enforce_single_root=args.enforce_single_root,
    )
    inference = WordLinguaInference.from_pretrained(
        model_dir=args.model_dir,
        device=device,
        statistics_path=args.statistics,
        config=config,
    )

    samples = load_samples(Path(args.data_path))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    totals: Counter[str] = Counter()
    conversion_errors: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(samples), args.batch_size), desc=f"ignore-type {args.split}"):
        batch = samples[start : start + args.batch_size]
        results = inference.parse_batch([sample["sentence"] for sample in batch])
        for sample, result in zip(batch, results):
            totals["total"] += 1

            word_metric = graph_match_metric(
                result.word_lingua_graph,
                sample["gold_word_graph"],
                edge_match=args.edge_match,
                with_type=False,
                with_spans=True,
            )
            add_metric_totals("word_lingua", totals, word_metric)

            if result.lingua_graph is None:
                totals["lingua_conversion_error"] += 1
                conversion_errors.append(
                    {
                        "idx": sample["idx"],
                        "line": sample["idx"] + 1,
                        "sentence": sample["sentence"],
                        "error": result.lingua_conversion_error or "pred_lingua_graph is None",
                    }
                )
                continue

            lingua_metric = graph_match_metric(
                result.lingua_graph,
                sample["gold_lingua_graph"],
                edge_match=args.edge_match,
                with_type=False,
                with_spans=True,
            )
            add_metric_totals("lingua", totals, lingua_metric)

    valid_lingua_total = totals["total"] - totals["lingua_conversion_error"]
    word_metrics = build_metrics("word_lingua", totals)
    lingua_metrics = build_metrics("lingua", totals)
    if valid_lingua_total != totals["total"]:
        lingua_metrics["exact_match_rate_valid_lingua"] = (
            totals["lingua_exact"] / valid_lingua_total if valid_lingua_total else 0.0
        )

    output = {
        "split": args.split,
        "data_path": args.data_path,
        "model_dir": args.model_dir,
        "total_examples": totals["total"],
        "lingua_conversion_errors": totals["lingua_conversion_error"],
        "metric_options": {
            "edge_match": args.edge_match,
            "with_type": False,
            "with_spans": True,
        },
        "decode_config": {
            "decoder_mode": args.decoder_mode,
            "edge_factor_decode_weight": args.edge_factor_decode_weight,
            "ilp_max_candidates_per_child": args.ilp_max_candidates_per_child,
            "ilp_max_labels_per_pair": args.ilp_max_labels_per_pair,
            "ilp_time_limit": args.ilp_time_limit,
            "ilp_mip_rel_gap": args.ilp_mip_rel_gap,
            "ilp_root_score_weight": args.ilp_root_score_weight,
            "enforce_connected": args.enforce_connected,
            "enable_phase4_rewrites": args.enable_phase4_rewrites,
            "enforce_single_root": args.enforce_single_root,
        },
        "word_lingua_metrics": word_metrics,
        "lingua_metrics": lingua_metrics,
        "conversion_errors": conversion_errors,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--statistics", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--arc_threshold", type=float, default=0.5)
    parser.add_argument("--edge_factor_decode_weight", type=float, default=0.2)
    parser.add_argument("--decoder_mode", choices=["ilp", "sdp_graph"], default="ilp")
    parser.add_argument("--ilp_max_candidates_per_child", type=int, default=16)
    parser.add_argument("--ilp_max_labels_per_pair", type=int, default=4)
    parser.add_argument("--ilp_time_limit", type=float, default=20.0)
    parser.add_argument("--ilp_mip_rel_gap", type=float, default=0.001)
    parser.add_argument("--ilp_root_score_weight", type=float, default=1.0)
    parser.add_argument("--enforce_connected", action="store_true")
    parser.add_argument("--enable_phase4_rewrites", action="store_true")
    parser.add_argument("--enforce_single_root", dest="enforce_single_root", action="store_true")
    parser.add_argument("--no_enforce_single_root", dest="enforce_single_root", action="store_false")
    parser.set_defaults(enforce_single_root=True)
    parser.add_argument("--edge_match", choices=["exact", "loose"], default="exact")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    output = evaluate(parse_args())
    print(
        json.dumps(
            {
                "split": output["split"],
                "total_examples": output["total_examples"],
                "lingua_conversion_errors": output["lingua_conversion_errors"],
                "word_lingua": {
                    key: output["word_lingua_metrics"][key]
                    for key in ("exact_match_count", "exact_match_rate", "node_f1", "edge_f1")
                },
                "lingua": {
                    key: output["lingua_metrics"][key]
                    for key in ("exact_match_count", "exact_match_rate", "node_f1", "edge_f1")
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
