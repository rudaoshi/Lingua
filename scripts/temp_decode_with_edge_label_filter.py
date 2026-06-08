#!/usr/bin/env python3
"""Temporary evaluation with a decoder-time edge-label allowlist.

This script keeps the trained model bundle unchanged, but masks relation logits
for labels that are not present in a supplied/current word-lingua label set
before `GreedyPatternDecoder.decode()` runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import jsonlines
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lingua.learn.wordgraph.decoding.inference import (  # noqa: E402
    InferenceConfig,
    WordLinguaInference,
)
from lingua.learn.wordgraph.modeler.gp2word import lingua2wordlingua  # noqa: E402
from lingua.learn.wordgraph.modeler.word2gp import wordlingua2lingua  # noqa: E402
from lingua.structure.gpgraph import GPGraph  # noqa: E402
from lingua.utils.metric import graph_match_metric  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def edge_f1_from_counts(matched_edge_num: int, pred_edge_num: int, true_edge_num: int) -> float:
    if pred_edge_num + true_edge_num == 0:
        return 1.0
    return 2 * matched_edge_num / (pred_edge_num + true_edge_num)


def edge_label_counts_from_lingua_jsonl(data_path: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    conversion_errors = 0
    with jsonlines.open(data_path, "r") as reader:
        for idx, item in enumerate(reader):
            try:
                lingua_graph = GPGraph.parse(item["lingua"])
                word_graph, _, _ = lingua2wordlingua(copy.deepcopy(lingua_graph))
                for _, edge, _ in word_graph.edges():
                    counts[str(edge.label)] += 1
            except Exception as exc:
                conversion_errors += 1
                logger.warning("Failed to derive labels for sample %s: %s", idx, exc)
    if conversion_errors:
        logger.warning("Label derivation conversion errors: %d", conversion_errors)
    if not counts:
        raise ValueError(f"No edge labels derived from {data_path}")
    return counts


def load_json_label_set(path: str) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(key) for key in data.keys()}
    if isinstance(data, list):
        return {str(item) for item in data}
    raise ValueError(f"Unsupported label JSON shape in {path}: {type(data).__name__}")


def load_allowed_edge_labels(args: argparse.Namespace) -> tuple[set[str], dict[str, Any]]:
    sources: list[str] = []
    counts: Counter[str] = Counter()
    allowed: set[str] = set()

    if args.allowed_label2id:
        allowed.update(load_json_label_set(args.allowed_label2id))
        sources.append(f"label2id:{args.allowed_label2id}")

    if args.allowed_edge_types:
        allowed.update(load_json_label_set(args.allowed_edge_types))
        sources.append(f"edge_types:{args.allowed_edge_types}")

    if args.allowed_labels_data_path:
        counts = edge_label_counts_from_lingua_jsonl(args.allowed_labels_data_path)
        allowed.update(counts.keys())
        sources.append(f"data:{args.allowed_labels_data_path}")

    if not allowed:
        raise ValueError(
            "No allowed labels were loaded. Pass --allowed_labels_data_path, "
            "--allowed_label2id, or --allowed_edge_types."
        )

    meta: dict[str, Any] = {
        "sources": sources,
        "count": len(allowed),
        "labels": sorted(allowed),
    }
    if counts:
        meta["label_counts"] = dict(counts.most_common())
    return allowed, meta


def attach_decoder_edge_label_filter(
    inference: WordLinguaInference,
    allowed_labels: Iterable[str],
    *,
    masked_logit: float = -1e9,
) -> dict[str, Any]:
    """Mask disallowed relation-label logits immediately before decoder.decode()."""
    allowed_set = {str(label) for label in allowed_labels}
    model_labels = {int(idx): str(label) for idx, label in inference.id2label.items()}
    disallowed_ids = sorted(idx for idx, label in model_labels.items() if label not in allowed_set)
    disallowed_labels = [model_labels[idx] for idx in disallowed_ids]
    allowed_missing_from_model = sorted(allowed_set - set(model_labels.values()))

    original_decode = inference.decoder.decode

    def decode_with_edge_label_filter(*decode_args: Any, **decode_kwargs: Any) -> Any:
        if "rel_logits" in decode_kwargs:
            rel_logits = np.array(decode_kwargs["rel_logits"], copy=True)
            if disallowed_ids:
                rel_logits[..., disallowed_ids] = masked_logit
            decode_kwargs = dict(decode_kwargs)
            decode_kwargs["rel_logits"] = rel_logits
            graph, info = original_decode(*decode_args, **decode_kwargs)
        else:
            # Bound method signature:
            # decode(words, arc_logits, rel_logits, node_type_preds, ...)
            if len(decode_args) < 3:
                raise TypeError("decode() rel_logits argument was not found")
            decode_args = list(decode_args)
            rel_logits = np.array(decode_args[2], copy=True)
            if disallowed_ids:
                rel_logits[..., disallowed_ids] = masked_logit
            decode_args[2] = rel_logits
            graph, info = original_decode(*decode_args, **decode_kwargs)

        if info is None:
            info = {}
        info["edge_label_filter"] = {
            "allowed_label_count": len(allowed_set),
            "model_label_count": len(model_labels),
            "disallowed_model_label_count": len(disallowed_ids),
            "disallowed_model_labels": disallowed_labels,
            "allowed_labels_missing_from_model": allowed_missing_from_model,
            "masked_logit": masked_logit,
        }
        return graph, info

    inference.decoder.decode = decode_with_edge_label_filter  # type: ignore[method-assign]

    return {
        "allowed_label_count": len(allowed_set),
        "model_label_count": len(model_labels),
        "disallowed_model_label_count": len(disallowed_ids),
        "disallowed_model_labels": disallowed_labels,
        "allowed_labels_missing_from_model": allowed_missing_from_model,
        "masked_logit": masked_logit,
    }


def load_samples(data_path: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with jsonlines.open(data_path, "r") as reader:
        for idx, item in enumerate(reader):
            sentence = item["sentence"]
            gold_lingua_graph = GPGraph.parse(item["lingua"])
            gold_word_lingua_graph = None
            try:
                gold_word_lingua_graph, _, _ = lingua2wordlingua(copy.deepcopy(gold_lingua_graph))
            except Exception as exc:
                logger.warning("Failed to convert gold lingua to word-lingua %s: %s", idx, exc)
            samples.append(
                {
                    "idx": idx,
                    "sentence": sentence,
                    "gold_lingua_graph": gold_lingua_graph,
                    "gold_word_lingua_graph": gold_word_lingua_graph,
                }
            )
    logger.info("Loaded %d samples from %s", len(samples), data_path)
    return samples


def edge_label_counter(graph: GPGraph | None) -> Counter[str]:
    counts: Counter[str] = Counter()
    if graph is None:
        return counts
    for _, edge, _ in graph.edges():
        counts[str(edge.label)] += 1
    return counts


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    allowed_labels, allowed_meta = load_allowed_edge_labels(args)
    logger.info("Loaded %d allowed edge labels from %s", len(allowed_labels), allowed_meta["sources"])

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info("Using device: %s", device)

    config = InferenceConfig(
        max_length=args.max_length,
        arc_threshold=args.arc_threshold,
        lam_parent_prob=args.lam_parent_prob,
        lam_sibling_prob=args.lam_sibling_prob,
        lam_pos_prob=args.lam_pos_prob,
        lam_parent_given_child_prob=args.lam_parent_given_child_prob,
        edge_factor_decode_weight=args.edge_factor_decode_weight,
        debug=args.debug,
        debug_dir=args.debug_dir,
    )
    inference = WordLinguaInference.from_pretrained(
        model_dir=args.model_dir,
        device=device,
        statistics_path=args.statistics,
        config=config,
    )
    filter_meta = attach_decoder_edge_label_filter(
        inference,
        allowed_labels,
        masked_logit=args.masked_logit,
    )
    logger.info("Decoder edge-label filter: %s", filter_meta)

    samples = load_samples(args.data_path)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        logger.info("Evaluating on %d samples due to --max_samples", len(samples))

    total_count = 0
    word_exact_match_count = 0
    total_pred_nodes = 0
    total_true_nodes = 0
    total_matched_nodes = 0
    total_pred_edges = 0
    total_true_edges = 0
    total_matched_edges = 0

    lingua_exact_match_count = 0
    lingua_conversion_errors = 0
    lingua_total_pred_nodes = 0
    lingua_total_true_nodes = 0
    lingua_total_matched_nodes = 0
    lingua_total_pred_edges = 0
    lingua_total_true_edges = 0
    lingua_total_matched_edges = 0

    pred_edge_label_counts: Counter[str] = Counter()
    pred_disallowed_label_counts: Counter[str] = Counter()
    examples_by_disallowed_label: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for batch_start in tqdm(range(0, len(samples), args.batch_size), desc="Evaluating filtered decode"):
        batch_samples = samples[batch_start : batch_start + args.batch_size]
        batch_results = inference.parse_batch([sample["sentence"] for sample in batch_samples], debug=args.debug)

        for sample, result in zip(batch_samples, batch_results):
            example_idx = sample["idx"]
            gold_word_lingua_graph = sample["gold_word_lingua_graph"]
            gold_lingua_graph = sample["gold_lingua_graph"]
            pred_word_lingua_graph = result.word_lingua_graph
            pred_lingua_graph = result.lingua_graph

            if pred_word_lingua_graph is None:
                logger.warning("No prediction for sample %s: %s", example_idx, result.lingua_conversion_error)
                continue

            pred_labels = edge_label_counter(pred_word_lingua_graph)
            pred_edge_label_counts.update(pred_labels)
            for label, count in pred_labels.items():
                if label not in allowed_labels:
                    pred_disallowed_label_counts[label] += count
                    if len(examples_by_disallowed_label[label]) < 5:
                        examples_by_disallowed_label[label].append(
                            {
                                "idx": example_idx,
                                "count": count,
                                "sentence": sample["sentence"],
                            }
                        )

            try:
                if pred_lingua_graph is not None:
                    lingua_metric = graph_match_metric(
                        pred_lingua_graph,
                        gold_lingua_graph,
                        edge_match=args.edge_match,
                        with_spans=True,
                    )
                    lingua_total_pred_nodes += lingua_metric["pred_node_num"]
                    lingua_total_true_nodes += lingua_metric["true_node_num"]
                    lingua_total_matched_nodes += lingua_metric["matched_node_num"]
                    lingua_total_pred_edges += lingua_metric["pred_edge_num"]
                    lingua_total_true_edges += lingua_metric["true_edge_num"]
                    lingua_total_matched_edges += lingua_metric["matched_edge_num"]
                    if lingua_metric["exact_same"]:
                        lingua_exact_match_count += 1
                else:
                    lingua_conversion_errors += 1
            except Exception as exc:
                lingua_conversion_errors += 1
                logger.debug("Error computing lingua metric for sample %s: %s", example_idx, exc)

            if gold_word_lingua_graph is not None:
                metric = graph_match_metric(
                    pred_word_lingua_graph,
                    gold_word_lingua_graph,
                    edge_match=args.edge_match,
                    with_spans=True,
                )
                total_pred_nodes += metric["pred_node_num"]
                total_true_nodes += metric["true_node_num"]
                total_matched_nodes += metric["matched_node_num"]
                total_pred_edges += metric["pred_edge_num"]
                total_true_edges += metric["true_edge_num"]
                total_matched_edges += metric["matched_edge_num"]
                if metric["exact_same"]:
                    word_exact_match_count += 1

            total_count += 1

    valid_lingua_count = total_count - lingua_conversion_errors

    word_node_precision = total_matched_nodes / (total_pred_nodes + 1e-8)
    word_node_recall = total_matched_nodes / (total_true_nodes + 1e-8)
    word_node_f1 = 2 * word_node_precision * word_node_recall / (word_node_precision + word_node_recall + 1e-8)
    word_edge_precision = total_matched_edges / (total_pred_edges + 1e-8)
    word_edge_recall = total_matched_edges / (total_true_edges + 1e-8)
    word_edge_f1 = 2 * word_edge_precision * word_edge_recall / (word_edge_precision + word_edge_recall + 1e-8)

    lingua_node_precision = lingua_total_matched_nodes / (lingua_total_pred_nodes + 1e-8)
    lingua_node_recall = lingua_total_matched_nodes / (lingua_total_true_nodes + 1e-8)
    lingua_node_f1 = 2 * lingua_node_precision * lingua_node_recall / (
        lingua_node_precision + lingua_node_recall + 1e-8
    )
    lingua_edge_precision = lingua_total_matched_edges / (lingua_total_pred_edges + 1e-8)
    lingua_edge_recall = lingua_total_matched_edges / (lingua_total_true_edges + 1e-8)
    lingua_edge_f1 = 2 * lingua_edge_precision * lingua_edge_recall / (
        lingua_edge_precision + lingua_edge_recall + 1e-8
    )

    results = {
        "total_examples": total_count,
        "word_lingua_metrics": {
            "exact_match_count": word_exact_match_count,
            "exact_match_rate": word_exact_match_count / (total_count + 1e-8),
            "node_precision": word_node_precision,
            "node_recall": word_node_recall,
            "node_f1": word_node_f1,
            "edge_precision": word_edge_precision,
            "edge_recall": word_edge_recall,
            "edge_f1": word_edge_f1,
        },
        "lingua_metrics": {
            "conversion_errors": lingua_conversion_errors,
            "exact_match_count": lingua_exact_match_count,
            "exact_match_rate": lingua_exact_match_count / (valid_lingua_count + 1e-8),
            "node_precision": lingua_node_precision,
            "node_recall": lingua_node_recall,
            "node_f1": lingua_node_f1,
            "edge_precision": lingua_edge_precision,
            "edge_recall": lingua_edge_recall,
            "edge_f1": lingua_edge_f1,
        },
        "edge_label_filter": filter_meta,
        "allowed_edge_labels": allowed_meta,
        "prediction_label_audit": {
            "predicted_label_counts": dict(pred_edge_label_counts.most_common()),
            "predicted_disallowed_label_counts": dict(pred_disallowed_label_counts.most_common()),
            "examples_by_predicted_disallowed_label": examples_by_disallowed_label,
        },
        "config": {
            "model_dir": args.model_dir,
            "data_path": args.data_path,
            "device": device,
            "max_length": args.max_length,
            "arc_threshold": args.arc_threshold,
            "edge_factor_decode_weight": args.edge_factor_decode_weight,
            "edge_match": args.edge_match,
            "masked_logit": args.masked_logit,
        },
    }

    logger.info("Total examples evaluated: %d", total_count)
    logger.info(
        "Word-lingua exact: %d (%.2f%%), node F1 %.4f, edge F1 %.4f",
        word_exact_match_count,
        results["word_lingua_metrics"]["exact_match_rate"] * 100,
        word_node_f1,
        word_edge_f1,
    )
    logger.info(
        "Lingua exact: %d (%.2f%%), node F1 %.4f, edge F1 %.4f, conversion errors %d",
        lingua_exact_match_count,
        results["lingua_metrics"]["exact_match_rate"] * 100,
        lingua_node_f1,
        lingua_edge_f1,
        lingua_conversion_errors,
    )
    logger.info("Predicted disallowed labels: %s", dict(pred_disallowed_label_counts.most_common()))

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Results saved to %s", output_path)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporary decoder-time edge-label filtering evaluation")
    parser.add_argument("--data_path", required=True, help="JSONL data to evaluate")
    parser.add_argument("--model_dir", required=True, help="Pretrained word-lingua model bundle")
    parser.add_argument("--output_path", default=None, help="Where to write metrics JSON")
    parser.add_argument("--allowed_labels_data_path", default=None, help="Derive allowed edge labels from this data")
    parser.add_argument("--allowed_label2id", default=None, help="JSON label2id whose keys are allowed labels")
    parser.add_argument("--allowed_edge_types", default=None, help="JSON list of allowed edge labels")
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--statistics", default=None, help="Optional wordlingua_statistics.json")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--arc_threshold", type=float, default=0.5)
    parser.add_argument("--lam_parent_prob", type=float, default=1.0)
    parser.add_argument("--lam_sibling_prob", type=float, default=1.0)
    parser.add_argument("--lam_pos_prob", type=float, default=1.0)
    parser.add_argument("--lam_parent_given_child_prob", type=float, default=1.0)
    parser.add_argument("--edge_factor_decode_weight", type=float, default=0.0)
    parser.add_argument("--edge_match", choices=["exact", "loose"], default="exact")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--masked_logit", type=float, default=-1e9)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_dir", default=None)
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
