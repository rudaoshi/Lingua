#!/usr/bin/env python3
"""Analyze train non-exact matches for filtered word-lingua decoding.

This is an analysis-only helper. It reuses the temporary decoder-time edge
label allowlist, runs inference, and writes per-case + aggregate diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jsonlines
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
SRC_DIR = REPO_ROOT / "src"
for path in (str(SCRIPT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from lingua.learn.wordgraph.decoding.inference import (  # noqa: E402
    InferenceConfig,
    WordLinguaInference,
)
from lingua.learn.wordgraph.modeler.gp2word import lingua2wordlingua  # noqa: E402
from lingua.structure.gpgraph import GPGraph  # noqa: E402
from lingua.utils.metric import graph_match_metric  # noqa: E402
from temp_decode_with_edge_label_filter import (  # noqa: E402
    attach_decoder_edge_label_filter,
    load_allowed_edge_labels,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_value(val) for key, val in value.items()}
    return str(value)


def node_info_json(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": str(info.get("text")),
        "type": None if info.get("type") is None else str(info.get("type")),
        "spans": normalize_value(info.get("spans")),
    }


def node_key(info: dict[str, Any]) -> tuple[str, str | None, str]:
    info_json = node_info_json(info)
    return (
        info_json["text"],
        info_json["type"],
        json.dumps(info_json["spans"], ensure_ascii=False, sort_keys=True),
    )


def edge_json(edge: tuple[dict[str, Any], Any, dict[str, Any]]) -> dict[str, Any]:
    src, label, tgt = edge
    return {
        "source": node_info_json(src),
        "label": str(label),
        "target": node_info_json(tgt),
    }


def edge_key(edge: tuple[dict[str, Any], Any, dict[str, Any]]) -> tuple[Any, str, Any]:
    src, label, tgt = edge
    return (node_key(src), str(label), node_key(tgt))


def endpoint_key(edge: tuple[dict[str, Any], Any, dict[str, Any]]) -> tuple[Any, Any]:
    src, _, tgt = edge
    return (node_key(src), node_key(tgt))


def edge_type_key(edge: tuple[dict[str, Any], Any, dict[str, Any]]) -> str:
    src, label, tgt = edge
    src_type = node_info_json(src)["type"]
    tgt_type = node_info_json(tgt)["type"]
    return f"{src_type or 'None'} --{str(label)}--> {tgt_type or 'None'}"


def label_family(label: str) -> str:
    parts = label.split("+")
    if any("punctuation" in part for part in parts):
        return "punctuation"
    if any(part in {"@expression", "@list", "func.arg"} for part in parts):
        return "expression_list"
    if any(part in {"@modification_of", "as:modification", "variable", "body"} for part in parts):
        return "modification"
    if any(part in {"@attribute_of", "@appositive_of", "@ref_of", "as:attribute", "as:appositive", "as:ref"} for part in parts):
        return "attribute_appos_ref"
    if any(part.startswith("pred.arg.") for part in parts):
        return "predicate_args"
    if any(part in {"@lower", "@upper"} for part in parts):
        return "lower_upper"
    if any(part in {"as:discourse", "@discourse_of"} for part in parts):
        return "discourse"
    if any(part == "next_word" for part in parts):
        return "linearization"
    return "other"


def node_type_family(edge: tuple[dict[str, Any], Any, dict[str, Any]]) -> set[str]:
    src, _, tgt = edge
    types = {str(src.get("type")), str(tgt.get("type"))}
    families: set[str] = set()
    if any("PunctuationalConstant" in t for t in types):
        families.add("punctuation")
    if any(t in {"ExpressionFunctor", "ListFunctor", "ExpressionOperator", "ListOperator"} for t in types):
        families.add("expression_list")
    if any("ModificationalFunctor" in t or "Modification" in t for t in types):
        families.add("modification")
    if any(t in {"Ref", "Apposition", "AttributePredicator", "Attribute"} for t in types):
        families.add("attribute_appos_ref")
    if any("Predicator" in t for t in types):
        families.add("predicate_args")
    return families


def edge_families(edge: tuple[dict[str, Any], Any, dict[str, Any]]) -> set[str]:
    label = str(edge[1])
    families = {label_family(label)}
    families.update(node_type_family(edge))
    families.discard("other")
    if not families:
        families.add("other")
    return families


def pair_label_confusions(
    unmatched_pred_edges: list[tuple[dict[str, Any], Any, dict[str, Any]]],
    unmatched_true_edges: list[tuple[dict[str, Any], Any, dict[str, Any]]],
) -> list[dict[str, Any]]:
    pred_by_endpoint: dict[Any, list[tuple[dict[str, Any], Any, dict[str, Any]]]] = defaultdict(list)
    true_by_endpoint: dict[Any, list[tuple[dict[str, Any], Any, dict[str, Any]]]] = defaultdict(list)
    for edge in unmatched_pred_edges:
        pred_by_endpoint[endpoint_key(edge)].append(edge)
    for edge in unmatched_true_edges:
        true_by_endpoint[endpoint_key(edge)].append(edge)

    pairs: list[dict[str, Any]] = []
    for key in sorted(set(pred_by_endpoint) & set(true_by_endpoint), key=str):
        pred_edges = pred_by_endpoint[key]
        true_edges = true_by_endpoint[key]
        for true_edge, pred_edge in zip(true_edges, pred_edges):
            true_label = str(true_edge[1])
            pred_label = str(pred_edge[1])
            if true_label == pred_label:
                continue
            pairs.append(
                {
                    "gold_label": true_label,
                    "pred_label": pred_label,
                    "source": node_info_json(true_edge[0]),
                    "target": node_info_json(true_edge[2]),
                }
            )
    return pairs


def pair_same_label_endpoint_errors(
    unmatched_pred_edges: list[tuple[dict[str, Any], Any, dict[str, Any]]],
    unmatched_true_edges: list[tuple[dict[str, Any], Any, dict[str, Any]]],
) -> list[dict[str, Any]]:
    pred_by_label: dict[str, list[tuple[dict[str, Any], Any, dict[str, Any]]]] = defaultdict(list)
    true_by_label: dict[str, list[tuple[dict[str, Any], Any, dict[str, Any]]]] = defaultdict(list)
    for edge in unmatched_pred_edges:
        pred_by_label[str(edge[1])].append(edge)
    for edge in unmatched_true_edges:
        true_by_label[str(edge[1])].append(edge)

    pairs: list[dict[str, Any]] = []
    for label in sorted(set(pred_by_label) & set(true_by_label)):
        for true_edge, pred_edge in zip(true_by_label[label], pred_by_label[label]):
            if endpoint_key(true_edge) == endpoint_key(pred_edge):
                continue
            pairs.append(
                {
                    "label": label,
                    "gold_source": node_info_json(true_edge[0]),
                    "gold_target": node_info_json(true_edge[2]),
                    "pred_source": node_info_json(pred_edge[0]),
                    "pred_target": node_info_json(pred_edge[2]),
                }
            )
    return pairs


def classify_case(
    word_metric: dict[str, Any] | None,
    lingua_metric: dict[str, Any] | None,
    lingua_conversion_error: str | None,
) -> list[str]:
    categories: set[str] = set()
    if word_metric is None:
        categories.add("gold_word_conversion_error")
        return sorted(categories)
    if word_metric["unmatched_pred_nodes"] or word_metric["unmatched_true_nodes"]:
        categories.add("word_node_mismatch")
    if word_metric["unmatched_pred_edges"] or word_metric["unmatched_true_edges"]:
        categories.add("word_edge_mismatch")
    label_pairs = pair_label_confusions(
        word_metric["unmatched_pred_edges"],
        word_metric["unmatched_true_edges"],
    )
    if label_pairs:
        categories.add("same_endpoint_label_confusion")
    endpoint_pairs = pair_same_label_endpoint_errors(
        word_metric["unmatched_pred_edges"],
        word_metric["unmatched_true_edges"],
    )
    if endpoint_pairs:
        categories.add("same_label_scope_attachment")
    for edge in word_metric["unmatched_pred_edges"] + word_metric["unmatched_true_edges"]:
        categories.update(edge_families(edge))
    if lingua_conversion_error:
        categories.add("lingua_conversion_error")
    elif lingua_metric is not None and not lingua_metric["exact_same"]:
        categories.add("lingua_mismatch")
    if not categories:
        categories.add("unknown")
    return sorted(categories)


def load_samples(data_path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with jsonlines.open(data_path, "r") as reader:
        for idx, item in enumerate(reader):
            gold_lingua_graph = GPGraph.parse(item["lingua"])
            gold_word_lingua_graph = None
            gold_word_error = None
            try:
                gold_word_lingua_graph, _, _ = lingua2wordlingua(copy.deepcopy(gold_lingua_graph))
            except Exception as exc:
                gold_word_error = str(exc)
                logger.warning("Gold word-lingua conversion failed at %s: %s", idx, exc)
            samples.append(
                {
                    "idx": idx,
                    "sentence": item["sentence"],
                    "gold_lingua_graph": gold_lingua_graph,
                    "gold_word_lingua_graph": gold_word_lingua_graph,
                    "gold_word_error": gold_word_error,
                }
            )
    return samples


def compact_counter(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(limit)]


def add_example(bucket: dict[str, list[dict[str, Any]]], key: str, example: dict[str, Any], limit: int) -> None:
    if len(bucket[key]) < limit:
        bucket[key].append(example)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    allowed_labels, allowed_meta = load_allowed_edge_labels(args)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
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
    logger.info("Decoder label filter: %s", filter_meta)

    samples = load_samples(Path(args.data_path))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    logger.info("Loaded %d samples", len(samples))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "nonexact_cases.jsonl"

    totals = Counter()
    category_counts: Counter[str] = Counter()
    word_missing_label_counts: Counter[str] = Counter()
    word_extra_label_counts: Counter[str] = Counter()
    word_missing_type_counts: Counter[str] = Counter()
    word_extra_type_counts: Counter[str] = Counter()
    label_confusion_counts: Counter[str] = Counter()
    same_label_scope_counts: Counter[str] = Counter()
    node_missing_type_counts: Counter[str] = Counter()
    node_extra_type_counts: Counter[str] = Counter()
    lingua_missing_label_counts: Counter[str] = Counter()
    lingua_extra_label_counts: Counter[str] = Counter()
    examples_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_label_confusion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_missing_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_extra_label: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with jsonlines.open(cases_path, "w") as writer:
        for batch_start in tqdm(range(0, len(samples), args.batch_size), desc="Analyzing train"):
            batch_samples = samples[batch_start : batch_start + args.batch_size]
            batch_results = inference.parse_batch([sample["sentence"] for sample in batch_samples], debug=args.debug)
            for sample, result in zip(batch_samples, batch_results):
                totals["total"] += 1
                idx = sample["idx"]
                sentence = sample["sentence"]
                gold_word_graph = sample["gold_word_lingua_graph"]
                gold_lingua_graph = sample["gold_lingua_graph"]
                pred_word_graph = result.word_lingua_graph
                pred_lingua_graph = result.lingua_graph

                word_metric = None
                if pred_word_graph is not None and gold_word_graph is not None:
                    word_metric = graph_match_metric(
                        pred_word_graph,
                        gold_word_graph,
                        edge_match=args.edge_match,
                        with_spans=True,
                    )
                    if word_metric["exact_same"]:
                        totals["word_exact"] += 1
                    else:
                        totals["word_nonexact"] += 1
                else:
                    totals["word_missing_graph"] += 1

                lingua_metric = None
                lingua_conversion_error = None
                if pred_lingua_graph is None:
                    lingua_conversion_error = result.lingua_conversion_error or "pred_lingua_graph is None"
                    totals["lingua_conversion_error"] += 1
                else:
                    try:
                        lingua_metric = graph_match_metric(
                            pred_lingua_graph,
                            gold_lingua_graph,
                            edge_match=args.edge_match,
                            with_spans=True,
                        )
                        if lingua_metric["exact_same"]:
                            totals["lingua_exact"] += 1
                        else:
                            totals["lingua_nonexact"] += 1
                    except Exception as exc:
                        lingua_conversion_error = str(exc)
                        totals["lingua_metric_error"] += 1

                if word_metric is not None and word_metric["exact_same"] and (
                    lingua_metric is not None and lingua_metric["exact_same"]
                ):
                    continue
                if word_metric is None and not lingua_conversion_error:
                    continue

                categories = classify_case(word_metric, lingua_metric, lingua_conversion_error)
                category_counts.update(categories)

                label_pairs: list[dict[str, Any]] = []
                endpoint_pairs: list[dict[str, Any]] = []
                unmatched_pred_edges_json: list[dict[str, Any]] = []
                unmatched_true_edges_json: list[dict[str, Any]] = []
                unmatched_pred_nodes_json: list[dict[str, Any]] = []
                unmatched_true_nodes_json: list[dict[str, Any]] = []

                if word_metric is not None:
                    unmatched_pred_edges = word_metric["unmatched_pred_edges"]
                    unmatched_true_edges = word_metric["unmatched_true_edges"]
                    unmatched_pred_edges_json = [edge_json(edge) for edge in unmatched_pred_edges]
                    unmatched_true_edges_json = [edge_json(edge) for edge in unmatched_true_edges]
                    unmatched_pred_nodes_json = [node_info_json(node) for node in word_metric["unmatched_pred_nodes"]]
                    unmatched_true_nodes_json = [node_info_json(node) for node in word_metric["unmatched_true_nodes"]]

                    word_extra_label_counts.update(str(edge[1]) for edge in unmatched_pred_edges)
                    word_missing_label_counts.update(str(edge[1]) for edge in unmatched_true_edges)
                    word_extra_type_counts.update(edge_type_key(edge) for edge in unmatched_pred_edges)
                    word_missing_type_counts.update(edge_type_key(edge) for edge in unmatched_true_edges)
                    node_extra_type_counts.update(str(node.get("type")) for node in word_metric["unmatched_pred_nodes"])
                    node_missing_type_counts.update(str(node.get("type")) for node in word_metric["unmatched_true_nodes"])

                    label_pairs = pair_label_confusions(unmatched_pred_edges, unmatched_true_edges)
                    endpoint_pairs = pair_same_label_endpoint_errors(unmatched_pred_edges, unmatched_true_edges)
                    for pair in label_pairs:
                        key = f"{pair['gold_label']} -> {pair['pred_label']}"
                        label_confusion_counts[key] += 1
                        add_example(
                            examples_by_label_confusion,
                            key,
                            {"idx": idx, "sentence": sentence, "pair": pair},
                            args.example_limit,
                        )
                    for pair in endpoint_pairs:
                        same_label_scope_counts[pair["label"]] += 1

                    for edge in unmatched_true_edges:
                        add_example(
                            examples_by_missing_label,
                            str(edge[1]),
                            {"idx": idx, "sentence": sentence, "edge": edge_json(edge)},
                            args.example_limit,
                        )
                    for edge in unmatched_pred_edges:
                        add_example(
                            examples_by_extra_label,
                            str(edge[1]),
                            {"idx": idx, "sentence": sentence, "edge": edge_json(edge)},
                            args.example_limit,
                        )

                if lingua_metric is not None and not lingua_metric["exact_same"]:
                    lingua_extra_label_counts.update(str(edge[1]) for edge in lingua_metric["unmatched_pred_edges"])
                    lingua_missing_label_counts.update(str(edge[1]) for edge in lingua_metric["unmatched_true_edges"])

                case_example = {"idx": idx, "sentence": sentence, "categories": categories}
                for category in categories:
                    add_example(examples_by_category, category, case_example, args.example_limit)

                writer.write(
                    {
                        "idx": idx,
                        "sentence": sentence,
                        "categories": categories,
                        "word_exact": None if word_metric is None else word_metric["exact_same"],
                        "word_counts": None
                        if word_metric is None
                        else {
                            "pred_nodes": word_metric["pred_node_num"],
                            "true_nodes": word_metric["true_node_num"],
                            "matched_nodes": word_metric["matched_node_num"],
                            "pred_edges": word_metric["pred_edge_num"],
                            "true_edges": word_metric["true_edge_num"],
                            "matched_edges": word_metric["matched_edge_num"],
                            "node_f1": word_metric["node_f1"],
                            "edge_f1": word_metric["edge_f1"],
                        },
                        "unmatched_pred_nodes": unmatched_pred_nodes_json,
                        "unmatched_true_nodes": unmatched_true_nodes_json,
                        "unmatched_pred_edges": unmatched_pred_edges_json,
                        "unmatched_true_edges": unmatched_true_edges_json,
                        "same_endpoint_label_confusions": label_pairs,
                        "same_label_scope_attachment": endpoint_pairs[: args.case_pair_limit],
                        "lingua_exact": None if lingua_metric is None else lingua_metric["exact_same"],
                        "lingua_conversion_error": lingua_conversion_error,
                    }
                )

    valid_lingua = totals["total"] - totals["lingua_conversion_error"] - totals["lingua_metric_error"]
    summary = {
        "total": totals["total"],
        "word_exact": totals["word_exact"],
        "word_nonexact": totals["word_nonexact"],
        "word_exact_rate": totals["word_exact"] / totals["total"] if totals["total"] else 0,
        "word_missing_graph": totals["word_missing_graph"],
        "lingua_exact": totals["lingua_exact"],
        "lingua_nonexact": totals["lingua_nonexact"],
        "lingua_conversion_error": totals["lingua_conversion_error"],
        "lingua_metric_error": totals["lingua_metric_error"],
        "lingua_valid": valid_lingua,
        "lingua_exact_rate": totals["lingua_exact"] / valid_lingua if valid_lingua else 0,
        "category_counts": dict(category_counts.most_common()),
        "word_missing_label_counts": compact_counter(word_missing_label_counts, args.top_k),
        "word_extra_label_counts": compact_counter(word_extra_label_counts, args.top_k),
        "word_missing_type_counts": compact_counter(word_missing_type_counts, args.top_k),
        "word_extra_type_counts": compact_counter(word_extra_type_counts, args.top_k),
        "word_same_endpoint_label_confusions": compact_counter(label_confusion_counts, args.top_k),
        "word_same_label_scope_attachment": compact_counter(same_label_scope_counts, args.top_k),
        "word_unmatched_true_node_type_counts": compact_counter(node_missing_type_counts, args.top_k),
        "word_unmatched_pred_node_type_counts": compact_counter(node_extra_type_counts, args.top_k),
        "lingua_missing_label_counts": compact_counter(lingua_missing_label_counts, args.top_k),
        "lingua_extra_label_counts": compact_counter(lingua_extra_label_counts, args.top_k),
        "examples_by_category": examples_by_category,
        "examples_by_label_confusion": examples_by_label_confusion,
        "examples_by_missing_label": examples_by_missing_label,
        "examples_by_extra_label": examples_by_extra_label,
        "edge_label_filter": filter_meta,
        "allowed_edge_labels": allowed_meta,
        "config": {
            "data_path": args.data_path,
            "model_dir": args.model_dir,
            "statistics": args.statistics,
            "device": device,
            "batch_size": args.batch_size,
            "edge_match": args.edge_match,
            "masked_logit": args.masked_logit,
            "edge_factor_decode_weight": args.edge_factor_decode_weight,
        },
        "outputs": {
            "cases_jsonl": str(cases_path),
            "summary_json": str(output_dir / "nonexact_summary.json"),
        },
    }
    summary_path = output_dir / "nonexact_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", cases_path)
    logger.info("Wrote %s", summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze filtered train non-exact matches")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allowed_labels_data_path", default=None)
    parser.add_argument("--allowed_label2id", default=None)
    parser.add_argument("--allowed_edge_types", default=None)
    parser.add_argument("--statistics", default=None)
    parser.add_argument("--device", default=None)
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
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--example_limit", type=int, default=5)
    parser.add_argument("--case_pair_limit", type=int, default=30)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_dir", default=None)
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(
        json.dumps(
            {
                "total": summary["total"],
                "word_exact": summary["word_exact"],
                "word_nonexact": summary["word_nonexact"],
                "lingua_exact": summary["lingua_exact"],
                "lingua_nonexact": summary["lingua_nonexact"],
                "lingua_conversion_error": summary["lingua_conversion_error"],
                "top_categories": list(summary["category_counts"].items())[:10],
                "summary_json": summary["outputs"]["summary_json"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
