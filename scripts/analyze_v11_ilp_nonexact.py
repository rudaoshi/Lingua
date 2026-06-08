#!/usr/bin/env python3
"""Detailed non-exact analysis for the V11 ILP word-lingua decoder."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
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

from analyze_v11_train_edge_confidence import (  # noqa: E402
    classify_case,
    edge_key,
    edge_tuple_json,
    endpoint_key,
    graph_edges_with_conf,
    label_family,
    load_samples,
    node_info_json,
    node_key,
    percentiles,
    write_jsonl,
)
from lingua.learn.wordgraph.decoding.inference import (  # noqa: E402
    InferenceConfig,
    WordLinguaInference,
    combine_rel_logits_with_edge_factor_logits,
    encode_batch_words_like_training,
)
from lingua.learn.wordgraph.modeler.word2gp import wordlingua2lingua  # noqa: E402
from lingua.structure.gpgraph import GPGraph  # noqa: E402
from lingua.utils.metric import get_node_info, graph_match_metric  # noqa: E402


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    return str(value)


def write_safe_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def compact_counter(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(limit)]


def add_example(bucket: dict[str, list[dict[str, Any]]], key: str, row: dict[str, Any], limit: int) -> None:
    if len(bucket[key]) < limit:
        bucket[key].append(row)


def edge_type_key(edge_tuple: tuple[dict[str, Any], Any, dict[str, Any]]) -> str:
    src, label, tgt = edge_tuple
    src_type = node_info_json(src)["type"]
    tgt_type = node_info_json(tgt)["type"]
    return f"{src_type or 'None'} --{str(label)}--> {tgt_type or 'None'}"


def node_text_span_key(node_info: dict[str, Any]) -> tuple[str, str]:
    info = node_info_json(node_info)
    return (
        str(info.get("text")),
        json.dumps(info.get("spans"), ensure_ascii=False, sort_keys=True),
    )


def node_text_key(node_info: dict[str, Any]) -> str:
    return str(node_info_json(node_info).get("text"))


def node_type_confusions(
    unmatched_pred_nodes: list[dict[str, Any]],
    unmatched_true_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pred_by_text_span: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    true_by_text_span: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for node in unmatched_pred_nodes:
        pred_by_text_span[node_text_span_key(node)].append(node)
    for node in unmatched_true_nodes:
        true_by_text_span[node_text_span_key(node)].append(node)

    rows: list[dict[str, Any]] = []
    for key in sorted(set(pred_by_text_span) & set(true_by_text_span), key=str):
        for gold, pred in zip(true_by_text_span[key], pred_by_text_span[key]):
            gold_info = node_info_json(gold)
            pred_info = node_info_json(pred)
            if gold_info.get("type") == pred_info.get("type"):
                continue
            rows.append({"gold": gold_info, "pred": pred_info})
    return rows


def node_span_split_merges(
    unmatched_pred_nodes: list[dict[str, Any]],
    unmatched_true_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pred_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    true_by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in unmatched_pred_nodes:
        pred_by_text[node_text_key(node)].append(node)
    for node in unmatched_true_nodes:
        true_by_text[node_text_key(node)].append(node)

    rows: list[dict[str, Any]] = []
    for text in sorted(set(pred_by_text) & set(true_by_text)):
        for gold, pred in zip(true_by_text[text], pred_by_text[text]):
            gold_info = node_info_json(gold)
            pred_info = node_info_json(pred)
            if gold_info.get("spans") == pred_info.get("spans"):
                continue
            rows.append({"gold": gold_info, "pred": pred_info})
    return rows


def pair_label_confusions(
    unmatched_pred_edges: list[tuple[dict[str, Any], Any, dict[str, Any]]],
    unmatched_true_edges: list[tuple[dict[str, Any], Any, dict[str, Any]]],
) -> list[dict[str, Any]]:
    pred_by_endpoint: dict[Any, list[Any]] = defaultdict(list)
    true_by_endpoint: dict[Any, list[Any]] = defaultdict(list)
    for edge in unmatched_pred_edges:
        pred_by_endpoint[endpoint_key(edge)].append(edge)
    for edge in unmatched_true_edges:
        true_by_endpoint[endpoint_key(edge)].append(edge)

    rows: list[dict[str, Any]] = []
    for endpoint in sorted(set(pred_by_endpoint) & set(true_by_endpoint), key=str):
        for gold, pred in zip(true_by_endpoint[endpoint], pred_by_endpoint[endpoint]):
            if str(gold[1]) == str(pred[1]):
                continue
            rows.append({"gold": edge_tuple_json(gold), "pred": edge_tuple_json(pred)})
    return rows


def pair_same_label_scope_errors(
    unmatched_pred_edges: list[tuple[dict[str, Any], Any, dict[str, Any]]],
    unmatched_true_edges: list[tuple[dict[str, Any], Any, dict[str, Any]]],
) -> list[dict[str, Any]]:
    pred_by_label: dict[str, list[Any]] = defaultdict(list)
    true_by_label: dict[str, list[Any]] = defaultdict(list)
    for edge in unmatched_pred_edges:
        pred_by_label[str(edge[1])].append(edge)
    for edge in unmatched_true_edges:
        true_by_label[str(edge[1])].append(edge)

    rows: list[dict[str, Any]] = []
    for label in sorted(set(pred_by_label) & set(true_by_label)):
        for gold, pred in zip(true_by_label[label], pred_by_label[label]):
            if endpoint_key(gold) == endpoint_key(pred):
                continue
            rows.append({"label": label, "gold": edge_tuple_json(gold), "pred": edge_tuple_json(pred)})
    return rows


def graph_roots(graph: GPGraph | None) -> list[dict[str, Any]]:
    if graph is None:
        return []
    rows = []
    for node in graph.nodes():
        try:
            if graph.g.in_degree(node.ID) != 0:
                continue
        except Exception:
            continue
        rows.append(node_info_json(get_node_info(node, graph, with_type=True, with_spans=True)))
    return rows


def root_signature(roots: list[dict[str, Any]]) -> str:
    return json.dumps(roots, ensure_ascii=False, sort_keys=True)


def selected_edge_source(edge: dict[str, Any], result: Any) -> str | None:
    if edge.get("graph") != "pred":
        return None
    graph = result.word_lingua_graph
    if graph is None:
        return None
    head = edge.get("head")
    dep = edge.get("dep")
    if head is None or dep is None:
        return None
    for parent, g_edge, child in graph.edges():
        try:
            p_span = getattr(parent, "spans", [None])[0]
            c_span = getattr(child, "spans", [None])[0]
            p_idx = p_span[0] if isinstance(p_span, (tuple, list)) else p_span
            c_idx = c_span[0] if isinstance(c_span, (tuple, list)) else c_span
        except Exception:
            continue
        if int(p_idx) == int(head) and int(c_idx) == int(dep) and str(g_edge.label) == str(edge.get("label")):
            return str(getattr(g_edge, "decoder_source", "unknown"))
    return None


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    category_counts: Counter[str] = Counter()
    word_missing_label_counts: Counter[str] = Counter()
    word_extra_label_counts: Counter[str] = Counter()
    word_missing_type_counts: Counter[str] = Counter()
    word_extra_type_counts: Counter[str] = Counter()
    lingua_missing_label_counts: Counter[str] = Counter()
    lingua_extra_label_counts: Counter[str] = Counter()
    node_missing_type_counts: Counter[str] = Counter()
    node_extra_type_counts: Counter[str] = Counter()
    node_type_confusion_counts: Counter[str] = Counter()
    node_span_mismatch_counts: Counter[str] = Counter()
    label_confusion_counts: Counter[str] = Counter()
    same_label_scope_counts: Counter[str] = Counter()
    low_edge_label_counts: Counter[str] = Counter()
    low_edge_family_counts: Counter[str] = Counter()
    low_edge_status_counts: Counter[str] = Counter()
    low_pred_unmatched_label_counts: Counter[str] = Counter()
    low_pred_unmatched_source_counts: Counter[str] = Counter()
    ilp_status_counts: Counter[str] = Counter()
    validation_error_counts: Counter[str] = Counter()
    conversion_error_counts: Counter[str] = Counter()
    root_mismatch_counts: Counter[str] = Counter()
    selected_root_word_counts: Counter[str] = Counter()

    examples_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_missing_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_extra_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_label_confusion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_low_pred_extra: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_node_type_confusion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples_by_conversion_error: dict[str, list[dict[str, Any]]] = defaultdict(list)

    case_records: list[dict[str, Any]] = []
    low_case_records: list[dict[str, Any]] = []
    conversion_error_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    all_pred_conf: list[float] = []
    all_gold_conf: list[float] = []
    nonexact_pred_conf: list[float] = []
    nonexact_gold_conf: list[float] = []
    nonexact_case_min_conf: list[float] = []
    ilp_candidate_edges: list[float] = []
    ilp_selected_edges: list[float] = []

    for batch_start in tqdm(range(0, len(samples), args.batch_size), desc=f"Analyzing {args.split}"):
        batch = samples[batch_start : batch_start + args.batch_size]
        results = inference.parse_batch([sample["sentence"] for sample in batch])
        batch_words = [result.words if result.words else sample["sentence"].split() for sample, result in zip(batch, results)]
        encoded = encode_batch_words_like_training(inference.tokenizer, batch_words, max_length=args.max_length)
        with torch.no_grad():
            model_output = inference.model(
                encoded["input_ids"].to(inference.device),
                encoded["attention_mask"].to(inference.device),
                encoded["word_to_subword"].to(inference.device),
                encoded["word_mask"].to(inference.device),
            )

        for batch_idx, (sample, result, words) in enumerate(zip(batch, results, batch_words)):
            idx = int(sample["idx"])
            sentence = sample["sentence"]
            totals["total"] += 1

            num_nodes = int(encoded["num_nodes"][batch_idx].item())
            arc_prob = torch.sigmoid(
                model_output["arc_logits"][batch_idx, :num_nodes, :num_nodes].float()
            ).detach().cpu().numpy()
            rel_logits_t = model_output["rel_logits"][batch_idx, :num_nodes, :num_nodes, :].float()
            edge_factor_logits_t = {
                name: logits[batch_idx, :num_nodes, :num_nodes, :].float()
                for name, logits in (model_output.get("edge_factor_logits") or {}).items()
            }
            rel_logits = combine_rel_logits_with_edge_factor_logits(
                rel_logits=rel_logits_t,
                edge_factor_logits=edge_factor_logits_t,
                edge_factor_targets_by_label=inference.edge_factor_targets_by_label,
                weight=args.edge_factor_decode_weight,
            )
            rel_prob = torch.softmax(rel_logits, dim=-1).detach().cpu().numpy()

            gold_lingua_graph = sample["gold_lingua_graph"]
            gold_word_graph = sample["gold_word_graph"]
            pred_word_graph = result.word_lingua_graph
            pred_lingua_graph = result.lingua_graph
            conversion_error = result.lingua_conversion_error
            decoding_info = result.decoding_info or {}
            ilp_info = decoding_info.get("ilp") or {}
            ilp_status_counts[f"{ilp_info.get('status')}:{ilp_info.get('success')}"] += 1
            if "candidate_edges" in ilp_info:
                ilp_candidate_edges.append(float(ilp_info["candidate_edges"]))
            if "selected_edges" in ilp_info:
                ilp_selected_edges.append(float(ilp_info["selected_edges"]))
            if ilp_info.get("selected_root_word") is not None:
                selected_root_word_counts[str(ilp_info.get("selected_root_word"))] += 1
            for err in decoding_info.get("validation_errors") or []:
                validation_error_counts[str(err)] += 1

            word_metric = None
            if pred_word_graph is not None and gold_word_graph is not None:
                word_metric = graph_match_metric(pred_word_graph, gold_word_graph, edge_match=args.edge_match, with_spans=True)
                totals["word_exact" if word_metric["exact_same"] else "word_nonexact"] += 1
            else:
                totals["word_metric_missing"] += 1

            lingua_metric = None
            if pred_lingua_graph is not None:
                try:
                    lingua_metric = graph_match_metric(pred_lingua_graph, gold_lingua_graph, edge_match=args.edge_match, with_spans=True)
                    totals["lingua_exact" if lingua_metric["exact_same"] else "lingua_nonexact"] += 1
                except Exception as exc:
                    conversion_error = str(exc)
                    totals["lingua_metric_error"] += 1
            else:
                totals["lingua_conversion_error"] += 1

            if conversion_error:
                conversion_error_counts[conversion_error] += 1
                row = {
                    "idx": idx,
                    "line": idx + 1,
                    "sentence": sentence,
                    "error": conversion_error,
                    "word_edges": result.edges,
                    "node_types": result.node_types,
                    "decoding_info": decoding_info,
                }
                conversion_error_records.append(row)
                add_example(examples_by_conversion_error, conversion_error, row, args.example_limit)

            if pred_word_graph is not None:
                pred_all_edges = graph_edges_with_conf(
                    pred_word_graph,
                    graph_kind="pred",
                    idx=idx,
                    sentence=sentence,
                    words=words,
                    arc_prob=arc_prob,
                    rel_prob=rel_prob,
                    label2id=inference.label2id,
                    unmatched_key_counts=Counter(),
                    low_threshold=args.low_threshold,
                )
                all_pred_conf.extend(float(edge["edge_confidence"]) for edge in pred_all_edges if edge["edge_confidence"] is not None)
            if gold_word_graph is not None:
                gold_all_edges = graph_edges_with_conf(
                    gold_word_graph,
                    graph_kind="gold",
                    idx=idx,
                    sentence=sentence,
                    words=words,
                    arc_prob=arc_prob,
                    rel_prob=rel_prob,
                    label2id=inference.label2id,
                    unmatched_key_counts=Counter(),
                    low_threshold=args.low_threshold,
                )
                all_gold_conf.extend(float(edge["edge_confidence"]) for edge in gold_all_edges if edge["edge_confidence"] is not None)

            if word_metric is None or word_metric["exact_same"]:
                continue

            categories = classify_case(word_metric, lingua_metric, conversion_error)
            category_counts.update(categories)
            for category in categories:
                add_example(examples_by_category, category, {"idx": idx, "sentence": sentence, "categories": categories}, args.example_limit)

            pred_roots = graph_roots(pred_word_graph)
            gold_roots = graph_roots(gold_word_graph)
            if root_signature(pred_roots) != root_signature(gold_roots):
                root_mismatch_counts["word_root_mismatch"] += 1

            unmatched_pred_nodes_json = [node_info_json(node) for node in word_metric["unmatched_pred_nodes"]]
            unmatched_true_nodes_json = [node_info_json(node) for node in word_metric["unmatched_true_nodes"]]
            node_extra_type_counts.update(str(node.get("type")) for node in word_metric["unmatched_pred_nodes"])
            node_missing_type_counts.update(str(node.get("type")) for node in word_metric["unmatched_true_nodes"])

            type_confusions = node_type_confusions(word_metric["unmatched_pred_nodes"], word_metric["unmatched_true_nodes"])
            for pair in type_confusions:
                key = f"{pair['gold'].get('type')} -> {pair['pred'].get('type')}"
                node_type_confusion_counts[key] += 1
                add_example(examples_by_node_type_confusion, key, {"idx": idx, "sentence": sentence, "pair": pair}, args.example_limit)

            span_mismatches = node_span_split_merges(word_metric["unmatched_pred_nodes"], word_metric["unmatched_true_nodes"])
            for pair in span_mismatches:
                key = f"{pair['gold'].get('type')} -> {pair['pred'].get('type')}"
                node_span_mismatch_counts[key] += 1

            word_extra_label_counts.update(str(edge[1]) for edge in word_metric["unmatched_pred_edges"])
            word_missing_label_counts.update(str(edge[1]) for edge in word_metric["unmatched_true_edges"])
            word_extra_type_counts.update(edge_type_key(edge) for edge in word_metric["unmatched_pred_edges"])
            word_missing_type_counts.update(edge_type_key(edge) for edge in word_metric["unmatched_true_edges"])

            label_confusions = pair_label_confusions(word_metric["unmatched_pred_edges"], word_metric["unmatched_true_edges"])
            for pair in label_confusions:
                key = f"{pair['gold']['label']} -> {pair['pred']['label']}"
                label_confusion_counts[key] += 1
                add_example(examples_by_label_confusion, key, {"idx": idx, "sentence": sentence, "pair": pair}, args.example_limit)

            same_label_scope = pair_same_label_scope_errors(word_metric["unmatched_pred_edges"], word_metric["unmatched_true_edges"])
            for pair in same_label_scope:
                same_label_scope_counts[pair["label"]] += 1

            if lingua_metric is not None and not lingua_metric["exact_same"]:
                lingua_extra_label_counts.update(str(edge[1]) for edge in lingua_metric["unmatched_pred_edges"])
                lingua_missing_label_counts.update(str(edge[1]) for edge in lingua_metric["unmatched_true_edges"])

            pred_edges = graph_edges_with_conf(
                pred_word_graph,
                graph_kind="pred",
                idx=idx,
                sentence=sentence,
                words=words,
                arc_prob=arc_prob,
                rel_prob=rel_prob,
                label2id=inference.label2id,
                unmatched_key_counts=Counter(edge_key(edge) for edge in word_metric["unmatched_pred_edges"]),
                low_threshold=args.low_threshold,
            )
            gold_edges = graph_edges_with_conf(
                gold_word_graph,
                graph_kind="gold",
                idx=idx,
                sentence=sentence,
                words=words,
                arc_prob=arc_prob,
                rel_prob=rel_prob,
                label2id=inference.label2id,
                unmatched_key_counts=Counter(edge_key(edge) for edge in word_metric["unmatched_true_edges"]),
                low_threshold=args.low_threshold,
            )
            for edge in pred_edges:
                edge["decoder_source"] = selected_edge_source(edge, result)
            for edge in pred_edges + gold_edges:
                edge["categories"] = categories
                edge_records.append(edge)
                if edge["edge_confidence"] is not None:
                    if edge["graph"] == "pred":
                        nonexact_pred_conf.append(float(edge["edge_confidence"]))
                    else:
                        nonexact_gold_conf.append(float(edge["edge_confidence"]))
                if not edge["low"]:
                    continue
                low_edge_label_counts[str(edge["label"])] += 1
                low_edge_family_counts[str(edge["family"])] += 1
                low_edge_status_counts[f"{edge['graph']}:{edge['status']}"] += 1
                if edge["graph"] == "pred" and edge["status"] == "unmatched":
                    low_pred_unmatched_label_counts[str(edge["label"])] += 1
                    low_pred_unmatched_source_counts[str(edge.get("decoder_source"))] += 1
                    add_example(
                        examples_by_low_pred_extra,
                        str(edge["label"]),
                        {"idx": idx, "sentence": sentence, "edge": edge},
                        args.example_limit,
                    )

            pred_conf = [float(edge["edge_confidence"]) for edge in pred_edges if edge["edge_confidence"] is not None]
            gold_conf = [float(edge["edge_confidence"]) for edge in gold_edges if edge["edge_confidence"] is not None]
            all_case_conf = pred_conf + gold_conf
            min_pred = min(pred_conf) if pred_conf else None
            min_gold = min(gold_conf) if gold_conf else None
            min_any = min(all_case_conf) if all_case_conf else None
            if min_any is not None:
                nonexact_case_min_conf.append(min_any)
            low_edges = sorted(
                [edge for edge in pred_edges + gold_edges if edge["low"]],
                key=lambda edge: (edge["edge_confidence"] is None, edge["edge_confidence"] if edge["edge_confidence"] is not None else 999),
            )
            low_unmatched_edges = [edge for edge in low_edges if edge["status"] == "unmatched"]

            for edge in word_metric["unmatched_true_edges"]:
                add_example(examples_by_missing_label, str(edge[1]), {"idx": idx, "sentence": sentence, "edge": edge_tuple_json(edge)}, args.example_limit)
            for edge in word_metric["unmatched_pred_edges"]:
                add_example(examples_by_extra_label, str(edge[1]), {"idx": idx, "sentence": sentence, "edge": edge_tuple_json(edge)}, args.example_limit)

            case_record = {
                "idx": idx,
                "line": idx + 1,
                "split": args.split,
                "sentence": sentence,
                "categories": categories,
                "word_counts": {
                    "pred_nodes": word_metric["pred_node_num"],
                    "true_nodes": word_metric["true_node_num"],
                    "matched_nodes": word_metric["matched_node_num"],
                    "pred_edges": word_metric["pred_edge_num"],
                    "true_edges": word_metric["true_edge_num"],
                    "matched_edges": word_metric["matched_edge_num"],
                    "node_f1": word_metric["node_f1"],
                    "edge_f1": word_metric["edge_f1"],
                },
                "lingua_exact": None if lingua_metric is None else lingua_metric["exact_same"],
                "lingua_conversion_error": conversion_error,
                "pred_roots": pred_roots,
                "gold_roots": gold_roots,
                "ilp": ilp_info,
                "validation_errors": decoding_info.get("validation_errors"),
                "min_pred_edge_confidence": min_pred,
                "min_gold_edge_confidence": min_gold,
                "min_edge_confidence": min_any,
                "low_edge_count": len(low_edges),
                "low_unmatched_edge_count": len(low_unmatched_edges),
                "lowest_edges": low_edges[: args.case_edge_limit],
                "lowest_unmatched_edges": low_unmatched_edges[: args.case_edge_limit],
                "unmatched_pred_nodes": unmatched_pred_nodes_json,
                "unmatched_true_nodes": unmatched_true_nodes_json,
                "node_type_confusions": type_confusions[: args.case_pair_limit],
                "node_span_mismatches": span_mismatches[: args.case_pair_limit],
                "unmatched_pred_edges": [edge_tuple_json(edge) for edge in word_metric["unmatched_pred_edges"]],
                "unmatched_true_edges": [edge_tuple_json(edge) for edge in word_metric["unmatched_true_edges"]],
                "same_endpoint_label_confusions": label_confusions[: args.case_pair_limit],
                "same_label_scope_attachment": same_label_scope[: args.case_pair_limit],
            }
            case_records.append(case_record)
            if low_edges:
                low_case_records.append(case_record)

    case_records.sort(
        key=lambda row: (
            row["min_edge_confidence"] is None,
            row["min_edge_confidence"] if row["min_edge_confidence"] is not None else 999,
            row["idx"],
        )
    )
    low_case_records.sort(
        key=lambda row: (
            row["min_edge_confidence"] is None,
            row["min_edge_confidence"] if row["min_edge_confidence"] is not None else 999,
            row["idx"],
        )
    )
    edge_records.sort(
        key=lambda row: (
            row["idx"],
            row["graph"],
            row["edge_confidence"] is None,
            row["edge_confidence"] if row["edge_confidence"] is not None else 999,
        )
    )

    write_safe_jsonl(output_dir / "nonexact_cases_with_confidence.jsonl", case_records)
    write_safe_jsonl(output_dir / f"low_confidence_cases_lt{args.low_threshold:.2f}.jsonl", low_case_records)
    write_safe_jsonl(output_dir / "lowest_edge_confidence_cases_top200.jsonl", case_records[:200])
    write_safe_jsonl(output_dir / "edge_confidences_nonexact.jsonl", edge_records)
    write_safe_jsonl(output_dir / "conversion_errors.jsonl", conversion_error_records)

    threshold_counts: dict[str, dict[str, int]] = {}
    for threshold in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]:
        threshold_counts[str(threshold)] = {
            "nonexact_cases_with_any_edge_below": sum(
                1 for row in case_records if row["min_edge_confidence"] is not None and row["min_edge_confidence"] < threshold
            ),
            "nonexact_pred_edges_below": sum(1 for confidence in nonexact_pred_conf if confidence < threshold),
            "nonexact_gold_edges_below": sum(1 for confidence in nonexact_gold_conf if confidence < threshold),
        }

    valid_lingua = totals["total"] - totals["lingua_conversion_error"] - totals["lingua_metric_error"]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "model_dir": args.model_dir,
        "data_path": args.data_path,
        "device": device,
        "confidence_formula": "sigmoid(arc_logit[head,dep]) * softmax(combined_rel_logits[head,dep])[label]",
        "low_threshold": args.low_threshold,
        "totals": dict(totals),
        "word_exact_rate": totals["word_exact"] / totals["total"] if totals["total"] else None,
        "lingua_exact_rate": totals["lingua_exact"] / valid_lingua if valid_lingua else None,
        "word_nonexact_case_records": len(case_records),
        "low_confidence_case_records": len(low_case_records),
        "category_counts": dict(category_counts.most_common()),
        "word_missing_label_counts": compact_counter(word_missing_label_counts, args.top_k),
        "word_extra_label_counts": compact_counter(word_extra_label_counts, args.top_k),
        "word_missing_type_counts": compact_counter(word_missing_type_counts, args.top_k),
        "word_extra_type_counts": compact_counter(word_extra_type_counts, args.top_k),
        "lingua_missing_label_counts": compact_counter(lingua_missing_label_counts, args.top_k),
        "lingua_extra_label_counts": compact_counter(lingua_extra_label_counts, args.top_k),
        "word_unmatched_true_node_type_counts": compact_counter(node_missing_type_counts, args.top_k),
        "word_unmatched_pred_node_type_counts": compact_counter(node_extra_type_counts, args.top_k),
        "node_type_confusion_counts": compact_counter(node_type_confusion_counts, args.top_k),
        "node_span_mismatch_counts": compact_counter(node_span_mismatch_counts, args.top_k),
        "word_same_endpoint_label_confusions": compact_counter(label_confusion_counts, args.top_k),
        "word_same_label_scope_attachment": compact_counter(same_label_scope_counts, args.top_k),
        "low_edge_label_counts": compact_counter(low_edge_label_counts, args.top_k),
        "low_edge_family_counts": compact_counter(low_edge_family_counts, args.top_k),
        "low_edge_status_counts": compact_counter(low_edge_status_counts, args.top_k),
        "low_pred_unmatched_label_counts": compact_counter(low_pred_unmatched_label_counts, args.top_k),
        "low_pred_unmatched_source_counts": compact_counter(low_pred_unmatched_source_counts, args.top_k),
        "ilp_status_counts": compact_counter(ilp_status_counts, args.top_k),
        "ilp_candidate_edges_quantiles": percentiles(ilp_candidate_edges),
        "ilp_selected_edges_quantiles": percentiles(ilp_selected_edges),
        "validation_error_counts": compact_counter(validation_error_counts, args.top_k),
        "conversion_error_counts": compact_counter(conversion_error_counts, args.top_k),
        "root_mismatch_counts": compact_counter(root_mismatch_counts, args.top_k),
        "selected_root_word_counts": compact_counter(selected_root_word_counts, args.top_k),
        "confidence_quantiles": {
            "all_pred_edges": percentiles(all_pred_conf),
            "all_gold_edges": percentiles(all_gold_conf),
            "nonexact_pred_edges": percentiles(nonexact_pred_conf),
            "nonexact_gold_edges": percentiles(nonexact_gold_conf),
            "nonexact_case_min_edge": percentiles(nonexact_case_min_conf),
        },
        "threshold_counts": threshold_counts,
        "examples_by_category": examples_by_category,
        "examples_by_missing_label": examples_by_missing_label,
        "examples_by_extra_label": examples_by_extra_label,
        "examples_by_label_confusion": examples_by_label_confusion,
        "examples_by_low_pred_extra": examples_by_low_pred_extra,
        "examples_by_node_type_confusion": examples_by_node_type_confusion,
        "examples_by_conversion_error": examples_by_conversion_error,
        "config": {
            "max_length": args.max_length,
            "arc_threshold": args.arc_threshold,
            "edge_factor_decode_weight": args.edge_factor_decode_weight,
            "decoder_mode": args.decoder_mode,
            "ilp_max_candidates_per_child": args.ilp_max_candidates_per_child,
            "ilp_max_labels_per_pair": args.ilp_max_labels_per_pair,
            "ilp_time_limit": args.ilp_time_limit,
            "ilp_mip_rel_gap": args.ilp_mip_rel_gap,
            "ilp_root_score_weight": args.ilp_root_score_weight,
            "enforce_connected": args.enforce_connected,
            "enable_phase4_rewrites": args.enable_phase4_rewrites,
            "enforce_single_root": args.enforce_single_root,
            "batch_size": args.batch_size,
            "edge_match": args.edge_match,
        },
        "outputs": {
            "nonexact_cases": str(output_dir / "nonexact_cases_with_confidence.jsonl"),
            "low_confidence_cases": str(output_dir / f"low_confidence_cases_lt{args.low_threshold:.2f}.jsonl"),
            "lowest_cases": str(output_dir / "lowest_edge_confidence_cases_top200.jsonl"),
            "edge_confidences": str(output_dir / "edge_confidences_nonexact.jsonl"),
            "conversion_errors": str(output_dir / "conversion_errors.jsonl"),
            "summary": str(output_dir / "nonexact_summary.json"),
        },
    }
    safe_summary = to_jsonable(summary)
    (output_dir / "nonexact_summary.json").write_text(json.dumps(safe_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(safe_summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--statistics", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--arc_threshold", type=float, default=0.5)
    parser.add_argument("--edge_factor_decode_weight", type=float, default=0.2)
    parser.add_argument("--decoder_mode", choices=["ilp", "sdp_graph"], default="ilp")
    parser.add_argument("--ilp_max_candidates_per_child", type=int, default=64)
    parser.add_argument("--ilp_max_labels_per_pair", type=int, default=6)
    parser.add_argument("--ilp_time_limit", type=float, default=20.0)
    parser.add_argument("--ilp_mip_rel_gap", type=float, default=0.001)
    parser.add_argument("--ilp_root_score_weight", type=float, default=1.0)
    parser.add_argument("--enforce_connected", action="store_true")
    parser.add_argument("--enable_phase4_rewrites", action="store_true")
    parser.add_argument("--enforce_single_root", dest="enforce_single_root", action="store_true")
    parser.add_argument("--no_enforce_single_root", dest="enforce_single_root", action="store_false")
    parser.set_defaults(enforce_single_root=True)
    parser.add_argument("--edge_match", choices=["exact", "loose"], default="exact")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--low_threshold", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--example_limit", type=int, default=8)
    parser.add_argument("--case_edge_limit", type=int, default=12)
    parser.add_argument("--case_pair_limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    analyze(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
