#!/usr/bin/env python3
"""Analyze V11 train non-exact word-lingua cases with per-edge confidence."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from lingua.learn.wordgraph.decoding.inference import (
    InferenceConfig,
    WordLinguaInference,
    encode_batch_words_like_training,
)
from lingua.learn.wordgraph.modeler.gp2word import lingua2wordlingua
from lingua.structure.gpgraph import GPGraph, GPGPhraseNode
from lingua.utils.metric import get_node_info, graph_match_metric


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        str(p): float(np.percentile(arr, p))
        for p in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
    }


def safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [safe_json_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    return str(value)


def node_info_json(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": str(info.get("text")),
        "type": None if info.get("type") is None else str(info.get("type")),
        "spans": safe_json_value(info.get("spans")),
    }


def node_key(info: dict[str, Any]) -> tuple[str, str | None, str]:
    info_json = node_info_json(info)
    return (
        info_json["text"],
        info_json["type"],
        json.dumps(info_json["spans"], ensure_ascii=False, sort_keys=True),
    )


def edge_key(edge_tuple: tuple[dict[str, Any], Any, dict[str, Any]]) -> tuple[Any, str, Any]:
    src, label, tgt = edge_tuple
    return (node_key(src), str(label), node_key(tgt))


def endpoint_key(edge_tuple: tuple[dict[str, Any], Any, dict[str, Any]]) -> tuple[Any, Any]:
    src, _label, tgt = edge_tuple
    return (node_key(src), node_key(tgt))


def edge_tuple_json(edge_tuple: tuple[dict[str, Any], Any, dict[str, Any]]) -> dict[str, Any]:
    src, label, tgt = edge_tuple
    return {"source": node_info_json(src), "label": str(label), "target": node_info_json(tgt)}


def label_family(label: str) -> str:
    parts = label.split("+")
    if any("punctuation" in part for part in parts):
        return "punctuation"
    if any(part in {"@expression", "@list", "func.arg"} for part in parts):
        return "expression_list"
    if any(part in {"@modification_of", "as:modification", "variable", "body"} for part in parts):
        return "modification"
    if any(
        part in {"@attribute_of", "@appositive_of", "@ref_of", "as:attribute", "as:appositive", "as:ref"}
        for part in parts
    ):
        return "attribute_appos_ref"
    if any(part.startswith("pred.arg.") for part in parts):
        return "predicate_args"
    if any(part in {"@lower", "@upper"} for part in parts):
        return "lower_upper"
    if any(part in {"as:discourse", "@discourse_of"} for part in parts):
        return "discourse"
    if any(part == "next_word" for part in parts):
        return "linearization"
    if any(part == "@copula" for part in parts):
        return "copula"
    if any(part == "@parataxis" for part in parts):
        return "parataxis"
    return "other"


def node_type_family_from_edge(edge_tuple: tuple[dict[str, Any], Any, dict[str, Any]]) -> set[str]:
    src, _label, tgt = edge_tuple
    types = {str(src.get("type")), str(tgt.get("type"))}
    families: set[str] = set()
    if any("PunctuationalConstant" in t for t in types):
        families.add("punctuation")
    if any(t in {"ExpressionFunctor", "ListFunctor", "ExpressionOperator", "ListOperator"} for t in types):
        families.add("expression_list")
    if any("ModificationalFunctor" in t or "Modification" in t for t in types):
        families.add("modification")
    if any(t in {"Ref", "Apposition", "AttributePredicator", "Attribute", "AppositionalPredicator"} for t in types):
        families.add("attribute_appos_ref")
    if any("Predicator" in t for t in types):
        families.add("predicate_args")
    return families


def edge_families(edge_tuple: tuple[dict[str, Any], Any, dict[str, Any]]) -> set[str]:
    families = {label_family(str(edge_tuple[1]))}
    families.update(node_type_family_from_edge(edge_tuple))
    families.discard("other")
    return families or {"other"}


def classify_case(word_metric: dict[str, Any], lingua_metric: dict[str, Any] | None, conversion_error: str | None) -> list[str]:
    categories = set()
    if word_metric["unmatched_pred_nodes"] or word_metric["unmatched_true_nodes"]:
        categories.add("word_node_mismatch")
    if word_metric["unmatched_pred_edges"] or word_metric["unmatched_true_edges"]:
        categories.add("word_edge_mismatch")

    pred_by_endpoint: dict[Any, list[Any]] = defaultdict(list)
    true_by_endpoint: dict[Any, list[Any]] = defaultdict(list)
    for edge in word_metric["unmatched_pred_edges"]:
        pred_by_endpoint[endpoint_key(edge)].append(edge)
    for edge in word_metric["unmatched_true_edges"]:
        true_by_endpoint[endpoint_key(edge)].append(edge)
    for endpoint in set(pred_by_endpoint) & set(true_by_endpoint):
        if any(str(pred[1]) != str(true[1]) for pred in pred_by_endpoint[endpoint] for true in true_by_endpoint[endpoint]):
            categories.add("same_endpoint_label_confusion")

    pred_by_label: dict[str, list[Any]] = defaultdict(list)
    true_by_label: dict[str, list[Any]] = defaultdict(list)
    for edge in word_metric["unmatched_pred_edges"]:
        pred_by_label[str(edge[1])].append(edge)
    for edge in word_metric["unmatched_true_edges"]:
        true_by_label[str(edge[1])].append(edge)
    for label in set(pred_by_label) & set(true_by_label):
        if any(endpoint_key(pred) != endpoint_key(true) for pred in pred_by_label[label] for true in true_by_label[label]):
            categories.add("same_label_scope_attachment")

    for edge in word_metric["unmatched_pred_edges"] + word_metric["unmatched_true_edges"]:
        categories.update(edge_families(edge))
    if conversion_error:
        categories.add("lingua_conversion_error")
    elif lingua_metric is not None and not lingua_metric["exact_same"]:
        categories.add("lingua_mismatch")
    return sorted(categories or {"unknown"})


def single_word_idx(node: Any, num_words: int) -> int | None:
    if not isinstance(node, GPGPhraseNode):
        return None
    try:
        word_indices = list(node.words(with_aux=False))
        if len(word_indices) == 1:
            idx = int(word_indices[0])
            if 0 <= idx < num_words:
                return idx
    except Exception:
        pass
    spans = getattr(node, "spans", None)
    if spans:
        span = spans[0]
        idx = int(span[0] if isinstance(span, (list, tuple)) else span)
        if 0 <= idx < num_words:
            return idx
    return None


def graph_edges_with_conf(
    graph,
    *,
    graph_kind: str,
    idx: int,
    sentence: str,
    words: list[str],
    arc_prob: np.ndarray,
    rel_prob: np.ndarray,
    label2id: dict[str, int],
    unmatched_key_counts: Counter,
    low_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    num_words = len(words)
    for parent, edge, child in graph.edges():
        head = single_word_idx(parent, num_words)
        dep = single_word_idx(child, num_words)
        label = str(edge.label)
        source_info = get_node_info(parent, graph, with_type=True, with_spans=True)
        target_info = get_node_info(child, graph, with_type=True, with_spans=True)
        tuple_key = edge_key((source_info, label, target_info))
        status = "unmatched" if unmatched_key_counts.get(tuple_key, 0) > 0 else "matched"
        if status == "unmatched":
            unmatched_key_counts[tuple_key] -= 1

        arc_p = None
        label_p = None
        confidence = None
        if head is not None and dep is not None and head < arc_prob.shape[0] and dep < arc_prob.shape[1] and label in label2id:
            arc_p = float(arc_prob[head, dep])
            label_p = float(rel_prob[head, dep, label2id[label]])
            confidence = arc_p * label_p

        rows.append(
            {
                "idx": idx,
                "sentence": sentence,
                "graph": graph_kind,
                "status": status,
                "head": head,
                "dep": dep,
                "head_word": words[head] if head is not None and head < len(words) else None,
                "dep_word": words[dep] if dep is not None and dep < len(words) else None,
                "label": label,
                "family": label_family(label),
                "source": node_info_json(source_info),
                "target": node_info_json(target_info),
                "arc_prob": arc_p,
                "label_prob": label_p,
                "edge_confidence": confidence,
                "low": confidence is not None and confidence < low_threshold,
            }
        )
    return rows


def load_samples(data_path: Path) -> list[dict[str, Any]]:
    samples = []
    for idx, item in enumerate(read_jsonl(data_path)):
        gold_lingua_graph = GPGraph.parse(item["lingua"])
        try:
            gold_word_graph, _, _ = lingua2wordlingua(copy.deepcopy(gold_lingua_graph))
            gold_error = None
        except Exception as exc:
            gold_word_graph = None
            gold_error = str(exc)
        samples.append(
            {
                "idx": idx,
                "sentence": item["sentence"],
                "gold_lingua_graph": gold_lingua_graph,
                "gold_word_graph": gold_word_graph,
                "gold_error": gold_error,
            }
        )
    return samples


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    inference = WordLinguaInference.from_pretrained(
        args.model_dir,
        device=device,
        config=InferenceConfig(max_length=args.max_length, edge_factor_decode_weight=args.edge_factor_decode_weight),
    )
    samples = load_samples(Path(args.data_path))
    if args.max_samples:
        samples = samples[: args.max_samples]

    edge_records = []
    case_records = []
    low_case_records = []
    all_pred_conf = []
    all_gold_conf = []
    nonexact_pred_conf = []
    nonexact_gold_conf = []
    nonexact_case_min_conf = []

    totals = Counter()
    category_counts = Counter()
    missing_label_counts = Counter()
    extra_label_counts = Counter()
    low_edge_label_counts = Counter()
    low_edge_family_counts = Counter()
    low_edge_status_counts = Counter()
    same_endpoint_label_confusions = Counter()

    for batch_start in tqdm(range(0, len(samples), args.batch_size), desc="V11 train confidence"):
        batch = samples[batch_start : batch_start + args.batch_size]
        results = inference.parse_batch([sample["sentence"] for sample in batch])
        batch_words = [result.words if result.words else sample["sentence"].split() for sample, result in zip(batch, results)]
        encoded = encode_batch_words_like_training(inference.tokenizer, batch_words, max_length=args.max_length)
        with torch.no_grad():
            output = inference.model(
                encoded["input_ids"].to(inference.device),
                encoded["attention_mask"].to(inference.device),
                encoded["word_to_subword"].to(inference.device),
                encoded["word_mask"].to(inference.device),
            )

        for batch_idx, (sample, result, words) in enumerate(zip(batch, results, batch_words)):
            idx = int(sample["idx"])
            sentence = sample["sentence"]
            num_nodes = int(encoded["num_nodes"][batch_idx].item())
            arc_prob = torch.sigmoid(output["arc_logits"][batch_idx, :num_nodes, :num_nodes].float()).detach().cpu().numpy()
            rel_prob = torch.softmax(
                output["rel_logits"][batch_idx, :num_nodes, :num_nodes, :].float(),
                dim=-1,
            ).detach().cpu().numpy()

            totals["total"] += 1
            pred_word_graph = result.word_lingua_graph
            pred_lingua_graph = result.lingua_graph
            gold_word_graph = sample["gold_word_graph"]
            gold_lingua_graph = sample["gold_lingua_graph"]
            conversion_error = result.lingua_conversion_error

            word_metric = None
            if pred_word_graph is not None and gold_word_graph is not None:
                word_metric = graph_match_metric(pred_word_graph, gold_word_graph, edge_match="exact", with_spans=True)
                totals["word_exact" if word_metric["exact_same"] else "word_nonexact"] += 1
            else:
                totals["word_metric_missing"] += 1

            lingua_metric = None
            if pred_lingua_graph is not None:
                try:
                    lingua_metric = graph_match_metric(pred_lingua_graph, gold_lingua_graph, edge_match="exact", with_spans=True)
                    totals["lingua_exact" if lingua_metric["exact_same"] else "lingua_nonexact"] += 1
                except Exception as exc:
                    conversion_error = str(exc)
                    totals["lingua_metric_error"] += 1
            else:
                totals["lingua_conversion_error"] += 1

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
            missing_label_counts.update(str(edge[1]) for edge in word_metric["unmatched_true_edges"])
            extra_label_counts.update(str(edge[1]) for edge in word_metric["unmatched_pred_edges"])

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

            nonexact_pred_conf.extend(float(edge["edge_confidence"]) for edge in pred_edges if edge["edge_confidence"] is not None)
            nonexact_gold_conf.extend(float(edge["edge_confidence"]) for edge in gold_edges if edge["edge_confidence"] is not None)
            for edge in pred_edges + gold_edges:
                edge_record = dict(edge)
                edge_record["categories"] = categories
                edge_records.append(edge_record)
                if edge["low"]:
                    low_edge_label_counts[str(edge["label"])] += 1
                    low_edge_family_counts[str(edge["family"])] += 1
                    low_edge_status_counts[f"{edge['graph']}:{edge['status']}"] += 1

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

            pred_by_endpoint: dict[Any, list[Any]] = defaultdict(list)
            true_by_endpoint: dict[Any, list[Any]] = defaultdict(list)
            for edge in word_metric["unmatched_pred_edges"]:
                pred_by_endpoint[endpoint_key(edge)].append(edge)
            for edge in word_metric["unmatched_true_edges"]:
                true_by_endpoint[endpoint_key(edge)].append(edge)
            same_endpoint_pairs = []
            for endpoint in sorted(set(pred_by_endpoint) & set(true_by_endpoint), key=str):
                for pred_edge in pred_by_endpoint[endpoint]:
                    for true_edge in true_by_endpoint[endpoint]:
                        if str(pred_edge[1]) != str(true_edge[1]):
                            same_endpoint_label_confusions[f"{str(true_edge[1])} -> {str(pred_edge[1])}"] += 1
                            same_endpoint_pairs.append(
                                {"gold": edge_tuple_json(true_edge), "pred": edge_tuple_json(pred_edge)}
                            )

            case_record = {
                "idx": idx,
                "line": idx + 1,
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
                "min_pred_edge_confidence": min_pred,
                "min_gold_edge_confidence": min_gold,
                "min_edge_confidence": min_any,
                "low_edge_count": len(low_edges),
                "low_unmatched_edge_count": len(low_unmatched_edges),
                "lowest_edges": low_edges[:12],
                "lowest_unmatched_edges": low_unmatched_edges[:12],
                "unmatched_pred_edges": [edge_tuple_json(edge) for edge in word_metric["unmatched_pred_edges"]],
                "unmatched_true_edges": [edge_tuple_json(edge) for edge in word_metric["unmatched_true_edges"]],
                "same_endpoint_label_confusions": same_endpoint_pairs[:20],
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

    nonexact_cases_path = output_dir / "nonexact_cases_with_confidence.jsonl"
    edge_conf_path = output_dir / "edge_confidences_train_nonexact.jsonl"
    low_cases_path = output_dir / f"low_confidence_cases_lt{args.low_threshold:.2f}.jsonl"
    lowest_top_path = output_dir / "lowest_edge_confidence_cases_top200.jsonl"
    write_jsonl(nonexact_cases_path, case_records)
    write_jsonl(edge_conf_path, edge_records)
    write_jsonl(low_cases_path, low_case_records)
    write_jsonl(lowest_top_path, case_records[:200])

    threshold_counts = {}
    for threshold in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]:
        threshold_counts[str(threshold)] = {
            "nonexact_cases_with_any_edge_below": sum(
                1
                for row in case_records
                if row["min_edge_confidence"] is not None and row["min_edge_confidence"] < threshold
            ),
            "nonexact_pred_edges_below": sum(1 for confidence in nonexact_pred_conf if confidence < threshold),
            "nonexact_gold_edges_below": sum(1 for confidence in nonexact_gold_conf if confidence < threshold),
        }

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_dir": args.model_dir,
        "data_path": args.data_path,
        "device": device,
        "confidence_formula": "edge_confidence = sigmoid(arc_logit[head,dep]) * softmax(rel_logits[head,dep])[label]",
        "low_threshold": args.low_threshold,
        "totals": dict(totals),
        "word_exact_rate": totals["word_exact"] / totals["total"] if totals["total"] else None,
        "word_nonexact_case_records": len(case_records),
        "low_confidence_case_records": len(low_case_records),
        "confidence_quantiles": {
            "all_train_pred_edges": percentiles(all_pred_conf),
            "all_train_gold_edges": percentiles(all_gold_conf),
            "nonexact_pred_edges": percentiles(nonexact_pred_conf),
            "nonexact_gold_edges": percentiles(nonexact_gold_conf),
            "nonexact_case_min_edge": percentiles(nonexact_case_min_conf),
        },
        "threshold_counts": threshold_counts,
        "category_counts": dict(category_counts.most_common()),
        "missing_label_counts": dict(missing_label_counts.most_common(50)),
        "extra_label_counts": dict(extra_label_counts.most_common(50)),
        "low_edge_label_counts": dict(low_edge_label_counts.most_common(50)),
        "low_edge_family_counts": dict(low_edge_family_counts.most_common()),
        "low_edge_status_counts": dict(low_edge_status_counts.most_common()),
        "same_endpoint_label_confusions": dict(same_endpoint_label_confusions.most_common(50)),
        "outputs": {
            "summary": str(output_dir / "summary.json"),
            "nonexact_cases": str(nonexact_cases_path),
            "edge_confidences": str(edge_conf_path),
            "low_confidence_cases": str(low_cases_path),
            "lowest_top200": str(lowest_top_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--low_threshold", type=float, default=0.50)
    parser.add_argument("--edge_factor_decode_weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    summary = analyze(parse_args())
    print(
        json.dumps(
            {
                "summary": summary["outputs"]["summary"],
                "total": summary["totals"].get("total"),
                "word_exact": summary["totals"].get("word_exact"),
                "word_nonexact": summary["totals"].get("word_nonexact"),
                "low_confidence_case_records": summary["low_confidence_case_records"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
