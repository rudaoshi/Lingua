#!/usr/bin/env python3
"""Visualize V11 low-confidence @lower non-exact cases."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analyze_v11_train_edge_confidence import (
    edge_key,
    edge_tuple_json,
    graph_edges_with_conf,
    load_samples,
    node_info_json,
    safe_json_value,
    single_word_idx,
)
from lingua.learn.wordgraph.decoding.inference import (
    InferenceConfig,
    WordLinguaInference,
    combine_rel_logits_with_edge_factor_logits,
    encode_words_like_training,
)
from lingua.structure.gpgraph import GPGPhraseNode
from lingua.utils.metric import get_node_info, graph_match_metric


def dot_quote(value: Any) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{text}"'


def node_word_idx(node: Any, num_words: int) -> int | None:
    return single_word_idx(node, num_words)


def short_text(text: str, max_len: int = 28) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def graph_node_label(node: Any, graph: Any, words: list[str]) -> str:
    info = get_node_info(node, graph, with_type=True, with_spans=True)
    idx = node_word_idx(node, len(words))
    if idx is not None:
        word = words[idx] if idx < len(words) else str(info.get("text"))
        return f"{idx}: {short_text(word)}\n{short_text(str(info.get('type')))}"
    return f"{short_text(str(info.get('text')))}\n{short_text(str(info.get('type')))}"


def edge_statuses(metric_edges: list[Any]) -> Counter:
    return Counter(edge_key(edge) for edge in metric_edges)


def graph_edge_rows(
    graph: Any,
    *,
    graph_kind: str,
    idx: int,
    sentence: str,
    words: list[str],
    arc_prob: np.ndarray,
    rel_prob: np.ndarray,
    label2id: dict[str, int],
    unmatched_edges: list[Any],
    low_threshold: float,
) -> list[dict[str, Any]]:
    return graph_edges_with_conf(
        graph,
        graph_kind=graph_kind,
        idx=idx,
        sentence=sentence,
        words=words,
        arc_prob=arc_prob,
        rel_prob=rel_prob,
        label2id=label2id,
        unmatched_key_counts=edge_statuses(unmatched_edges),
        low_threshold=low_threshold,
    )


def edge_row_key(row: dict[str, Any]) -> tuple[Any, str, Any]:
    return (
        (
            row["source"]["text"],
            row["source"]["type"],
            json.dumps(row["source"]["spans"], ensure_ascii=False, sort_keys=True),
        ),
        str(row["label"]),
        (
            row["target"]["text"],
            row["target"]["type"],
            json.dumps(row["target"]["spans"], ensure_ascii=False, sort_keys=True),
        ),
    )


def node_id_to_word_idx(graph: Any, words: list[str]) -> dict[Any, int]:
    mapping = {}
    for node in graph.nodes():
        idx = node_word_idx(node, len(words))
        if idx is not None:
            mapping[node.ID] = idx
    return mapping


def word_idx_to_node(graph: Any, words: list[str]) -> dict[int, Any]:
    mapping = {}
    for node in graph.nodes():
        idx = node_word_idx(node, len(words))
        if idx is not None:
            mapping[idx] = node
    return mapping


def is_primary_at_child(graph: Any, words: list[str], parent_idx: int, child_idx: int) -> bool:
    nodes = word_idx_to_node(graph, words)
    parent = nodes.get(int(parent_idx))
    child = nodes.get(int(child_idx))
    if parent is None or child is None:
        return False
    edge = graph.get_edge(parent, child)
    if edge is None:
        return False
    labels = set(str(edge.label).split("+"))
    primary_at_edges = {
        "@modification_of", "@appositive_of", "@attribute_of",
        "@copula", "@discourse_of", "@expression", "@list",
        "@parataxis", "@ref_of", "@vocative_of",
    }
    return bool(labels & primary_at_edges)


def provenance_for_row(
    row: dict[str, Any],
    decoding_info: dict[str, Any],
    *,
    node_id_word_idx: dict[Any, int],
    graph: Any,
    words: list[str],
) -> list[dict[str, Any]]:
    out = []
    head = row.get("head")
    dep = row.get("dep")
    if head is None or dep is None:
        return out
    for item in decoding_info.get("phase5_upper_lower_edges", []) or []:
        from_idx = node_id_word_idx.get(item.get("from"))
        to_idx = node_id_word_idx.get(item.get("to"))
        if from_idx is None:
            from_idx = node_id_word_idx.get(str(item.get("from")))
        if to_idx is None:
            to_idx = node_id_word_idx.get(str(item.get("to")))
        if from_idx == int(head) and to_idx == int(dep):
            out.append({"phase": "phase5", **safe_json_value(item)})
    for item in decoding_info.get("phase4_multihead_fixes", []) or []:
        moved = item.get("moved_parent_edges") or []
        for moved_item in moved:
            if int(moved_item.get("from_parent", -1)) != int(head):
                continue
            phase4_head = int(item.get("head", -1))
            if is_primary_at_child(graph, words, phase4_head, int(dep)):
                out.append({
                    "phase": "phase4_multihead_direct_lower",
                    "head": phase4_head,
                    "from_parent": int(head),
                    "target_existing_at_child": int(dep),
                    "moved_label": str(moved_item.get("label")),
                })
        for child in item.get("path", []) or []:
            if int(child) == int(dep):
                out.append({"phase": "phase4_multihead_context", **safe_json_value(item)})
                break
    for failure in decoding_info.get("phase4_add_edge_failures", []) or []:
        if failure.get("context") != "phase4_move":
            continue
        meta = failure.get("meta") or {}
        phase4_head = meta.get("head")
        if phase4_head is None:
            continue
        if int(failure.get("parent", -1)) != int(head):
            continue
        if "@" not in str(failure.get("label")):
            continue
        if is_primary_at_child(graph, words, int(phase4_head), int(dep)):
            out.append({
                "phase": "phase4_failed_move_leak",
                "failed_parent": int(head),
                "failed_child": int(failure.get("child", -1)),
                "failed_label": str(failure.get("label")),
                "validator_code": str(failure.get("validator_code")),
                "target_existing_at_child": int(dep),
            })
    return out


def render_graph(
    graph: Any,
    *,
    graph_kind: str,
    idx: int,
    sentence: str,
    words: list[str],
    edge_rows: list[dict[str, Any]],
    output_dir: Path,
    low_threshold: float,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_counts = Counter(edge_row_key(row) for row in edge_rows)
    row_by_key: dict[Any, list[dict[str, Any]]] = {}
    for row in edge_rows:
        row_by_key.setdefault(edge_row_key(row), []).append(row)

    stem = f"case_{idx:05d}_{graph_kind}"
    dot_path = output_dir / f"{stem}.dot"
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    lines = [
        "digraph G {",
        "  graph [rankdir=LR, bgcolor=white, pad=0.2, nodesep=0.28, ranksep=0.48];",
        '  node [shape=box, style="rounded,filled", color="#475569", fillcolor="#f8fafc", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9, color="#64748b", arrowsize=0.7];',
        f"  label={dot_quote(f'idx={idx} {graph_kind}: {sentence}')} ;",
        '  labelloc="t";',
        '  fontsize=12;',
    ]

    for node in graph.nodes():
        lines.append(f"  {dot_quote(node.ID)} [label={dot_quote(graph_node_label(node, graph, words))}];")

    for parent, edge, child in graph.edges():
        source_info = node_info_json(get_node_info(parent, graph, with_type=True, with_spans=True))
        target_info = node_info_json(get_node_info(child, graph, with_type=True, with_spans=True))
        key = (
            (
                source_info["text"],
                source_info["type"],
                json.dumps(source_info["spans"], ensure_ascii=False, sort_keys=True),
            ),
            str(edge.label),
            (
                target_info["text"],
                target_info["type"],
                json.dumps(target_info["spans"], ensure_ascii=False, sort_keys=True),
            ),
        )
        row = row_by_key.get(key, [{}])[0]
        label = str(edge.label)
        conf = row.get("edge_confidence")
        status = row.get("status")
        attrs = {"label": label}
        if conf is not None:
            attrs["label"] = f"{label}\nconf={conf:.2e}"
        if status == "unmatched" and label == "@lower" and conf is not None and conf < low_threshold:
            attrs.update({"color": "#dc2626", "fontcolor": "#dc2626", "penwidth": "3.0"})
        elif status == "unmatched":
            attrs.update({"color": "#f97316", "fontcolor": "#ea580c", "penwidth": "2.0"})
        elif label == "@lower":
            attrs.update({"color": "#7c3aed", "fontcolor": "#6d28d9", "penwidth": "1.6"})
        attr_text = ", ".join(f"{k}={dot_quote(v)}" for k, v in attrs.items())
        lines.append(f"  {dot_quote(parent.ID)} -> {dot_quote(child.ID)} [{attr_text}];")
        row_counts[key] -= 1

    lines.append("}")
    dot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
    subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
    return {"dot": str(dot_path), "svg": str(svg_path), "png": str(png_path)}


def score_words(inference: WordLinguaInference, words: list[str]) -> tuple[np.ndarray, np.ndarray]:
    encoded = encode_words_like_training(inference.tokenizer, words, max_length=inference.config.max_length)
    num_nodes = int(encoded["num_nodes"][0].item())
    with torch.no_grad():
        out = inference.model(
            encoded["input_ids"].to(inference.device),
            encoded["attention_mask"].to(inference.device),
            encoded["word_to_subword"].to(inference.device),
            encoded["word_mask"].to(inference.device),
        )
    arc_prob = torch.sigmoid(out["arc_logits"][0, :num_nodes, :num_nodes].float()).detach().cpu().numpy()
    rel_logits_t = out["rel_logits"][0, :num_nodes, :num_nodes, :].float()
    edge_factor_logits_t = {
        name: logits[0, :num_nodes, :num_nodes, :].float()
        for name, logits in (out.get("edge_factor_logits") or {}).items()
    }
    rel_logits = combine_rel_logits_with_edge_factor_logits(
        rel_logits=rel_logits_t,
        edge_factor_logits=edge_factor_logits_t,
        edge_factor_targets_by_label=inference.edge_factor_targets_by_label,
        weight=inference.config.edge_factor_decode_weight,
    )
    rel_prob = torch.softmax(rel_logits, dim=-1).detach().cpu().numpy()
    return arc_prob, rel_prob


def analyze_case(
    sample: dict[str, Any],
    *,
    inference: WordLinguaInference,
    output_dir: Path,
    low_threshold: float,
) -> dict[str, Any]:
    idx = int(sample["idx"])
    sentence = sample["sentence"]
    result = inference.parse(sentence)
    words = result.words
    arc_prob, rel_prob = score_words(inference, words)

    pred_word_graph = result.word_lingua_graph
    gold_word_graph = sample["gold_word_graph"]
    metric = graph_match_metric(pred_word_graph, gold_word_graph, edge_match="exact", with_spans=True)
    pred_rows = graph_edge_rows(
        pred_word_graph,
        graph_kind="pred",
        idx=idx,
        sentence=sentence,
        words=words,
        arc_prob=arc_prob,
        rel_prob=rel_prob,
        label2id=inference.label2id,
        unmatched_edges=metric["unmatched_pred_edges"],
        low_threshold=low_threshold,
    )
    gold_rows = graph_edge_rows(
        gold_word_graph,
        graph_kind="gold",
        idx=idx,
        sentence=sentence,
        words=words,
        arc_prob=arc_prob,
        rel_prob=rel_prob,
        label2id=inference.label2id,
        unmatched_edges=metric["unmatched_true_edges"],
        low_threshold=low_threshold,
    )
    pred_node_id_word_idx = node_id_to_word_idx(pred_word_graph, words)
    for row in pred_rows:
        if row["status"] == "unmatched":
            row["provenance"] = provenance_for_row(
                row,
                result.decoding_info,
                node_id_word_idx=pred_node_id_word_idx,
                graph=pred_word_graph,
                words=words,
            )

    paths = {
        "pred": render_graph(
            pred_word_graph,
            graph_kind="pred",
            idx=idx,
            sentence=sentence,
            words=words,
            edge_rows=pred_rows,
            output_dir=output_dir,
            low_threshold=low_threshold,
        ),
        "gold": render_graph(
            gold_word_graph,
            graph_kind="gold",
            idx=idx,
            sentence=sentence,
            words=words,
            edge_rows=gold_rows,
            output_dir=output_dir,
            low_threshold=low_threshold,
        ),
    }
    low_unmatched_lower = [
        row for row in pred_rows
        if row["status"] == "unmatched"
        and row["label"] == "@lower"
        and row["edge_confidence"] is not None
        and row["edge_confidence"] < low_threshold
    ]
    return {
        "idx": idx,
        "line": idx + 1,
        "sentence": sentence,
        "words": words,
        "exact_same": bool(metric["exact_same"]),
        "word_counts": {
            "pred_nodes": metric["pred_node_num"],
            "true_nodes": metric["true_node_num"],
            "matched_nodes": metric["matched_node_num"],
            "pred_edges": metric["pred_edge_num"],
            "true_edges": metric["true_edge_num"],
            "matched_edges": metric["matched_edge_num"],
            "node_f1": metric["node_f1"],
            "edge_f1": metric["edge_f1"],
        },
        "low_unmatched_lower": low_unmatched_lower,
        "unmatched_pred_edges": [edge_tuple_json(edge) for edge in metric["unmatched_pred_edges"]],
        "unmatched_true_edges": [edge_tuple_json(edge) for edge in metric["unmatched_true_edges"]],
        "phase5_upper_lower_edges": safe_json_value(result.decoding_info.get("phase5_upper_lower_edges", [])),
        "phase4_multihead_fixes": safe_json_value(result.decoding_info.get("phase4_multihead_fixes", [])),
        "phase4_force_connects": safe_json_value(result.decoding_info.get("phase4_force_connects", [])),
        "phase4_add_edge_failures": safe_json_value(result.decoding_info.get("phase4_add_edge_failures", [])),
        "single_root_repairs": safe_json_value(result.decoding_info.get("single_root_repairs", [])),
        "decoding_info_keys": sorted(result.decoding_info.keys()),
        "paths": paths,
    }


def write_index(records: list[dict[str, Any]], output_dir: Path) -> None:
    lines = ["# V11 low-confidence @lower case visualizations", ""]
    for record in records:
        lines.append(f"## idx {record['idx']} line {record['line']}")
        lines.append("")
        lines.append(record["sentence"])
        lines.append("")
        lines.append("Low unmatched @lower edges:")
        for row in record["low_unmatched_lower"]:
            prov = row.get("provenance") or []
            parts = []
            for p in prov:
                if p.get("case") is not None:
                    parts.append(f"{p.get('phase')}:{p.get('case')}")
                else:
                    parts.append(str(p.get("phase")))
            prov_text = ", ".join(parts) or "unknown"
            lines.append(
                f"- {row.get('head')}:{row.get('head_word')} -> {row.get('dep')}:{row.get('dep_word')} "
                f"{row.get('label')} conf={row.get('edge_confidence'):.3e} "
                f"arc={row.get('arc_prob'):.3e} label={row.get('label_prob'):.3e} provenance={prov_text}"
            )
        lines.append("")
        lines.append(f"- pred SVG: `{record['paths']['pred']['svg']}`")
        lines.append(f"- gold SVG: `{record['paths']['gold']['svg']}`")
        lines.append("")
    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--idx", type=int, action="append", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--low_threshold", type=float, default=0.50)
    parser.add_argument("--edge_factor_decode_weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    inference = WordLinguaInference.from_pretrained(
        args.model_dir,
        device=device,
        config=InferenceConfig(
            max_length=args.max_length,
            edge_factor_decode_weight=args.edge_factor_decode_weight,
        ),
    )
    inference.config.debug = True
    inference.decoder.debug = True
    inference.decoder._debug_visualize_step = lambda *args, **kwargs: None
    inference.decoder._debug_log_phase34_report = lambda *args, **kwargs: None

    wanted = set(args.idx)
    samples_by_idx = {sample["idx"]: sample for sample in load_samples(Path(args.data_path)) if sample["idx"] in wanted}
    missing = sorted(wanted - set(samples_by_idx))
    if missing:
        raise SystemExit(f"missing sample idx: {missing}")

    records = [
        analyze_case(samples_by_idx[idx], inference=inference, output_dir=output_dir, low_threshold=args.low_threshold)
        for idx in args.idx
    ]
    (output_dir / "case_analysis.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    write_index(records, output_dir)
    print(json.dumps({
        "output_dir": str(output_dir),
        "case_analysis": str(output_dir / "case_analysis.json"),
        "index": str(output_dir / "index.md"),
        "cases": [
            {
                "idx": record["idx"],
                "low_unmatched_lower_count": len(record["low_unmatched_lower"]),
                "phase5_edge_count": len(record["phase5_upper_lower_edges"]),
            }
            for record in records
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
