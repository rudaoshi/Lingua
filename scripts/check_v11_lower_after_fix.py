#!/usr/bin/env python3
"""Recount old low-confidence @lower cases after decoder changes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import time

import torch
from tqdm import tqdm

from analyze_v11_train_edge_confidence import (
    edge_key,
    graph_edges_with_conf,
    load_samples,
)
from lingua.learn.wordgraph.decoding.inference import (
    InferenceConfig,
    WordLinguaInference,
    encode_batch_words_like_training,
)
from lingua.utils.metric import graph_match_metric


def old_bad_indices(edge_path: Path) -> tuple[list[int], int]:
    idxs = set()
    edge_count = 0
    with edge_path.open(encoding="utf-8") as f:
        for line in f:
            edge = json.loads(line)
            if (
                edge.get("graph") == "pred"
                and edge.get("status") == "unmatched"
                and edge.get("label") == "@lower"
                and edge.get("edge_confidence") is not None
                and edge.get("edge_confidence") < 0.5
            ):
                idxs.add(int(edge["idx"]))
                edge_count += 1
    return sorted(idxs), edge_count


def infer_decoder_source(decoding_info: dict, parent: int, child: int, label: str) -> str:
    for component in decoding_info.get("phase3_components", []) or []:
        selected = component.get("selected_edge") or {}
        if (
            selected.get("parent") == parent
            and selected.get("child") == child
            and selected.get("label") == label
        ):
            return "phase3"

    for repair in decoding_info.get("single_root_repairs", []) or []:
        if (
            repair.get("parent") == parent
            and repair.get("child") == child
            and repair.get("label") == label
        ):
            return "single_root_repair"

    for fix in decoding_info.get("phase4_multihead_fixes", []) or []:
        for moved in fix.get("moved_parent_edges", []) or []:
            if (
                moved.get("from_parent") == parent
                and moved.get("new_child") == child
                and moved.get("label") == label
            ):
                return "phase4_move"
            for added in moved.get("added_lower_edges", []) or []:
                if (
                    added.get("from_word_idx") == parent
                    and added.get("to_word_idx") == child
                    and added.get("label") == label
                ):
                    return "phase4_added_lower"

    for added in decoding_info.get("phase5_upper_lower_edges", []) or []:
        if (
            added.get("from_word_idx") == parent
            and added.get("to_word_idx") == child
            and added.get("label") == label
        ):
            return "phase5"

    return "phase2_or_existing"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--old_edge_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--low_threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    idxs, old_edge_count = old_bad_indices(Path(args.old_edge_path))
    samples = {sample["idx"]: sample for sample in load_samples(Path(args.data_path)) if sample["idx"] in idxs}

    inference = WordLinguaInference.from_pretrained(
        args.model_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        config=InferenceConfig(max_length=args.max_length, edge_factor_decode_weight=0.0, debug=True),
    )
    inference.decoder.debug = True
    inference.decoder._debug_visualize_step = lambda *args, **kwargs: None
    inference.decoder._debug_log_phase34_report = lambda *args, **kwargs: None

    summary = Counter()
    skipped = Counter()
    remaining_sources = Counter()
    remaining_rows = []
    started = time.time()

    for start in tqdm(range(0, len(idxs), args.batch_size), desc="after-fix old lower cases"):
        batch_idxs = idxs[start : start + args.batch_size]
        batch = [samples[idx] for idx in batch_idxs]
        results = inference.parse_batch([sample["sentence"] for sample in batch])
        batch_words = [
            result.words if result.words else sample["sentence"].split()
            for sample, result in zip(batch, results)
        ]
        encoded = encode_batch_words_like_training(
            inference.tokenizer,
            batch_words,
            max_length=args.max_length,
        )
        with torch.no_grad():
            output = inference.model(
                encoded["input_ids"].to(inference.device),
                encoded["attention_mask"].to(inference.device),
                encoded["word_to_subword"].to(inference.device),
                encoded["word_mask"].to(inference.device),
            )

        for batch_idx, (idx, sample, result, words) in enumerate(zip(batch_idxs, batch, results, batch_words)):
            summary["cases"] += 1
            pred_graph = result.word_lingua_graph
            gold_graph = sample["gold_word_graph"]
            if pred_graph is None or gold_graph is None:
                summary["metric_missing"] += 1
                continue

            metric = graph_match_metric(pred_graph, gold_graph, edge_match="exact", with_spans=True)
            if not metric["exact_same"]:
                summary["word_nonexact_after"] += 1

            num_nodes = int(encoded["num_nodes"][batch_idx].item())
            arc_prob = torch.sigmoid(
                output["arc_logits"][batch_idx, :num_nodes, :num_nodes].float()
            ).detach().cpu().numpy()
            rel_prob = torch.softmax(
                output["rel_logits"][batch_idx, :num_nodes, :num_nodes, :].float(),
                dim=-1,
            ).detach().cpu().numpy()
            pred_edges = graph_edges_with_conf(
                pred_graph,
                graph_kind="pred",
                idx=idx,
                sentence=sample["sentence"],
                words=words,
                arc_prob=arc_prob,
                rel_prob=rel_prob,
                label2id=inference.label2id,
                unmatched_key_counts=Counter(edge_key(edge) for edge in metric["unmatched_pred_edges"]),
                low_threshold=args.low_threshold,
            )
            lows = [
                edge for edge in pred_edges
                if edge["status"] == "unmatched"
                and edge["label"] == "@lower"
                and edge["edge_confidence"] is not None
                and edge["edge_confidence"] < args.low_threshold
            ]
            if lows:
                summary["cases_with_low_unmatched_lower_after"] += 1
                summary["low_unmatched_lower_after"] += len(lows)
                for edge in lows:
                    source = infer_decoder_source(
                        result.decoding_info,
                        int(edge["head"]),
                        int(edge["dep"]),
                        str(edge["label"]),
                    )
                    edge["decoder_source"] = source
                    remaining_sources[source] += 1
                remaining_rows.extend(lows[:10])

            for item in result.decoding_info.get("skipped_hierarchy_edges", []) or []:
                skipped[f"{item.get('source')}:{item.get('reason')}"] += 1

    summary_json = {
        "old_bad_case_count": len(idxs),
        "old_low_unmatched_lower_edges": old_edge_count,
        "created_seconds": round(time.time() - started, 2),
        "summary": dict(summary),
        "remaining_low_lower_sources": dict(remaining_sources.most_common()),
        "skipped_hierarchy_edges": dict(skipped.most_common()),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "remaining_low_lower_examples.jsonl").open("w", encoding="utf-8") as f:
        for row in remaining_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
