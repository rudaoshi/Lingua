#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_UD_UPOS_DIR = DEFAULT_DATA_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lingua.learn.hierarchical.dataset import (
    HierarchicalOracleDataset,
    HierarchicalTrainingExample,
    HierarchicalVocab,
    build_hierarchical_examples_from_oracle,
    collate_hierarchical_batch,
    load_ud_upos_by_line,
)
from lingua.learn.hierarchical.model import HierarchicalTokenSetParser
from lingua.learn.hierarchical.oracle import DEFAULT_VIEW, NON_AUX_KIND, extract_hierarchical_oracle
from lingua.learn.hierarchical.training import (
    forward_hierarchical_model,
    move_batch_to_device,
    save_metrics,
)


@dataclass(frozen=True)
class AnalysisRecord:
    example: HierarchicalTrainingExample
    sentence: str | None
    words: tuple[str, ...]
    root_token_sets: tuple[tuple[int, ...], ...]
    unit: dict[str, Any]


class MetricAccumulator:
    def __init__(self) -> None:
        self.total_units = 0
        self.head_pos_correct = 0
        self.head_aux_kind_correct = 0
        self.head_identity_correct = 0
        self.decision_correct = 0
        self.unit_exact = 0
        self.role_tp = 0
        self.role_fp = 0
        self.role_fn = 0
        self.role_token_exact = 0
        self.role_token_total = 0

    def update(
        self,
        *,
        pos_match: bool,
        aux_match: bool,
        decision_match: bool,
        unit_role_match: bool,
        role_tp: int,
        role_fp: int,
        role_fn: int,
        role_token_exact: int,
        role_token_total: int,
    ) -> None:
        self.total_units += 1
        self.head_pos_correct += int(pos_match)
        self.head_aux_kind_correct += int(aux_match)
        self.head_identity_correct += int(pos_match and aux_match)
        self.decision_correct += int(decision_match)
        self.unit_exact += int(pos_match and aux_match and decision_match and unit_role_match)
        self.role_tp += role_tp
        self.role_fp += role_fp
        self.role_fn += role_fn
        self.role_token_exact += role_token_exact
        self.role_token_total += role_token_total

    def to_dict(self) -> dict[str, float]:
        role_precision = self.role_tp / (self.role_tp + self.role_fp + 1e-8)
        role_recall = self.role_tp / (self.role_tp + self.role_fn + 1e-8)
        role_f1 = 2 * role_precision * role_recall / (role_precision + role_recall + 1e-8)
        return {
            "head_pos_accuracy": self.head_pos_correct / (self.total_units + 1e-8),
            "head_aux_kind_accuracy": self.head_aux_kind_correct / (self.total_units + 1e-8),
            "head_identity_accuracy": self.head_identity_correct / (self.total_units + 1e-8),
            "decision_accuracy": self.decision_correct / (self.total_units + 1e-8),
            "role_precision": role_precision,
            "role_recall": role_recall,
            "role_f1": role_f1,
            "role_token_exact": self.role_token_exact / (self.role_token_total + 1e-8),
            "unit_exact": self.unit_exact / (self.total_units + 1e-8),
            "num_units": float(self.total_units),
            "role_tp": float(self.role_tp),
            "role_fp": float(self.role_fp),
            "role_fn": float(self.role_fn),
            "role_token_total": float(self.role_token_total),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate and analyze saved hierarchical token-set oracle predictions."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config_path", type=Path)
    parser.add_argument("--vocab_path", type=Path)
    parser.add_argument("--train_path", type=Path)
    parser.add_argument("--dev_path", type=Path)
    parser.add_argument("--test_path", type=Path)
    parser.add_argument("--ud_upos_dir", type=Path)
    parser.add_argument("--disable_ud_upos", action="store_true")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_dev_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--max_train_units", type=int)
    parser.add_argument("--max_dev_units", type=int)
    parser.add_argument("--max_test_units", type=int)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--no_predictions", action="store_true")
    parser.add_argument("--no_errors", action="store_true")
    parser.add_argument("--no_force_one_role_per_unit_token", action="store_true")
    args = parser.parse_args()

    config_path = args.config_path or args.checkpoint.parent / "config.json"
    config = load_json(config_path) if config_path.exists() else {}
    run_args = dict(config.get("args", {}))
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    checkpoint_config = dict(checkpoint.get("config", {}))
    merged_args = {**run_args, **checkpoint_config}

    train_path = args.train_path or Path(merged_args.get("train_path", DEFAULT_DATA_DIR / "train.jsonl"))
    dev_path = args.dev_path or Path(merged_args.get("dev_path", DEFAULT_DATA_DIR / "dev.jsonl"))
    test_path = args.test_path or Path(merged_args.get("test_path", DEFAULT_DATA_DIR / "test.jsonl"))
    vocab_path = args.vocab_path or args.checkpoint.parent / "vocab.json"
    batch_size = args.batch_size or int(merged_args.get("batch_size", 64))
    threshold = args.threshold if args.threshold is not None else float(merged_args.get("threshold", 0.5))
    device = resolve_device(args.device)

    vocab = HierarchicalVocab.load(vocab_path)
    model = build_model_from_checkpoint(checkpoint, vocab, merged_args).to(device)
    model.load_state_dict(compatible_model_state_dict(checkpoint, model))
    tokenizer = build_tokenizer_from_config(merged_args)
    ud_upos_by_split = load_ud_upos_maps(args, merged_args)

    splits = {
        "train": (train_path, args.max_train_samples, args.max_train_units),
        "dev": (dev_path, args.max_dev_samples, args.max_dev_units),
        "test": (test_path, args.max_test_samples, args.max_test_units),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "config_path": str(config_path),
        "vocab_path": str(vocab_path),
        "output_dir": str(args.output_dir),
        "batch_size": batch_size,
        "threshold": threshold,
        "force_one_role_per_unit_token": not args.no_force_one_role_per_unit_token,
        "device": str(device),
        "splits": {},
        "files": [],
    }

    all_metrics: dict[str, Any] = {}
    for split_name, (data_path, max_samples, max_units) in splits.items():
        records = build_analysis_records(
            data_path,
            max_samples=max_samples,
            max_units=max_units,
            ud_upos_by_line=ud_upos_by_split.get(split_name),
        )
        split_output = evaluate_split(
            split_name=split_name,
            records=records,
            vocab=vocab,
            model=model,
            tokenizer=tokenizer,
            max_subword_length=int(merged_args.get("max_subword_length", 512)),
            device=device,
            batch_size=batch_size,
            threshold=threshold,
            force_one_role_per_unit_token=not args.no_force_one_role_per_unit_token,
            output_dir=args.output_dir,
            top_k=args.top_k,
            write_predictions=not args.no_predictions,
            write_errors=not args.no_errors,
        )
        all_metrics[split_name] = split_output["summary"]
        manifest["splits"][split_name] = {
            "data_path": str(data_path),
            "num_units": len(records),
        }
        manifest["files"].extend(split_output["files"])
        print(json.dumps({split_name: split_output["summary"]["metrics"]}, ensure_ascii=False))

    metrics_path = args.output_dir / "metrics_by_split.json"
    save_metrics(metrics_path, all_metrics)
    manifest["files"].append(str(metrics_path))
    manifest_path = args.output_dir / "manifest.json"
    save_metrics(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


def build_analysis_records(
    data_path: Path,
    max_samples: int | None = None,
    max_units: int | None = None,
    ud_upos_by_line: dict[int, tuple[str, ...]] | None = None,
) -> list[AnalysisRecord]:
    records: list[AnalysisRecord] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_samples is not None and line_number > max_samples:
                break
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample_id = str(sample.get("id") or f"{data_path.name}:{line_number}")
            oracle = extract_hierarchical_oracle(sample).to_dict()
            words = tuple(str(word) for word in oracle["words"])
            root_token_sets = tuple(tuple(int(token_id) for token_id in item) for item in oracle["root_token_sets"])
            examples = build_hierarchical_examples_from_oracle(
                sample_id,
                oracle,
                upos_tags=(ud_upos_by_line or {}).get(line_number),
            )

            for example in examples:
                unit = oracle["units"][example.unit_index]
                records.append(
                    AnalysisRecord(
                        example=example,
                        sentence=sample.get("sentence"),
                        words=words,
                        root_token_sets=root_token_sets,
                        unit=unit,
                    )
                )
                if max_units is not None and len(records) >= max_units:
                    return records
    return records


@torch.no_grad()
def evaluate_split(
    *,
    split_name: str,
    records: list[AnalysisRecord],
    vocab: HierarchicalVocab,
    model: HierarchicalTokenSetParser,
    tokenizer: Any | None,
    max_subword_length: int,
    device: torch.device,
    batch_size: int,
    threshold: float,
    force_one_role_per_unit_token: bool,
    output_dir: Path,
    top_k: int,
    write_predictions: bool,
    write_errors: bool,
) -> dict[str, Any]:
    model.eval()
    dataset = HierarchicalOracleDataset([record.example for record in records], vocab)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_hierarchical_batch(
            batch,
            pad_id=vocab.pad_id,
            upos_pad_id=vocab.upos_pad_id,
            tokenizer=tokenizer,
            max_subword_length=max_subword_length,
        ),
    )

    id_to_unit_pos = invert_mapping(vocab.unit_pos2id)
    id_to_aux_kind = invert_mapping(vocab.aux_kind2id)
    id_to_decision = invert_mapping(vocab.decision2id)
    id_to_role = invert_mapping(vocab.role_label2id)
    role_ids = sorted(id_to_role)

    metrics = MetricAccumulator()
    role_label_metrics: dict[str, Counter[str]] = defaultdict(Counter)
    role_family_metrics: dict[str, Counter[str]] = defaultdict(Counter)
    head_pos_confusion: Counter[tuple[str, str]] = Counter()
    head_aux_kind_confusion: Counter[tuple[str, str]] = Counter()
    decision_confusion: Counter[tuple[str, str]] = Counter()
    role_set_confusion: Counter[tuple[str, str]] = Counter()
    bio_prefix_confusion: Counter[tuple[str, str]] = Counter()
    missing_role_labels: Counter[str] = Counter()
    spurious_role_labels: Counter[str] = Counter()
    missing_role_families: Counter[str] = Counter()
    spurious_role_families: Counter[str] = Counter()
    error_type_counts: Counter[str] = Counter()
    breakdowns: dict[str, dict[str, MetricAccumulator]] = defaultdict(lambda: defaultdict(MetricAccumulator))
    top_error_records: list[dict[str, Any]] = []

    prediction_path = output_dir / f"{split_name}_predictions.jsonl"
    error_path = output_dir / f"{split_name}_errors.jsonl"
    prediction_handle = prediction_path.open("w", encoding="utf-8") if write_predictions else None
    error_handle = error_path.open("w", encoding="utf-8") if write_errors else None

    cursor = 0
    try:
        for batch in loader:
            batch_size_actual = len(batch["sample_ids"])
            batch_records = records[cursor : cursor + batch_size_actual]
            cursor += batch_size_actual

            batch = move_batch_to_device(batch, device)
            output = forward_hierarchical_model(model, batch)
            unit_pos_pred = output["head_pos_logits"].argmax(dim=-1).cpu()
            aux_kind_pred = output["head_aux_kind_logits"].argmax(dim=-1).cpu()
            decision_pred = output["decision_logits"].argmax(dim=-1).cpu()
            role_pred = model.decode_role_labels(
                output["role_logits"],
                batch["active_unit_mask"],
            ).cpu()

            for row_index, record in enumerate(batch_records):
                prediction = build_unit_prediction(
                    split_name=split_name,
                    record=record,
                    pred_head_pos=id_to_unit_pos[int(unit_pos_pred[row_index])],
                    pred_head_aux_kind=id_to_aux_kind[int(aux_kind_pred[row_index])],
                    pred_decision=id_to_decision[int(decision_pred[row_index])],
                    pred_role_sets=predicted_role_sets(
                        role_pred[row_index],
                        record.example.input_token_ids,
                        id_to_role,
                        role_ids,
                    ),
                )
                update_analysis_counters(
                    prediction=prediction,
                    metrics=metrics,
                    role_label_metrics=role_label_metrics,
                    role_family_metrics=role_family_metrics,
                    head_pos_confusion=head_pos_confusion,
                    head_aux_kind_confusion=head_aux_kind_confusion,
                    decision_confusion=decision_confusion,
                    role_set_confusion=role_set_confusion,
                    bio_prefix_confusion=bio_prefix_confusion,
                    missing_role_labels=missing_role_labels,
                    spurious_role_labels=spurious_role_labels,
                    missing_role_families=missing_role_families,
                    spurious_role_families=spurious_role_families,
                    error_type_counts=error_type_counts,
                    breakdowns=breakdowns,
                )

                if prediction_handle is not None:
                    prediction_handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

                if not prediction["matches"]["unit_exact"]:
                    if error_handle is not None:
                        error_handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    maybe_add_top_error(top_error_records, prediction, top_k)
    finally:
        if prediction_handle is not None:
            prediction_handle.close()
        if error_handle is not None:
            error_handle.close()

    summary = {
        "metrics": metrics.to_dict(),
        "error_type_counts": dict(error_type_counts.most_common()),
        "role_label_metrics": counter_metrics(role_label_metrics),
        "role_family_metrics": counter_metrics(role_family_metrics),
        "head_pos_confusion": confusion_to_rows(head_pos_confusion),
        "head_aux_kind_confusion": confusion_to_rows(head_aux_kind_confusion),
        "decision_confusion": confusion_to_rows(decision_confusion),
        "role_set_confusion_top": confusion_to_rows(role_set_confusion, limit=top_k),
        "bio_prefix_confusion_top": confusion_to_rows(bio_prefix_confusion, limit=top_k),
        "missing_role_labels_top": counter_rows(missing_role_labels, limit=top_k),
        "spurious_role_labels_top": counter_rows(spurious_role_labels, limit=top_k),
        "missing_role_families_top": counter_rows(missing_role_families, limit=top_k),
        "spurious_role_families_top": counter_rows(spurious_role_families, limit=top_k),
        "breakdowns": format_breakdowns(breakdowns),
        "top_error_examples": top_error_records,
    }

    summary_path = output_dir / f"{split_name}_summary.json"
    save_metrics(summary_path, summary)
    files = [str(summary_path)]
    if write_predictions:
        files.append(str(prediction_path))
    if write_errors:
        files.append(str(error_path))
    return {"summary": summary, "files": files}


def build_unit_prediction(
    *,
    split_name: str,
    record: AnalysisRecord,
    pred_head_pos: str,
    pred_head_aux_kind: str,
    pred_decision: str,
    pred_role_sets: dict[int, tuple[str, ...]],
) -> dict[str, Any]:
    gold_role_sets = {
        token_id: tuple(label for label in labels if label != "O")
        for token_id, labels in record.example.role_labels_by_token
    }
    role_errors = []
    role_tp = role_fp = role_fn = 0
    role_token_exact = 0
    role_token_total = 0

    gold_roles_by_token = []
    pred_roles_by_token = []
    for token_id in record.example.input_token_ids:
        gold_labels = tuple(sorted(gold_role_sets.get(token_id, ())))
        pred_labels = tuple(sorted(pred_role_sets.get(token_id, ())))
        missing = tuple(label for label in gold_labels if label not in pred_labels)
        spurious = tuple(label for label in pred_labels if label not in gold_labels)
        role_tp += len([label for label in pred_labels if label in gold_labels])
        role_fp += len(spurious)
        role_fn += len(missing)
        role_token_total += 1
        role_token_exact += int(not missing and not spurious)

        token_payload = {
            "token_id": token_id,
            "token": record.words[token_id] if 0 <= token_id < len(record.words) else "",
            "labels": list(gold_labels) if gold_labels else ["O"],
        }
        pred_payload = {
            "token_id": token_id,
            "token": record.words[token_id] if 0 <= token_id < len(record.words) else "",
            "labels": list(pred_labels) if pred_labels else ["O"],
        }
        gold_roles_by_token.append(token_payload)
        pred_roles_by_token.append(pred_payload)
        if missing or spurious:
            role_errors.append(
                {
                    "token_id": token_id,
                    "token": token_payload["token"],
                    "gold_labels": token_payload["labels"],
                    "pred_labels": pred_payload["labels"],
                    "missing": list(missing),
                    "spurious": list(spurious),
                }
            )

    pos_match = pred_head_pos == record.example.unit_pos
    aux_match = pred_head_aux_kind == record.example.aux_kind
    decision_match = pred_decision == record.example.decision
    unit_role_match = role_token_exact == role_token_total
    unit_exact = pos_match and aux_match and decision_match and unit_role_match
    error_types = []
    if not pos_match:
        error_types.append("head_pos")
    if not aux_match:
        error_types.append("head_aux_kind")
    if not decision_match:
        error_types.append("decision")
    if role_fn:
        error_types.append("role_missing")
    if role_fp:
        error_types.append("role_spurious")

    return {
        "split": split_name,
        "sample_id": record.example.sample_id,
        "unit_index": record.example.unit_index,
        "example_index": record.example.example_index,
        "parent_unit_index": record.example.parent_unit_index,
        "sentence": record.sentence,
        "words": list(record.words),
        "root_token_sets": [list(token_ids) for token_ids in record.root_token_sets],
        "unit_features": unit_features(record.unit),
        "gold": {
            "view_kind": record.example.view_kind,
            "active_unit_pos": record.example.unit_pos,
            "head_pos": record.example.unit_pos,
            "head_aux_kind": record.example.aux_kind,
            "decision": record.example.decision,
            "input_token_ids": list(record.example.input_token_ids),
            "input_tokens": [
                record.words[token_id] if 0 <= token_id < len(record.words) else ""
                for token_id in record.example.input_token_ids
            ],
            "roles_by_token": gold_roles_by_token,
            "children": record.unit.get("children", []),
            "shared_links": record.unit.get("shared_links", []),
            "external_links": record.unit.get("external_links", []),
        },
        "pred": {
            "head_pos": pred_head_pos,
            "head_aux_kind": pred_head_aux_kind,
            "decision": pred_decision,
            "roles_by_token": pred_roles_by_token,
        },
        "matches": {
            "head_pos": pos_match,
            "head_aux_kind": aux_match,
            "head_identity": pos_match and aux_match,
            "decision": decision_match,
            "roles": unit_role_match,
            "unit_exact": unit_exact,
        },
        "role_counts": {
            "tp": role_tp,
            "fp": role_fp,
            "fn": role_fn,
            "token_exact": role_token_exact,
            "token_total": role_token_total,
        },
        "role_errors": role_errors,
        "error_types": error_types,
    }


def update_analysis_counters(
    *,
    prediction: dict[str, Any],
    metrics: MetricAccumulator,
    role_label_metrics: dict[str, Counter[str]],
    role_family_metrics: dict[str, Counter[str]],
    head_pos_confusion: Counter[tuple[str, str]],
    head_aux_kind_confusion: Counter[tuple[str, str]],
    decision_confusion: Counter[tuple[str, str]],
    role_set_confusion: Counter[tuple[str, str]],
    bio_prefix_confusion: Counter[tuple[str, str]],
    missing_role_labels: Counter[str],
    spurious_role_labels: Counter[str],
    missing_role_families: Counter[str],
    spurious_role_families: Counter[str],
    error_type_counts: Counter[str],
    breakdowns: dict[str, dict[str, MetricAccumulator]],
) -> None:
    matches = prediction["matches"]
    counts = prediction["role_counts"]
    metrics.update(
        pos_match=matches["head_pos"],
        aux_match=matches["head_aux_kind"],
        decision_match=matches["decision"],
        unit_role_match=matches["roles"],
        role_tp=int(counts["tp"]),
        role_fp=int(counts["fp"]),
        role_fn=int(counts["fn"]),
        role_token_exact=int(counts["token_exact"]),
        role_token_total=int(counts["token_total"]),
    )
    update_breakdowns(prediction, breakdowns)

    head_pos_confusion[(prediction["gold"]["head_pos"], prediction["pred"]["head_pos"])] += 1
    head_aux_kind_confusion[(prediction["gold"]["head_aux_kind"], prediction["pred"]["head_aux_kind"])] += 1
    decision_confusion[(prediction["gold"]["decision"], prediction["pred"]["decision"])] += 1
    for error_type in prediction["error_types"]:
        error_type_counts[error_type] += 1
    if not prediction["error_types"]:
        error_type_counts["exact"] += 1

    for gold_token, pred_token in zip(prediction["gold"]["roles_by_token"], prediction["pred"]["roles_by_token"]):
        gold_labels = set(without_o(gold_token["labels"]))
        pred_labels = set(without_o(pred_token["labels"]))
        role_set_confusion[(role_set_key(gold_labels), role_set_key(pred_labels))] += 1
        for label in gold_labels & pred_labels:
            role_label_metrics[label]["tp"] += 1
            role_family_metrics[label_family(label)]["tp"] += 1
        for label in gold_labels - pred_labels:
            role_label_metrics[label]["fn"] += 1
            role_family_metrics[label_family(label)]["fn"] += 1
            missing_role_labels[label] += 1
            missing_role_families[label_family(label)] += 1
        for label in pred_labels - gold_labels:
            role_label_metrics[label]["fp"] += 1
            role_family_metrics[label_family(label)]["fp"] += 1
            spurious_role_labels[label] += 1
            spurious_role_families[label_family(label)] += 1
        for gold_label in gold_labels:
            for pred_label in pred_labels:
                if gold_label != pred_label and label_family(gold_label) == label_family(pred_label):
                    bio_prefix_confusion[(gold_label, pred_label)] += 1


def update_breakdowns(
    prediction: dict[str, Any],
    breakdowns: dict[str, dict[str, MetricAccumulator]],
) -> None:
    features = prediction["unit_features"]
    keys = {
        "view_kind": prediction["gold"]["view_kind"],
        "active_unit_pos": prediction["gold"]["active_unit_pos"],
        "head_aux_kind": prediction["gold"]["head_aux_kind"],
        "is_aux": str(prediction["gold"]["head_aux_kind"] != NON_AUX_KIND).lower(),
        "decision": prediction["gold"]["decision"],
        "depth_bucket": depth_bucket(features["depth"]),
        "input_size_bucket": count_bucket(features["input_token_count"]),
        "child_count_bucket": count_bucket(features["child_count"]),
        "is_root": str(features["is_root"]).lower(),
        "incoming_shared": str(features["incoming_count"] > 1).lower(),
        "has_shared_child": str(features["has_shared_child"]).lower(),
        "has_shared_arg": str(features["has_shared_arg"]).lower(),
        "has_external_child": str(features["has_external_child"]).lower(),
        "has_overlapping_roles": str(features["has_overlapping_roles"]).lower(),
        "is_discontinuous": str(features["is_discontinuous"]).lower(),
    }
    for breakdown_name, value in keys.items():
        breakdowns[breakdown_name][value].update(
            pos_match=prediction["matches"]["head_pos"],
            aux_match=prediction["matches"]["head_aux_kind"],
            decision_match=prediction["matches"]["decision"],
            unit_role_match=prediction["matches"]["roles"],
            role_tp=int(prediction["role_counts"]["tp"]),
            role_fp=int(prediction["role_counts"]["fp"]),
            role_fn=int(prediction["role_counts"]["fn"]),
            role_token_exact=int(prediction["role_counts"]["token_exact"]),
            role_token_total=int(prediction["role_counts"]["token_total"]),
        )


def predicted_role_sets(
    role_pred_row: torch.Tensor,
    input_token_ids: Iterable[int],
    id_to_role: dict[int, str],
    role_ids: list[int],
) -> dict[int, tuple[str, ...]]:
    output: dict[int, tuple[str, ...]] = {}
    for token_id in input_token_ids:
        if token_id < role_pred_row.size(0):
            label = id_to_role[int(role_pred_row[token_id])]
            output[int(token_id)] = () if label == "O" else (label,)
        else:
            output[int(token_id)] = ()
    return output


def load_ud_upos_maps(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, dict[int, tuple[str, ...]]]:
    if args.disable_ud_upos or bool(config.get("disable_ud_upos", False)):
        return {}
    ud_upos_dir = args.ud_upos_dir or Path(
        config.get("ud_upos_dir", DEFAULT_UD_UPOS_DIR)
    )
    if not ud_upos_dir.exists():
        return {}
    return {
        split: load_ud_upos_by_line(ud_upos_path(ud_upos_dir, split))
        for split in ("train", "dev", "test")
        if ud_upos_path(ud_upos_dir, split).exists()
    }


def ud_upos_path(base_dir: Path, split: str) -> Path:
    preferred = base_dir / f"{split}_ud.jsonl"
    if preferred.exists():
        return preferred
    return base_dir / f"{split}.jsonl"


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    vocab: HierarchicalVocab,
    config: dict[str, Any],
) -> HierarchicalTokenSetParser:
    state = checkpoint["model_state_dict"]
    encoder_type = str(config.get("encoder_type", "gru"))
    embedding_dim = int(config.get("embedding_dim") or state.get("word_embedding.weight", torch.empty(0, 128)).shape[1])
    if encoder_type == "gru":
        hidden_size = int(config.get("hidden_size") or state["encoder.weight_hh_l0"].shape[1] // 3)
    else:
        hidden_size = int(config.get("hidden_size", 128))
    dropout = float(config.get("dropout", 0.2))
    use_crf = config.get("use_crf")
    if use_crf is None:
        use_crf = any(key.startswith("role_crf.") for key in state)
    return HierarchicalTokenSetParser(
        vocab_size=len(vocab.word2id),
        num_unit_pos=len(vocab.unit_pos2id),
        num_aux_kinds=len(vocab.aux_kind2id),
        num_role_labels=len(vocab.role_label2id),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        dropout=dropout,
        pad_id=vocab.pad_id,
        num_decisions=len(vocab.decision2id),
        num_view_kinds=len(vocab.view_kind2id),
        num_upos=len(vocab.upos2id),
        use_crf=bool(use_crf),
        role_label_names=labels_by_id(vocab.role_label2id),
        encoder_type=encoder_type,
        bert_model_name=str(config.get("bert_model", "bert-base-cased")) if encoder_type == "bert" else None,
        bert_local_files_only=bool(config.get("bert_local_files_only", False)),
    )


def build_tokenizer_from_config(config: dict[str, Any]):
    if str(config.get("encoder_type", "gru")) != "bert":
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.get("bert_model", "bert-base-cased")),
        use_fast=True,
        local_files_only=bool(config.get("bert_local_files_only", False)),
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("hierarchical BERT analysis requires a fast tokenizer")
    return tokenizer


def compatible_model_state_dict(
    checkpoint: dict[str, Any],
    model: HierarchicalTokenSetParser,
) -> dict[str, torch.Tensor]:
    state = dict(checkpoint["model_state_dict"])
    legacy_key_map = {
        "unit_pos_classifier.weight": "head_pos_classifier.weight",
        "unit_pos_classifier.bias": "head_pos_classifier.bias",
        "aux_kind_classifier.weight": "head_aux_kind_classifier.weight",
        "aux_kind_classifier.bias": "head_aux_kind_classifier.bias",
    }
    for old_key, new_key in legacy_key_map.items():
        if new_key not in state and old_key in state:
            state[new_key] = state[old_key]
        state.pop(old_key, None)
    model_state = model.state_dict()
    compatible_state = {}
    for key, value in state.items():
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
            compatible_state[key] = value
    for key, value in model_state.items():
        if key not in compatible_state:
            compatible_state[key] = value.detach().cpu()
    return compatible_state


def load_checkpoint(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def labels_by_id(label2id: dict[str, int]) -> list[str]:
    return [
        label
        for label, _index in sorted(label2id.items(), key=lambda item: item[1])
    ]


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(value)


def invert_mapping(mapping: dict[str, int]) -> dict[int, str]:
    return {int(value): str(key) for key, value in mapping.items()}


def without_o(labels: Iterable[str]) -> list[str]:
    return [label for label in labels if label != "O"]


def label_family(label: str) -> str:
    if len(label) > 2 and label[1] == "-":
        return label[2:]
    return label


def role_set_key(labels: Iterable[str]) -> str:
    labels = sorted(labels)
    return "+".join(labels) if labels else "O"


def unit_features(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_text": unit.get("input_text", ""),
        "depth": int(unit.get("depth", 0)),
        "incoming_count": int(unit.get("incoming_count", 0)),
        "is_root": bool(unit.get("is_root", False)),
        "input_token_count": len(unit.get("input_token_ids", [])),
        "self_token_count": len(unit.get("self_token_ids", [])),
        "child_count": len(unit.get("children", [])),
        "has_shared_child": bool(unit.get("has_shared_child", False)),
        "has_shared_arg": bool(unit.get("has_shared_arg", False)),
        "shared_arg_link_count": len(unit.get("shared_arg_links", [])),
        "has_external_child": bool(unit.get("has_external_child", False)),
        "has_overlapping_roles": bool(unit.get("has_overlapping_roles", False)),
        "is_discontinuous": bool(unit.get("is_discontinuous", False)),
    }


def depth_bucket(depth: int) -> str:
    if depth <= 5:
        return str(depth)
    return "6+"


def count_bucket(count: int) -> str:
    if count <= 3:
        return str(count)
    if count <= 5:
        return "4-5"
    if count <= 10:
        return "6-10"
    if count <= 20:
        return "11-20"
    return "21+"


def counter_metrics(counters: dict[str, Counter[str]]) -> list[dict[str, Any]]:
    rows = []
    for label, counter in counters.items():
        tp = int(counter.get("tp", 0))
        fp = int(counter.get("fp", 0))
        fn = int(counter.get("fn", 0))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": tp + fn,
                "predicted": tp + fp,
            }
        )
    return sorted(rows, key=lambda row: (-row["support"], row["label"]))


def confusion_to_rows(counter: Counter[tuple[str, str]], limit: int | None = None) -> list[dict[str, Any]]:
    rows = [
        {"gold": gold, "pred": pred, "count": int(count)}
        for (gold, pred), count in counter.most_common(limit)
    ]
    return rows


def counter_rows(counter: Counter[str], limit: int | None = None) -> list[dict[str, Any]]:
    return [
        {"label": label, "count": int(count)}
        for label, count in counter.most_common(limit)
    ]


def format_breakdowns(
    breakdowns: dict[str, dict[str, MetricAccumulator]]
) -> dict[str, list[dict[str, Any]]]:
    formatted = {}
    for breakdown_name, value_map in breakdowns.items():
        formatted[breakdown_name] = [
            {"value": value, **accumulator.to_dict()}
            for value, accumulator in sorted(
                value_map.items(),
                key=lambda item: (-item[1].total_units, item[0]),
            )
        ]
    return formatted


def maybe_add_top_error(
    top_error_records: list[dict[str, Any]],
    prediction: dict[str, Any],
    top_k: int,
) -> None:
    if top_k <= 0:
        return
    compact = {
        "sample_id": prediction["sample_id"],
        "unit_index": prediction["unit_index"],
        "sentence": prediction["sentence"],
        "unit_features": prediction["unit_features"],
        "gold": prediction["gold"],
        "pred": prediction["pred"],
        "matches": prediction["matches"],
        "role_counts": prediction["role_counts"],
        "role_errors": prediction["role_errors"],
        "error_types": prediction["error_types"],
    }
    top_error_records.append(compact)
    top_error_records.sort(key=error_rank, reverse=True)
    del top_error_records[top_k:]


def error_rank(prediction: dict[str, Any]) -> tuple[int, int, int, int]:
    counts = prediction["role_counts"]
    features = prediction["unit_features"]
    return (
        len(prediction["error_types"]),
        int(counts["fp"]) + int(counts["fn"]),
        int(features["input_token_count"]),
        int(features["depth"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
