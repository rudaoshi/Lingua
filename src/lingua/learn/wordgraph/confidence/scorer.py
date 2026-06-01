"""Gold-graph confidence scoring for trained Lingua wordgraph parsers."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ConfidenceWeights:
    """Weights matching the supervised wordgraph training objectives."""

    arc: float = 1.0
    rel: float = 1.0
    child_of_whether: float = 2.0
    node_type: float = 1.0
    is_root: float = 1.0


@dataclass
class GoldWordGraphExample:
    """Gold labels aligned to word-level model nodes."""

    sentence: str
    words: List[str]
    arc_labels: List[List[int]]
    is_root: List[int]
    child_of_whether: List[int]
    node_type: List[int]
    unknown_edge_labels: List[str]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    if path is None:
        return {}
    manifest: Dict[int, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        manifest[int(row["fold_local_idx"])] = row
    return manifest


def _single_phrase_word_idx(node: Any, num_words: int) -> Optional[int]:
    from lingua.structure.gpgraph import GPGPhraseNode

    if not isinstance(node, GPGPhraseNode):
        return None
    word_indices = list(node.words(with_aux=False))
    if len(word_indices) != 1:
        return None
    word_idx = int(word_indices[0])
    if 0 <= word_idx < num_words:
        return word_idx
    return None


def build_gold_wordgraph_example(
    item: Dict[str, Any],
    *,
    label2id: Dict[str, int],
    node_type2id: Dict[str, int],
) -> GoldWordGraphExample:
    """Convert a serialized Lingua sample into wordgraph gold labels."""

    from lingua.learn.wordgraph.modeler.gp2word import lingua2wordlingua
    from lingua.structure.gpgraph import GPGraph

    sentence = item["sentence"]
    lingua_graph = GPGraph.parse(item["lingua"])
    word_lingua_graph, _, _ = lingua2wordlingua(lingua_graph)

    words = list(word_lingua_graph.words)
    num_words = len(words)
    arc_labels = [[-1] * num_words for _ in range(num_words)]
    child_of_whether = [0] * num_words
    node_type = [-1] * num_words
    unknown_edge_labels: List[str] = []

    for node in word_lingua_graph.nodes():
        word_idx = _single_phrase_word_idx(node, num_words)
        if word_idx is None:
            continue
        if getattr(node, "child_of_whether", False):
            child_of_whether[word_idx] = 1
        pos = getattr(node, "pos", None)
        if pos not in node_type2id:
            raise ValueError(f"Unknown node type in gold sample: {pos}")
        node_type[word_idx] = node_type2id[pos]

    missing_node_type = [idx for idx, type_id in enumerate(node_type) if type_id < 0]
    if missing_node_type:
        raise ValueError(f"Missing node type labels for word indices: {missing_node_type}")

    for parent, edge, child in word_lingua_graph.edges():
        parent_idx = _single_phrase_word_idx(parent, num_words)
        child_idx = _single_phrase_word_idx(child, num_words)
        if parent_idx is None or child_idx is None:
            continue
        edge_label = edge.label
        if edge_label in label2id:
            label_id = label2id[edge_label]
        else:
            unknown_edge_labels.append(edge_label)
            raise ValueError(f"Unknown edge label in gold sample: {edge_label}")
        arc_labels[parent_idx][child_idx] = label_id

    indegree = [0] * num_words
    for parent_idx in range(num_words):
        for child_idx in range(num_words):
            if parent_idx == child_idx:
                continue
            if arc_labels[parent_idx][child_idx] >= 0:
                indegree[child_idx] += 1
    is_root = [int(degree == 0) for degree in indegree]

    return GoldWordGraphExample(
        sentence=sentence,
        words=words,
        arc_labels=arc_labels,
        is_root=is_root,
        child_of_whether=child_of_whether,
        node_type=node_type,
        unknown_edge_labels=unknown_edge_labels,
    )


def _label_tensors(
    example: GoldWordGraphExample,
    *,
    num_nodes: int,
    device: str,
) -> Dict[str, Any]:
    import torch

    arc_labels = torch.full((num_nodes, num_nodes), -1, dtype=torch.long, device=device)
    for parent_idx in range(num_nodes):
        for child_idx in range(num_nodes):
            arc_labels[parent_idx, child_idx] = example.arc_labels[parent_idx][child_idx]

    return {
        "arc_labels": arc_labels,
        "is_root": torch.tensor(example.is_root[:num_nodes], dtype=torch.long, device=device),
        "child_of_whether": torch.tensor(
            example.child_of_whether[:num_nodes], dtype=torch.long, device=device
        ),
        "node_type": torch.tensor(example.node_type[:num_nodes], dtype=torch.long, device=device),
    }


def score_gold_example(
    *,
    inference: Any,
    example: GoldWordGraphExample,
    weights: ConfidenceWeights,
) -> Dict[str, Any]:
    """Score one gold example with a held-out model."""

    import torch
    import torch.nn.functional as F
    from lingua.learn.wordgraph.decoding.inference import encode_words_like_training

    encoded = encode_words_like_training(
        inference.tokenizer,
        example.words,
        max_length=inference.config.max_length,
    )
    num_nodes = int(encoded["num_nodes"][0].item())
    if num_nodes <= 0:
        raise ValueError("Encoded example contains no word nodes")

    with torch.no_grad():
        output = inference.model(
            encoded["input_ids"].to(inference.device),
            encoded["attention_mask"].to(inference.device),
            encoded["word_to_subword"].to(inference.device),
            encoded["word_mask"].to(inference.device),
        )

    labels = _label_tensors(example, num_nodes=num_nodes, device=inference.device)

    arc_logits = output["arc_logits"][0, :num_nodes, :num_nodes]
    rel_logits = output["rel_logits"][0, :num_nodes, :num_nodes, :]
    pairwise_mask = torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=inference.device)
    edge_mask = labels["arc_labels"] >= 0

    component_nll: Dict[str, float] = {}
    active_weights: Dict[str, float] = {}

    arc_targets = edge_mask.float()
    component_nll["arc"] = float(
        F.binary_cross_entropy_with_logits(
            arc_logits[pairwise_mask],
            arc_targets[pairwise_mask],
            reduction="mean",
        ).item()
    )
    active_weights["arc"] = weights.arc

    if edge_mask.any():
        component_nll["rel"] = float(
            F.cross_entropy(
                rel_logits[edge_mask],
                labels["arc_labels"][edge_mask],
                reduction="mean",
            ).item()
        )
        active_weights["rel"] = weights.rel
    else:
        component_nll["rel"] = 0.0
        active_weights["rel"] = 0.0

    if output.get("child_of_whether_logits") is not None:
        cow_logits = output["child_of_whether_logits"][0, :num_nodes, :]
        component_nll["child_of_whether"] = float(
            F.cross_entropy(cow_logits, labels["child_of_whether"], reduction="mean").item()
        )
        active_weights["child_of_whether"] = weights.child_of_whether
    else:
        component_nll["child_of_whether"] = 0.0
        active_weights["child_of_whether"] = 0.0

    if output.get("node_type_logits") is not None:
        node_type_logits = output["node_type_logits"][0, :num_nodes, :]
        component_nll["node_type"] = float(
            F.cross_entropy(node_type_logits, labels["node_type"], reduction="mean").item()
        )
        active_weights["node_type"] = weights.node_type
    else:
        component_nll["node_type"] = 0.0
        active_weights["node_type"] = 0.0

    if output.get("is_root_logits") is not None:
        is_root_logits = output["is_root_logits"][0, :num_nodes, :]
        component_nll["is_root"] = float(
            F.cross_entropy(is_root_logits, labels["is_root"], reduction="mean").item()
        )
        active_weights["is_root"] = weights.is_root
    else:
        component_nll["is_root"] = 0.0
        active_weights["is_root"] = 0.0

    active_weight_sum = sum(weight for weight in active_weights.values() if weight > 0)
    weighted_sum_nll = sum(component_nll[name] * active_weights[name] for name in component_nll)
    normalized_nll = weighted_sum_nll / max(active_weight_sum, 1e-8)
    confidence = math.exp(-min(normalized_nll, 100.0))

    return {
        "nll": normalized_nll,
        "weighted_sum_nll": weighted_sum_nll,
        "active_weight_sum": active_weight_sum,
        "confidence": confidence,
        "component_nll": component_nll,
        "component_weights": active_weights,
        "num_words": len(example.words),
        "num_scored_words": num_nodes,
        "truncated": num_nodes < len(example.words),
        "unknown_edge_labels": sorted(set(example.unknown_edge_labels)),
        "unknown_edge_label_count": len(example.unknown_edge_labels),
    }


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_file(
    *,
    model_dir: Path | str,
    data_path: Path | str,
    output_path: Path | str,
    manifest_path: Optional[Path | str] = None,
    summary_path: Optional[Path | str] = None,
    device: Optional[str] = None,
    max_length: int = 256,
    weights: Optional[ConfidenceWeights] = None,
    fail_on_error: bool = False,
) -> Dict[str, Any]:
    """Score a JSONL file and write per-sample confidence records."""

    model_dir = Path(model_dir)
    data_path = Path(data_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path) if summary_path else None
    manifest = load_manifest(Path(manifest_path) if manifest_path else None)
    weights = weights or ConfidenceWeights()
    device = device or _default_device()

    from tqdm import tqdm
    from lingua.learn.wordgraph.decoding.inference import InferenceConfig, WordLinguaInference

    inference = WordLinguaInference.from_pretrained(
        str(model_dir),
        device=device,
        config=InferenceConfig(max_length=max_length),
    )

    rows = read_jsonl(data_path)
    scored_records: List[Dict[str, Any]] = []
    error_count = 0

    for local_idx, item in enumerate(tqdm(rows, desc="Scoring confidence")):
        manifest_row = manifest.get(local_idx, {})
        base_record = {
            "fold": manifest_row.get("fold"),
            "fold_local_idx": local_idx,
            "global_idx": manifest_row.get("global_idx", local_idx),
            "source_split": manifest_row.get("source_split"),
            "source_line": manifest_row.get("source_line"),
            "sentence": item.get("sentence", manifest_row.get("sentence", "")),
        }
        try:
            gold = build_gold_wordgraph_example(
                item,
                label2id=inference.label2id,
                node_type2id=inference.node_type2id,
            )
            score = score_gold_example(inference=inference, example=gold, weights=weights)
            scored_records.append({**base_record, **score, "error": None})
        except Exception as exc:  # pragma: no cover - exercised by CLI failure mode
            if fail_on_error:
                raise
            error_count += 1
            scored_records.append(
                {
                    **base_record,
                    "confidence": None,
                    "nll": None,
                    "component_nll": {},
                    "error": str(exc),
                }
            )

    write_jsonl(output_path, scored_records)

    valid_confidences = [
        float(record["confidence"])
        for record in scored_records
        if record.get("confidence") is not None
    ]
    summary = {
        "model_dir": str(model_dir),
        "data_path": str(data_path),
        "output_path": str(output_path),
        "total_samples": len(scored_records),
        "scored_samples": len(valid_confidences),
        "error_samples": error_count,
        "mean_confidence": (
            sum(valid_confidences) / len(valid_confidences) if valid_confidences else None
        ),
        "min_confidence": min(valid_confidences) if valid_confidences else None,
        "max_confidence": max(valid_confidences) if valid_confidences else None,
        "max_length": max_length,
        "weights": weights.__dict__,
    }

    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score held-out Lingua wordgraph confidence")
    parser.add_argument("--model_dir", required=True, help="Trained wordgraph model directory")
    parser.add_argument("--data_path", required=True, help="Held-out JSONL data path")
    parser.add_argument("--output_path", required=True, help="Output JSONL path")
    parser.add_argument("--manifest_path", default=None, help="Optional fold manifest JSONL")
    parser.add_argument("--summary_path", default=None, help="Optional summary JSON path")
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu; auto if omitted")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum tokenized length")
    parser.add_argument("--arc_weight", type=float, default=1.0)
    parser.add_argument("--rel_weight", type=float, default=1.0)
    parser.add_argument("--cow_weight", type=float, default=2.0)
    parser.add_argument("--node_type_weight", type=float, default=1.0)
    parser.add_argument("--is_root_weight", type=float, default=1.0)
    parser.add_argument("--fail_on_error", action="store_true")
    args = parser.parse_args(argv)

    score_file(
        model_dir=args.model_dir,
        data_path=args.data_path,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        summary_path=args.summary_path,
        device=args.device,
        max_length=args.max_length,
        weights=ConfidenceWeights(
            arc=args.arc_weight,
            rel=args.rel_weight,
            child_of_whether=args.cow_weight,
            node_type=args.node_type_weight,
            is_root=args.is_root_weight,
        ),
        fail_on_error=args.fail_on_error,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
