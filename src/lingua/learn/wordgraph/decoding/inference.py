#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word-Lingua Inference Module

This module provides:
1. Shared utilities for running Word-Lingua model inference and extracting scores
2. WordLinguaInference class for end-to-end sentence parsing

The WordLinguaInference class:
- Loads a trained HuggingFace model and all dependencies
- Parses sentences through the model
- Decodes the output to generate word-lingua graphs
- Converts word-lingua graphs to lingua graphs

Usage:
    from lingua.learn.wordgraph.decoding.inference import WordLinguaInference
    
    inference = WordLinguaInference.from_checkpoint(
        checkpoint_path="/path/to/best_model.pt",
        device="cuda"
    )
    
    result = inference.parse("The cat sat on the mat.")
    lingua_graph = result.lingua_graph
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from lingua.structure.gpgraph import GPGraph, GPGPhraseNode, GPGEdge
from lingua.learn.wordgraph.model.edge_label_factors import FACTOR_NAMES

logger = logging.getLogger(__name__)


# ============================================================================
# Shared Utilities (decoding-agnostic)
# ============================================================================

@dataclass(frozen=True)
class WordLinguaModelOutputs:
    """Standardized model output container."""
    arc_logits: torch.Tensor  # [B, N, N]
    rel_logits: torch.Tensor  # [B, N, N, L]
    edge_factor_logits: Optional[Dict[str, torch.Tensor]] = None  # [B, N, N, F]
    node_type_logits: Optional[torch.Tensor] = None  # [B, N, T]
    child_of_whether_logits: Optional[torch.Tensor] = None  # [B, N, 2]
    is_root_logits: Optional[torch.Tensor] = None  # [B, N, 2]


def encode_words_like_training(tokenizer, words: List[str], *, max_length: int = 256) -> Dict[str, torch.Tensor]:
    """
    Encoding aligned with v2 training:
    - Graph nodes correspond 1:1 to `words` (NO synthetic ROOT node)
    - [CLS]/[SEP] exist as transformer tokens but are NOT graph nodes
    """
    all_tokens: List[str] = ["[CLS]"]
    word_to_subword: List[List[int]] = []

    for w in words:
        toks = tokenizer.tokenize(w) or ["[UNK]"]
        start = len(all_tokens)
        idxs = list(range(start, start + len(toks)))
        word_to_subword.append(idxs)
        all_tokens.extend(toks)

    all_tokens.append("[SEP]")

    # Truncate if needed
    if len(all_tokens) > max_length:
        all_tokens = all_tokens[: max_length - 1] + ["[SEP]"]
        new_w2s: List[List[int]] = []
        for idxs in word_to_subword:
            valid = [i for i in idxs if i < max_length - 1]
            if valid:
                new_w2s.append(valid)
            else:
                break
        word_to_subword = new_w2s

    input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    attention_mask = [1] * len(input_ids)
    num_nodes = len(word_to_subword)

    max_subwords = max(len(x) for x in word_to_subword) if word_to_subword else 1
    input_ids_t = torch.tensor([input_ids], dtype=torch.long)
    attention_mask_t = torch.tensor([attention_mask], dtype=torch.bool)
    w2s_t = torch.full((1, num_nodes, max_subwords), -1, dtype=torch.long)
    word_mask_t = torch.zeros((1, num_nodes), dtype=torch.bool)
    word_mask_t[0, :num_nodes] = True

    for w_i, idxs in enumerate(word_to_subword):
        for s_i, s in enumerate(idxs):
            w2s_t[0, w_i, s_i] = s

    return {
        "input_ids": input_ids_t,
        "attention_mask": attention_mask_t,
        "word_to_subword": w2s_t,
        "word_mask": word_mask_t,
        "num_nodes": torch.tensor([num_nodes], dtype=torch.long),
    }


def encode_batch_words_like_training(
    tokenizer, 
    batch_words: List[List[str]], 
    *, 
    max_length: int = 256
) -> Dict[str, torch.Tensor]:
    """
    Batch encoding aligned with v2 training.
    
    Args:
        tokenizer: HuggingFace tokenizer
        batch_words: List of word lists, each word list is a tokenized sentence
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of batched tensors
    """
    batch_size = len(batch_words)
    
    # First pass: encode all examples and collect dimensions
    all_encodings = []
    max_seq_len = 0
    max_num_words = 0
    max_subwords_per_word = 1
    
    for words in batch_words:
        all_tokens: List[str] = ["[CLS]"]
        word_to_subword: List[List[int]] = []
        
        for w in words:
            toks = tokenizer.tokenize(w) or ["[UNK]"]
            start = len(all_tokens)
            idxs = list(range(start, start + len(toks)))
            word_to_subword.append(idxs)
            all_tokens.extend(toks)
            max_subwords_per_word = max(max_subwords_per_word, len(toks))
        
        all_tokens.append("[SEP]")
        
        # Truncate if needed
        if len(all_tokens) > max_length:
            all_tokens = all_tokens[: max_length - 1] + ["[SEP]"]
            new_w2s: List[List[int]] = []
            for idxs in word_to_subword:
                valid = [i for i in idxs if i < max_length - 1]
                if valid:
                    new_w2s.append(valid)
                else:
                    break
            word_to_subword = new_w2s
        
        input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
        num_nodes = len(word_to_subword)
        
        max_seq_len = max(max_seq_len, len(input_ids))
        max_num_words = max(max_num_words, num_nodes)
        
        all_encodings.append({
            "input_ids": input_ids,
            "word_to_subword": word_to_subword,
            "num_nodes": num_nodes,
        })
    
    # Second pass: create padded tensors
    input_ids_t = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    attention_mask_t = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    w2s_t = torch.full((batch_size, max_num_words, max_subwords_per_word), -1, dtype=torch.long)
    word_mask_t = torch.zeros((batch_size, max_num_words), dtype=torch.bool)
    num_nodes_t = torch.zeros(batch_size, dtype=torch.long)
    
    for b_idx, enc in enumerate(all_encodings):
        seq_len = len(enc["input_ids"])
        num_nodes = enc["num_nodes"]
        
        input_ids_t[b_idx, :seq_len] = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask_t[b_idx, :seq_len] = True
        word_mask_t[b_idx, :num_nodes] = True
        num_nodes_t[b_idx] = num_nodes
        
        for w_i, idxs in enumerate(enc["word_to_subword"]):
            for s_i, s in enumerate(idxs):
                w2s_t[b_idx, w_i, s_i] = s
    
    return {
        "input_ids": input_ids_t,
        "attention_mask": attention_mask_t,
        "word_to_subword": w2s_t,
        "word_mask": word_mask_t,
        "num_nodes": num_nodes_t,
    }


def extract_outputs(out: Dict[str, Any]) -> WordLinguaModelOutputs:
    """Standardize model outputs across scripts."""
    return WordLinguaModelOutputs(
        arc_logits=out["arc_logits"],
        rel_logits=out["rel_logits"],
        edge_factor_logits=out.get("edge_factor_logits"),
        node_type_logits=out.get("node_type_logits"),
        child_of_whether_logits=out.get("child_of_whether_logits"),
        is_root_logits=out.get("is_root_logits"),
    )


@torch.no_grad()
def forward_encoded(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_to_subword: torch.Tensor,
    word_mask: torch.Tensor,
    device: str,
) -> WordLinguaModelOutputs:
    """Run the model forward given already-encoded tensors."""
    out = model(
        input_ids.to(device),
        attention_mask.to(device),
        word_to_subword.to(device),
        word_mask.to(device),
    )
    return extract_outputs(out)


@torch.no_grad()
def forward_words(
    *,
    model,
    tokenizer,
    words: List[str],
    device: str,
    max_length: int,
) -> Dict[str, Any]:
    """
    Convenience wrapper: encode words like training, forward, and return a dict with
    encoded tensors + standardized outputs.
    """
    enc = encode_words_like_training(tokenizer, words, max_length=max_length)
    outs = forward_encoded(
        model=model,
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        word_to_subword=enc["word_to_subword"],
        word_mask=enc["word_mask"],
        device=device,
    )
    return {"enc": enc, "outs": outs}


def logits_to_lists_for_decoder(
    *,
    arc_logits_1: torch.Tensor,
    rel_logits_1: torch.Tensor,
    id2label: Dict[int, str],
) -> Dict[str, Any]:
    """Convert single-example logits into python lists for downstream decoders."""
    num_nodes = int(arc_logits_1.shape[0])
    arc_scores = arc_logits_1[:num_nodes, :num_nodes].float().cpu().numpy().tolist()
    label_scores = rel_logits_1[:num_nodes, :num_nodes, :].float().cpu().numpy().tolist()
    labels = [id2label[i] for i in range(len(id2label))]
    return {"arc_scores": arc_scores, "label_scores": label_scores, "labels": labels}


def node_pos_from_node_type_logits(
    node_type_logits_1: Optional[torch.Tensor],
    *,
    id2node_type: Dict[int, str],
) -> Optional[List[str]]:
    if node_type_logits_1 is None:
        return None
    node_type_pred = node_type_logits_1.argmax(dim=-1).tolist()
    node_pos: List[str] = []
    for i, t in enumerate(node_type_pred):
        type_id = int(t)
        if type_id not in id2node_type:
            raise ValueError(f"Predicted node type id is not in model vocabulary: {type_id}")
        node_pos.append(id2node_type[type_id])
    return node_pos


def build_edge_factor_targets_for_labels(
    *,
    id2label: Mapping[int, str],
    edge_factor_vocabs: Optional[Mapping[str, List[str]]],
    edge_label_factors: Optional[Mapping[str, Mapping[str, str]]],
) -> Dict[str, torch.Tensor]:
    """Build factor target ids for each legal edge label id."""
    if not edge_factor_vocabs or not edge_label_factors:
        return {}

    ordered_labels = [id2label[i] for i in range(len(id2label))]
    targets: Dict[str, torch.Tensor] = {}
    for factor_name in FACTOR_NAMES:
        if factor_name not in edge_factor_vocabs:
            continue
        value2id = {
            value: idx
            for idx, value in enumerate(edge_factor_vocabs[factor_name])
        }
        factor_targets: List[int] = []
        for label in ordered_labels:
            label_factors = edge_label_factors.get(label)
            if label_factors is None:
                raise ValueError(f"missing edge factor decomposition for label: {label}")
            value = label_factors.get(factor_name)
            if value not in value2id:
                raise ValueError(
                    f"edge factor value {value!r} for {factor_name}/{label} is not in vocabulary"
                )
            factor_targets.append(value2id[value])
        targets[factor_name] = torch.tensor(factor_targets, dtype=torch.long)
    return targets


def combine_rel_logits_with_edge_factor_logits(
    *,
    rel_logits: torch.Tensor,
    edge_factor_logits: Optional[Mapping[str, torch.Tensor]],
    edge_factor_targets_by_label: Mapping[str, torch.Tensor],
    weight: float,
) -> torch.Tensor:
    """Combine relation logits with legal-label factor evidence for decoding."""
    if weight <= 0 or not edge_factor_logits or not edge_factor_targets_by_label:
        return rel_logits

    factor_score = torch.zeros_like(rel_logits)
    factor_count = 0
    for factor_name, targets in edge_factor_targets_by_label.items():
        if factor_name not in edge_factor_logits:
            continue
        logits = edge_factor_logits[factor_name]
        target_ids = targets.to(device=logits.device)
        label_factor_scores = torch.log_softmax(logits, dim=-1).index_select(
            dim=-1,
            index=target_ids,
        )
        factor_score = factor_score + label_factor_scores.to(device=rel_logits.device)
        factor_count += 1

    if factor_count == 0:
        return rel_logits
    return rel_logits + weight * (factor_score / factor_count)


# ============================================================================
# Inference Configuration and Result
# ============================================================================

@dataclass
class InferenceConfig:
    """Configuration for word-lingua inference."""
    
    # Model config (loaded from config.json, can be overridden)
    arc_hidden_size: int = 500
    rel_hidden_size: int = 100
    node_hidden_size: int = 256
    dropout: float = 0.1
    word_pooling: str = "mean"
    max_length: int = 256
    
    # Decoder config
    arc_threshold: float = 0.5
    lam_parent_prob: float = 1.0
    lam_sibling_prob: float = 1.0
    lam_pos_prob: float = 1.0
    lam_parent_given_child_prob: float = 1.0
    edge_factor_decode_weight: float = 0.0
    enforce_connected: bool = False
    enable_phase4_rewrites: bool = False
    enforce_single_root: bool = True
    decoder_mode: str = "ilp"
    ilp_max_candidates_per_child: int = 16
    ilp_max_labels_per_pair: int = 4
    ilp_time_limit: float = 30.0
    ilp_mip_rel_gap: float = 0.001
    ilp_root_score_weight: float = 1.0
    ilp_lazy_constraints: bool = True
    ilp_max_lazy_rounds: int = 4
    hybrid_fallback_policy: str = "conversion"
    
    # Debug config
    debug: bool = False
    debug_dir: Optional[str] = None


@dataclass
class InferenceResult:
    """Result of word-lingua inference."""
    
    # Input
    sentence: str
    words: List[str]
    
    # Word-lingua graph (intermediate)
    word_lingua_graph: Optional[GPGraph] = None
    
    # Lingua graph (final)
    lingua_graph: Optional[GPGraph] = None
    lingua_conversion_error: Optional[str] = None
    
    # Raw predictions
    edges: List[Tuple[int, int, str]] = field(default_factory=list)
    node_types: List[str] = field(default_factory=list)
    child_of_whether: List[bool] = field(default_factory=list)
    
    # Debug info
    decoding_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "sentence": self.sentence,
            "words": self.words,
            "edges": [{"head": h, "dep": d, "label": l} for h, d, l in self.edges],
            "node_types": self.node_types,
            "child_of_whether": self.child_of_whether,
            "lingua": self.lingua_graph.data() if self.lingua_graph else None,
            "word_lingua": self.word_lingua_graph.data() if self.word_lingua_graph else None,
            "lingua_conversion_error": self.lingua_conversion_error,
            "decoding_info": self.decoding_info if self.decoding_info else None,
        }


# ============================================================================
# WordLinguaInference Class
# ============================================================================

class WordLinguaInference:
    """
    Word-Lingua inference class.
    
    Loads a trained model and provides methods to parse sentences into lingua graphs.
    """
    
    def __init__(
        self,
        model: Any,  # WordLinguaParserV2
        tokenizer: Any,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        config: InferenceConfig,
        statistics: Optional[Any] = None,  # WordLinguaStatistics
        id2node_type: Optional[Dict[int, str]] = None,
        edge_factor_vocabs: Optional[Mapping[str, List[str]]] = None,
        edge_label_factors: Optional[Mapping[str, Mapping[str, str]]] = None,
        device: str = "cpu",
    ):
        """
        Initialize the inference engine.
        
        Args:
            model: Trained WordLinguaParserV2 model
            tokenizer: HuggingFace tokenizer
            id2label: Mapping from label ID to label string
            label2id: Mapping from label string to label ID
            config: Inference configuration
            statistics: Optional word-lingua statistics for structural constraints
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.label2id = label2id
        self.config = config
        self.statistics = statistics
        if id2node_type is None:
            raise ValueError("id2node_type must be provided by the loaded model vocabulary")
        self.id2node_type = id2node_type
        self.node_type2id = {v: k for k, v in id2node_type.items()}
        self.edge_factor_vocabs = edge_factor_vocabs or {}
        self.edge_label_factors = edge_label_factors or {}
        self.edge_factor_targets_by_label = build_edge_factor_targets_for_labels(
            id2label=id2label,
            edge_factor_vocabs=self.edge_factor_vocabs,
            edge_label_factors=self.edge_label_factors,
        )
        self.device = device
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        self.decoder_mode = str(config.decoder_mode or "hybrid")
        if self.decoder_mode not in {"hybrid", "ilp", "sdp_graph"}:
            raise ValueError(f"Unsupported decoder_mode: {self.decoder_mode}")
        self.hybrid_fallback_policy = str(config.hybrid_fallback_policy or "conversion")
        if self.hybrid_fallback_policy not in {"conversion", "issues"}:
            raise ValueError(f"Unsupported hybrid_fallback_policy: {self.hybrid_fallback_policy}")

        # Lazy import to avoid circular dependencies
        from lingua.learn.wordgraph.decoding.graph_decoder import GreedyPatternDecoder

        def _make_decoder(mode: str) -> GreedyPatternDecoder:
            return GreedyPatternDecoder(
                stats=None,  # Legacy parameter, ignored
                id2label=id2label,
                label2id=label2id,
                arc_threshold=config.arc_threshold,
                edge_prob_tables=statistics,
                lam_parent_prob=config.lam_parent_prob,
                lam_sibling_prob=config.lam_sibling_prob,
                lam_pos_prob=config.lam_pos_prob,
                lam_parent_given_child_prob=config.lam_parent_given_child_prob,
                id2node_type=self.id2node_type,
                debug=config.debug,
                debug_dir=config.debug_dir,
                enforce_connected=config.enforce_connected,
                enable_phase4_rewrites=config.enable_phase4_rewrites,
                enforce_single_root=config.enforce_single_root,
                decoder_mode=mode,
                ilp_max_candidates_per_child=config.ilp_max_candidates_per_child,
                ilp_max_labels_per_pair=config.ilp_max_labels_per_pair,
                ilp_time_limit=config.ilp_time_limit,
                ilp_mip_rel_gap=config.ilp_mip_rel_gap,
                ilp_root_score_weight=config.ilp_root_score_weight,
                ilp_lazy_constraints=config.ilp_lazy_constraints,
                ilp_max_lazy_rounds=config.ilp_max_lazy_rounds,
            )

        primary_mode = "sdp_graph" if self.decoder_mode == "hybrid" else self.decoder_mode
        self.decoder = _make_decoder(primary_mode)
        self.ilp_decoder = _make_decoder("ilp") if self.decoder_mode == "hybrid" else None
        
        self._example_counter = 0

    def _result_decoding_info(self, decoding_info: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.debug:
            return decoding_info

        keep_keys = {
            "decoder_mode",
            "decoder_options",
            "errors",
            "warnings",
            "forced_connections",
            "disconnected_nodes",
            "phase3_skipped",
            "phase3_failures",
            "phase4_force_connects",
            "phase4_multihead_fixes",
            "single_root_repair_skipped",
            "single_root_repairs",
            "ilp",
            "hybrid",
            "validation_errors",
            "has_issues",
        }
        return {
            key: value
            for key, value in decoding_info.items()
            if key in keep_keys and value
        }

    def _set_debug_state(self, debug: Optional[bool]) -> None:
        if debug is not None:
            self.decoder.debug = debug
            if self.ilp_decoder is not None:
                self.ilp_decoder.debug = debug

    def _convert_word_lingua_graph(self, word_lingua_graph: GPGraph) -> Tuple[Optional[GPGraph], Optional[str]]:
        from lingua.learn.wordgraph.modeler.word2gp import wordlingua2lingua

        try:
            import copy
            lingua_graph, _, _ = wordlingua2lingua(copy.deepcopy(word_lingua_graph))
            return lingua_graph, None
        except Exception as e:
            return None, str(e)

    def _decode_with_optional_ilp_fallback(
        self,
        *,
        words: List[str],
        arc_logits: np.ndarray,
        rel_logits: np.ndarray,
        node_type_preds: np.ndarray,
        cow_preds: List[bool],
        is_root_logits: Optional[np.ndarray],
    ) -> Tuple[GPGraph, Optional[GPGraph], Optional[str], Dict[str, Any]]:
        child_of_whether = np.array([int(c) for c in cow_preds])

        fast_graph, fast_info = self.decoder.decode(
            words=words,
            arc_logits=arc_logits,
            rel_logits=rel_logits,
            node_type_preds=node_type_preds,
            child_of_whether_preds=child_of_whether,
            is_root_logits=is_root_logits,
        )
        fast_lingua_graph, fast_conversion_error = self._convert_word_lingua_graph(fast_graph)

        if self.decoder_mode != "hybrid":
            return fast_graph, fast_lingua_graph, fast_conversion_error, fast_info

        fallback_reason = None
        if fast_conversion_error:
            fallback_reason = "conversion_error"
        elif fast_info.get("validation_errors"):
            fallback_reason = "validation_errors"
        elif self.hybrid_fallback_policy == "issues" and fast_info.get("has_issues"):
            fallback_reason = "decoder_issues"

        hybrid_info = {
            "primary_decoder_mode": "sdp_graph",
            "fallback_decoder_mode": "ilp",
            "fallback_policy": self.hybrid_fallback_policy,
            "fallback_used": False,
            "fallback_reason": fallback_reason,
            "accepted_decoder_mode": "sdp_graph",
            "primary_conversion_error": fast_conversion_error,
        }

        if fallback_reason and self.ilp_decoder is not None:
            ilp_graph, ilp_info = self.ilp_decoder.decode(
                words=words,
                arc_logits=arc_logits,
                rel_logits=rel_logits,
                node_type_preds=node_type_preds,
                child_of_whether_preds=child_of_whether,
                is_root_logits=is_root_logits,
            )
            ilp_lingua_graph, ilp_conversion_error = self._convert_word_lingua_graph(ilp_graph)

            hybrid_info.update({
                "fallback_used": True,
                "accepted_decoder_mode": "ilp",
                "fallback_conversion_error": ilp_conversion_error,
            })
            if self.config.debug:
                hybrid_info["primary_decoding_info"] = fast_info
                hybrid_info["fallback_decoding_info"] = ilp_info

            final_info = dict(ilp_info)
            final_info["decoder_mode"] = "hybrid"
            final_info["hybrid"] = hybrid_info
            return ilp_graph, ilp_lingua_graph, ilp_conversion_error, final_info

        final_info = dict(fast_info)
        final_info["decoder_mode"] = "hybrid"
        final_info["hybrid"] = hybrid_info
        return fast_graph, fast_lingua_graph, fast_conversion_error, final_info

    @staticmethod
    def _extract_edges_and_apply_child_flags(
        word_lingua_graph: GPGraph,
        cow_preds: List[bool],
    ) -> List[Tuple[int, int, str]]:
        edges: List[Tuple[int, int, str]] = []
        node_id_to_idx = {}
        for node in word_lingua_graph.nodes():
            if hasattr(node, "spans") and node.spans:
                span = node.spans[0]
                idx = span[0] if isinstance(span, (tuple, list)) else span
                node_id_to_idx[node.ID] = idx

        for parent, edge, child in word_lingua_graph.edges():
            p_idx = node_id_to_idx.get(parent.ID)
            c_idx = node_id_to_idx.get(child.ID)
            if p_idx is not None and c_idx is not None:
                edges.append((p_idx, c_idx, edge.label))

        for node in word_lingua_graph.nodes():
            idx = node_id_to_idx.get(node.ID)
            if idx is not None and idx < len(cow_preds) and cow_preds[idx]:
                node.child_of_whether = True

        return edges
    
    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        statistics_path: Optional[str] = None,
        config: Optional[InferenceConfig] = None,
    ) -> "WordLinguaInference":
        """
        Load inference engine from a pretrained model directory.
        
        The model directory should contain:
        - config.json: model configuration (bert_model_name, hidden sizes, etc.)
        - pytorch_model.bin or model.safetensors: model weights
        - tokenizer.json, vocab.txt, etc.: tokenizer files
        - label2id.json: label mapping
        - wordlingua_statistics.json (optional): statistics for decoding
        
        Args:
            model_dir: Path to model directory
            device: Device to run inference on
            statistics_path: Optional path to wordlingua_statistics.json
            config: Optional inference configuration (loaded from config.json if not provided)
        
        Returns:
            Initialized WordLinguaInference instance
        """
        from transformers import AutoTokenizer
        from lingua.learn.wordgraph.decoding.statistics import WordLinguaStatistics
        from lingua.learn.wordgraph.model.biaffine import (
            WordLinguaParserV2,
            build_node_type_maps,
        )
        
        if not os.path.isdir(model_dir):
            raise ValueError(f"model_dir must be a directory: {model_dir}")
        
        logger.info(f"Loading model from: {model_dir}")
        
        # Load config.json
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {model_dir}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        
        logger.info(f"Model config: {model_config}")
        
        edge_types = model_config.get("edge_types")
        if edge_types is None:
            edge_types_path = os.path.join(model_dir, "edge_types.json")
            if os.path.exists(edge_types_path):
                with open(edge_types_path, "r", encoding="utf-8") as f:
                    edge_types = json.load(f)

        # Load edge vocabulary
        label2id_path = os.path.join(model_dir, "label2id.json")
        if os.path.exists(label2id_path):
            with open(label2id_path, "r", encoding="utf-8") as f:
                label2id = {str(k): int(v) for k, v in json.load(f).items()}
        elif edge_types:
            label2id = {str(label): idx for idx, label in enumerate(edge_types)}
        else:
            raise FileNotFoundError(f"Neither label2id.json nor edge_types was found in {model_dir}")
        id2label = {v: k for k, v in label2id.items()}

        if edge_types is not None:
            expected_label2id = {str(label): idx for idx, label in enumerate(edge_types)}
            if expected_label2id != label2id:
                logger.warning("label2id.json differs from config edge_types; using label2id.json for compatibility")

        edge_factor_vocabs = model_config.get("edge_factor_vocabs")
        if edge_factor_vocabs is None:
            edge_factor_vocabs_path = os.path.join(model_dir, "edge_factor_vocabs.json")
            if os.path.exists(edge_factor_vocabs_path):
                with open(edge_factor_vocabs_path, "r", encoding="utf-8") as f:
                    edge_factor_vocabs = json.load(f)

        edge_label_factors = model_config.get("edge_label_factors")
        if edge_label_factors is None:
            edge_label_factors_path = os.path.join(model_dir, "edge_label_factors.json")
            if os.path.exists(edge_label_factors_path):
                with open(edge_label_factors_path, "r", encoding="utf-8") as f:
                    edge_label_factors = json.load(f)

        # Load weights before constructing the model.
        weights_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(model_dir, "model.safetensors")

        if os.path.exists(weights_path):
            logger.info(f"Loading weights from: {weights_path}")
            if weights_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(weights_path, device="cpu")
            else:
                state_dict = torch.load(weights_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model weights found in {model_dir} (tried pytorch_model.bin and model.safetensors)")

        node_types = model_config.get("node_types")
        if not node_types:
            node_types_path = os.path.join(model_dir, "node_types.json")
            if os.path.exists(node_types_path):
                with open(node_types_path, "r", encoding="utf-8") as f:
                    node_types = json.load(f)
        if not node_types:
            raise ValueError("Model bundle is missing node_types; retrain/export it with a sample-derived node type vocabulary")
        node_type2id, id2node_type = build_node_type_maps(node_types)
        
        # Build InferenceConfig - always use hidden sizes from model_config for model architecture
        arc_hidden_size = model_config.get("arc_hidden_size", 500)
        rel_hidden_size = model_config.get("rel_hidden_size", 100)
        node_hidden_size = model_config.get("node_hidden_size", 256)
        dropout = model_config.get("dropout", 0.1)
        word_pooling = model_config.get("word_pooling", "mean")
        
        if config is not None:
            inference_config = InferenceConfig(
                arc_hidden_size=arc_hidden_size,
                rel_hidden_size=rel_hidden_size,
                node_hidden_size=node_hidden_size,
                dropout=dropout,
                word_pooling=word_pooling,
                # Copy other settings from passed config
                max_length=config.max_length,
                arc_threshold=config.arc_threshold,
                lam_parent_prob=config.lam_parent_prob,
                lam_sibling_prob=config.lam_sibling_prob,
                lam_pos_prob=config.lam_pos_prob,
                lam_parent_given_child_prob=config.lam_parent_given_child_prob,
                edge_factor_decode_weight=config.edge_factor_decode_weight,
                enforce_connected=config.enforce_connected,
                enable_phase4_rewrites=config.enable_phase4_rewrites,
                enforce_single_root=config.enforce_single_root,
                decoder_mode=config.decoder_mode,
                ilp_max_candidates_per_child=config.ilp_max_candidates_per_child,
                ilp_max_labels_per_pair=config.ilp_max_labels_per_pair,
                ilp_time_limit=config.ilp_time_limit,
                ilp_mip_rel_gap=config.ilp_mip_rel_gap,
                ilp_root_score_weight=config.ilp_root_score_weight,
                ilp_lazy_constraints=config.ilp_lazy_constraints,
                ilp_max_lazy_rounds=config.ilp_max_lazy_rounds,
                hybrid_fallback_policy=config.hybrid_fallback_policy,
                debug=config.debug,
                debug_dir=config.debug_dir,
            )
        else:
            inference_config = InferenceConfig(
                arc_hidden_size=arc_hidden_size,
                rel_hidden_size=rel_hidden_size,
                node_hidden_size=node_hidden_size,
                dropout=dropout,
                word_pooling=word_pooling,
            )
        
        logger.info(f"Hidden sizes - arc: {arc_hidden_size}, rel: {rel_hidden_size}, node: {node_hidden_size}")
        
        # Load tokenizer from model directory
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Create model (load BERT config from pretrained model directory)
        model_edge_types = [id2label[i] for i in range(len(id2label))]
        model = WordLinguaParserV2(
            edge_types=model_edge_types,
            node_types=node_types,
            edge_factor_vocabs=edge_factor_vocabs,
            arc_hidden_size=inference_config.arc_hidden_size,
            rel_hidden_size=inference_config.rel_hidden_size,
            node_hidden_size=inference_config.node_hidden_size,
            dropout=inference_config.dropout,
            word_pooling=inference_config.word_pooling,
            pretrained_model_path=model_dir,
        )
        model.load_state_dict(state_dict, strict=False)
        
        # Load statistics
        statistics = None
        if statistics_path is None:
            candidate = os.path.join(model_dir, "wordlingua_statistics.json")
            if os.path.exists(candidate):
                statistics_path = candidate
        
        if statistics_path and os.path.exists(statistics_path):
            logger.info(f"Loading statistics from: {statistics_path}")
            statistics = WordLinguaStatistics.load(statistics_path, verbose=False)
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            label2id=label2id,
            config=inference_config,
            statistics=statistics,
            id2node_type=id2node_type,
            edge_factor_vocabs=edge_factor_vocabs,
            edge_label_factors=edge_label_factors,
            device=device,
        )
    
    def _encode_words(self, words: List[str]) -> Dict[str, Any]:
        """Encode words into model input tensors."""
        return encode_words_like_training(self.tokenizer, words, max_length=self.config.max_length)
    
    @torch.no_grad()
    def _forward(self, encoded: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Run model forward pass."""
        out = self.model(
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
            encoded["word_to_subword"].to(self.device),
            encoded["word_mask"].to(self.device),
        )
        return out
    
    def parse(
        self,
        sentence: Union[str, List[str]],
        debug: Optional[bool] = None,
    ) -> InferenceResult:
        """
        Parse a sentence into a lingua graph.
        
        Args:
            sentence: Input sentence (string) or pre-tokenized words (list)
            debug: Override debug setting for this call
        
        Returns:
            InferenceResult containing the parsed graphs and metadata
        """
        from lingua.utils.segement import segment
        
        # Tokenize if needed
        if isinstance(sentence, str):
            words = segment(sentence)
            sentence_str = sentence
        else:
            words = sentence
            sentence_str = " ".join(words)
        
        if not words:
            raise ValueError("Empty input sentence")
        
        # Update debug settings
        self._set_debug_state(debug)
        
        self._example_counter += 1
        self.decoder.debug_example_id = self._example_counter
        if self.ilp_decoder is not None:
            self.ilp_decoder.debug_example_id = self._example_counter
        
        if self.config.debug:
            logger.info(f"Parsing sentence {self._example_counter}: {sentence_str[:100]}...")
        
        # Encode
        encoded = self._encode_words(words)
        num_nodes = int(encoded["num_nodes"][0].item())
        
        # Forward pass
        out = self._forward(encoded)
        
        # Extract logits
        arc_logits = out["arc_logits"][0, :num_nodes, :num_nodes].float().cpu().numpy()
        rel_logits_t = out["rel_logits"][0, :num_nodes, :num_nodes, :].float()
        edge_factor_logits_t = {
            name: logits[0, :num_nodes, :num_nodes, :].float()
            for name, logits in (out.get("edge_factor_logits") or {}).items()
        }
        rel_logits = combine_rel_logits_with_edge_factor_logits(
            rel_logits=rel_logits_t,
            edge_factor_logits=edge_factor_logits_t,
            edge_factor_targets_by_label=self.edge_factor_targets_by_label,
            weight=self.config.edge_factor_decode_weight,
        ).cpu().numpy()
        node_type_logits = out["node_type_logits"][0, :num_nodes, :].float().cpu().numpy()
        cow_logits = out["child_of_whether_logits"][0, :num_nodes, :].float().cpu().numpy()
        is_root_logits = out["is_root_logits"][0, :num_nodes, :].float().cpu().numpy()
        
        # Get predictions; decoder maps IDs through the model bundle's node type vocab.
        node_type_preds = np.array([int(np.argmax(node_type_logits[i])) for i in range(num_nodes)])
        cow_preds = [bool(np.argmax(cow_logits[i]) == 1) for i in range(num_nodes)]
        
        word_lingua_graph, lingua_graph, conversion_error, decoding_info = self._decode_with_optional_ilp_fallback(
            words=words,
            arc_logits=arc_logits,
            rel_logits=rel_logits,
            node_type_preds=node_type_preds,
            cow_preds=cow_preds,
            is_root_logits=is_root_logits,
        )
        edges = self._extract_edges_and_apply_child_flags(word_lingua_graph, cow_preds)
        
        return InferenceResult(
            sentence=sentence_str,
            words=words,
            word_lingua_graph=word_lingua_graph,
            lingua_graph=lingua_graph,
            lingua_conversion_error=conversion_error,
            edges=edges,
            node_types=node_type_preds,
            child_of_whether=cow_preds,
            decoding_info=self._result_decoding_info(decoding_info),
        )
    
    def parse_batch(
        self,
        sentences: List[Union[str, List[str]]],
        debug: Optional[bool] = None,
    ) -> List[InferenceResult]:
        """
        Parse a batch of sentences using batched inference.
        
        Uses a single forward pass through the model for all sentences,
        then decodes each output individually.
        
        Args:
            sentences: List of sentences (strings or pre-tokenized word lists)
            debug: Override debug setting for this call
        
        Returns:
            List of InferenceResult for each sentence
        """
        from lingua.utils.segement import segment
        
        if not sentences:
            return []
        
        # Update debug settings
        self._set_debug_state(debug)
        
        # Step 1: Tokenize all sentences
        batch_words: List[List[str]] = []
        sentence_strs: List[str] = []
        valid_indices: List[int] = []  # Track which sentences are valid
        
        for i, sentence in enumerate(sentences):
            try:
                if isinstance(sentence, str):
                    words = segment(sentence)
                    sentence_str = sentence
                else:
                    words = sentence
                    sentence_str = " ".join(words)
                
                if not words:
                    continue
                    
                batch_words.append(words)
                sentence_strs.append(sentence_str)
                valid_indices.append(i)
            except Exception as e:
                logger.debug(f"Failed to tokenize sentence {i}: {e}")
                continue
        
        if not batch_words:
            # Return empty results for all invalid sentences
            return [
                InferenceResult(
                    sentence=s if isinstance(s, str) else " ".join(s),
                    words=s.split() if isinstance(s, str) else s,
                    word_lingua_graph=None,
                    lingua_graph=None,
                    lingua_conversion_error="Failed to tokenize",
                )
                for s in sentences
            ]
        
        # Step 2: Batch encode
        encoded = encode_batch_words_like_training(
            self.tokenizer, 
            batch_words, 
            max_length=self.config.max_length
        )
        
        # Step 3: Single forward pass for the entire batch
        with torch.no_grad():
            out = self.model(
                encoded["input_ids"].to(self.device),
                encoded["attention_mask"].to(self.device),
                encoded["word_to_subword"].to(self.device),
                encoded["word_mask"].to(self.device),
            )
        
        # Step 4: Decode each example
        batch_results: List[InferenceResult] = []
        
        for b_idx, (words, sentence_str) in enumerate(zip(batch_words, sentence_strs)):
            try:
                self._example_counter += 1
                self.decoder.debug_example_id = self._example_counter
                if self.ilp_decoder is not None:
                    self.ilp_decoder.debug_example_id = self._example_counter
                
                num_nodes = int(encoded["num_nodes"][b_idx].item())
                
                # Extract logits for this example
                arc_logits = out["arc_logits"][b_idx, :num_nodes, :num_nodes].float().cpu().numpy()
                rel_logits_t = out["rel_logits"][b_idx, :num_nodes, :num_nodes, :].float()
                edge_factor_logits_t = {
                    name: logits[b_idx, :num_nodes, :num_nodes, :].float()
                    for name, logits in (out.get("edge_factor_logits") or {}).items()
                }
                rel_logits = combine_rel_logits_with_edge_factor_logits(
                    rel_logits=rel_logits_t,
                    edge_factor_logits=edge_factor_logits_t,
                    edge_factor_targets_by_label=self.edge_factor_targets_by_label,
                    weight=self.config.edge_factor_decode_weight,
                ).cpu().numpy()
                node_type_logits = out["node_type_logits"][b_idx, :num_nodes, :].float().cpu().numpy()
                cow_logits = out["child_of_whether_logits"][b_idx, :num_nodes, :].float().cpu().numpy()
                is_root_logits = out["is_root_logits"][b_idx, :num_nodes, :].float().cpu().numpy()
                
                # Get predictions; decoder maps IDs through the model bundle's node type vocab.
                node_type_preds = np.array([int(np.argmax(node_type_logits[i])) for i in range(num_nodes)])
                cow_preds = [bool(np.argmax(cow_logits[i]) == 1) for i in range(num_nodes)]
                
                word_lingua_graph, lingua_graph, conversion_error, decoding_info = self._decode_with_optional_ilp_fallback(
                    words=words,
                    arc_logits=arc_logits,
                    rel_logits=rel_logits,
                    node_type_preds=node_type_preds,
                    cow_preds=cow_preds,
                    is_root_logits=is_root_logits,
                )
                edges = self._extract_edges_and_apply_child_flags(word_lingua_graph, cow_preds)
                
                batch_results.append(InferenceResult(
                    sentence=sentence_str,
                    words=words,
                    word_lingua_graph=word_lingua_graph,
                    lingua_graph=lingua_graph,
                    lingua_conversion_error=conversion_error,
                    edges=edges,
                    node_types=node_type_preds,
                    child_of_whether=cow_preds,
                    decoding_info=self._result_decoding_info(decoding_info),
                ))
                
            except Exception as e:
                import traceback
                logger.error(f"Failed to decode sentence {b_idx}: {e}")
                logger.error(f"Stack trace:\n{traceback.format_exc()}")
                batch_results.append(InferenceResult(
                    sentence=sentence_str,
                    words=words,
                    word_lingua_graph=None,
                    lingua_graph=None,
                    lingua_conversion_error=str(e),
                ))
        
        # Reconstruct full results list with None for invalid indices
        results: List[InferenceResult] = []
        valid_result_idx = 0
        for i, sentence in enumerate(sentences):
            if i in valid_indices:
                results.append(batch_results[valid_result_idx])
                valid_result_idx += 1
            else:
                words = sentence.split() if isinstance(sentence, str) else sentence
                results.append(InferenceResult(
                    sentence=" ".join(words) if isinstance(sentence, list) else sentence,
                    words=words,
                    word_lingua_graph=None,
                    lingua_graph=None,
                    lingua_conversion_error="Failed to process",
                ))
        
        return results


def main():
    """Command-line interface for word-lingua inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Word-Lingua Inference")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--statistics", type=str, default=None, help="Path to wordlingua_statistics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input", type=str, default=None, help="Input sentence")
    parser.add_argument("--input-file", type=str, default=None, help="JSONL file with sentences")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file")
    parser.add_argument("--decoder-mode", type=str, default="ilp", choices=["hybrid", "ilp", "sdp_graph"], help="Decoder backend")
    parser.add_argument("--hybrid-fallback-policy", type=str, default="conversion", choices=["conversion", "issues"], help="Hybrid fallback trigger")
    parser.add_argument("--no-ilp-lazy-constraints", action="store_true", help="Disable lazy ILP high-order cuts")
    parser.add_argument("--ilp-max-lazy-rounds", type=int, default=4, help="Maximum lazy-cut rounds before eager fallback")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--debug-dir", type=str, default=None, help="Directory for debug outputs")
    
    args = parser.parse_args()
    
    # Create config
    config = InferenceConfig(
        decoder_mode=args.decoder_mode,
        hybrid_fallback_policy=args.hybrid_fallback_policy,
        ilp_lazy_constraints=not args.no_ilp_lazy_constraints,
        ilp_max_lazy_rounds=args.ilp_max_lazy_rounds,
        debug=args.debug,
        debug_dir=args.debug_dir,
    )
    
    # Load inference engine
    inference = WordLinguaInference.from_pretrained(
        model_dir=args.model_dir,
        device=args.device,
        statistics_path=args.statistics,
        config=config,
    )
    
    # Process input
    results = []
    
    if args.input:
        result = inference.parse(args.input)
        results.append(result)
    elif args.input_file:
        import jsonlines
        with jsonlines.open(args.input_file, "r") as f:
            for item in f:
                sentence = item.get("sentence") or item.get("text")
                if sentence:
                    result = inference.parse(sentence)
                    results.append(result)
    else:
        print("Enter sentences (one per line, empty line to quit):")
        while True:
            try:
                line = input().strip()
                if not line:
                    break
                result = inference.parse(line)
                results.append(result)
                print(json.dumps(result.to_dict(), ensure_ascii=False))
            except EOFError:
                break
    
    # Output results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    elif results and args.input:
        print(json.dumps(results[0].to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
