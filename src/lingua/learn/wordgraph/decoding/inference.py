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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from lingua.structure.gpgraph import GPGraph, GPGPhraseNode, GPGEdge

logger = logging.getLogger(__name__)


# ============================================================================
# Shared Utilities (decoding-agnostic)
# ============================================================================

@dataclass(frozen=True)
class WordLinguaModelOutputs:
    """Standardized model output container."""
    arc_logits: torch.Tensor  # [B, N, N]
    rel_logits: torch.Tensor  # [B, N, N, L]
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
        node_pos.append(id2node_type.get(int(t), "UNK"))
    return node_pos


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
        self.device = device
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        # Lazy import to avoid circular dependencies
        from lingua.learn.wordgraph.decoding.graph_decoder import GreedyPatternDecoder
        from lingua.learn.wordgraph.model.biaffine import ID2NODE_TYPE
        
        self._id2node_type = ID2NODE_TYPE
        
        # Initialize decoder
        self.decoder = GreedyPatternDecoder(
            stats=None,  # Legacy parameter, ignored
            id2label=id2label,
            label2id=label2id,
            arc_threshold=config.arc_threshold,
            edge_prob_tables=statistics,
            lam_parent_prob=config.lam_parent_prob,
            lam_sibling_prob=config.lam_sibling_prob,
            lam_pos_prob=config.lam_pos_prob,
            lam_parent_given_child_prob=config.lam_parent_given_child_prob,
            debug=config.debug,
            debug_dir=config.debug_dir,
        )
        
        self._example_counter = 0
    
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
        from lingua.learn.wordgraph.model.biaffine import WordLinguaParserV2
        
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
        
        # Load label2id.json
        label2id_path = os.path.join(model_dir, "label2id.json")
        if not os.path.exists(label2id_path):
            raise FileNotFoundError(f"label2id.json not found in {model_dir}")
        
        with open(label2id_path, "r", encoding="utf-8") as f:
            label2id = {str(k): int(v) for k, v in json.load(f).items()}
        id2label = {v: k for k, v in label2id.items()}
        
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
        model = WordLinguaParserV2(
            label_num=len(label2id),
            arc_hidden_size=inference_config.arc_hidden_size,
            rel_hidden_size=inference_config.rel_hidden_size,
            node_hidden_size=inference_config.node_hidden_size,
            dropout=inference_config.dropout,
            word_pooling=inference_config.word_pooling,
            pretrained_model_path=model_dir,
        )
        
        # Load weights from pytorch_model.bin or model.safetensors
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
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"No model weights found in {model_dir} (tried pytorch_model.bin and model.safetensors)")
        
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
        from lingua.learn.wordgraph.modeler.word2gp import wordlingua2lingua
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
        if debug is not None:
            self.decoder.debug = debug
        
        self._example_counter += 1
        self.decoder.debug_example_id = self._example_counter
        
        if self.config.debug:
            logger.info(f"Parsing sentence {self._example_counter}: {sentence_str[:100]}...")
        
        # Encode
        encoded = self._encode_words(words)
        num_nodes = int(encoded["num_nodes"][0].item())
        
        # Forward pass
        out = self._forward(encoded)
        
        # Extract logits
        arc_logits = out["arc_logits"][0, :num_nodes, :num_nodes].float().cpu().numpy()
        rel_logits = out["rel_logits"][0, :num_nodes, :num_nodes, :].float().cpu().numpy()
        node_type_logits = out["node_type_logits"][0, :num_nodes, :].float().cpu().numpy()
        cow_logits = out["child_of_whether_logits"][0, :num_nodes, :].float().cpu().numpy()
        is_root_logits = out["is_root_logits"][0, :num_nodes, :].float().cpu().numpy()
        
        # Get predictions - pass integer IDs to decoder (it looks them up in ID2NODE_TYPE)
        node_type_preds = np.array([int(np.argmax(node_type_logits[i])) for i in range(num_nodes)])
        cow_preds = [bool(np.argmax(cow_logits[i]) == 1) for i in range(num_nodes)]
        
        # Decode
        word_lingua_graph, decoding_info = self.decoder.decode(
            words=words,
            arc_logits=arc_logits,
            rel_logits=rel_logits,
            node_type_preds=node_type_preds,
            child_of_whether_preds=np.array([int(c) for c in cow_preds]),
            is_root_logits=is_root_logits,
        )
        
        # Extract edges from decoded graph
        edges = []
        node_id_to_idx = {}
        for node in word_lingua_graph.nodes():
            if hasattr(node, 'spans') and node.spans:
                span = node.spans[0]
                idx = span[0] if isinstance(span, (tuple, list)) else span
                node_id_to_idx[node.ID] = idx
        
        for parent, edge, child in word_lingua_graph.edges():
            p_idx = node_id_to_idx.get(parent.ID)
            c_idx = node_id_to_idx.get(child.ID)
            if p_idx is not None and c_idx is not None:
                edges.append((p_idx, c_idx, edge.label))
        
        # Set child_of_whether attribute
        for node in word_lingua_graph.nodes():
            idx = node_id_to_idx.get(node.ID)
            if idx is not None and idx < len(cow_preds) and cow_preds[idx]:
                node.child_of_whether = True
        
        # Convert to lingua graph (deepcopy to preserve word_lingua_graph)
        lingua_graph = None
        conversion_error = None
        try:
            import copy
            lingua_graph, _, _ = wordlingua2lingua(copy.deepcopy(word_lingua_graph))
        except Exception as e:
            conversion_error = str(e)
            if self.config.debug:
                logger.warning(f"Word-lingua to lingua conversion failed: {e}")
        
        return InferenceResult(
            sentence=sentence_str,
            words=words,
            word_lingua_graph=word_lingua_graph,
            lingua_graph=lingua_graph,
            lingua_conversion_error=conversion_error,
            edges=edges,
            node_types=node_type_preds,
            child_of_whether=cow_preds,
            decoding_info=decoding_info if self.config.debug else {},
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
        from lingua.learn.wordgraph.modeler.word2gp import wordlingua2lingua
        from lingua.utils.segement import segment
        
        if not sentences:
            return []
        
        # Update debug settings
        if debug is not None:
            self.decoder.debug = debug
        
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
                
                num_nodes = int(encoded["num_nodes"][b_idx].item())
                
                # Extract logits for this example
                arc_logits = out["arc_logits"][b_idx, :num_nodes, :num_nodes].float().cpu().numpy()
                rel_logits = out["rel_logits"][b_idx, :num_nodes, :num_nodes, :].float().cpu().numpy()
                node_type_logits = out["node_type_logits"][b_idx, :num_nodes, :].float().cpu().numpy()
                cow_logits = out["child_of_whether_logits"][b_idx, :num_nodes, :].float().cpu().numpy()
                is_root_logits = out["is_root_logits"][b_idx, :num_nodes, :].float().cpu().numpy()
                
                # Get predictions - pass integer IDs to decoder (it looks them up in ID2NODE_TYPE)
                node_type_preds = np.array([int(np.argmax(node_type_logits[i])) for i in range(num_nodes)])
                cow_preds = [bool(np.argmax(cow_logits[i]) == 1) for i in range(num_nodes)]
                
                # Decode
                word_lingua_graph, decoding_info = self.decoder.decode(
                    words=words,
                    arc_logits=arc_logits,
                    rel_logits=rel_logits,
                    node_type_preds=node_type_preds,
                    child_of_whether_preds=np.array([int(c) for c in cow_preds]),
                    is_root_logits=is_root_logits,
                )
                
                # Extract edges from decoded graph
                edges = []
                node_id_to_idx = {}
                for node in word_lingua_graph.nodes():
                    if hasattr(node, 'spans') and node.spans:
                        span = node.spans[0]
                        idx = span[0] if isinstance(span, (tuple, list)) else span
                        node_id_to_idx[node.ID] = idx
                
                for parent, edge, child in word_lingua_graph.edges():
                    p_idx = node_id_to_idx.get(parent.ID)
                    c_idx = node_id_to_idx.get(child.ID)
                    if p_idx is not None and c_idx is not None:
                        edges.append((p_idx, c_idx, edge.label))
                
                # Set child_of_whether attribute
                for node in word_lingua_graph.nodes():
                    idx = node_id_to_idx.get(node.ID)
                    if idx is not None and idx < len(cow_preds) and cow_preds[idx]:
                        node.child_of_whether = True
                
                # Convert to lingua graph (deepcopy to preserve word_lingua_graph)
                lingua_graph = None
                conversion_error = None
                try:
                    import copy
                    lingua_graph, _, _ = wordlingua2lingua(copy.deepcopy(word_lingua_graph))
                except Exception as e:
                    conversion_error = str(e)
                    if self.config.debug:
                        logger.warning(f"Word-lingua to lingua conversion failed: {e}")
                
                batch_results.append(InferenceResult(
                    sentence=sentence_str,
                    words=words,
                    word_lingua_graph=word_lingua_graph,
                    lingua_graph=lingua_graph,
                    lingua_conversion_error=conversion_error,
                    edges=edges,
                    node_types=node_type_preds,
                    child_of_whether=cow_preds,
                    decoding_info=decoding_info if self.config.debug else {},
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
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--debug-dir", type=str, default=None, help="Directory for debug outputs")
    
    args = parser.parse_args()
    
    # Create config
    config = InferenceConfig(
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
