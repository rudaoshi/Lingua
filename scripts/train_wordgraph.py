"""
Word-Lingua Graph Parser using BERT and Biaffine Attention (v2)

This version reads lingua graph data directly and converts to word_lingua format,
properly handling the child_of_whether attribute and node type prediction.

Key additions:
1. Reads from lingua graph data (not pre-converted word_lingua)
2. Predicts child_of_whether attribute (boolean per node)
3. Predicts node type/pos (e.g., NominalConstant, FactualPredicator, etc.)
4. Root node is learned through graph structure (added in gp2word.py)
"""

import os
import sys
import json
import copy
import logging
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import jsonlines
from tqdm import tqdm
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "wordgraph-baseline"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lingua.structure.gpgraph import GPGraph, GPGPhraseNode, GPGAuxNode
from lingua.learn.wordgraph.modeler.gp2word import lingua2wordlingua
from lingua.learn.wordgraph.model.biaffine import (
    WordLinguaParserV2,
    build_node_type_maps,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset (v2 - reads lingua data directly)
# ============================================================================

@dataclass
class WordLinguaExampleV2:
    """A single training example for word-lingua parsing (v2)."""
    sentence: str
    words: List[str]
    # For each node pair (i, j), store edge label id if exists (-1 otherwise).
    # Note: node indices correspond 1:1 to word indices (NO synthetic/virtual ROOT node).
    arc_labels: List[List[int]]
    # is_root[i] = True if node i is a (topological) root (in-degree == 0)
    is_root: List[bool]
    # child_of_whether[i] = True if word i has child_of_whether attribute
    child_of_whether: List[bool]
    # node_type[i] = type_id for word i (e.g., NominalConstant, FactualPredicator)
    node_type: List[int] = field(default_factory=list)
    # Original graph for evaluation
    original_graph: Optional[Any] = None


class WordLinguaDatasetV2(Dataset):
    """Dataset for word-lingua-graph parsing (v2) - reads lingua data directly."""
    
    def __init__(self, data_path: str, tokenizer, label2id: Dict[str, int],
                 node_type2id: Dict[str, int], max_length: int = 512):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.node_type2id = node_type2id
        self.max_length = max_length
        self.examples: List[WordLinguaExampleV2] = []
        
        self._load_data(data_path)
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def _load_data(self, data_path: str):
        """Load and process data from JSONL file."""
        with jsonlines.open(data_path, "r") as reader:
            for idx, item in enumerate(reader):
                try:
                    example = self._process_item(item)
                    if example is not None:
                        self.examples.append(example)
                except Exception as e:
                    logger.warning(f"Error processing item {idx}: {e}")
                    continue
    
    def _process_item(self, item: dict) -> Optional[WordLinguaExampleV2]:
        """Process a single item from the dataset."""
        sentence = item["sentence"]
        
        # Parse lingua graph from the "lingua" field
        lingua_graph = GPGraph.parse(item["lingua"])
        original_graph = copy.deepcopy(lingua_graph)
        
        # Convert to word-lingua-graph
        try:
            word_lingua_graph, _, _ = lingua2wordlingua(lingua_graph)
        except Exception as e:
            logger.debug(f"Failed to convert graph: {e}")
            raise e 
        
        # Get words
        words = word_lingua_graph.words
        if not words:
            return None
        
        num_words = len(words)
        # IMPORTANT: input word-lingua graphs do NOT contain a synthetic/virtual ROOT node.
        # Keep node indexing aligned with words: node_idx == word_idx.
        num_nodes = num_words
        
        # Build adjacency matrix with edge labels
        arc_labels = [[-1] * num_nodes for _ in range(num_nodes)]
        
        # Track child_of_whether attribute and node type
        child_of_whether = [False] * num_nodes
        node_type = [-1] * num_nodes
        
        def _phrase_word_idx_to_node_idx(word_idx: int) -> int:
            return word_idx

        # Extract node attributes for phrase nodes
        for node in word_lingua_graph.nodes():
            if isinstance(node, GPGPhraseNode):
                word_indices = list(node.words(with_aux=False))
                if len(word_indices) == 1:
                    word_idx = word_indices[0]
                    if word_idx < num_words:
                        node_idx = _phrase_word_idx_to_node_idx(word_idx)
                        # Check for child_of_whether attribute
                        if hasattr(node, 'child_of_whether') and node.child_of_whether:
                            child_of_whether[node_idx] = True
                        # Get node type (pos)
                        if node.pos not in self.node_type2id:
                            raise ValueError(f"Unknown node type in sample: {node.pos}")
                        node_type[node_idx] = self.node_type2id[node.pos]

        missing_node_type = [idx for idx, type_id in enumerate(node_type) if type_id < 0]
        if missing_node_type:
            raise ValueError(f"Missing node type labels for word indices: {missing_node_type}")
        
        # Process edges (root node is now part of graph structure, learned through arc/relation prediction)
        for parent, edge, child in word_lingua_graph.edges():
            parent_idx = None
            child_idx = None

            if isinstance(parent, GPGPhraseNode):
                parent_words = list(parent.words(with_aux=False))
                if len(parent_words) == 1 and parent_words[0] < num_words:
                    parent_idx = _phrase_word_idx_to_node_idx(parent_words[0])

            if isinstance(child, GPGPhraseNode):
                child_words = list(child.words(with_aux=False))
                if len(child_words) == 1 and child_words[0] < num_words:
                    child_idx = _phrase_word_idx_to_node_idx(child_words[0])

            if parent_idx is None or child_idx is None:
                continue
            
            # Get edge label
            edge_label = edge.label
            if edge_label not in self.label2id:
                raise ValueError(f"Unknown edge label in sample: {edge_label}")
            label_id = self.label2id[edge_label]
            
            arc_labels[parent_idx][child_idx] = label_id

        # Derive is_root labels from gold adjacency: is_root if in-degree == 0.
        indeg = [0] * num_nodes
        for p in range(num_nodes):
            for c in range(num_nodes):
                if p == c:
                    continue
                if arc_labels[p][c] >= 0:
                    indeg[c] += 1
        is_root = [d == 0 for d in indeg]
        
        return WordLinguaExampleV2(
            sentence=sentence,
            words=words,
            arc_labels=arc_labels,
            is_root=is_root,
            child_of_whether=child_of_whether,
            node_type=node_type,
            original_graph=original_graph
        )
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        return self._encode_example(example)
    
    def _encode_example(self, example: WordLinguaExampleV2) -> Dict[str, Any]:
        """Encode a single example for the model."""
        words = example.words
        
        # Tokenize each word and track word-to-subword mapping
        all_tokens = ["[CLS]"]
        # NOTE: [CLS] is present as a transformer token, but NOT a graph node.
        word_to_subword = []
        
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = ["[UNK]"]
            
            start_idx = len(all_tokens)
            subword_indices = list(range(start_idx, start_idx + len(word_tokens)))
            word_to_subword.append(subword_indices)
            
            all_tokens.extend(word_tokens)
        
        all_tokens.append("[SEP]")
        
        # Truncate if necessary
        if len(all_tokens) > self.max_length:
            all_tokens = all_tokens[:self.max_length - 1] + ["[SEP]"]
            new_word_to_subword = []
            for subword_indices in word_to_subword:
                valid_indices = [idx for idx in subword_indices if idx < self.max_length - 1]
                if valid_indices:
                    new_word_to_subword.append(valid_indices)
                else:
                    break
            word_to_subword = new_word_to_subword
        
        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(all_tokens)
        attention_mask = [1] * len(input_ids)
        
        trunc_num_words = len(word_to_subword)
        num_nodes = trunc_num_words

        # Build labels aligned with truncation (no synthetic ROOT node).
        arc_labels = [row[:trunc_num_words] for row in example.arc_labels[:trunc_num_words]]
        child_of_whether = example.child_of_whether[:trunc_num_words]
        node_type = example.node_type[:trunc_num_words]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_to_subword": word_to_subword,
            "arc_labels": arc_labels,
            "is_root": [int(x) for x in example.is_root[:trunc_num_words]],
            "child_of_whether": child_of_whether,
            "node_type": node_type,
            "num_words": num_nodes,
            "sentence": example.sentence,
        }


def collate_fn_v2(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader (v2)."""
    batch_size = len(batch)
    
    # Find max lengths
    max_seq_len = max(len(item["input_ids"]) for item in batch)
    max_words = max(item["num_words"] for item in batch)
    
    # Find max subwords per word
    max_subwords = 1
    for item in batch:
        for subword_indices in item["word_to_subword"]:
            max_subwords = max(max_subwords, len(subword_indices))
    
    # Prepare tensors
    input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    word_to_subword = torch.full((batch_size, max_words, max_subwords), -1, dtype=torch.long)
    word_mask = torch.zeros(batch_size, max_words, dtype=torch.bool)
    arc_labels = torch.full((batch_size, max_words, max_words), -1, dtype=torch.long)
    is_root = torch.zeros(batch_size, max_words, dtype=torch.long)
    child_of_whether = torch.zeros(batch_size, max_words, dtype=torch.long)
    node_type = torch.zeros(batch_size, max_words, dtype=torch.long)
    
    sentences = []
    
    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        num_words = item["num_words"]
        
        input_ids[i, :seq_len] = torch.tensor(item["input_ids"])
        attention_mask[i, :seq_len] = torch.tensor(item["attention_mask"], dtype=torch.bool)
        word_mask[i, :num_words] = True
        
        # Fill word_to_subword mapping
        for w_idx, subword_indices in enumerate(item["word_to_subword"]):
            for s_idx, subword_idx in enumerate(subword_indices):
                word_to_subword[i, w_idx, s_idx] = subword_idx
        
        # Fill arc labels
        for j in range(num_words):
            for k in range(num_words):
                if j < len(item["arc_labels"]) and k < len(item["arc_labels"][j]):
                    arc_labels[i, j, k] = item["arc_labels"][j][k]
        
        # Fill child_of_whether and node_type
        for j in range(num_words):
            is_root[i, j] = int(item["is_root"][j])
            child_of_whether[i, j] = int(item["child_of_whether"][j])
            node_type[i, j] = item["node_type"][j]
        
        sentences.append(item["sentence"])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "word_to_subword": word_to_subword,
        "word_mask": word_mask,
        "arc_labels": arc_labels,
        "is_root": is_root,
        "child_of_whether": child_of_whether,
        "node_type": node_type,
        "sentences": sentences,
    }


# ============================================================================
# Training Functions
# ============================================================================

def compute_loss_v2(model_output: Dict[str, torch.Tensor],
                    arc_labels: torch.Tensor,
                    is_root: torch.Tensor,
                    child_of_whether: torch.Tensor,
                    node_type: torch.Tensor,
                    word_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute loss for all prediction tasks.
    """
    arc_logits = model_output["arc_logits"]
    rel_logits = model_output["rel_logits"]
    child_of_whether_logits = model_output["child_of_whether_logits"]
    node_type_logits = model_output["node_type_logits"]
    is_root_logits = model_output["is_root_logits"]
    
    device = arc_logits.device
    
    # Create valid position mask
    pairwise_mask = word_mask.unsqueeze(-1) & word_mask.unsqueeze(-2)
    
    # Arc loss
    if pairwise_mask.any():
        arc_logits_selected = arc_logits[pairwise_mask]
        arc_targets_selected = (arc_labels[pairwise_mask] >= 0).float()
        arc_loss = F.binary_cross_entropy_with_logits(
            arc_logits_selected, arc_targets_selected, reduction='mean'
        )
    else:
        arc_loss = torch.tensor(0.0, device=device)
    
    # Relation loss
    edge_mask = (arc_labels >= 0) & pairwise_mask
    if edge_mask.any():
        rel_logits_selected = rel_logits[edge_mask]
        rel_targets = arc_labels[edge_mask]
        rel_loss = F.cross_entropy(rel_logits_selected, rel_targets)
    else:
        rel_loss = torch.tensor(0.0, device=device)
    
    # Child of whether loss (node-level)
    if word_mask.any():
        cow_logits_selected = child_of_whether_logits[word_mask]
        cow_targets = child_of_whether[word_mask]
        # Use class weights to handle imbalance (child_of_whether is rare)
        cow_loss = F.cross_entropy(cow_logits_selected, cow_targets)
    else:
        cow_loss = torch.tensor(0.0, device=device)
    
    # Node type loss (node-level multi-class classification)
    if word_mask.any():
        node_type_logits_selected = node_type_logits[word_mask]
        node_type_targets = node_type[word_mask]
        node_type_loss = F.cross_entropy(node_type_logits_selected, node_type_targets)
    else:
        node_type_loss = torch.tensor(0.0, device=device)

    # is_root loss (node-level binary classification)
    if word_mask.any():
        is_root_logits_selected = is_root_logits[word_mask]
        is_root_targets = is_root[word_mask]
        is_root_loss = F.cross_entropy(is_root_logits_selected, is_root_targets)
    else:
        is_root_loss = torch.tensor(0.0, device=device)
    
    return {
        "arc_loss": arc_loss,
        "rel_loss": rel_loss,
        "child_of_whether_loss": cow_loss,
        "node_type_loss": node_type_loss,
        "is_root_loss": is_root_loss,
    }


def evaluate_v2(model: nn.Module, dataloader: DataLoader, 
                device: torch.device, id2label: Dict[int, str]) -> Dict[str, float]:
    """Evaluate the model on a dataset (v2)."""
    model.eval()
    
    total_arc_correct = 0
    total_arc_pred = 0
    total_arc_gold = 0
    
    total_rel_correct = 0
    total_rel_gold = 0
    
    total_cow_correct = 0
    total_cow_pred = 0
    total_cow_gold = 0
    
    total_node_type_correct = 0
    total_node_type_total = 0

    total_is_root_correct = 0
    total_is_root_pred = 0
    total_is_root_gold = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_to_subword = batch["word_to_subword"].to(device)
            word_mask = batch["word_mask"].to(device)
            arc_labels = batch["arc_labels"].to(device)
            is_root_gt = batch["is_root"].to(device)
            child_of_whether_gt = batch["child_of_whether"].to(device)
            node_type_gt = batch["node_type"].to(device)
            
            output = model(input_ids, attention_mask, word_to_subword, word_mask)
            
            arc_logits = output["arc_logits"]
            rel_logits = output["rel_logits"]
            cow_logits = output["child_of_whether_logits"]
            node_type_logits = output["node_type_logits"]
            is_root_logits = output["is_root_logits"]
            
            # Arc predictions
            arc_preds = (arc_logits > 0)
            arc_gold = (arc_labels >= 0)
            pairwise_mask = word_mask.unsqueeze(-1) & word_mask.unsqueeze(-2)
            
            total_arc_correct += (arc_preds & arc_gold & pairwise_mask).sum().item()
            total_arc_pred += (arc_preds & pairwise_mask).sum().item()
            total_arc_gold += (arc_gold & pairwise_mask).sum().item()
            
            # Relation predictions
            rel_preds = rel_logits.argmax(dim=-1)
            edge_match = arc_preds & arc_gold & pairwise_mask
            if edge_match.any():
                rel_correct = (rel_preds == arc_labels) & edge_match
                total_rel_correct += rel_correct.sum().item()
                total_rel_gold += edge_match.sum().item()
            
            # Child of whether predictions
            cow_preds = cow_logits.argmax(dim=-1)
            cow_gold = child_of_whether_gt
            total_cow_correct += ((cow_preds == 1) & (cow_gold == 1) & word_mask).sum().item()
            total_cow_pred += ((cow_preds == 1) & word_mask).sum().item()
            total_cow_gold += ((cow_gold == 1) & word_mask).sum().item()
            
            # Node type predictions
            node_type_preds = node_type_logits.argmax(dim=-1)
            total_node_type_correct += ((node_type_preds == node_type_gt) & word_mask).sum().item()
            total_node_type_total += word_mask.sum().item()

            # is_root predictions
            is_root_preds = is_root_logits.argmax(dim=-1)
            total_is_root_correct += ((is_root_preds == 1) & (is_root_gt == 1) & word_mask).sum().item()
            total_is_root_pred += ((is_root_preds == 1) & word_mask).sum().item()
            total_is_root_gold += ((is_root_gt == 1) & word_mask).sum().item()
    
    # Compute F1 scores
    arc_precision = total_arc_correct / (total_arc_pred + 1e-8)
    arc_recall = total_arc_correct / (total_arc_gold + 1e-8)
    arc_f1 = 2 * arc_precision * arc_recall / (arc_precision + arc_recall + 1e-8)
    
    rel_accuracy = total_rel_correct / (total_rel_gold + 1e-8)
    
    labeled_precision = total_rel_correct / (total_arc_pred + 1e-8)
    labeled_recall = total_rel_correct / (total_arc_gold + 1e-8)
    labeled_f1 = 2 * labeled_precision * labeled_recall / (labeled_precision + labeled_recall + 1e-8)
    
    cow_precision = total_cow_correct / (total_cow_pred + 1e-8)
    cow_recall = total_cow_correct / (total_cow_gold + 1e-8)
    cow_f1 = 2 * cow_precision * cow_recall / (cow_precision + cow_recall + 1e-8)
    
    node_type_accuracy = total_node_type_correct / (total_node_type_total + 1e-8)

    is_root_precision = total_is_root_correct / (total_is_root_pred + 1e-8)
    is_root_recall = total_is_root_correct / (total_is_root_gold + 1e-8)
    is_root_f1 = 2 * is_root_precision * is_root_recall / (is_root_precision + is_root_recall + 1e-8)
    
    return {
        "arc_precision": arc_precision,
        "arc_recall": arc_recall,
        "arc_f1": arc_f1,
        "rel_accuracy": rel_accuracy,
        "labeled_precision": labeled_precision,
        "labeled_recall": labeled_recall,
        "labeled_f1": labeled_f1,
        "child_of_whether_precision": cow_precision,
        "child_of_whether_recall": cow_recall,
        "child_of_whether_f1": cow_f1,
        "node_type_accuracy": node_type_accuracy,
        "is_root_precision": is_root_precision,
        "is_root_recall": is_root_recall,
        "is_root_f1": is_root_f1,
    }


def train_epoch_v2(model: nn.Module, dataloader: DataLoader,
                   optimizer: torch.optim.Optimizer, 
                   device: torch.device,
                   arc_weight: float = 1.0,
                   rel_weight: float = 1.0,
                   cow_weight: float = 1.0,
                   node_type_weight: float = 1.0,
                   is_root_weight: float = 1.0) -> Dict[str, float]:
    """Train for one epoch (v2)."""

    model.train()
    
    total_arc_loss = 0.0
    total_rel_loss = 0.0
    total_cow_loss = 0.0
    total_node_type_loss = 0.0
    total_is_root_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        word_to_subword = batch["word_to_subword"].to(device)
        word_mask = batch["word_mask"].to(device)
        arc_labels = batch["arc_labels"].to(device)
        is_root = batch["is_root"].to(device)
        child_of_whether = batch["child_of_whether"].to(device)
        node_type = batch["node_type"].to(device)
        
        # Forward pass
        output = model(input_ids, attention_mask, word_to_subword, word_mask)
        
        # Compute loss
        losses = compute_loss_v2(output, arc_labels, is_root, child_of_whether, node_type, word_mask)
        
        loss = (arc_weight * losses["arc_loss"] + 
                rel_weight * losses["rel_loss"] +
                cow_weight * losses["child_of_whether_loss"] +
                node_type_weight * losses["node_type_loss"] +
                is_root_weight * losses["is_root_loss"])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_arc_loss += losses["arc_loss"].item()
        total_rel_loss += losses["rel_loss"].item()
        total_cow_loss += losses["child_of_whether_loss"].item()
        total_node_type_loss += losses["node_type_loss"].item()
        total_is_root_loss += losses["is_root_loss"].item()
        num_batches += 1
        
        progress_bar.set_postfix({
            "arc": f"{losses['arc_loss'].item():.4f}",
            "rel": f"{losses['rel_loss'].item():.4f}",
            "cow": f"{losses['child_of_whether_loss'].item():.4f}",
            "ntype": f"{losses['node_type_loss'].item():.4f}"
        })
    
    return {
        "arc_loss": total_arc_loss / num_batches,
        "rel_loss": total_rel_loss / num_batches,
        "child_of_whether_loss": total_cow_loss / num_batches,
        "node_type_loss": total_node_type_loss / num_batches,
        "is_root_loss": total_is_root_loss / num_batches,
    }


# ============================================================================
# Main Training Function
# ============================================================================

def _ordered_vocab(counter: Counter) -> List[str]:
    """Return labels in first-seen order from a Counter."""
    return list(counter.keys())


def build_vocab_v2(
    data_paths: List[str],
) -> Tuple[List[str], Dict[str, int], Dict[int, str], List[str], Dict[str, int], Dict[int, str]]:
    """Build edge and node type vocabularies from word-lingua samples."""
    edge_counter = Counter()
    node_type_counter = Counter()
    
    for data_path in data_paths:
        with jsonlines.open(data_path, "r") as reader:
            for item in reader:
                try:
                    lingua_graph = GPGraph.parse(item["lingua"])
                    word_lingua_graph, _, _ = lingua2wordlingua(lingua_graph)

                    for node in word_lingua_graph.nodes():
                        if isinstance(node, GPGPhraseNode):
                            word_indices = list(node.words(with_aux=False))
                            if len(word_indices) == 1:
                                node_type_counter[node.pos] += 1
                    
                    for _, edge, _ in word_lingua_graph.edges():
                        edge_counter[edge.label] += 1
                except Exception:
                    raise 

    node_types = _ordered_vocab(node_type_counter)
    edge_types = _ordered_vocab(edge_counter)
    if not node_types:
        raise ValueError("No node types found in training data")
    if not edge_types:
        raise ValueError("No edge types found in training data")

    node_type2id, id2node_type = build_node_type_maps(node_types)
    label2id = {label: idx for idx, label in enumerate(edge_types)}
    
    id2label = {v: k for k, v in label2id.items()}
    
    logger.info(f"Built edge vocabulary with {len(edge_types)} labels")
    for label, count in edge_counter.most_common(20):
        logger.info(f"  {label}: {count}")
    logger.info(f"Built node type vocabulary with {len(node_types)} types")
    for node_type, count in node_type_counter.most_common():
        logger.info(f"  {node_type}: {count}")
    
    return edge_types, label2id, id2label, node_types, node_type2id, id2node_type


def export_pretrained_bundle(
    *,
    output_dir: str,
    model: nn.Module,
    tokenizer,
    label2id: Dict[str, int],
    edge_types: List[str],
    node_types: List[str],
    bert_model_name: str,
    arc_hidden_size: int,
    rel_hidden_size: int,
    node_hidden_size: int,
    dropout: float,
    word_pooling: str,
) -> None:
    """Write a Hugging Face style bundle that inference scripts can load directly."""
    os.makedirs(output_dir, exist_ok=True)

    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "edge_types.json"), "w", encoding="utf-8") as f:
        json.dump(edge_types, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "node_types.json"), "w", encoding="utf-8") as f:
        json.dump(node_types, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "bert_model_name": bert_model_name,
                "arc_hidden_size": arc_hidden_size,
                "rel_hidden_size": rel_hidden_size,
                "node_hidden_size": node_hidden_size,
                "dropout": dropout,
                "word_pooling": word_pooling,
                "edge_types": edge_types,
                "node_types": node_types,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))


def save_model_variant(
    *,
    output_dir: str,
    checkpoint_name: str,
    bundle_name: str,
    model: nn.Module,
    tokenizer,
    optimizer,
    epoch: int,
    train_metrics: Dict[str, float],
    dev_metrics: Dict[str, float],
    label2id: Dict[str, int],
    edge_types: List[str],
    node_types: List[str],
    bert_model_name: str,
    arc_hidden_size: int,
    rel_hidden_size: int,
    node_hidden_size: int,
    dropout: float,
    word_pooling: str,
    selection_metric: str,
    selection_score: float,
    export_legacy_root_bundle: bool = False,
    legacy_checkpoint_name: Optional[str] = None,
) -> None:
    """Save a reloadable checkpoint and an inference-ready pretrained bundle."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "selection_metric": selection_metric,
        "selection_score": selection_score,
        "label2id": label2id,
        "edge_types": edge_types,
        "node_types": node_types,
    }
    torch.save(checkpoint, os.path.join(output_dir, checkpoint_name))
    if legacy_checkpoint_name:
        torch.save(checkpoint, os.path.join(output_dir, legacy_checkpoint_name))

    export_pretrained_bundle(
        output_dir=os.path.join(output_dir, bundle_name),
        model=model,
        tokenizer=tokenizer,
        label2id=label2id,
        edge_types=edge_types,
        node_types=node_types,
        bert_model_name=bert_model_name,
        arc_hidden_size=arc_hidden_size,
        rel_hidden_size=rel_hidden_size,
        node_hidden_size=node_hidden_size,
        dropout=dropout,
        word_pooling=word_pooling,
    )

    if export_legacy_root_bundle:
        export_pretrained_bundle(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            label2id=label2id,
            edge_types=edge_types,
            node_types=node_types,
            bert_model_name=bert_model_name,
            arc_hidden_size=arc_hidden_size,
            rel_hidden_size=rel_hidden_size,
            node_hidden_size=node_hidden_size,
            dropout=dropout,
            word_pooling=word_pooling,
        )


def main(args):
    """Main training function."""
    
    # Set up device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Build vocabularies from the training samples.
    logger.info("Building edge and node type vocabularies...")
    train_path = os.path.join(args.data_dir, "train.jsonl")
    dev_path = os.path.join(args.data_dir, "dev.jsonl")
    test_path = os.path.join(args.data_dir, "test.jsonl")
    
    edge_types, label2id, id2label, node_types, node_type2id, _ = build_vocab_v2([train_path])
    
    # Save vocabularies at the run root as well as inside exported bundles.
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "edge_types.json"), "w", encoding="utf-8") as f:
        json.dump(edge_types, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "node_types.json"), "w", encoding="utf-8") as f:
        json.dump(node_types, f, indent=2, ensure_ascii=False)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.bert_model}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    tokenizer.save_pretrained(args.output_dir)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = WordLinguaDatasetV2(train_path, tokenizer, label2id, node_type2id, args.max_length)
    dev_dataset = WordLinguaDatasetV2(dev_path, tokenizer, label2id, node_type2id, args.max_length)
    test_dataset = WordLinguaDatasetV2(test_path, tokenizer, label2id, node_type2id, args.max_length)
    
    # Log statistics
    train_cow_count = sum(sum(ex.child_of_whether) for ex in train_dataset.examples)
    logger.info(f"Train set: {len(train_dataset)} examples, "
                f"{train_cow_count} child_of_whether nodes")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=collate_fn_v2, num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_v2, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_v2, num_workers=0
    )
    
    # Create model
    logger.info("Creating model...")
    model = WordLinguaParserV2(
        init_bert_model_name=args.bert_model,
        edge_types=edge_types,
        node_types=node_types,
        arc_hidden_size=args.arc_hidden_size,
        rel_hidden_size=args.rel_hidden_size,
        node_hidden_size=args.node_hidden_size,
        dropout=args.dropout,
        word_pooling=args.word_pooling
    )
    model.to(device)
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Training loop
    best_combined_f1 = 0.0
    best_labeled_f1 = 0.0
    
    for epoch in range(args.num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch_v2(
            model, train_loader, optimizer, device,
            arc_weight=args.arc_weight, rel_weight=args.rel_weight,
            cow_weight=args.cow_weight,
            node_type_weight=args.node_type_weight,
            is_root_weight=args.is_root_weight,
        )
        logger.info(f"Train - Arc Loss: {train_metrics['arc_loss']:.4f}, "
                   f"Rel Loss: {train_metrics['rel_loss']:.4f}, "
                   f"CoW Loss: {train_metrics['child_of_whether_loss']:.4f}, "
                   f"NodeType Loss: {train_metrics['node_type_loss']:.4f}, "
                   f"IsRoot Loss: {train_metrics['is_root_loss']:.4f}")
        
        # Evaluate on train
        train_eval_metrics = evaluate_v2(model, train_loader, device, id2label)
        logger.info(f"Train - Arc F1: {train_eval_metrics['arc_f1']:.4f}, "
                   f"Labeled F1: {train_eval_metrics['labeled_f1']:.4f}, "
                   f"CoW F1: {train_eval_metrics['child_of_whether_f1']:.4f}, "
                   f"NodeType Acc: {train_eval_metrics['node_type_accuracy']:.4f}, "
                   f"IsRoot F1: {train_eval_metrics['is_root_f1']:.4f}")
        
        # Evaluate on dev
        dev_metrics = evaluate_v2(model, dev_loader, device, id2label)
        logger.info(f"Dev   - Arc F1: {dev_metrics['arc_f1']:.4f}, "
                   f"Labeled F1: {dev_metrics['labeled_f1']:.4f}, "
                   f"CoW F1: {dev_metrics['child_of_whether_f1']:.4f}, "
                   f"NodeType Acc: {dev_metrics['node_type_accuracy']:.4f}, "
                   f"IsRoot F1: {dev_metrics['is_root_f1']:.4f}")
        
        # Save best model variants.
        combined_f1 = (dev_metrics['labeled_f1'] + 
                       dev_metrics['child_of_whether_f1'] + 
                       dev_metrics['node_type_accuracy'] +
                       dev_metrics['is_root_f1']) / 4
        
        if combined_f1 > best_combined_f1:
            best_combined_f1 = combined_f1
            save_model_variant(
                output_dir=args.output_dir,
                checkpoint_name="best_combined.pt",
                bundle_name="best_combined",
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_eval_metrics,
                dev_metrics=dev_metrics,
                label2id=label2id,
                edge_types=edge_types,
                node_types=node_types,
                bert_model_name=args.bert_model,
                arc_hidden_size=args.arc_hidden_size,
                rel_hidden_size=args.rel_hidden_size,
                node_hidden_size=args.node_hidden_size,
                dropout=args.dropout,
                word_pooling=args.word_pooling,
                selection_metric="combined_f1",
                selection_score=best_combined_f1,
                export_legacy_root_bundle=True,
                legacy_checkpoint_name="best_model.pt",
            )
            logger.info(f"Saved best-combined model with Combined F1: {best_combined_f1:.4f}")

        if dev_metrics["labeled_f1"] > best_labeled_f1:
            best_labeled_f1 = dev_metrics["labeled_f1"]
            save_model_variant(
                output_dir=args.output_dir,
                checkpoint_name="best_labeled_f1.pt",
                bundle_name="best_labeled_f1",
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_eval_metrics,
                dev_metrics=dev_metrics,
                label2id=label2id,
                edge_types=edge_types,
                node_types=node_types,
                bert_model_name=args.bert_model,
                arc_hidden_size=args.arc_hidden_size,
                rel_hidden_size=args.rel_hidden_size,
                node_hidden_size=args.node_hidden_size,
                dropout=args.dropout,
                word_pooling=args.word_pooling,
                selection_metric="dev_labeled_f1",
                selection_score=best_labeled_f1,
            )
            logger.info(f"Saved best-labeled-F1 model with Dev Labeled F1: {best_labeled_f1:.4f}")

    save_model_variant(
        output_dir=args.output_dir,
        checkpoint_name="last_epoch.pt",
        bundle_name="last_epoch",
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        epoch=args.num_epochs - 1,
        train_metrics=train_eval_metrics,
        dev_metrics=dev_metrics,
        label2id=label2id,
        edge_types=edge_types,
        node_types=node_types,
        bert_model_name=args.bert_model,
        arc_hidden_size=args.arc_hidden_size,
        rel_hidden_size=args.rel_hidden_size,
        node_hidden_size=args.node_hidden_size,
        dropout=args.dropout,
        word_pooling=args.word_pooling,
        selection_metric="last_epoch",
        selection_score=dev_metrics["labeled_f1"],
    )
    logger.info(
        f"Saved last-epoch model from epoch {args.num_epochs} "
        f"with Dev Labeled F1: {dev_metrics['labeled_f1']:.4f}"
    )
    
    # Final evaluation on test
    logger.info("\n" + "="*50)
    logger.info("Final Evaluation on Test Set")
    logger.info("="*50)
    
    # Load best-combined model for backward-compatible final test reporting.
    checkpoint = torch.load(os.path.join(args.output_dir, "best_combined.pt"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    test_metrics = evaluate_v2(model, test_loader, device, id2label)
    
    logger.info(f"Test - Arc F1: {test_metrics['arc_f1']:.4f}")
    logger.info(f"Test - Labeled F1: {test_metrics['labeled_f1']:.4f}")
    logger.info(f"Test - Child of Whether P/R/F1: "
                f"{test_metrics['child_of_whether_precision']:.4f}/"
                f"{test_metrics['child_of_whether_recall']:.4f}/"
                f"{test_metrics['child_of_whether_f1']:.4f}")
    logger.info(f"Test - Node Type Accuracy: {test_metrics['node_type_accuracy']:.4f}")
    logger.info(f"Test - IsRoot P/R/F1: "
                f"{test_metrics['is_root_precision']:.4f}/"
                f"{test_metrics['is_root_recall']:.4f}/"
                f"{test_metrics['is_root_f1']:.4f}")
    
    # Save test results
    with open(os.path.join(args.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    export_pretrained_bundle(
        output_dir=args.output_dir,
        model=model,
        tokenizer=tokenizer,
        label2id=label2id,
        edge_types=edge_types,
        node_types=node_types,
        bert_model_name=args.bert_model,
        arc_hidden_size=args.arc_hidden_size,
        rel_hidden_size=args.rel_hidden_size,
        node_hidden_size=args.node_hidden_size,
        dropout=args.dropout,
        word_pooling=args.word_pooling,
    )
    
    logger.info(f"\nTraining complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Lingua word-graph parser")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, 
                       default=str(DEFAULT_DATA_DIR),
                       help="Directory containing train.jsonl, dev.jsonl, test.jsonl (lingua format)")
    parser.add_argument("--output_dir", type=str,
                       default=str(DEFAULT_OUTPUT_DIR),
                       help="Directory to save model and results")
    
    # Model arguments
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                       help="Pretrained BERT model name")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--arc_hidden_size", type=int, default=500,
                       help="Hidden size for arc prediction MLP")
    parser.add_argument("--rel_hidden_size", type=int, default=100,
                       help="Hidden size for relation prediction MLP")
    parser.add_argument("--node_hidden_size", type=int, default=256,
                       help="Hidden size for node-level classifiers")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--word_pooling", type=str, default="mean",
                       choices=["mean", "first", "last", "max"],
                       help="How to aggregate subword embeddings to word representations")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--arc_weight", type=float, default=1.0,
                       help="Weight for arc loss")
    parser.add_argument("--rel_weight", type=float, default=1.0,
                       help="Weight for relation loss")
    parser.add_argument("--cow_weight", type=float, default=2.0,
                       help="Weight for child_of_whether loss")
    parser.add_argument("--node_type_weight", type=float, default=1.0,
                       help="Weight for node_type loss")
    parser.add_argument("--is_root_weight", type=float, default=1.0,
                       help="Weight for is_root loss")
    
    args = parser.parse_args()
    main(args)
