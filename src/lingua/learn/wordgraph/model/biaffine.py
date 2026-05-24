from typing import Dict, Optional
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
# ============================================================================
# Edge Label Constants
# ============================================================================

# ============================================================================
# Node Type Constants
# ============================================================================

# Node types (pos) in word-lingua-graph
NODE_TYPES = [
    "NominalConstant",
    "FactualPredicator", 
    "ModificationalFunctor",
    "PunctuationalConstant",
    "DeterminerConstant",
    "ModificationalConstant",
    "ConjunctionalFunctor",
    "GeneralFunctor",
    "LogicalPredicator",
    "ExpressionFunctor",
    "SymbolConstant",
    "OtherConstant",
    "InterjectionConstant",
    "AttributePredicator",
]

# Build node type mappings
NODE_TYPE2ID = {nt: i for i, nt in enumerate(NODE_TYPES)}
NODE_TYPE2ID["UNK"] = len(NODE_TYPES)
ID2NODE_TYPE = {v: k for k, v in NODE_TYPE2ID.items()}


# ============================================================================
# Bilinear Module
# ============================================================================

class PairwiseBilinear(nn.Module):
    """Pairwise bilinear layer for biaffine attention."""
    
    def __init__(self, in1_features: int, in2_features: int, 
                 out_features: int, bias_x: bool = True, bias_y: bool = True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            torch.zeros(out_features, in1_features + int(bias_x), in2_features + int(bias_y))
        )
        bound = 1 / (self.in1_features * self.in2_features) ** 0.25
        nn.init.uniform_(self.weight, -bound, bound)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.bias_x:
            x1 = torch.cat([x1, torch.ones_like(x1[..., :1])], dim=-1)
        if self.bias_y:
            x2 = torch.cat([x2, torch.ones_like(x2[..., :1])], dim=-1)
        
        return torch.einsum('bxi,oij,byj->bxyo', x1, self.weight, x2)


# ============================================================================
# MLP Module
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron with dropout."""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 dropout: float = 0.1, activation: nn.Module = nn.ReLU):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation(),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ============================================================================
# Biaffine Layer
# ============================================================================

class BiaffineLayer(nn.Module):
    """Biaffine attention layer for arc and relation prediction."""
    
    def __init__(self, input_size: int, label_num: int,
                 arc_hidden_size: int = 500, rel_hidden_size: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        self.label_num = label_num
        
        # MLPs for arc prediction
        self.mlp_arc_h = MLP(input_size, arc_hidden_size, dropout)
        self.mlp_arc_d = MLP(input_size, arc_hidden_size, dropout)
        
        # MLPs for relation prediction
        self.mlp_rel_h = MLP(input_size, rel_hidden_size, dropout)
        self.mlp_rel_d = MLP(input_size, rel_hidden_size, dropout)
        
        # Bilinear layers
        self.arc_atten = PairwiseBilinear(arc_hidden_size, arc_hidden_size, 1)
        self.rel_atten = PairwiseBilinear(rel_hidden_size, rel_hidden_size, label_num)
    
    def forward(self, input: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) - True for valid tokens
        
        Returns:
            dict with arc_logits (batch, seq_len, seq_len) and 
            rel_logits (batch, seq_len, seq_len, label_num)
        """
        # Compute representations for arc prediction
        arc_h = self.mlp_arc_h(input)  # head
        arc_d = self.mlp_arc_d(input)  # dependent
        
        # Compute representations for relation prediction
        rel_h = self.mlp_rel_h(input)
        rel_d = self.mlp_rel_d(input)
        
        # Compute arc scores: (batch, seq_len, seq_len, 1) -> (batch, seq_len, seq_len)
        s_arc = self.arc_atten(arc_d, arc_h).squeeze(-1)
        
        # Compute relation scores: (batch, seq_len, seq_len, label_num)
        s_rel = self.rel_atten(rel_d, rel_h)
        
        # Create pairwise mask if attention_mask is provided
        if attention_mask is not None:
            pairwise_mask = attention_mask.unsqueeze(-1) & attention_mask.unsqueeze(-2)
            s_rel = s_rel.masked_fill(~pairwise_mask.unsqueeze(-1), -1e9)
        else:
            pairwise_mask = None
        
        return {
            "arc_logits": s_arc,
            "rel_logits": s_rel,
            "mask": pairwise_mask
        }


# ============================================================================
# Node Classifier (for child_of_whether)
# ============================================================================

class NodeClassifier(nn.Module):
    """Classifier for node-level binary attributes."""
    
    def __init__(self, input_size: int, hidden_size: int = 256, 
                 dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            logits: (batch, seq_len, num_classes)
        """
        return self.classifier(x)


# ============================================================================
# Word-Lingua Parser Model (v2)
# ============================================================================

class WordLinguaParserV2(nn.Module):
    """
    BERT-based parser for word-lingua-graph prediction.
    
    This version adds:
    - child_of_whether prediction (node-level binary classification)
    - node_type prediction (node-level multi-class classification)
    - is_root prediction (node-level binary classification)
    - Root node is learned through graph structure (arc/relation prediction)
    """
    
    def __init__(self, label_num: int,
                 init_bert_model_name: Optional[str] = None,
                 num_node_types: int = len(NODE_TYPE2ID),
                 arc_hidden_size: int = 500, rel_hidden_size: int = 100,
                 node_hidden_size: int = 256,
                 dropout: float = 0.1, word_pooling: str = "mean",
                 pretrained_model_path: Optional[str] = None):
        """
        Args:
            label_num: Number of relation labels
            init_bert_model_name: Name of BERT model to download from HuggingFace Hub
                                 for model initialization (e.g., 'bert-base-cased').
                                 Used only during training. Ignored if pretrained_model_path is provided.
            num_node_types: Number of node types
            arc_hidden_size: Hidden size for arc prediction
            rel_hidden_size: Hidden size for relation prediction
            node_hidden_size: Hidden size for node classifiers
            dropout: Dropout rate
            word_pooling: Strategy for aggregating subwords ('mean', 'first', 'last', 'max')
            pretrained_model_path: Path to a saved model directory for inference.
                                  If provided, loads BERT config from this directory
                                  and weights are loaded separately via load_state_dict.
        """
        super().__init__()
        
        from transformers import AutoModel, AutoConfig
        import json
        import os
        
        # Load BERT from local pretrained path or download from HuggingFace Hub
        if pretrained_model_path is not None:
            # Load our model's config.json to get the bert_model_name
            model_config_path = os.path.join(pretrained_model_path, "config.json")
            if os.path.exists(model_config_path):
                with open(model_config_path, "r") as f:
                    model_config = json.load(f)
                bert_model_name = model_config.get("bert_model_name")
                if bert_model_name:
                    logger.info(f"Loading BERT from bert_model_name in config: {bert_model_name}")
                    # Check if BERT model files exist locally in the directory
                    bert_model_local = os.path.join(pretrained_model_path, "bert")
                    if os.path.exists(bert_model_local):
                        logger.info(f"Found local BERT model at: {bert_model_local}")
                        config = AutoConfig.from_pretrained(bert_model_local)
                        self.bert = AutoModel.from_config(config)
                    else:
                        # Load from HuggingFace Hub using bert_model_name
                        logger.info(f"Loading BERT config from HuggingFace Hub: {bert_model_name}")
                        config = AutoConfig.from_pretrained(bert_model_name)
                        self.bert = AutoModel.from_config(config)
                else:
                    raise ValueError(f"config.json at {model_config_path} does not contain 'bert_model_name'")
            else:
                # Try loading directly as a HuggingFace model (for compatibility)
                logger.info(f"Loading BERT config from pretrained model: {pretrained_model_path}")
                config = AutoConfig.from_pretrained(pretrained_model_path)
                self.bert = AutoModel.from_config(config)
        elif init_bert_model_name is not None:
            logger.info(f"Initializing BERT from HuggingFace Hub: {init_bert_model_name}")
            self.bert = AutoModel.from_pretrained(init_bert_model_name)
        else:
            raise ValueError("Either init_bert_model_name or pretrained_model_path must be provided")
        
        hidden_size = self.bert.config.hidden_size
        self.word_pooling = word_pooling
        
        # Biaffine layer for arc and relation prediction
        self.biaffine = BiaffineLayer(
            input_size=hidden_size,
            label_num=label_num,
            arc_hidden_size=arc_hidden_size,
            rel_hidden_size=rel_hidden_size,
            dropout=dropout
        )
        
        # Node classifiers
        self.child_of_whether_classifier = NodeClassifier(
            input_size=hidden_size,
            hidden_size=node_hidden_size,
            dropout=dropout,
            num_classes=2
        )
        
        # Node type classifier (multi-class)
        self.node_type_classifier = NodeClassifier(
            input_size=hidden_size,
            hidden_size=node_hidden_size,
            dropout=dropout,
            num_classes=num_node_types
        )

        # Root classifier (binary): predicts whether a node is a topological root (in-degree == 0)
        self.is_root_classifier = NodeClassifier(
            input_size=hidden_size,
            hidden_size=node_hidden_size,
            dropout=dropout,
            num_classes=2
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def aggregate_subwords(self, hidden_states: torch.Tensor, 
                           word_to_subword: torch.Tensor,
                           word_mask: torch.Tensor) -> torch.Tensor:
        """
        Aggregate subword embeddings to word-level representations.
        
        Memory-efficient implementation that avoids creating large intermediate tensors.
        """
        batch_size, max_words, max_subwords = word_to_subword.shape
        hidden_size = hidden_states.size(-1)
        device = hidden_states.device
        
        # Create output tensor
        word_repr = torch.zeros(batch_size, max_words, hidden_size, device=device, dtype=hidden_states.dtype)
        
        # Get valid subword mask
        subword_mask = (word_to_subword >= 0)  # [batch, max_words, max_subwords]
        safe_indices = word_to_subword.clamp(min=0)  # [batch, max_words, max_subwords]
        
        if self.word_pooling == "first":
            # Simply take the first subword for each word
            first_indices = safe_indices[:, :, 0]  # [batch, max_words]
            # Gather from hidden_states: [batch, seq_len, hidden] -> [batch, max_words, hidden]
            gather_indices = first_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
            word_repr = torch.gather(hidden_states, dim=1, index=gather_indices)
            
        elif self.word_pooling == "last":
            # Take the last valid subword for each word
            num_subwords = subword_mask.sum(dim=2).clamp(min=1)  # [batch, max_words]
            last_subword_pos = num_subwords - 1  # [batch, max_words]
            # Get the index from safe_indices at the last valid position
            last_indices = torch.gather(safe_indices, dim=2, 
                                        index=last_subword_pos.unsqueeze(-1)).squeeze(-1)  # [batch, max_words]
            gather_indices = last_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
            word_repr = torch.gather(hidden_states, dim=1, index=gather_indices)
            
        elif self.word_pooling in ["mean", "max"]:
            # For mean/max pooling, we need to gather all subwords
            # But we do it more efficiently by flattening and using scatter/gather
            
            # Flatten indices: [batch, max_words * max_subwords]
            flat_indices = safe_indices.view(batch_size, -1)  # [batch, max_words * max_subwords]
            flat_mask = subword_mask.view(batch_size, -1)  # [batch, max_words * max_subwords]
            
            # Gather all subword embeddings at once
            gather_indices = flat_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
            flat_embeds = torch.gather(hidden_states, dim=1, index=gather_indices)  # [batch, max_words * max_subwords, hidden]
            
            # Reshape back to [batch, max_words, max_subwords, hidden]
            subword_embeds = flat_embeds.view(batch_size, max_words, max_subwords, hidden_size)
            subword_mask_expanded = subword_mask.unsqueeze(-1)  # [batch, max_words, max_subwords, 1]
            
            if self.word_pooling == "mean":
                subword_embeds = subword_embeds * subword_mask_expanded.float()
                word_repr = subword_embeds.sum(dim=2)
                num_subwords = subword_mask.sum(dim=2, keepdim=True).clamp(min=1).float()
                word_repr = word_repr / num_subwords
            else:  # max
                subword_embeds = subword_embeds.masked_fill(~subword_mask_expanded, float('-inf'))
                word_repr = subword_embeds.max(dim=2)[0]
        else:
            raise ValueError(f"Unknown word_pooling: {self.word_pooling}")
        
        return word_repr
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                word_to_subword: torch.Tensor,
                word_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with arc_logits, rel_logits, child_of_whether_logits, node_type_logits
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        hidden_states = self.dropout(hidden_states)
        
        # Aggregate subword embeddings to word-level representations
        word_repr = self.aggregate_subwords(hidden_states, word_to_subword, word_mask)
        
        # Apply biaffine attention for arc/relation prediction
        biaffine_result = self.biaffine(word_repr, word_mask)
        
        # Node-level predictions
        child_of_whether_logits = self.child_of_whether_classifier(word_repr)
        node_type_logits = self.node_type_classifier(word_repr)
        is_root_logits = self.is_root_classifier(word_repr)
        
        return {
            "arc_logits": biaffine_result["arc_logits"],
            "rel_logits": biaffine_result["rel_logits"],
            "mask": biaffine_result["mask"],
            "child_of_whether_logits": child_of_whether_logits,
            "node_type_logits": node_type_logits,
            "is_root_logits": is_root_logits,
        }

