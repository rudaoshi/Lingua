from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalTokenSetParser(nn.Module):
    """Lightweight baseline for recursive token-set Lingua supervision."""

    def __init__(
        self,
        vocab_size: int,
        num_unit_pos: int,
        num_aux_kinds: int,
        num_role_labels: int,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        dropout: float = 0.2,
        pad_id: int = 0,
        num_decisions: int = 2,
        num_view_kinds: int = 1,
        num_query_roles: int = 1,
        num_shared_roles: int = 1,
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.unit_membership_embedding = nn.Embedding(2, embedding_dim)
        self.query_role_embedding = nn.Embedding(
            num_query_roles,
            embedding_dim,
            padding_idx=0,
        )
        self.shared_role_embedding = nn.Embedding(
            num_shared_roles,
            embedding_dim,
            padding_idx=0,
        )
        self.encoder = nn.GRU(
            input_size=embedding_dim * 4,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        encoded_size = hidden_size * 2
        self.dropout = nn.Dropout(dropout)
        self.view_kind_embedding = nn.Embedding(num_view_kinds, encoded_size)
        nn.init.zeros_(self.view_kind_embedding.weight)
        self.unit_pos_classifier = nn.Linear(encoded_size, num_unit_pos)
        self.aux_kind_classifier = nn.Linear(encoded_size, num_aux_kinds)
        self.decision_classifier = nn.Linear(encoded_size, num_decisions)
        self.role_classifier = nn.Linear(encoded_size, num_role_labels)

    def forward(
        self,
        word_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_mask: torch.Tensor,
        view_kind: torch.Tensor | None = None,
        query_role_ids: torch.Tensor | None = None,
        shared_role_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        membership_ids = unit_mask.long()
        if query_role_ids is None:
            query_role_ids = torch.zeros_like(word_ids)
        if shared_role_ids is None:
            shared_role_ids = torch.zeros_like(word_ids)
        embeddings = torch.cat(
            [
                self.word_embedding(word_ids),
                self.unit_membership_embedding(membership_ids),
                self.query_role_embedding(query_role_ids),
                self.shared_role_embedding(shared_role_ids),
            ],
            dim=-1,
        )
        encoded, _ = self.encoder(embeddings)
        encoded = self.dropout(encoded)

        if view_kind is None:
            view_kind = torch.zeros(
                word_ids.size(0),
                dtype=torch.long,
                device=word_ids.device,
            )
        view_repr = self.view_kind_embedding(view_kind)
        conditioned_encoded = encoded + view_repr.unsqueeze(1)

        unit_mask_float = unit_mask.float().unsqueeze(-1)
        unit_lengths = unit_mask_float.sum(dim=1).clamp_min(1.0)
        unit_repr = (conditioned_encoded * unit_mask_float).sum(dim=1) / unit_lengths
        unit_repr = self.dropout(unit_repr)

        return {
            "unit_pos_logits": self.unit_pos_classifier(unit_repr),
            "aux_kind_logits": self.aux_kind_classifier(unit_repr),
            "decision_logits": self.decision_classifier(unit_repr),
            "role_logits": self.role_classifier(conditioned_encoded).masked_fill(
                ~attention_mask.unsqueeze(-1),
                0.0,
            ),
        }
