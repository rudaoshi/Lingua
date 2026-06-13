from __future__ import annotations

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    """Linear-chain CRF over the active-unit token order."""

    def __init__(
        self,
        num_tags: int,
        role_label_names: list[str] | None = None,
        constraint_value: float = -10000.0,
    ):
        super().__init__()
        self.num_tags = num_tags
        self.constraint_value = constraint_value
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))

        start_constraints, transition_constraints = _bio_transition_constraints(
            role_label_names,
            num_tags,
            constraint_value,
        )
        self.register_buffer(
            "start_constraints",
            start_constraints,
            persistent=False,
        )
        self.register_buffer(
            "transition_constraints",
            transition_constraints,
            persistent=False,
        )

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        active_unit_mask: torch.Tensor,
        tag_constraints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emissions = self._apply_tag_constraints(emissions, tag_constraints)
        packed_emissions, packed_tags, packed_mask = self._pack_active_sequences(
            emissions,
            tags,
            active_unit_mask,
        )
        if packed_emissions.size(1) == 0:
            return emissions.sum() * 0.0

        log_partition = self._log_partition(packed_emissions, packed_mask)
        gold_score = self._gold_score(packed_emissions, packed_tags, packed_mask)
        token_count = packed_mask.sum().clamp_min(1).to(emissions.dtype)
        return (log_partition - gold_score).sum() / token_count

    @torch.no_grad()
    def decode(
        self,
        emissions: torch.Tensor,
        active_unit_mask: torch.Tensor,
        tag_constraints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        paths, _scores = self.decode_with_score(
            emissions,
            active_unit_mask,
            tag_constraints=tag_constraints,
        )
        return paths

    @torch.no_grad()
    def decode_with_score(
        self,
        emissions: torch.Tensor,
        active_unit_mask: torch.Tensor,
        tag_constraints: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emissions = self._apply_tag_constraints(emissions, tag_constraints)
        packed_emissions, _, packed_mask = self._pack_active_sequences(
            emissions,
            None,
            active_unit_mask,
        )
        dense_paths = torch.zeros(
            emissions.shape[:2],
            dtype=torch.long,
            device=emissions.device,
        )
        path_scores = emissions.new_full((emissions.size(0),), float("-inf"))
        if packed_emissions.size(1) == 0:
            return dense_paths, path_scores

        transition_scores = self.transitions + self.transition_constraints
        start_scores = self.start_transitions + self.start_constraints
        end_scores = self.end_transitions
        lengths = packed_mask.long().sum(dim=1).tolist()

        for row_index, length in enumerate(lengths):
            if length <= 0:
                continue
            sequence = packed_emissions[row_index, :length]
            score = start_scores + sequence[0]
            backpointers = []
            for time_index in range(1, length):
                next_score = (
                    score.unsqueeze(1)
                    + transition_scores
                    + sequence[time_index].unsqueeze(0)
                )
                score, backpointer = next_score.max(dim=0)
                backpointers.append(backpointer)

            score = score + end_scores
            best_score, best_tag_tensor = score.max(dim=0)
            best_tag = int(best_tag_tensor.item())
            path_scores[row_index] = best_score
            best_path = [best_tag]
            for backpointer in reversed(backpointers):
                best_tag = int(backpointer[best_tag].item())
                best_path.append(best_tag)
            best_path.reverse()

            active_token_ids = active_unit_mask[row_index].nonzero(as_tuple=False).flatten()
            dense_paths[row_index, active_token_ids[:length]] = torch.tensor(
                best_path,
                dtype=torch.long,
                device=emissions.device,
            )
        return dense_paths, path_scores

    def _apply_tag_constraints(
        self,
        emissions: torch.Tensor,
        tag_constraints: torch.Tensor | None,
    ) -> torch.Tensor:
        if tag_constraints is None:
            return emissions
        return emissions.masked_fill(
            ~tag_constraints.to(device=emissions.device).unsqueeze(1),
            self.constraint_value,
        )

    def _log_partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        transition_scores = self.transitions + self.transition_constraints
        score = self.start_transitions + self.start_constraints + emissions[:, 0]
        for time_index in range(1, emissions.size(1)):
            next_score = torch.logsumexp(
                score.unsqueeze(2)
                + transition_scores.unsqueeze(0)
                + emissions[:, time_index].unsqueeze(1),
                dim=1,
            )
            score = torch.where(mask[:, time_index].unsqueeze(1), next_score, score)
        return torch.logsumexp(score + self.end_transitions, dim=1)

    def _gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        transition_scores = self.transitions + self.transition_constraints
        batch_indices = torch.arange(emissions.size(0), device=emissions.device)
        score = (
            self.start_transitions[tags[:, 0]]
            + self.start_constraints[tags[:, 0]]
            + emissions[batch_indices, 0, tags[:, 0]]
        )
        for time_index in range(1, emissions.size(1)):
            current_mask = mask[:, time_index]
            previous_tags = tags[:, time_index - 1]
            current_tags = tags[:, time_index]
            transition_score = transition_scores[previous_tags, current_tags]
            emission_score = emissions[batch_indices, time_index, current_tags]
            score = score + (transition_score + emission_score) * current_mask.to(emissions.dtype)

        lengths = mask.long().sum(dim=1).clamp_min(1)
        last_tags = tags[batch_indices, lengths - 1]
        score = score + self.end_transitions[last_tags]
        return score

    def _pack_active_sequences(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor | None,
        active_unit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = active_unit_mask.long().sum(dim=1)
        max_length = int(lengths.max().item()) if lengths.numel() else 0
        packed_emissions = emissions.new_zeros(
            emissions.size(0),
            max_length,
            emissions.size(-1),
        )
        packed_tags = torch.zeros(
            emissions.size(0),
            max_length,
            dtype=torch.long,
            device=emissions.device,
        )
        packed_mask = torch.zeros(
            emissions.size(0),
            max_length,
            dtype=torch.bool,
            device=emissions.device,
        )

        for row_index in range(emissions.size(0)):
            token_ids = active_unit_mask[row_index].nonzero(as_tuple=False).flatten()
            length = int(token_ids.numel())
            if length == 0:
                continue
            packed_emissions[row_index, :length] = emissions[row_index, token_ids]
            if tags is not None:
                packed_tags[row_index, :length] = tags[row_index, token_ids]
            packed_mask[row_index, :length] = True
        return packed_emissions, packed_tags, packed_mask


class HierarchicalTokenSetParser(nn.Module):
    """Recursive token-set Lingua parser with GRU or BERT word encoder."""

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
        num_upos: int = 2,
        use_crf: bool = True,
        role_label_names: list[str] | None = None,
        unit_pos_names: list[str] | None = None,
        aux_kind_names: list[str] | None = None,
        decision_names: list[str] | None = None,
        encoder_type: str = "gru",
        bert_model_name: str | None = None,
        freeze_bert: bool = False,
        bert_local_files_only: bool = False,
        bert_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if encoder_type not in {"gru", "bert"}:
            raise ValueError(f"unsupported hierarchical encoder_type: {encoder_type}")
        self.encoder_type = encoder_type

        if encoder_type == "bert":
            if not bert_model_name:
                raise ValueError("bert_model_name is required when encoder_type='bert'")
            from transformers import AutoModel

            self.bert = AutoModel.from_pretrained(
                bert_model_name,
                local_files_only=bert_local_files_only,
            )
            if bert_gradient_checkpointing and hasattr(self.bert, "gradient_checkpointing_enable"):
                self.bert.gradient_checkpointing_enable()
            if freeze_bert:
                for parameter in self.bert.parameters():
                    parameter.requires_grad = False
            encoded_size = int(self.bert.config.hidden_size)
            self.word_embedding = None
            self.upos_embedding = nn.Embedding(num_upos, encoded_size, padding_idx=0)
            self.encoder = None
            self.unit_membership_embedding = nn.Embedding(2, encoded_size)
        else:
            self.bert = None
            self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
            self.upos_embedding = nn.Embedding(num_upos, embedding_dim, padding_idx=0)
            self.unit_membership_embedding = nn.Embedding(2, embedding_dim)
            self.encoder = nn.GRU(
                input_size=embedding_dim * 3,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=True,
            )
            encoded_size = hidden_size * 2

        self.encoded_size = encoded_size
        self.dropout = nn.Dropout(dropout)
        self.view_kind_embedding = nn.Embedding(num_view_kinds, encoded_size)
        nn.init.zeros_(self.view_kind_embedding.weight)
        nn.init.zeros_(self.upos_embedding.weight)
        self.current_unit_pos_embedding = nn.Embedding(
            num_unit_pos,
            encoded_size,
            padding_idx=0,
        )
        nn.init.zeros_(self.current_unit_pos_embedding.weight)
        self.current_aux_kind_embedding = nn.Embedding(
            num_aux_kinds,
            encoded_size,
            padding_idx=0,
        )
        nn.init.zeros_(self.current_aux_kind_embedding.weight)
        self.head_pos_classifier = nn.Linear(encoded_size, num_unit_pos)
        self.head_aux_kind_classifier = nn.Linear(encoded_size, num_aux_kinds)
        self.decision_classifier = nn.Linear(encoded_size, num_decisions)
        self.role_classifier = nn.Linear(encoded_size, num_role_labels)
        self.role_crf = (
            LinearChainCRF(num_role_labels, role_label_names=role_label_names)
            if use_crf
            else None
        )
        role_constraints = _build_role_tag_constraints(
            unit_pos_names=unit_pos_names,
            aux_kind_names=aux_kind_names,
            decision_names=decision_names,
            role_label_names=role_label_names,
            num_unit_pos=num_unit_pos,
            num_aux_kinds=num_aux_kinds,
            num_decisions=num_decisions,
            num_role_labels=num_role_labels,
        )
        self.register_buffer(
            "role_tag_constraints",
            role_constraints,
            persistent=False,
        )

    def forward(
        self,
        word_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        active_unit_mask: torch.Tensor,
        view_kind: torch.Tensor | None = None,
        active_unit_pos: torch.Tensor | None = None,
        role_target_ids: torch.Tensor | None = None,
        bert_input_ids: torch.Tensor | None = None,
        bert_attention_mask: torch.Tensor | None = None,
        bert_token_type_ids: torch.Tensor | None = None,
        bert_word_starts: torch.Tensor | None = None,
        active_aux_kind: torch.Tensor | None = None,
        active_decision: torch.Tensor | None = None,
        upos_ids: torch.Tensor | None = None,
        bert_upos_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if view_kind is None:
            view_kind = torch.zeros(
                word_ids.size(0),
                dtype=torch.long,
                device=word_ids.device,
            )
        if upos_ids is None:
            upos_ids = torch.zeros_like(word_ids)

        if self.encoder_type == "bert":
            encoded = self._encode_bert_words(
                word_ids=word_ids,
                active_unit_mask=active_unit_mask,
                view_kind=view_kind,
                bert_upos_ids=bert_upos_ids,
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask,
                bert_token_type_ids=bert_token_type_ids,
                bert_word_starts=bert_word_starts,
            )
            conditioned_encoded = self.dropout(encoded)
        else:
            encoded = self._encode_gru_words(word_ids, active_unit_mask, upos_ids)
            encoded = self.dropout(encoded)
            view_repr = self.view_kind_embedding(view_kind)
            conditioned_encoded = encoded + view_repr.unsqueeze(1)

        unit_mask_float = active_unit_mask.float().unsqueeze(-1)
        unit_lengths = unit_mask_float.sum(dim=1).clamp_min(1.0)
        unit_repr = (conditioned_encoded * unit_mask_float).sum(dim=1) / unit_lengths
        unit_repr = self.dropout(unit_repr)

        head_pos_logits = self.head_pos_classifier(unit_repr)
        head_aux_kind_logits = self.head_aux_kind_classifier(unit_repr)
        decision_logits = self.decision_classifier(unit_repr)

        if active_unit_pos is None:
            active_unit_pos_condition = torch.matmul(
                torch.softmax(head_pos_logits, dim=-1),
                self.current_unit_pos_embedding.weight,
            )
        else:
            active_unit_pos_condition = self.current_unit_pos_embedding(active_unit_pos)
        if active_aux_kind is None:
            active_aux_kind_condition = torch.matmul(
                torch.softmax(head_aux_kind_logits, dim=-1),
                self.current_aux_kind_embedding.weight,
            )
        else:
            active_aux_kind_condition = self.current_aux_kind_embedding(active_aux_kind)
        role_encoded = (
            conditioned_encoded
            + active_unit_pos_condition.unsqueeze(1)
            + active_aux_kind_condition.unsqueeze(1)
        )
        role_logits = self.role_classifier(role_encoded).masked_fill(
            ~attention_mask.unsqueeze(-1),
            0.0,
        )
        output = {
            "head_pos_logits": head_pos_logits,
            "head_aux_kind_logits": head_aux_kind_logits,
            "decision_logits": decision_logits,
            "role_logits": role_logits,
        }
        if self.role_crf is not None and role_target_ids is not None:
            tag_constraints = None
            if (
                active_unit_pos is not None
                and active_aux_kind is not None
                and active_decision is not None
            ):
                tag_constraints = self.role_tag_constraints[
                    active_unit_pos,
                    active_aux_kind,
                    active_decision,
                ]
            output["role_crf_nll"] = self.role_crf.neg_log_likelihood(
                role_logits,
                role_target_ids,
                active_unit_mask,
                tag_constraints=tag_constraints,
            )
        return output

    def _encode_gru_words(
        self,
        word_ids: torch.Tensor,
        active_unit_mask: torch.Tensor,
        upos_ids: torch.Tensor,
    ) -> torch.Tensor:
        membership_ids = active_unit_mask.long()
        embeddings = torch.cat(
            [
                self.word_embedding(word_ids),
                self.upos_embedding(upos_ids),
                self.unit_membership_embedding(membership_ids),
            ],
            dim=-1,
        )
        encoded, _ = self.encoder(embeddings)
        return encoded

    def _encode_bert_words(
        self,
        *,
        word_ids: torch.Tensor,
        active_unit_mask: torch.Tensor,
        view_kind: torch.Tensor,
        bert_upos_ids: torch.Tensor | None,
        bert_input_ids: torch.Tensor | None,
        bert_attention_mask: torch.Tensor | None,
        bert_token_type_ids: torch.Tensor | None,
        bert_word_starts: torch.Tensor | None,
    ) -> torch.Tensor:
        if bert_input_ids is None or bert_attention_mask is None or bert_word_starts is None:
            raise ValueError("BERT encoder requires bert_input_ids, bert_attention_mask, and bert_word_starts")

        inputs_embeds = self.bert.get_input_embeddings()(bert_input_ids)
        view_repr = self.view_kind_embedding(view_kind).unsqueeze(1)
        if bert_upos_ids is None:
            bert_upos_ids = torch.zeros_like(bert_input_ids)
        upos_repr = self.upos_embedding(bert_upos_ids)
        inputs_embeds = (
            inputs_embeds
            + view_repr.to(dtype=inputs_embeds.dtype)
            + upos_repr.to(dtype=inputs_embeds.dtype)
        )

        bert_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": bert_attention_mask,
        }
        if bert_token_type_ids is not None:
            bert_kwargs["token_type_ids"] = bert_token_type_ids
        sequence_output = self.bert(**bert_kwargs).last_hidden_state

        starts = bert_word_starts.clamp_min(0)
        gather_index = starts.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))
        word_encoded = torch.gather(sequence_output, dim=1, index=gather_index)
        word_valid = bert_word_starts.ge(0)
        word_encoded = word_encoded.masked_fill(~word_valid.unsqueeze(-1), 0.0)
        return word_encoded + self.unit_membership_embedding(active_unit_mask.long())

    def decode_role_labels(
        self,
        role_logits: torch.Tensor,
        active_unit_mask: torch.Tensor,
        tag_constraints: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.role_crf is None:
            if tag_constraints is not None:
                role_logits = role_logits.masked_fill(
                    ~tag_constraints.to(device=role_logits.device).unsqueeze(1),
                    -10000.0,
                )
            return role_logits.argmax(dim=-1)
        return self.role_crf.decode(
            role_logits,
            active_unit_mask,
            tag_constraints=tag_constraints,
        )

    @torch.no_grad()
    def decode_structured(
        self,
        output: dict[str, torch.Tensor],
        active_unit_mask: torch.Tensor,
        strategy: str = "independent",
        top_k_unit_pos: int = 4,
        top_k_aux_kind: int = 4,
        role_score_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if strategy == "independent":
            head_pos = output["head_pos_logits"].argmax(dim=-1)
            head_aux_kind = output["head_aux_kind_logits"].argmax(dim=-1)
            decision = output["decision_logits"].argmax(dim=-1)
            constraints = self.role_tag_constraints[head_pos, head_aux_kind, decision]
            roles = self.decode_role_labels(
                output["role_logits"],
                active_unit_mask,
                tag_constraints=constraints,
            )
            return {
                "head_pos": head_pos,
                "head_aux_kind": head_aux_kind,
                "decision": decision,
                "role_labels": roles,
            }
        if strategy == "exhaustive":
            return self._decode_structured_exhaustive(
                output,
                active_unit_mask,
                top_k_unit_pos=top_k_unit_pos,
                top_k_aux_kind=top_k_aux_kind,
                role_score_weight=role_score_weight,
            )
        if strategy != "two_pass":
            raise ValueError(f"unsupported structured decode strategy: {strategy}")
        return self._decode_structured_two_pass(output, active_unit_mask)

    @torch.no_grad()
    def _decode_structured_two_pass(
        self,
        output: dict[str, torch.Tensor],
        active_unit_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        role_logits = output["role_logits"]
        initial_roles = self.decode_role_labels(role_logits, active_unit_mask)
        head_log_probs = torch.log_softmax(output["head_pos_logits"], dim=-1)
        aux_log_probs = torch.log_softmax(output["head_aux_kind_logits"], dim=-1)
        decision_log_probs = torch.log_softmax(output["decision_logits"], dim=-1)

        batch_size = role_logits.size(0)
        head_pos = output["head_pos_logits"].argmax(dim=-1).clone()
        head_aux_kind = output["head_aux_kind_logits"].argmax(dim=-1).clone()
        decision = output["decision_logits"].argmax(dim=-1).clone()

        flat_constraints = self.role_tag_constraints.view(
            -1,
            self.role_tag_constraints.size(-1),
        )
        num_aux = self.role_tag_constraints.size(1)
        num_decisions = self.role_tag_constraints.size(2)

        for row_index in range(batch_size):
            active_role_ids = initial_roles[row_index, active_unit_mask[row_index]]
            if active_role_ids.numel() == 0:
                continue
            candidate_valid = flat_constraints[:, active_role_ids].all(dim=1)
            if not bool(candidate_valid.any().item()):
                continue

            scores = (
                head_log_probs[row_index].view(-1, 1, 1)
                + aux_log_probs[row_index].view(1, -1, 1)
                + decision_log_probs[row_index].view(1, 1, -1)
            ).reshape(-1)
            scores = scores.masked_fill(~candidate_valid.to(scores.device), float("-inf"))
            best_flat = int(scores.argmax().item())
            head_pos[row_index] = best_flat // (num_aux * num_decisions)
            remainder = best_flat % (num_aux * num_decisions)
            head_aux_kind[row_index] = remainder // num_decisions
            decision[row_index] = remainder % num_decisions

        constraints = self.role_tag_constraints[head_pos, head_aux_kind, decision]
        roles = self.decode_role_labels(
            role_logits,
            active_unit_mask,
            tag_constraints=constraints,
        )
        return {
            "head_pos": head_pos,
            "head_aux_kind": head_aux_kind,
            "decision": decision,
            "role_labels": roles,
        }

    @torch.no_grad()
    def _decode_structured_exhaustive(
        self,
        output: dict[str, torch.Tensor],
        active_unit_mask: torch.Tensor,
        top_k_unit_pos: int,
        top_k_aux_kind: int,
        role_score_weight: float,
    ) -> dict[str, torch.Tensor]:
        role_logits = output["role_logits"]
        head_log_probs = torch.log_softmax(output["head_pos_logits"], dim=-1)
        aux_log_probs = torch.log_softmax(output["head_aux_kind_logits"], dim=-1)
        decision_log_probs = torch.log_softmax(output["decision_logits"], dim=-1)
        batch_size = role_logits.size(0)
        device = role_logits.device

        head_pos = torch.zeros(batch_size, dtype=torch.long, device=device)
        head_aux_kind = torch.zeros(batch_size, dtype=torch.long, device=device)
        decision = torch.zeros(batch_size, dtype=torch.long, device=device)
        roles = torch.zeros(role_logits.shape[:2], dtype=torch.long, device=device)

        pos_k = min(top_k_unit_pos, head_log_probs.size(1))
        aux_k = min(top_k_aux_kind, aux_log_probs.size(1))
        decision_ids = torch.arange(decision_log_probs.size(1), device=device)
        top_pos = head_log_probs.topk(pos_k, dim=-1).indices
        top_aux = aux_log_probs.topk(aux_k, dim=-1).indices

        for row_index in range(batch_size):
            best_score = None
            best_path = None
            best_candidate = None
            active_len = int(active_unit_mask[row_index].sum().item())
            score_denominator = max(active_len, 1)
            for pos_id in top_pos[row_index].tolist():
                for aux_id in top_aux[row_index].tolist():
                    for decision_id in decision_ids.tolist():
                        constraints = self.role_tag_constraints[
                            pos_id,
                            aux_id,
                            decision_id,
                        ].unsqueeze(0)
                        if self.role_crf is None:
                            candidate_logits = role_logits[row_index : row_index + 1]
                            candidate_logits = candidate_logits.masked_fill(
                                ~constraints.to(device=device).unsqueeze(1),
                                -10000.0,
                            )
                            path = candidate_logits.argmax(dim=-1)
                            role_score = candidate_logits[
                                0,
                                active_unit_mask[row_index],
                                path[0, active_unit_mask[row_index]],
                            ].sum()
                        else:
                            path, role_scores = self.role_crf.decode_with_score(
                                role_logits[row_index : row_index + 1],
                                active_unit_mask[row_index : row_index + 1],
                                tag_constraints=constraints,
                            )
                            role_score = role_scores[0]
                        total_score = (
                            head_log_probs[row_index, pos_id]
                            + aux_log_probs[row_index, aux_id]
                            + decision_log_probs[row_index, decision_id]
                            + role_score_weight * (role_score / score_denominator)
                        )
                        if best_score is None or total_score > best_score:
                            best_score = total_score
                            best_path = path[0]
                            best_candidate = (pos_id, aux_id, int(decision_id))
            if best_candidate is None or best_path is None:
                best_candidate = (
                    int(output["head_pos_logits"][row_index].argmax().item()),
                    int(output["head_aux_kind_logits"][row_index].argmax().item()),
                    int(output["decision_logits"][row_index].argmax().item()),
                )
                best_path = self.decode_role_labels(
                    role_logits[row_index : row_index + 1],
                    active_unit_mask[row_index : row_index + 1],
                )[0]
            head_pos[row_index], head_aux_kind[row_index], decision[row_index] = best_candidate
            roles[row_index] = best_path
        return {
            "head_pos": head_pos,
            "head_aux_kind": head_aux_kind,
            "decision": decision,
            "role_labels": roles,
        }


def _bio_transition_constraints(
    role_label_names: list[str] | None,
    num_tags: int,
    constraint_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    start_constraints = torch.zeros(num_tags)
    transition_constraints = torch.zeros(num_tags, num_tags)
    if role_label_names is None or len(role_label_names) != num_tags:
        return start_constraints, transition_constraints

    parsed_labels = [_parse_bio_label(label) for label in role_label_names]
    for tag_index, (prefix, _role_set) in enumerate(parsed_labels):
        if prefix == "I":
            start_constraints[tag_index] = constraint_value

    for previous_index, previous_label in enumerate(parsed_labels):
        for current_index, current_label in enumerate(parsed_labels):
            if not _is_allowed_bio_transition(previous_label, current_label):
                transition_constraints[previous_index, current_index] = constraint_value
    return start_constraints, transition_constraints


def _parse_bio_label(label: str) -> tuple[str, str]:
    if label == "O":
        return "O", ""
    if "-" not in label:
        return "B", label
    prefix, role_set = label.split("-", 1)
    return prefix, role_set


def _is_allowed_bio_transition(
    previous_label: tuple[str, str],
    current_label: tuple[str, str],
) -> bool:
    current_prefix, current_role_set = current_label
    if current_prefix != "I":
        return True
    previous_prefix, previous_role_set = previous_label
    return previous_prefix in {"B", "I"} and previous_role_set == current_role_set


def _build_role_tag_constraints(
    *,
    unit_pos_names: list[str] | None,
    aux_kind_names: list[str] | None,
    decision_names: list[str] | None,
    role_label_names: list[str] | None,
    num_unit_pos: int,
    num_aux_kinds: int,
    num_decisions: int,
    num_role_labels: int,
) -> torch.Tensor:
    constraints = torch.ones(
        num_unit_pos,
        num_aux_kinds,
        num_decisions,
        num_role_labels,
        dtype=torch.bool,
    )
    if (
        unit_pos_names is None
        or aux_kind_names is None
        or decision_names is None
        or role_label_names is None
        or len(unit_pos_names) != num_unit_pos
        or len(aux_kind_names) != num_aux_kinds
        or len(decision_names) != num_decisions
        or len(role_label_names) != num_role_labels
    ):
        return constraints

    for pos_index, unit_pos in enumerate(unit_pos_names):
        for aux_index, aux_kind in enumerate(aux_kind_names):
            for decision_index, decision in enumerate(decision_names):
                for role_index, role_label in enumerate(role_label_names):
                    constraints[pos_index, aux_index, decision_index, role_index] = (
                        _is_allowed_role_for_unit(
                            role_label,
                            unit_pos=unit_pos,
                            aux_kind=aux_kind,
                            decision=decision,
                        )
                    )
    return constraints


def _is_allowed_role_for_unit(
    role_label: str,
    *,
    unit_pos: str,
    aux_kind: str,
    decision: str,
) -> bool:
    families = _role_families(role_label)
    if not families:
        return True
    if decision == "leaf":
        return all(family == "HEAD" for family in families)

    role_profile = _allowed_role_profile(unit_pos=unit_pos, aux_kind=aux_kind)
    if role_profile is None:
        return True
    allowed_families, allow_arg_families = role_profile
    for family in families:
        if family in allowed_families:
            continue
        if allow_arg_families and family.startswith("ARG"):
            continue
        if family not in _KNOWN_ROLE_FAMILIES and not family.startswith("ARG"):
            continue
        return False
    return True


def _role_families(role_label: str) -> set[str]:
    if role_label == "O":
        return set()
    _prefix, role_set = _parse_bio_label(role_label)
    if not role_set:
        return set()
    return set(role_set.split("+"))


def _allowed_role_profile(
    *,
    unit_pos: str,
    aux_kind: str,
) -> tuple[set[str], bool] | None:
    if unit_pos == "<unk>" or aux_kind == "<unk>":
        return {"HEAD"}, False
    if aux_kind == "Modification":
        return {"BODY", "VARIABLE"}, False
    if aux_kind in {"Expression", "List", "Parataxis", "Whether"}:
        return {"FUNC_ARG"}, False
    if aux_kind == "Ref":
        return {"FUNC_ARG", "HEAD"}, True
    if aux_kind in {"Punctuation", "Apposition", "Attribute", "Copula"}:
        return {"HEAD", "FUNC_ARG"}, True
    if aux_kind in {"Discourse", "Vocative"}:
        return {"HEAD", "BODY", "VARIABLE", "FUNC_ARG"}, False
    if aux_kind != "<non_aux>":
        return None

    if unit_pos in {
        "FactualPredicator",
        "AttributePredicator",
        "AppositionalPredicator",
        "ReferentialPredicator",
    }:
        return {"HEAD"}, True
    if unit_pos == "LogicalPredicator":
        return {"HEAD", "BODY", "VARIABLE"}, True
    if unit_pos == "ModificationalFunctor":
        return {"HEAD", "BODY", "VARIABLE"}, False
    if unit_pos in {
        "ConjunctionalFunctor",
        "ExpressionFunctor",
        "ListFunctor",
    }:
        return {"HEAD", "FUNC_ARG"}, True
    if unit_pos == "GeneralFunctor":
        return {"HEAD", "BODY", "VARIABLE", "FUNC_ARG"}, True
    return None


_KNOWN_ROLE_FAMILIES = {
    "HEAD",
    "ARG1",
    "ARG2",
    "ARG3",
    "ARG4",
    "VARIABLE",
    "BODY",
    "FUNC_ARG",
}
