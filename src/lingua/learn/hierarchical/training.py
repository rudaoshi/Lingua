from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def compute_hierarchical_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    head_pos_weight: float = 1.0,
    head_aux_kind_weight: float = 1.0,
    decision_weight: float = 1.0,
    role_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    head_pos_loss = F.cross_entropy(output["head_pos_logits"], batch["head_pos"])
    head_aux_kind_loss = F.cross_entropy(
        output["head_aux_kind_logits"],
        batch["head_aux_kind"],
    )
    decision_loss = F.cross_entropy(output["decision_logits"], batch["decision"])
    role_mask = batch["active_unit_mask"]
    if "role_crf_nll" in output:
        role_loss = output["role_crf_nll"]
    elif role_mask.any():
        role_loss = F.cross_entropy(
            output["role_logits"][role_mask],
            batch["role_target_ids"][role_mask],
        )
    else:
        role_loss = torch.tensor(0.0, device=output["role_logits"].device)

    total_loss = (
        head_pos_weight * head_pos_loss
        + head_aux_kind_weight * head_aux_kind_loss
        + decision_weight * decision_loss
        + role_weight * role_loss
    )
    return {
        "loss": total_loss,
        "head_pos_loss": head_pos_loss,
        "head_aux_kind_loss": head_aux_kind_loss,
        "decision_loss": decision_loss,
        "role_loss": role_loss,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    head_pos_weight: float = 1.0,
    head_aux_kind_weight: float = 1.0,
    decision_weight: float = 1.0,
    role_weight: float = 1.0,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {
        "loss": 0.0,
        "head_pos_loss": 0.0,
        "head_aux_kind_loss": 0.0,
        "decision_loss": 0.0,
        "role_loss": 0.0,
    }
    batch_count = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        output = forward_hierarchical_model(
            model,
            batch,
            include_role_targets=True,
        )
        losses = compute_hierarchical_loss(
            output,
            batch,
            head_pos_weight=head_pos_weight,
            head_aux_kind_weight=head_aux_kind_weight,
            decision_weight=decision_weight,
            role_weight=role_weight,
        )
        losses["loss"].backward()
        optimizer.step()

        for key in totals:
            totals[key] += float(losses[key].item())
        batch_count += 1

    return {key: value / max(batch_count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_hierarchical_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    force_one_role_per_unit_token: bool = True,
    decode_strategy: str = "two_pass",
) -> dict[str, float]:
    model.eval()
    total_units = 0
    head_pos_correct = 0
    head_aux_kind_correct = 0
    head_identity_correct = 0
    decision_correct = 0
    unit_exact = 0

    role_token_exact = 0
    role_token_total = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        output = forward_hierarchical_model(model, batch)

        if hasattr(model, "decode_structured"):
            decoded = model.decode_structured(
                output,
                batch["active_unit_mask"],
                strategy=decode_strategy,
            )
            head_pos_pred = decoded["head_pos"]
            head_aux_kind_pred = decoded["head_aux_kind"]
            decision_pred = decoded["decision"]
            role_pred = decoded["role_labels"]
        else:
            head_pos_pred = output["head_pos_logits"].argmax(dim=-1)
            head_aux_kind_pred = output["head_aux_kind_logits"].argmax(dim=-1)
            decision_pred = output["decision_logits"].argmax(dim=-1)
            role_pred = model.decode_role_labels(
                output["role_logits"],
                batch["active_unit_mask"],
            )
        role_gold = batch["role_target_ids"]
        token_mask = batch["active_unit_mask"]

        pos_match = head_pos_pred == batch["head_pos"]
        aux_match = head_aux_kind_pred == batch["head_aux_kind"]
        identity_match = pos_match & aux_match
        decision_match = decision_pred == batch["decision"]
        head_pos_correct += int(pos_match.sum().item())
        head_aux_kind_correct += int(aux_match.sum().item())
        head_identity_correct += int(identity_match.sum().item())
        decision_correct += int(decision_match.sum().item())
        total_units += int(batch["head_pos"].numel())

        per_token_match = role_pred == role_gold
        role_token_exact += int((per_token_match & token_mask).sum().item())
        role_token_total += int(token_mask.sum().item())

        per_unit_role_match = (per_token_match | ~token_mask).all(dim=1)
        unit_exact += int((identity_match & decision_match & per_unit_role_match).sum().item())

    role_accuracy = role_token_exact / (role_token_total + 1e-8)

    return {
        "head_pos_accuracy": head_pos_correct / (total_units + 1e-8),
        "head_aux_kind_accuracy": head_aux_kind_correct / (total_units + 1e-8),
        "head_identity_accuracy": head_identity_correct / (total_units + 1e-8),
        "decision_accuracy": decision_correct / (total_units + 1e-8),
        "role_precision": role_accuracy,
        "role_recall": role_accuracy,
        "role_f1": role_accuracy,
        "role_token_exact": role_accuracy,
        "unit_exact": unit_exact / (total_units + 1e-8),
        "num_units": float(total_units),
    }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def forward_hierarchical_model(
    model: nn.Module,
    batch: dict[str, Any],
    include_role_targets: bool = False,
    condition_on_gold_types: bool = False,
) -> dict[str, torch.Tensor]:
    kwargs = {}
    for key in (
        "bert_input_ids",
        "bert_attention_mask",
        "bert_token_type_ids",
        "bert_word_starts",
        "upos_ids",
        "bert_upos_ids",
    ):
        if key in batch:
            kwargs[key] = batch[key]
    if include_role_targets:
        kwargs["role_target_ids"] = batch["role_target_ids"]
    if include_role_targets or condition_on_gold_types:
        kwargs["active_aux_kind"] = batch["head_aux_kind"]
        kwargs["active_decision"] = batch["decision"]
        active_unit_pos = batch["active_unit_pos"]
    else:
        active_unit_pos = None
    return model(
        batch["word_ids"],
        batch["attention_mask"],
        batch["active_unit_mask"],
        batch["view_kind"],
        active_unit_pos,
        **kwargs,
    )


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
