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
    unit_pos_weight: float = 1.0,
    aux_kind_weight: float = 1.0,
    decision_weight: float = 1.0,
    role_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    unit_pos_loss = F.cross_entropy(output["unit_pos_logits"], batch["unit_pos"])
    aux_kind_loss = F.cross_entropy(output["aux_kind_logits"], batch["aux_kind"])
    decision_loss = F.cross_entropy(output["decision_logits"], batch["decision"])
    role_mask = batch["unit_mask"].unsqueeze(-1)
    if role_mask.any():
        role_loss = F.binary_cross_entropy_with_logits(
            output["role_logits"][role_mask.expand_as(output["role_logits"])],
            batch["role_targets"][role_mask.expand_as(batch["role_targets"])],
        )
    else:
        role_loss = torch.tensor(0.0, device=output["role_logits"].device)

    total_loss = (
        unit_pos_weight * unit_pos_loss
        + aux_kind_weight * aux_kind_loss
        + decision_weight * decision_loss
        + role_weight * role_loss
    )
    return {
        "loss": total_loss,
        "unit_pos_loss": unit_pos_loss,
        "aux_kind_loss": aux_kind_loss,
        "decision_loss": decision_loss,
        "role_loss": role_loss,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    unit_pos_weight: float = 1.0,
    aux_kind_weight: float = 1.0,
    decision_weight: float = 1.0,
    role_weight: float = 1.0,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {
        "loss": 0.0,
        "unit_pos_loss": 0.0,
        "aux_kind_loss": 0.0,
        "decision_loss": 0.0,
        "role_loss": 0.0,
    }
    batch_count = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        output = model(
            batch["word_ids"],
            batch["attention_mask"],
            batch["unit_mask"],
            batch["view_kind"],
            batch["query_role_ids"],
            batch["shared_role_ids"],
        )
        losses = compute_hierarchical_loss(
            output,
            batch,
            unit_pos_weight=unit_pos_weight,
            aux_kind_weight=aux_kind_weight,
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
) -> dict[str, float]:
    model.eval()
    total_units = 0
    unit_pos_correct = 0
    aux_kind_correct = 0
    unit_identity_correct = 0
    decision_correct = 0
    unit_exact = 0

    role_tp = 0
    role_fp = 0
    role_fn = 0
    role_token_exact = 0
    role_token_total = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        output = model(
            batch["word_ids"],
            batch["attention_mask"],
            batch["unit_mask"],
            batch["view_kind"],
            batch["query_role_ids"],
            batch["shared_role_ids"],
        )

        unit_pos_pred = output["unit_pos_logits"].argmax(dim=-1)
        aux_kind_pred = output["aux_kind_logits"].argmax(dim=-1)
        decision_pred = output["decision_logits"].argmax(dim=-1)
        role_scores = torch.sigmoid(output["role_logits"])
        role_pred = role_scores >= threshold
        if force_one_role_per_unit_token and role_scores.size(-1) > 0:
            force_role_per_token = batch.get("force_role_per_token")
            if force_role_per_token is None:
                force_mask = batch["unit_mask"]
            else:
                force_mask = batch["unit_mask"] & force_role_per_token.unsqueeze(1)
            no_role_pred = ~role_pred.any(dim=-1) & force_mask
            best_role = role_scores.argmax(dim=-1)
            role_pred = role_pred.clone()
            role_pred[no_role_pred, best_role[no_role_pred]] = True
        role_gold = batch["role_targets"] > 0.5
        role_mask = batch["unit_mask"].unsqueeze(-1).expand_as(role_gold)
        token_mask = batch["unit_mask"]

        pos_match = unit_pos_pred == batch["unit_pos"]
        aux_match = aux_kind_pred == batch["aux_kind"]
        identity_match = pos_match & aux_match
        decision_match = decision_pred == batch["decision"]
        unit_pos_correct += int(pos_match.sum().item())
        aux_kind_correct += int(aux_match.sum().item())
        unit_identity_correct += int(identity_match.sum().item())
        decision_correct += int(decision_match.sum().item())
        total_units += int(batch["unit_pos"].numel())

        masked_pred = role_pred & role_mask
        masked_gold = role_gold & role_mask
        role_tp += int((masked_pred & masked_gold).sum().item())
        role_fp += int((masked_pred & ~masked_gold & role_mask).sum().item())
        role_fn += int((~masked_pred & masked_gold & role_mask).sum().item())

        per_token_match = ((role_pred == role_gold) | ~role_mask).all(dim=-1)
        role_token_exact += int((per_token_match & token_mask).sum().item())
        role_token_total += int(token_mask.sum().item())

        per_unit_role_match = ((role_pred == role_gold) | ~role_mask).all(dim=(1, 2))
        unit_exact += int((identity_match & decision_match & per_unit_role_match).sum().item())

    role_precision = role_tp / (role_tp + role_fp + 1e-8)
    role_recall = role_tp / (role_tp + role_fn + 1e-8)
    role_f1 = 2 * role_precision * role_recall / (role_precision + role_recall + 1e-8)

    return {
        "unit_pos_accuracy": unit_pos_correct / (total_units + 1e-8),
        "aux_kind_accuracy": aux_kind_correct / (total_units + 1e-8),
        "unit_identity_accuracy": unit_identity_correct / (total_units + 1e-8),
        "decision_accuracy": decision_correct / (total_units + 1e-8),
        "role_precision": role_precision,
        "role_recall": role_recall,
        "role_f1": role_f1,
        "role_token_exact": role_token_exact / (role_token_total + 1e-8),
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


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
