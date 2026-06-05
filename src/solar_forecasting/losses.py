"""Loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    sentinel_value: float,
) -> torch.Tensor:
    mask = targets != sentinel_value
    if not mask.any():
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    predictions_masked = torch.masked_select(predictions, mask)
    targets_masked = torch.masked_select(targets, mask)
    return F.mse_loss(predictions_masked, targets_masked)


def masked_rmse_scaled(pred: torch.Tensor, target: torch.Tensor, sentinel_value: float) -> torch.Tensor:
    valid_positions = target.squeeze(-1) != sentinel_value
    if not valid_positions.any():
        return torch.tensor(0.0, device=pred.device)
    diff = pred - target
    error = torch.masked_select(diff, valid_positions.unsqueeze(-1))
    return torch.sqrt(torch.mean(error**2))
