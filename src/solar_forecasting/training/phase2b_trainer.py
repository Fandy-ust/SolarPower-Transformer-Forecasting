"""Phase 2B scheduled sampling + continuity-aware fine-tuning (refactored from original script)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from solar_forecasting.constants import TARGET_FEATURE_INDEX
from solar_forecasting.losses import masked_rmse_scaled


def find_continuous_segments(mask: torch.Tensor, min_length: int = 3) -> torch.Tensor:
    continuity_mask = torch.zeros_like(mask)
    for batch_i in range(mask.size(0)):
        kernel = torch.ones(min_length, device=mask.device)
        padded_seq = F.pad(mask[batch_i].float(), (min_length - 1, 0))
        conv_res = F.conv1d(padded_seq.view(1, 1, -1), kernel.view(1, 1, -1)).squeeze()
        is_in_segment = conv_res >= min_length
        for j in torch.where(is_in_segment)[0]:
            continuity_mask[batch_i, j : j + min_length] = True
    return continuity_mask & mask


class SolarTransformerPhase2B:
    def __init__(
        self,
        model,
        optimizer,
        device: torch.device,
        lookforward_steps: int,
        sentinel_value: float,
        max_grad_norm: float,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lookforward_steps = lookforward_steps
        self.sentinel_value = sentinel_value
        self.max_grad_norm = max_grad_norm

    def process_batch(self, batch):
        src, tgt_in, tgt_out, tgt_mask, src_mask = batch
        return (
            src.to(self.device),
            tgt_in.to(self.device),
            tgt_out.to(self.device),
            tgt_mask.to(self.device),
            src_mask.to(self.device),
        )

    def train_epoch(self, loader, epoch, prediction_cache):
        self.model.train()
        total_loss = 0.0
        epsilon = max(0.05, 0.9 - epoch * 0.1)
        print(f"Scheduled sampling rate (epsilon): {epsilon:.2f}")

        progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Phase 2B Training", leave=True)
        for batch_idx, batch in progress_bar:
            src, tgt_in, tgt_out, tgt_mask, src_mask = self.process_batch(batch)

            if batch_idx not in prediction_cache:
                print(f"Warning: Missing prediction for batch {batch_idx} in cache. Skipping.")
                continue

            cached_preds = prediction_cache[batch_idx].to(self.device)

            mixed_input = tgt_in.clone()

            nan_mask = tgt_mask.clone()
            mixed_input[:, :, TARGET_FEATURE_INDEX][nan_mask] = cached_preds[nan_mask]

            if torch.rand(1).item() < epsilon:
                ground_truth_mask = ~nan_mask
                mixed_input[:, :, TARGET_FEATURE_INDEX][ground_truth_mask] = cached_preds[ground_truth_mask]

            self.optimizer.zero_grad()
            output = self.model(src, mixed_input, tgt_key_padding_mask=tgt_mask, src_key_padding_mask=src_mask)

            loss = self.continuous_aware_loss(output, tgt_out, tgt_mask, epoch)

            if not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(avg_loss=total_loss / (batch_idx + 1))

        return total_loss / len(loader)

    def generate_predictions_for_epoch(self, loader):
        self.model.eval()
        epoch_predictions = {}
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(loader),
                total=len(loader),
                desc="Generating Predictions for Next Epoch",
                leave=False,
            )
            for batch_idx, batch in progress_bar:
                src, tgt_in, tgt_out, tgt_mask, src_mask = self.process_batch(batch)

                filled_output = self.incremental_fill(src, src_mask, tgt_in, tgt_mask)

                expected_len = self.lookforward_steps
                if filled_output.shape[1] != expected_len:
                    raise ValueError(
                        f"`incremental_fill` wrong length in batch {batch_idx}: "
                        f"expected {expected_len}, got {filled_output.shape[1]}.",
                    )

                epoch_predictions[batch_idx] = filled_output[:, :, 0].cpu()

        return epoch_predictions

    def incremental_fill(self, src, src_mask, tgt_in, tgt_mask):
        self.model.eval()
        with torch.no_grad():
            working_input = tgt_in.clone()
            final_predictions = torch.zeros_like(tgt_in[:, :, 0]).unsqueeze(-1)

            for step in range(self.lookforward_steps):
                pred_all_steps = self.model(
                    src,
                    working_input,
                    tgt_key_padding_mask=tgt_mask,
                    src_key_padding_mask=src_mask,
                )
                pred_this_step = pred_all_steps[:, step, 0]
                final_predictions[:, step, 0] = pred_this_step
                if step < self.lookforward_steps - 1:
                    working_input[:, step + 1, TARGET_FEATURE_INDEX] = pred_this_step

        return final_predictions

    def continuous_aware_loss(self, pred, target, mask, epoch):
        valid_mask = target != self.sentinel_value
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        base_loss = F.mse_loss(pred[valid_mask], target[valid_mask])
        alpha = min(0.5, epoch * 0.05)
        continuity_mask = find_continuous_segments(valid_mask.squeeze(-1), min_length=5)

        if continuity_mask.any():
            continuity_mask_expanded = continuity_mask.unsqueeze(-1)
            cont_loss = F.mse_loss(pred[continuity_mask_expanded], target[continuity_mask_expanded])
            return base_loss * (1 - alpha) + cont_loss * alpha
        return base_loss

    def validate(self, loader):
        print("Running comprehensive validation...")
        return {
            "guided_rmse": self._run_validation(loader, mode="guided"),
            "continuous_5step_rmse": self._run_validation(loader, mode="continuous", steps=5),
            "full_autoregressive_rmse": self._run_validation(loader, mode="autoregressive"),
        }

    def _run_validation(self, loader, mode="guided", steps=None):
        self.model.eval()
        total_loss = 0.0
        sentinel = self.sentinel_value
        with torch.no_grad():
            progress_bar = tqdm(loader, desc=f"Validation (Mode: {mode})", leave=False)
            for batch in progress_bar:
                src, tgt_in, tgt_out, tgt_mask, src_mask = self.process_batch(batch)

                pred = torch.zeros_like(tgt_out)
                if mode == "guided":
                    pred = self.model(src, tgt_in, tgt_key_padding_mask=tgt_mask, src_key_padding_mask=src_mask)
                elif mode == "continuous":
                    pred = self.rolling_forecast(src, src_mask, tgt_in, tgt_mask, steps)
                else:
                    pred = self.incremental_fill(src, src_mask, tgt_in, tgt_mask)

                loss = masked_rmse_scaled(pred, tgt_out, sentinel)
                total_loss += loss.item()

        return total_loss / len(loader)

    def rolling_forecast(self, src, src_mask, tgt_in, tgt_mask, fixed_steps):
        """First ``fixed_steps`` guided, remainder autoregressive via incremental_fill."""
        pred_input = tgt_in.clone()
        full_preds = self.incremental_fill(src, src_mask, tgt_in, tgt_mask)
        final_preds = self.model(src, pred_input, tgt_key_padding_mask=tgt_mask, src_key_padding_mask=src_mask)
        final_preds[:, fixed_steps:] = full_preds[:, fixed_steps:]
        return final_preds
