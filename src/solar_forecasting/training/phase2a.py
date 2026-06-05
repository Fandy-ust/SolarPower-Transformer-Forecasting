#!/usr/bin/env python3
"""Phase 2A: semi-autoregressive gap filling with weighted multi-step supervision."""

from __future__ import annotations

import argparse
from pathlib import Path


import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from solar_forecasting.constants import DECODER_INPUT_FEATURES, ENCODER_FEATURES, TARGET_FEATURE_INDEX
from solar_forecasting.dataset import SolarPowerDatasetTransformer
from solar_forecasting.losses import masked_mse_loss
from solar_forecasting.model import TimeSeriesTransformer
from solar_forecasting.scaling import load_scaler_stats, scale_dataframe_by_campus
from solar_forecasting.utils import (
    add_config_argument,
    checkpoint_path,
    checkpoints_dir,
    ensure_dirs,
    maybe_load_checkpoint,
    parse_config_path,
    repo_root,
    resolve_path,
    set_seed,
    torch_device,
)


def train_epoch_phase2a(model, data_loader, optimizer, sentinel, device, max_grad_norm):
    model.train()
    total_loss_for_logging = 0.0
    num_batches_processed = 0
    progress_bar = tqdm(data_loader, desc="Phase 2A Training", leave=True, dynamic_ncols=True)

    for batch in progress_bar:
        src, tgt_input, tgt_output, tgt_padding_mask, src_padding_mask = (
            batch[0].to(device),
            batch[1].to(device),
            batch[2].to(device),
            batch[3].to(device),
            batch[4].to(device),
        )

        optimizer.zero_grad()

        nan_mask = tgt_output[:, :, 0] == sentinel
        max_nans_in_batch = int(torch.max(torch.sum(nan_mask, dim=1)).item())

        if max_nans_in_batch == 0:
            outputs = model(src, tgt_input, tgt_padding_mask, src_padding_mask)
            loss = masked_mse_loss(outputs, tgt_output, sentinel)
            if not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                total_loss_for_logging += loss.item()
        else:
            working_tgt_input = tgt_input.clone()
            nan_cumsum = torch.cumsum(nan_mask.int(), dim=1)

            for k in range(max_nans_in_batch):
                current_model_output = model(src, working_tgt_input, tgt_padding_mask, src_padding_mask)
                loss_k = masked_mse_loss(current_model_output, tgt_output, sentinel)

                if not torch.isnan(loss_k) and loss_k.item() > 0:
                    weight = (k + 1) / max_nans_in_batch
                    (loss_k * weight).backward()
                    if k == max_nans_in_batch - 1:
                        total_loss_for_logging += loss_k.item()

                if k == max_nans_in_batch - 1:
                    break

                is_kth_nan = (nan_cumsum == k + 1) & nan_mask
                predictions_for_kth_nan = current_model_output.detach()[:, :, 0][is_kth_nan]
                working_tgt_input[:, :, TARGET_FEATURE_INDEX][is_kth_nan] = predictions_for_kth_nan

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        num_batches_processed += 1
        progress_bar.set_postfix(avg_loss=total_loss_for_logging / max(num_batches_processed, 1))

    return total_loss_for_logging / num_batches_processed if num_batches_processed > 0 else 0.0


def validate_epoch_phase2a(model, data_loader, sentinel, device):
    model.eval()
    total_val_loss = 0.0
    progress_bar = tqdm(data_loader, desc="Phase 2A Validation", leave=False, dynamic_ncols=True)

    with torch.no_grad():
        for batch in progress_bar:
            src, tgt_input, tgt_output, tgt_padding_mask, src_padding_mask = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
                batch[3].to(device),
                batch[4].to(device),
            )

            nan_mask = tgt_output[:, :, 0] == sentinel
            max_nans_in_batch = int(torch.max(torch.sum(nan_mask, dim=1)).item())

            final_output = None
            if max_nans_in_batch == 0:
                final_output = model(src, tgt_input, tgt_padding_mask, src_padding_mask)
            else:
                working_tgt_input = tgt_input.clone()
                nan_cumsum = torch.cumsum(nan_mask.int(), dim=1)
                for k in range(max_nans_in_batch):
                    current_model_output = model(src, working_tgt_input, tgt_padding_mask, src_padding_mask)
                    if k == max_nans_in_batch - 1:
                        final_output = current_model_output
                        break
                    is_kth_nan = (nan_cumsum == k + 1) & nan_mask
                    predictions_for_kth_nan = current_model_output[:, :, 0][is_kth_nan]
                    working_tgt_input[:, :, TARGET_FEATURE_INDEX][is_kth_nan] = predictions_for_kth_nan

            loss = masked_mse_loss(final_output, tgt_output, sentinel)
            if not torch.isnan(loss):
                total_val_loss += loss.item()

    return total_val_loss / len(data_loader)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    repo = repo_root()
    cfg = parse_config_path(args)
    set_seed(cfg.get("seed", 42))

    device = torch_device(args.device)
    t_cfg = cfg["training"]
    p_cfg = cfg["phase2a_semi_autoregressive"]
    sentinel = float(t_cfg["sentinel_value"])

    ckpt_dir = checkpoints_dir(repo, cfg)
    ensure_dirs(ckpt_dir)

    phase1_path = checkpoint_path(repo, cfg, cfg["checkpoints"]["phase1_filename"])
    save_path = checkpoint_path(repo, cfg, cfg["checkpoints"]["phase2a_filename"])

    paths = cfg["paths"]
    train_csv = resolve_path(repo, paths["train_csv"])
    val_csv = resolve_path(repo, paths["validation_csv"])
    stats_path = resolve_path(repo, paths["campus_scale"])

    df_train_scaled = scale_dataframe_by_campus(pd.read_csv(train_csv), load_scaler_stats(stats_path))
    df_val_scaled = scale_dataframe_by_campus(pd.read_csv(val_csv), load_scaler_stats(stats_path))

    lb = int(t_cfg["lookback_steps"])
    lf = int(t_cfg["lookforward_steps"])
    batch_size = int(t_cfg["batch_size"])
    num_workers = int(p_cfg["num_workers"])

    train_loader = DataLoader(
        SolarPowerDatasetTransformer(df_train_scaled, lb, lf, sentinel),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        SolarPowerDatasetTransformer(df_val_scaled, lb, lf, sentinel),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = TimeSeriesTransformer(
        encoder_feature_dim=len(ENCODER_FEATURES),
        decoder_feature_dim=len(DECODER_INPUT_FEATURES),
        d_model=int(t_cfg["d_model"]),
        nhead=int(t_cfg["nhead"]),
        num_encoder_layers=int(t_cfg["num_encoder_layers"]),
        num_decoder_layers=int(t_cfg["num_decoder_layers"]),
        dim_feedforward=int(t_cfg["dim_feedforward"]),
        dropout=float(t_cfg["dropout_phase12"]),
    ).to(device)

    if maybe_load_checkpoint(model, phase1_path, map_location=device):
        print(f"Loaded Phase 1 weights from {phase1_path}")
    else:
        print(
            "Phase 1 checkpoint not found - training Phase 2A from scratch "
            f"(run phase1_teacher_forcing.py first). Expected at {phase1_path}",
        )

    optimizer = optim.Adam(model.parameters(), lr=float(p_cfg["learning_rate"]))
    best_val_loss = float("inf")
    patience_counter = 0

    epochs = int(p_cfg["epochs"])
    patience = int(p_cfg["patience"])

    for epoch in range(epochs):
        train_loss = train_epoch_phase2a(
            model,
            train_loader,
            optimizer,
            sentinel,
            device,
            float(p_cfg["max_grad_norm"]),
        )
        val_loss = validate_epoch_phase2a(model, val_loader, sentinel, device)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(f"  -> Saved best to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    print(f"Phase 2A done. Best val loss {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
