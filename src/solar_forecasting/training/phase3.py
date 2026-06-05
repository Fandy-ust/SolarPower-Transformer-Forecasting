#!/usr/bin/env python3
"""Phase 3: fully autoregressive fine-tuning."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from solar_forecasting.constants import DECODER_INPUT_FEATURES, ENCODER_FEATURES, TARGET_FEATURE_INDEX
from solar_forecasting.dataset import SolarPowerDatasetTransformer
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


def train_epoch_fully_autoregressive(model, data_loader, optimizer, sentinel, lf, grad_clip, device):
    model.train()
    total_loss = 0.0
    for src, tgt_in, tgt_out, _, src_key_padding_mask in tqdm(data_loader, desc="[Train] Fully AR"):
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        src_key_padding_mask = src_key_padding_mask.to(device)

        optimizer.zero_grad()
        working_input = tgt_in.clone()

        for step in range(lf):
            pred_all_steps = model(src, working_input, None, src_key_padding_mask)
            pred_this_step = pred_all_steps[:, step, 0]
            if step < lf - 1:
                working_input[:, step + 1, TARGET_FEATURE_INDEX] = pred_this_step.detach()

        final_preds = pred_all_steps
        valid_mask = tgt_out != sentinel
        loss = nn.functional.mse_loss(final_preds[valid_mask], tgt_out[valid_mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


@torch.no_grad()
def validate_fully_autoregressive(model, data_loader, sentinel, lf, device):
    model.eval()
    total_loss = 0.0
    for src, tgt_in, tgt_out, _, src_key_padding_mask in tqdm(data_loader, desc="[Valid] Fully AR"):
        src = src.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out = tgt_out.to(device)
        src_key_padding_mask = src_key_padding_mask.to(device)

        working_input = tgt_in.clone()
        for step in range(lf):
            pred_all_steps = model(src, working_input, None, src_key_padding_mask)
            pred_this_step = pred_all_steps[:, step, 0]
            if step < lf - 1:
                working_input[:, step + 1, TARGET_FEATURE_INDEX] = pred_this_step

        final_preds = pred_all_steps
        valid_mask = tgt_out != sentinel
        loss = nn.functional.mse_loss(final_preds[valid_mask], tgt_out[valid_mask])
        total_loss += loss.item()

    return total_loss / len(data_loader)


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
    t_global = cfg["training"]
    p_cfg = cfg["phase3_fully_autoregressive"]
    sentinel = float(t_global["sentinel_value"])

    lf = int(t_global["lookforward_steps"])
    lb = int(t_global["lookback_steps"])

    ckpt_dir = checkpoints_dir(repo, cfg)
    ensure_dirs(ckpt_dir)

    prev_path = checkpoint_path(repo, cfg, cfg["checkpoints"]["phase2b_filename"])
    out_path = checkpoint_path(repo, cfg, cfg["checkpoints"]["phase3_filename"])

    paths = cfg["paths"]
    train_csv = resolve_path(repo, paths["train_csv"])
    val_csv = resolve_path(repo, paths["validation_csv"])
    stats_path = resolve_path(repo, paths["campus_scale"])
    scaler = load_scaler_stats(stats_path)

    bs = int(p_cfg.get("batch_size", t_global["batch_size"]))

    train_ds = SolarPowerDatasetTransformer(scale_dataframe_by_campus(pd.read_csv(train_csv), scaler), lb, lf, sentinel)
    val_ds = SolarPowerDatasetTransformer(scale_dataframe_by_campus(pd.read_csv(val_csv), scaler), lb, lf, sentinel)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    dropout = float(p_cfg.get("dropout", t_global["dropout_phase12"]))
    model = TimeSeriesTransformer(
        encoder_feature_dim=len(ENCODER_FEATURES),
        decoder_feature_dim=len(DECODER_INPUT_FEATURES),
        d_model=int(t_global["d_model"]),
        nhead=int(t_global["nhead"]),
        num_encoder_layers=int(t_global["num_encoder_layers"]),
        num_decoder_layers=int(t_global["num_decoder_layers"]),
        dim_feedforward=int(t_global["dim_feedforward"]),
        dropout=dropout,
    ).to(device)

    if not maybe_load_checkpoint(model, prev_path, map_location=device):
        raise FileNotFoundError(f"Missing Phase 2B weights at {prev_path}")
    print(f"Loaded Phase 2B weights from {prev_path}")

    optimizer = optim.AdamW(model.parameters(), lr=float(p_cfg["learning_rate"]))

    epochs = int(p_cfg["epochs"])
    grad_clip = float(p_cfg["grad_clip"])
    best_val = float("inf")

    print("Starting Phase 3 fine-tuning...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch_fully_autoregressive(model, train_loader, optimizer, sentinel, lf, grad_clip, device)
        va = validate_fully_autoregressive(model, val_loader, sentinel, lf, device)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs} | {elapsed:.1f}s | train {tr:.6f} | val {va:.6f}")

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), out_path)
            print(f"  -> saved {out_path}")

    print(f"Phase 3 complete. Best val MSE (scaled targets): {best_val:.6f}")


if __name__ == "__main__":
    main()
