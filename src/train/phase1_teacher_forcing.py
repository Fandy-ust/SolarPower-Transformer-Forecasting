#!/usr/bin/env python3
"""Phase 1: teacher-forcing pre-training."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import DECODER_INPUT_FEATURES, ENCODER_FEATURES
from src.dataset import SolarPowerDatasetTransformer
from src.losses import masked_mse_loss
from src.model import TimeSeriesTransformer
from src.scaling import load_scaler_stats, scale_dataframe_by_campus
from src.utils import (
    add_config_argument,
    checkpoint_path,
    checkpoints_dir,
    ensure_dirs,
    parse_config_path,
    repo_root,
    resolve_path,
    set_seed,
    torch_device,
)


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
    p_cfg = cfg["phase1_teacher_forcing"]
    sentinel = float(t_cfg["sentinel_value"])
    paths = cfg["paths"]

    train_csv = resolve_path(repo, paths["train_csv"])
    val_csv = resolve_path(repo, paths["validation_csv"])
    stats_path = resolve_path(repo, paths["campus_scale"])

    ckpt_dir = checkpoints_dir(repo, cfg)
    ensure_dirs(ckpt_dir)

    ckpt_name = cfg["checkpoints"]["phase1_filename"]
    model_save_path = checkpoint_path(repo, cfg, ckpt_name)

    df_train_all = pd.read_csv(train_csv)
    df_val_all = pd.read_csv(val_csv)
    scaling_stats = load_scaler_stats(stats_path)

    df_train_scaled = scale_dataframe_by_campus(df_train_all, scaling_stats)
    df_val_scaled = scale_dataframe_by_campus(df_val_all, scaling_stats)

    if df_train_scaled.empty or df_val_scaled.empty:
        raise SystemExit(
            "Train or validation set empty after scaling. "
            "Place CSV files under paths in configs/default.yaml (see data/README.md).",
        )

    lb = int(t_cfg["lookback_steps"])
    lf = int(t_cfg["lookforward_steps"])
    batch_size = int(t_cfg["batch_size"])
    num_workers = int(p_cfg["num_workers"])

    train_ds = SolarPowerDatasetTransformer(df_train_scaled, lb, lf, sentinel)
    val_ds = SolarPowerDatasetTransformer(df_val_scaled, lb, lf, sentinel)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
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

    criterion = lambda pred, tgt: masked_mse_loss(pred, tgt, sentinel)  # noqa: E731
    optimizer = optim.Adam(model.parameters(), lr=float(p_cfg["learning_rate"]))

    epochs = int(p_cfg["epochs"])
    patience = int(p_cfg["patience"])
    max_grad = float(p_cfg["max_grad_norm"])

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for src, tgt_input, tgt_output, tgt_padding_mask, src_padding_mask in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} Train",
            leave=False,
        ):
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            tgt_padding_mask = tgt_padding_mask.to(device)
            src_padding_mask = src_padding_mask.to(device)

            optimizer.zero_grad()
            outputs = model(src, tgt_input, tgt_padding_mask, src_padding_mask)
            loss = criterion(outputs, tgt_output)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            optimizer.step()
            total_train_loss += loss.item()

        epoch_train_loss = total_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for src, tgt_input, tgt_output, tgt_padding_mask, src_padding_mask in val_loader:
                src = src.to(device)
                tgt_input = tgt_input.to(device)
                tgt_output = tgt_output.to(device)
                tgt_padding_mask = tgt_padding_mask.to(device)
                src_padding_mask = src_padding_mask.to(device)
                outputs = model(src, tgt_input, tgt_padding_mask, src_padding_mask)
                loss = criterion(outputs, tgt_output)
                total_val_loss += loss.item()
        epoch_val_loss = total_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            print(f"  -> Saved best checkpoint to {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    print(f"\nBest validation masked MSE: {best_val_loss:.6f}")

    plots_dir = resolve_path(repo, paths.get("figures_dir", "docs/figures"))
    ensure_dirs(plots_dir)
    plt.figure(figsize=(12, 7))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Transformer Phase 1 training history")
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "transformer_phase1_training_history.png"))
    plt.close()
    print("Phase 1 complete.")


if __name__ == "__main__":
    main()
