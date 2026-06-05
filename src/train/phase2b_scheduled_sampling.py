#!/usr/bin/env python3
"""Phase 2B: scheduled sampling refinement (loads Phase 2A weights)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.constants import DECODER_INPUT_FEATURES, ENCODER_FEATURES
from src.dataset import SolarPowerDatasetTransformer
from src.model import TimeSeriesTransformer
from src.phase2b_trainer import SolarTransformerPhase2B
from src.scaling import load_scaler_stats, scale_dataframe_by_campus
from src.utils import (
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
    p_cfg = cfg["phase2b_scheduled_sampling"]
    sentinel = float(t_cfg["sentinel_value"])

    ckpts = cfg["checkpoints"]
    ckpt_dir = checkpoints_dir(repo, cfg)
    ensure_dirs(ckpt_dir)

    phase2a_path = checkpoint_path(repo, cfg, ckpts["phase2a_filename"])
    phase2b_path = checkpoint_path(repo, cfg, ckpts["phase2b_filename"])

    paths = cfg["paths"]
    train_csv = resolve_path(repo, paths["train_csv"])
    val_csv = resolve_path(repo, paths["validation_csv"])
    stats_path = resolve_path(repo, paths["campus_scale"])

    df_train_scaled = scale_dataframe_by_campus(pd.read_csv(train_csv), load_scaler_stats(stats_path))
    df_val_scaled = scale_dataframe_by_campus(pd.read_csv(val_csv), load_scaler_stats(stats_path))

    lb = int(t_cfg["lookback_steps"])
    lf = int(t_cfg["lookforward_steps"])
    batch_size = int(t_cfg["batch_size"])

    nw = int(p_cfg["num_workers"])
    shuffle_train = False
    train_ds = SolarPowerDatasetTransformer(df_train_scaled, lb, lf, sentinel)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=nw,
        pin_memory=True,
    )
    val_loader = DataLoader(
        SolarPowerDatasetTransformer(df_val_scaled, lb, lf, sentinel),
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
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

    if not maybe_load_checkpoint(model, phase2a_path, map_location=device):
        raise FileNotFoundError(
            "Phase 2A checkpoint missing. Run phase2a_semi_autoregressive.py first.",
        )
    print(f"Loaded Phase 2A weights from {phase2a_path}")

    optimizer = optim.Adam(model.parameters(), lr=float(p_cfg["learning_rate"]))
    trainer = SolarTransformerPhase2B(
        model,
        optimizer,
        device=device,
        lookforward_steps=lf,
        sentinel_value=sentinel,
        max_grad_norm=float(p_cfg["max_grad_norm"]),
    )

    epochs = int(p_cfg["epochs"])
    patience = int(p_cfg["patience"])

    best_ar_loss = float("inf")
    patience_counter = 0

    print("Generating initial predictions for epoch 1...")
    prediction_cache = trainer.generate_predictions_for_epoch(train_loader)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = trainer.train_epoch(train_loader, epoch, prediction_cache)
        val_metrics = trainer.validate(val_loader)
        guided = val_metrics["guided_rmse"]
        cont = val_metrics["continuous_5step_rmse"]
        ar_loss = val_metrics["full_autoregressive_rmse"]

        print(f"Train MSE-ish: {train_loss:.6f}")
        print(f"Val RMSE (scaled) guided={guided:.6f} continuous={cont:.6f} AR={ar_loss:.6f}")

        if ar_loss < best_ar_loss:
            best_ar_loss = ar_loss
            torch.save(model.state_dict(), phase2b_path)
            patience_counter = 0
            print(f"  -> Saved best AR checkpoint to {phase2b_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

        if epoch < epochs - 1:
            prediction_cache = trainer.generate_predictions_for_epoch(train_loader)

    print(f"Phase 2B complete. Best AR val RMSE (scaled space): {best_ar_loss:.6f}")


if __name__ == "__main__":
    main()
