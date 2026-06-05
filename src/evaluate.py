#!/usr/bin/env python3
"""
Evaluate a trained Phase 3 checkpoint on the held-out test CSV.

Computes RMSE / R^2 in original power units (reverse per-campus z-score on SolarGeneration).
Optionally plots per-forecast-step RMSE and R^2 (see configs/default.yaml ``evaluation.plots``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import DECODER_INPUT_FEATURES, ENCODER_FEATURES, TARGET_COLUMN, TARGET_FEATURE_INDEX
from src.dataset import SolarPowerDatasetTransformer
from src.model import TimeSeriesTransformer
from src.scaling import load_scaler_stats, scale_dataframe_by_campus
from src.utils import (
    add_config_argument,
    checkpoint_path,
    maybe_load_checkpoint,
    parse_config_path,
    repo_root,
    resolve_path,
    set_seed,
    torch_device,
)


def campus_mean_std(scaler: dict, key) -> tuple[float, float]:
    """Map dataset campus key to scaler JSON keys like ``\"1\"``."""
    cid = str(int(np.asarray(key).item()))
    params = scaler[cid][TARGET_COLUMN]
    return float(params["mean"]), float(params["std"])


def inverse_scale_predictions(preds_scaled, tgts_scaled, campus_keys, scaler: dict):
    p = preds_scaled.detach().cpu().numpy().squeeze(-1)
    t = tgts_scaled.detach().cpu().numpy().squeeze(-1)
    if isinstance(campus_keys, torch.Tensor):
        ck = campus_keys.cpu().numpy()
    else:
        ck = np.asarray(campus_keys)

    po = np.empty_like(p, dtype=np.float64)
    to = np.empty_like(t, dtype=np.float64)
    for i in range(p.shape[0]):
        mu, sigma = campus_mean_std(scaler, ck[i])
        po[i] = p[i] * sigma + mu
        to[i] = t[i] * sigma + mu
    return po, to


@torch.no_grad()
def predict_autoregressive_batch(model, sentinel, lf, device, batch_without_campus):
    src, tgt_in, tgt_out, _tgt_pm, src_padding_mask = batch_without_campus
    src = src.to(device)
    tgt_in = tgt_in.to(device)
    tgt_out = tgt_out.to(device)
    src_padding_mask = src_padding_mask.to(device)

    working_input = tgt_in.clone()
    for step in range(lf):
        pred_all = model(src, working_input, None, src_padding_mask)
        pred_step = pred_all[:, step, 0]
        if step < lf - 1:
            working_input[:, step + 1, TARGET_FEATURE_INDEX] = pred_step

    preds = pred_all
    mask_valid = tgt_out.squeeze(-1) != sentinel
    return preds, tgt_out, mask_valid


def r2_rmse(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    residuals = obs - pred
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, rmse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path (defaults to Phase 3 file in configs/default.yaml).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo = repo_root()
    cfg = parse_config_path(args)
    set_seed(int(cfg.get("seed", 42)))

    device = torch_device(args.device)
    eval_cfg = cfg.get("evaluation", {})
    plots_cfg = eval_cfg.get("plots", {})

    t_cfg = cfg["training"]
    sentinel = float(t_cfg["sentinel_value"])
    lb = int(t_cfg["lookback_steps"])
    lf = int(t_cfg["lookforward_steps"])
    bs = int(eval_cfg.get("batch_size", t_cfg["batch_size"]))
    nw = int(eval_cfg.get("num_workers", 0))

    paths = cfg["paths"]
    test_csv = resolve_path(repo, paths["test_csv"])
    stats_path = resolve_path(repo, paths["campus_scale"])

    scaler = load_scaler_stats(stats_path)
    df_raw = pd.read_csv(test_csv)
    df_scaled = scale_dataframe_by_campus(df_raw, scaler)

    ds = SolarPowerDatasetTransformer(
        df_scaled,
        lb,
        lf,
        sentinel_value=sentinel,
        return_campus_key=True,
    )
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)

    ck_arg = args.checkpoint or str(checkpoint_path(repo, cfg, cfg["checkpoints"]["phase3_filename"]))
    ck_path = Path(ck_arg)
    if not ck_path.is_absolute():
        ck_path = (repo / ck_path).resolve()

    p3_dropout = float(cfg["phase3_fully_autoregressive"].get("dropout", t_cfg["dropout_phase12"]))
    model = TimeSeriesTransformer(
        encoder_feature_dim=len(ENCODER_FEATURES),
        decoder_feature_dim=len(DECODER_INPUT_FEATURES),
        d_model=int(t_cfg["d_model"]),
        nhead=int(t_cfg["nhead"]),
        num_encoder_layers=int(t_cfg["num_encoder_layers"]),
        num_decoder_layers=int(t_cfg["num_decoder_layers"]),
        dim_feedforward=int(t_cfg["dim_feedforward"]),
        dropout=p3_dropout,
    ).to(device)

    if not maybe_load_checkpoint(model, ck_path, map_location=device):
        raise FileNotFoundError(f"Checkpoint not found: {ck_path}")
    model.eval()
    print(f"Loaded weights from {ck_path}")

    all_preds, all_targets, valid_flags = [], [], []

    for batch in tqdm(loader, desc="Evaluating"):
        *rest, campus = batch
        preds, tgts, vmask = predict_autoregressive_batch(model, sentinel, lf, device, rest)

        preds_np, tgts_np = inverse_scale_predictions(preds, tgts, campus, scaler)
        m = vmask.cpu().numpy()

        for b in range(preds_np.shape[0]):
            all_preds.append(preds_np[b])
            all_targets.append(tgts_np[b])
            valid_flags.append(m[b])

    preds_a = np.stack(all_preds)
    tgts_a = np.stack(all_targets)
    vf = np.stack(valid_flags)

    p_flat = preds_a[vf].ravel()
    o_flat = tgts_a[vf].ravel()
    r2, rmse = r2_rmse(o_flat, p_flat)
    print(f"Test overall R^2={r2:.4f} RMSE={rmse:.4f} (original power units)")
    print(f"Valid supervisory timesteps counted: {int(vf.sum())}")

    r2_steps = []
    rmse_steps = []
    for s in range(lf):
        vv = vf[:, s]
        if not vv.any():
            r2_steps.append(float("nan"))
            rmse_steps.append(float("nan"))
            continue
        r2_s, rmse_s = r2_rmse(tgts_a[vv, s], preds_a[vv, s])
        r2_steps.append(r2_s)
        rmse_steps.append(rmse_s)

    rmse_png = plots_cfg.get("rmse_curve")
    r2_png = plots_cfg.get("r2_curve")
    if rmse_png:
        pth = resolve_path(repo, rmse_png)
        pth.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, lf + 1), rmse_steps, marker="o")
        plt.xlabel("Forecast step")
        plt.ylabel("RMSE (power units)")
        plt.title("RMSE vs forecast horizon (test)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pth)
        plt.close()
        print(f"Saved RMSE curve to {pth}")

    if r2_png:
        pth = resolve_path(repo, r2_png)
        pth.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, lf + 1), r2_steps, marker="o", color="orange")
        plt.xlabel("Forecast step")
        plt.ylabel(r"$R^2$")
        plt.title(r"$R^2$ vs forecast horizon (test)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pth)
        plt.close()
        print(f"Saved R^2 curve to {pth}")


if __name__ == "__main__":
    main()
