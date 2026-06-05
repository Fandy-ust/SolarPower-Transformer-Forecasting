"""Repository paths, RNG seeding, YAML config helpers."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    import yaml

    cfg_path = Path(path) if path else repo_root() / "configs" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(repo: Path, p: Path | str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return repo / path


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=str(repo_root() / "configs" / "default.yaml"),
        help="YAML config relative to cwd or absolute",
    )


def parse_config_path(args: argparse.Namespace) -> dict[str, Any]:
    repo = repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path_candidate = (Path.cwd() / cfg_path).resolve()
        cfg_path = cfg_path_candidate if cfg_path_candidate.exists() else (repo / args.config).resolve()
    return load_config(cfg_path)


def checkpoints_dir(repo: Path, cfg: dict[str, Any]) -> Path:
    return resolve_path(repo, cfg["paths"]["checkpoints_dir"])


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def checkpoint_path(repo: Path, cfg: dict[str, Any], filename: str) -> Path:
    return checkpoints_dir(repo, cfg) / filename


def torch_device(device_str: str | None = None):
    import torch

    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_load_checkpoint(model, path: Path, map_location):
    """Load weights if ``path`` exists; return True if loaded."""
    import torch

    if not path.exists():
        return False
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return True
