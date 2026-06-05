"""Per-campus z-score scaling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from solar_forecasting.constants import TARGET_COLUMN, WEATHER_FEATURES


def apply_zscore_scaling(df_data: pd.DataFrame, columns: list[str], stats: dict[str, Any]) -> pd.DataFrame:
    df_scaled = df_data.copy()
    for col in columns:
        if col in stats and stats[col]["std"] > 1e-6:
            df_scaled[col] = (df_scaled[col] - stats[col]["mean"]) / stats[col]["std"]
    return df_scaled


def scale_dataframe_by_campus(df: pd.DataFrame, scaler_stats: dict[str, dict]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    scaled_dfs: list[pd.DataFrame] = []
    columns_to_scale = WEATHER_FEATURES + [TARGET_COLUMN]

    for campus_id_num, group in df.groupby("CampusKey"):
        group_copy = group.copy()
        campus_id = str(campus_id_num)
        campus_params = scaler_stats.get(campus_id)

        if campus_params:
            for col in WEATHER_FEATURES:
                if col in group_copy.columns:
                    group_copy[col] = group_copy[col].fillna(0)

            scaled_group = apply_zscore_scaling(group_copy, columns_to_scale, campus_params)
            scaled_dfs.append(scaled_group)

    if not scaled_dfs:
        return pd.DataFrame()

    return pd.concat(scaled_dfs).sort_index()


def load_scaler_stats(path: Path | str) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)
