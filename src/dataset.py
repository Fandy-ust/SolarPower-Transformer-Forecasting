"""Dataset for sliding-window solar sequences."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.constants import (
    DECODER_INPUT_FEATURES,
    ENCODER_FEATURES,
    FUTURE_KNOWN_FEATURES,
    TARGET_COLUMN,
)


class SolarPowerDatasetTransformer(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int,
        lookforward: int,
        sentinel_value: float,
        return_campus_key: bool = False,
    ):
        self.df = df
        self.lookback = lookback
        self.lookforward = lookforward
        self.total_seq_len = lookback + lookforward
        self.sentinel_value = sentinel_value
        self.return_campus_key = return_campus_key
        self.campus_indices: list[dict] = []
        for campus_id in df["CampusKey"].unique():
            campus_df = df[df["CampusKey"] == campus_id]
            start_index = campus_df.index[0]
            num_samples = len(campus_df) - self.total_seq_len + 1
            if num_samples > 0:
                self.campus_indices.append(
                    {"start": start_index, "count": num_samples, "id": campus_id},
                )
        self.total_samples = sum(c["count"] for c in self.campus_indices)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        campus_idx = 0
        while idx >= self.campus_indices[campus_idx]["count"]:
            idx -= self.campus_indices[campus_idx]["count"]
            campus_idx += 1

        campus_info = self.campus_indices[campus_idx]
        start_pos = campus_info["start"] + idx
        end_pos = start_pos + self.total_seq_len
        sequence_slice = self.df.iloc[start_pos:end_pos]

        src_df = sequence_slice.iloc[: self.lookback].copy()
        src_key_padding_mask = torch.tensor(src_df[TARGET_COLUMN].isna().values, dtype=torch.bool)
        src_df.fillna(0, inplace=True)
        src = torch.tensor(src_df[ENCODER_FEATURES].values, dtype=torch.float32)

        future_slice = sequence_slice.iloc[self.lookback :]
        tgt_key_padding_mask = torch.tensor(future_slice[TARGET_COLUMN].isna().values, dtype=torch.bool)

        last_obs = sequence_slice.iloc[self.lookback - 1][TARGET_COLUMN]

        shifted = future_slice[TARGET_COLUMN].shift(1)
        shifted.iloc[0] = last_obs
        shifted = shifted.fillna(0)
        future_known_df = future_slice[FUTURE_KNOWN_FEATURES]
        tgt_input_df = pd.concat(
            [future_known_df.reset_index(drop=True), shifted.reset_index(drop=True).rename(TARGET_COLUMN)],
            axis=1,
        )
        tgt_input = torch.tensor(tgt_input_df[DECODER_INPUT_FEATURES].values, dtype=torch.float32)

        tgt_output_series = future_slice[TARGET_COLUMN].fillna(self.sentinel_value)
        tgt_output = torch.tensor(tgt_output_series.values, dtype=torch.float32).unsqueeze(-1)

        out = (src, tgt_input, tgt_output, tgt_key_padding_mask, src_key_padding_mask)
        if self.return_campus_key:
            return out + (campus_info["id"],)
        return out
