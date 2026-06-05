"""Feature names shared across model, dataset, and scaling."""

from __future__ import annotations

TARGET_COLUMN = "SolarGeneration"
WEATHER_FEATURES = [
    "AirTemperature",
    "Ghi_hourly",
    "CloudOpacity_hourly",
    "minutes_since_last_update",
    "RelativeHumidity",
]
TIME_FEATURES = ["zenith_sin", "azimuth_sin", "day_sin", "year_sin"]
ENCODER_FEATURES = WEATHER_FEATURES + TIME_FEATURES + [TARGET_COLUMN]
FUTURE_KNOWN_FEATURES = TIME_FEATURES
DECODER_INPUT_FEATURES = FUTURE_KNOWN_FEATURES + [TARGET_COLUMN]
TARGET_FEATURE_INDEX = DECODER_INPUT_FEATURES.index(TARGET_COLUMN)
