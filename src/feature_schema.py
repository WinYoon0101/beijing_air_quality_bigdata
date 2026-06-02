from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


TIME_COLUMNS = ["station", "year", "month", "day", "hour"]
DEFAULT_EXCLUDED_COLUMNS = {
    "No",
    "year",
    "month",
    "day",
    "hour",
    "station",
    "wd",
}

LIGHTGBM_CATEGORICAL_COLUMNS = {"station", "wd"}


def normalize_pm25_column(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "PM2.5" in data.columns and "PM2_5" not in data.columns:
        data = data.rename(columns={"PM2.5": "PM2_5"})
    return data


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in TIME_COLUMNS if column in df.columns]
    if sort_columns:
        return df.sort_values(sort_columns).reset_index(drop=True)
    return df.reset_index(drop=True)


def add_next_hour_target(df: pd.DataFrame, target_col: str = "Target_PM2.5_next_1h") -> pd.DataFrame:
    data = normalize_pm25_column(df)
    data = sort_by_time(data)

    if target_col in data.columns:
        return data

    if "PM2_5" not in data.columns:
        raise RuntimeError("Gold layer phải có cột PM2_5 hoặc PM2.5 để tạo target")

    if "station" in data.columns:
        data[target_col] = data.groupby("station")["PM2_5"].shift(-1)
    else:
        data[target_col] = data["PM2_5"].shift(-1)

    return data


def select_numeric_features(
    df: pd.DataFrame,
    target_col: str,
    excluded_columns: Iterable[str] | None = None,
) -> list[str]:
    excluded = set(DEFAULT_EXCLUDED_COLUMNS)
    excluded.add(target_col)
    if excluded_columns:
        excluded.update(excluded_columns)

    feature_candidates = [column for column in df.columns if column not in excluded]
    return df[feature_candidates].select_dtypes(include=["number"]).columns.tolist()


def resolve_inference_features(
    df: pd.DataFrame,
    metadata: dict,
    model_name: str,
) -> list[str]:
    feature_columns = metadata.get("feature_columns")
    if feature_columns:
        return list(feature_columns)

    if model_name.lower() == "xgboost":
        excluded = {"No", "Target_PM2.5_next_1h"}
        return df[[column for column in df.columns if column not in excluded]].select_dtypes(include=["number"]).columns.tolist()

    excluded = {"No", "Target_PM2.5_next_1h"}
    return [column for column in df.columns if column not in excluded]


def prepare_inference_frame(df: pd.DataFrame, features: list[str], model_name: str) -> pd.DataFrame:
    frame = normalize_pm25_column(df).copy()

    for column in features:
        if column not in frame.columns:
            frame[column] = 0

    frame = frame[features].copy()
    if model_name.lower() == "lightgbm":
        for column in LIGHTGBM_CATEGORICAL_COLUMNS:
            if column in frame.columns:
                frame[column] = frame[column].astype("category")

    return frame
