from __future__ import annotations

import math
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
                # LightGBM có thể xử lý NaN, nhưng để demo ổn định và đồng nhất schema:
                # thay NaN categorical bằng nhãn "unknown".
                frame[column] = frame[column].cat.add_categories(["unknown"]).fillna("unknown")

    # Chặn NaN trong numeric feature để tránh lỗi inference (đặc biệt với một số bản XGBoost/DMatrix setup).
    numeric_cols = frame.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        frame[numeric_cols] = frame[numeric_cols].fillna(0)

    # Còn lại các cột không phải numeric (thường là categorical với LightGBM) đã được xử lý ở trên.
    return frame


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính các đặc trưng vật lý/chu kỳ thời gian tương tự batch ETL (Spark):
    - hour_sin, hour_cos
    - month_sin, month_cos
    - saturated_vapor_pressure, actual_vapor_pressure

    Hàm này an toàn cho realtime (thiếu cột thì sẽ bỏ qua).
    """
    data = normalize_pm25_column(df).copy()

    if "hour" in data.columns:
        hour = pd.to_numeric(data["hour"], errors="coerce").fillna(0)
        data["hour_sin"] = (2 * math.pi * hour / 24).map(math.sin)
        data["hour_cos"] = (2 * math.pi * hour / 24).map(math.cos)

    if "month" in data.columns:
        month = pd.to_numeric(data["month"], errors="coerce").fillna(0)
        data["month_sin"] = (2 * math.pi * month / 12).map(math.sin)
        data["month_cos"] = (2 * math.pi * month / 12).map(math.cos)

    # Công thức giống Spark ETL
    if "TEMP" in data.columns:
        temp = pd.to_numeric(data["TEMP"], errors="coerce")
        data["saturated_vapor_pressure"] = 61.1 * ((7.5 * temp) / (237.3 + temp))

    if "DEWP" in data.columns:
        dewp = pd.to_numeric(data["DEWP"], errors="coerce")
        data["actual_vapor_pressure"] = 61.1 * ((7.5 * dewp) / (237.3 + dewp))

    return data
