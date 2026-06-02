from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import lightgbm as lgb
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

LIGHTGBM_CATEGORICAL_COLUMNS = {
    "station",
    "wd",
    "station_lag_1",
    "station_lag_2",
    "wd_lag_1",
    "wd_lag_2",
}

# Cột nguồn để tạo lag (khớp metadata LightGBM / Spark ETL)
LAG_SOURCE_COLUMNS = [
    "PM2_5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "wd",
    "WSPM",
    "station",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

# Chỉ các cột này có rolling_2 trong Gold (không có PM2_5, wd, station)
ROLLING_COLUMNS = [
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

CATEGORICAL_LAG_SOURCES = {"wd", "station"}


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


def load_lightgbm_booster(model_path: Path | str) -> lgb.Booster:
    """Nạp model từ bytes trong RAM — tránh đọc file đang bị ghi dở khi retrain."""
    path = Path(model_path)
    model_text = path.read_bytes().decode("utf-8")
    return lgb.Booster(model_str=model_text)


def _map_to_training_categories(values: pd.Series, categories: list) -> pd.Series:
    if not categories:
        return pd.to_numeric(values, errors="coerce").fillna(0)
    allowed = set(categories)
    fallback = categories[0]
    mapped = values.astype(str).replace({"nan": fallback, "None": fallback, "": fallback})
    mapped = mapped.map(lambda item: item if item in allowed else fallback)
    return pd.Categorical(mapped, categories=categories)


def categorical_levels_by_feature(booster) -> dict[str, list]:
    """Ánh xạ tên cột -> category levels theo thứ tự feature trong file model."""
    feature_names = list(booster.feature_name() or [])
    pandas_categorical = list(getattr(booster, "pandas_categorical", None) or [])
    levels: dict[str, list] = {}
    cat_idx = 0
    for name in feature_names:
        if name not in LIGHTGBM_CATEGORICAL_COLUMNS:
            continue
        if cat_idx < len(pandas_categorical):
            levels[name] = list(pandas_categorical[cat_idx])
        cat_idx += 1
    return levels


def prepare_lightgbm_frame(
    df: pd.DataFrame,
    expected_features: list[str],
    booster=None,
) -> pd.DataFrame:
    """Chuẩn hóa input khớp model LightGBM (dtype + pandas_categorical từ file model)."""
    frame = normalize_pm25_column(df).copy()
    feature_order = list(booster.feature_name()) if booster is not None else list(expected_features)
    if not feature_order:
        feature_order = list(expected_features)

    frame = frame.reindex(columns=feature_order, fill_value=0)
    cat_levels = categorical_levels_by_feature(booster) if booster is not None else {}

    for column in feature_order:
        if column in cat_levels:
            frame[column] = _map_to_training_categories(frame[column], cat_levels[column])
        elif column in LIGHTGBM_CATEGORICAL_COLUMNS:
            frame[column] = _map_to_training_categories(frame[column], ["unknown"])
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    # Tránh LightGBM tự nhận thêm cột object là categorical (gây lệch số lượng)
    for column in feature_order:
        dtype_name = frame[column].dtype.name
        if dtype_name not in {"category", "float64", "float32", "int64", "int32"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    return frame[feature_order]


def predict_lightgbm(booster: lgb.Booster, df: pd.DataFrame) -> list[float]:
    """Dự báo với DataFrame đã có đủ feature (sau enrich), khớp categorical lúc train."""
    feature_order = list(booster.feature_name())
    infer_df = prepare_lightgbm_frame(df, feature_order, booster=booster)
    return list(booster.predict(infer_df))


def prepare_inference_frame(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    booster=None,
) -> pd.DataFrame:
    if model_name.lower() == "lightgbm":
        return prepare_lightgbm_frame(df, features, booster=booster)

    frame = normalize_pm25_column(df).copy()
    for column in features:
        if column not in frame.columns:
            frame[column] = 0
    frame = frame[features].copy()
    numeric_cols = frame.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        frame[numeric_cols] = frame[numeric_cols].fillna(0)
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


def _snapshot_from_enriched_row(row: dict) -> dict:
    """Trích trạng thái cần lưu history để tính lag/rolling cho event kế tiếp."""
    snapshot = {}
    for column in LAG_SOURCE_COLUMNS:
        if column in row:
            snapshot[column] = row[column]
    return snapshot


def enrich_realtime_row(
    raw_row: dict,
    prev1: dict | None,
    prev2: dict | None,
    cum_wspm: float,
) -> tuple[dict, dict]:
    """
    Tính đặc trưng realtime khớp metadata LightGBM:
    physical -> lag/rolling -> cum_wspm.

    Dòng đầu tiên: lag/rolling numeric = 0; categorical lag dùng giá trị hiện tại (tránh lệch category).
    """
    base_df = normalize_pm25_column(pd.DataFrame([raw_row]))
    current_df = add_physical_features(base_df)
    current = current_df.iloc[0].to_dict()

    enriched = dict(current)
    enriched["cum_wspm"] = float(cum_wspm)

    for column in LAG_SOURCE_COLUMNS:
        if column not in current:
            continue
        lag1_key = f"{column}_lag_1"
        lag2_key = f"{column}_lag_2"
        if column in CATEGORICAL_LAG_SOURCES:
            enriched[lag1_key] = (
                str(prev1[column]) if prev1 and column in prev1 else str(current[column])
            )
            enriched[lag2_key] = (
                str(prev2[column]) if prev2 and column in prev2 else str(current[column])
            )
        else:
            enriched[lag1_key] = float(prev1[column]) if prev1 and column in prev1 else 0.0
            enriched[lag2_key] = float(prev2[column]) if prev2 and column in prev2 else 0.0

    for column in ROLLING_COLUMNS:
        roll_key = f"{column}_rolling_2"
        if column not in current:
            continue
        if prev1 and column in prev1:
            prev_val = float(prev1[column])
            cur_val = float(current[column])
            enriched[roll_key] = (prev_val + cur_val) / 2.0
        else:
            enriched[roll_key] = 0.0

    snapshot = _snapshot_from_enriched_row(enriched)
    return enriched, snapshot
