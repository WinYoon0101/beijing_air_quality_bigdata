from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


TIME_COLUMNS = ["station", "year", "month", "day", "hour"]
HISTORY_BASE_COLUMNS = [
    "No",
    "year",
    "month",
    "day",
    "hour",
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
]
DEFAULT_EXCLUDED_COLUMNS = {
    "No",
    "year",
    "month",
    "day",
    "hour",
    "station",
    "wd",
    "PM2.5",
    "PM2_5",
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
        excluded = {"No", "PM2_5", "PM2.5", "Target_PM2.5_next_1h"}
        return df[[column for column in df.columns if column not in excluded]].select_dtypes(include=["number"]).columns.tolist()

    excluded = {"No", "PM2_5", "PM2.5", "Target_PM2.5_next_1h"}
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


def build_realtime_feature_frame(
    history: pd.DataFrame,
    payload: dict | None,
    window_size: int = 48,
) -> pd.DataFrame:
    data = normalize_pm25_column(history)
    data = data[[column for column in HISTORY_BASE_COLUMNS if column in data.columns]].copy()

    if data.empty:
        raise RuntimeError("Không có dữ liệu Gold lịch sử để dựng cửa sổ realtime")

    payload_data = payload or {}
    station_value = payload_data.get("station")
    if station_value is not None and pd.notna(station_value) and "station" in data.columns:
        station_history = data.loc[data["station"] == station_value].copy()
        if station_history.empty:
            raise RuntimeError(f"Không tìm thấy lịch sử cho station={station_value} để tạo cửa sổ realtime")
    else:
        station_history = data.copy()

    sort_columns = [column for column in TIME_COLUMNS if column in station_history.columns]
    if sort_columns:
        station_history = station_history.sort_values(sort_columns).reset_index(drop=True)

    station_history = station_history.ffill().fillna(0)
    context_size = window_size + 2
    if len(station_history) < context_size - 1:
        raise RuntimeError(f"Cần ít nhất {context_size - 1} bản ghi lịch sử để dựng snapshot {window_size} giờ")

    context_history = station_history.tail(context_size - 1).copy()
    current_row = context_history.iloc[-1].to_dict()

    for column, value in payload_data.items():
        if pd.notna(value):
            current_row[column if column != "PM2.5" else "PM2_5"] = value

    if "PM2.5" in current_row and "PM2_5" not in current_row:
        current_row["PM2_5"] = current_row.pop("PM2.5")

    if all(column in current_row and pd.notna(current_row[column]) for column in ["year", "month", "day", "hour"]):
        current_timestamp = pd.Timestamp(
            year=int(current_row["year"]),
            month=int(current_row["month"]),
            day=int(current_row["day"]),
            hour=int(current_row["hour"]),
        )
    else:
        current_timestamp = None

    if current_timestamp is not None:
        current_row.update(
            {
                "year": current_timestamp.year,
                "month": current_timestamp.month,
                "day": current_timestamp.day,
                "hour": current_timestamp.hour,
            }
        )

    if "No" in current_row and pd.isna(current_row["No"]):
        current_row["No"] = context_history["No"].iloc[-1] + 1 if "No" in context_history.columns else 0

    context = pd.concat([context_history, pd.DataFrame([current_row])], ignore_index=True)
    sort_columns = [column for column in TIME_COLUMNS if column in context.columns]
    if sort_columns:
        context = context.sort_values(sort_columns)
    context = context.drop_duplicates(subset=[column for column in TIME_COLUMNS if column in context.columns], keep="last")
    return context.ffill().fillna(0).reset_index(drop=True)


def add_realtime_feature_history(df: pd.DataFrame) -> pd.DataFrame:
    working = normalize_pm25_column(df)
    working = sort_by_time(working)

    group_columns = "station" if "station" in working.columns else None
    lag_columns = [column for column in working.columns if column not in {"No", "year", "month", "day", "hour"}]
    rolling_columns = [
        column
        for column in working.columns
        if column not in {"No", "station", "wd", "year", "month", "day", "hour", "PM2_5"}
    ]

    if group_columns:
        grouped = working.groupby(group_columns, sort=False)
        for column in lag_columns:
            working[f"{column}_lag_1"] = grouped[column].shift(1)
            working[f"{column}_lag_2"] = grouped[column].shift(2)

        for column in rolling_columns:
            working[f"{column}_rolling_2"] = grouped[column].transform(
                lambda series: series.rolling(window=2, min_periods=2).mean()
            )

        working["cum_wspm"] = grouped["WSPM"].cumsum()
    else:
        for column in lag_columns:
            working[f"{column}_lag_1"] = working[column].shift(1)
            working[f"{column}_lag_2"] = working[column].shift(2)

        for column in rolling_columns:
            working[f"{column}_rolling_2"] = working[column].rolling(window=2, min_periods=2).mean()

        working["cum_wspm"] = working["WSPM"].cumsum()

    temp = pd.to_numeric(working["TEMP"], errors="coerce") if "TEMP" in working.columns else pd.Series(index=working.index, dtype="float64")
    dewp = pd.to_numeric(working["DEWP"], errors="coerce") if "DEWP" in working.columns else pd.Series(index=working.index, dtype="float64")
    if not temp.empty:
        working["saturated_vapor_pressure"] = 61.1 * ((7.5 * temp) / (237.3 + temp))
    if not dewp.empty:
        working["actual_vapor_pressure"] = 61.1 * ((7.5 * dewp) / (237.3 + dewp))
    return working
