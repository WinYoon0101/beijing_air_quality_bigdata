"""Dữ liệu đánh giá model dùng chung cho dashboard và script visualize."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import s3fs
import torch
import torch.nn as nn
import xgboost as xgb
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import (
    BUCKET_NAME,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MODEL_LIGHTGBM_PATH,
    MODEL_LSTM_PATH,
    MODEL_METRICS_PATH,
    MODEL_XGBOOST_PATH,
)
from feature_schema import (
    add_next_hour_target,
    load_lightgbm_booster,
    normalize_pm25_column,
    prepare_lightgbm_frame,
    sort_by_time,
)

_LIGHTGBM_BOOSTER_CACHE = None
_LIGHTGBM_MODEL_MTIME = None


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=3, target_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, target_size)

    def forward(self, x):
        _, (h_out, _) = self.lstm(x)
        out = torch.cat((h_out[-2, :, :], h_out[-1, :, :]), dim=1)
        return self.fc(out)


def _get_fs():
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY,
    )


def load_gold_frame():
    fs = _get_fs()
    df = pd.read_parquet(f"{BUCKET_NAME}/gold/features.parquet", filesystem=fs)
    return sort_by_time(normalize_pm25_column(df))


def _get_cached_lightgbm_booster():
    global _LIGHTGBM_BOOSTER_CACHE, _LIGHTGBM_MODEL_MTIME
    if not Path(MODEL_LIGHTGBM_PATH).exists():
        return None
    current_mtime = MODEL_LIGHTGBM_PATH.stat().st_mtime
    if _LIGHTGBM_BOOSTER_CACHE is None or _LIGHTGBM_MODEL_MTIME != current_mtime:
        _LIGHTGBM_BOOSTER_CACHE = load_lightgbm_booster(MODEL_LIGHTGBM_PATH)
        _LIGHTGBM_MODEL_MTIME = current_mtime
    return _LIGHTGBM_BOOSTER_CACHE


def _predict_lightgbm(df_input):
    model = _get_cached_lightgbm_booster()
    if model is None:
        return None
    expected_features = list(model.feature_name())
    frame = prepare_lightgbm_frame(df_input, expected_features, booster=model)
    return model.predict(frame)


def _predict_xgboost(df_input):
    if not Path(MODEL_XGBOOST_PATH).exists():
        return None
    model = xgb.Booster()
    model.load_model(str(MODEL_XGBOOST_PATH))
    expected_features = model.feature_names
    df_input = df_input.reindex(columns=expected_features, fill_value=0)
    df_numeric = df_input.select_dtypes(include=[np.number])
    return model.predict(xgb.DMatrix(df_numeric))


def _predict_lstm(df_input):
    if not Path(MODEL_LSTM_PATH).exists():
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(str(MODEL_LSTM_PATH), map_location=device)
    scaler_x_path = Path(MODEL_LSTM_PATH).parent / "model_lstm_pm25_scaler_X.joblib"
    scaler_y_path = Path(MODEL_LSTM_PATH).parent / "model_lstm_pm25_scaler_y.joblib"
    if not scaler_x_path.exists():
        return None
    scaler_x, scaler_y = load(str(scaler_x_path)), load(str(scaler_y_path))
    saved_cols = checkpoint["feature_cols"]
    seq_len = checkpoint.get("seq_length", 48)
    df_seq = df_input.reindex(columns=saved_cols, fill_value=0)
    data_scaled = scaler_x.transform(df_seq)
    if len(data_scaled) <= seq_len:
        return None
    x_seq = np.array([data_scaled[i : i + seq_len] for i in range(len(data_scaled) - seq_len)])
    model = LSTMModel(
        checkpoint["input_size"],
        checkpoint["hidden_size"],
        num_layers=checkpoint.get("num_layers", 3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(x_seq).float().to(device)).cpu().numpy()
    return scaler_y.inverse_transform(preds).flatten()


PREDICTORS = {
    "XGBoost": _predict_xgboost,
    "LightGBM": _predict_lightgbm,
    "LSTM": _predict_lstm,
}


def _timestamp_labels(frame):
    if all(column in frame.columns for column in ["year", "month", "day", "hour"]):
        return [
            pd.Timestamp(
                year=int(row.year),
                month=int(row.month),
                day=int(row.day),
                hour=int(row.hour),
            ).isoformat()
            for row in frame[["year", "month", "day", "hour"]].itertuples(index=False)
        ]
    return [str(index) for index in range(len(frame))]


def _model_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2 Score": float(r2_score(y_true, y_pred)),
    }


def build_evaluation_payload(
    station: str | None = None,
    test_ratio: float = 0.1,
    line_points: int = 150,
    max_scatter_points: int = 800,
):
    frame = load_gold_frame()
    target_col = "Target_PM2.5_next_1h"
    frame = add_next_hour_target(frame, target_col).ffill().fillna(0)

    resolved_station = None
    if station and station.strip().lower() not in {"", "all", "__all__"}:
        station_value = station.strip()
        if "station" in frame.columns:
            station_frame = frame.loc[frame["station"].astype(str) == station_value].copy()
            if station_frame.empty:
                raise ValueError(f"Không tìm thấy station={station_value}")
            frame = sort_by_time(station_frame)
            resolved_station = station_value

    _, test_df = train_test_split(frame, test_size=test_ratio, shuffle=False)
    test_df = test_df.reset_index(drop=True)

    preds_dict = {name: fn(test_df.copy()) for name, fn in PREDICTORS.items()}
    available = {k: v for k, v in preds_dict.items() if v is not None}
    if not available:
        raise RuntimeError("Không có model nào trả về kết quả dự báo.")

    min_len = min(len(v) for v in available.values())
    y_true = test_df[target_col].values[-min_len:]
    timestamps = _timestamp_labels(test_df.iloc[-min_len:])

    models = []
    timeline_models = {}
    for name, pred in available.items():
        pred_aligned = pred[-min_len:]
        metrics = _model_metrics(y_true, pred_aligned)
        scatter_len = min(len(y_true), max_scatter_points)
        models.append(
            {
                "Model": name,
                **metrics,
                "scatter": {
                    "actual": y_true[-scatter_len:].tolist(),
                    "predicted": pred_aligned[-scatter_len:].tolist(),
                },
                "residuals": (y_true - pred_aligned).tolist(),
            }
        )
        timeline_models[name] = pred_aligned.tolist()

    models = sorted(models, key=lambda row: row["RMSE"])
    line_tail = min(line_points, min_len)
    line_start = min_len - line_tail

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "station": resolved_station or "Tất cả trạm (test set)",
        "target_col": target_col,
        "sample_size": int(len(y_true)),
        "models": models,
        "timeline": {
            "timestamps": timestamps[line_start:],
            "actual": y_true[line_start:].tolist(),
            "models": {name: values[line_start:] for name, values in timeline_models.items()},
        },
    }


def load_metrics_file():
    metrics_path = Path(MODEL_METRICS_PATH)
    if not metrics_path.exists():
        return {"generated_at": None, "models": []}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {"generated_at": None, "models": []}
