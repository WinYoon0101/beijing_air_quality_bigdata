import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import s3fs
import lightgbm as lgb
import torch
import xgboost as xgb
from cassandra.cluster import Cluster
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from feature_schema import add_next_hour_target, normalize_pm25_column, prepare_inference_frame, resolve_inference_features, sort_by_time

from config import (
    BUCKET_NAME,
    CASSANDRA_FORECAST_TABLE,
    CASSANDRA_HOST,
    CASSANDRA_KEYSPACE,
    CASSANDRA_PORT,
    GOLD_PATH,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MODEL_LIGHTGBM_PATH,
    MODEL_LSTM_PATH,
    MODEL_METADATA_PATH,
    MODEL_XGBOOST_PATH,
    MODEL_METRICS_PATH,
)


app = FastAPI(title="PM2.5 Forecast Dashboard", version="1.1.0")
cluster = None
session = None
DASHBOARD_HTML_PATH = Path(__file__).with_name("dashboard.html")
SUPPORTED_MODELS = ("xgboost", "lightgbm", "lstm")


def _forecast_schema_sql():
    return f"""
        CREATE TABLE IF NOT EXISTS {CASSANDRA_KEYSPACE}.{CASSANDRA_FORECAST_TABLE} (
            station text,
            forecast_timestamp timestamp,
            id UUID,
            observed_timestamp timestamp,
            predicted float,
            model_name text,
            PRIMARY KEY ((station), forecast_timestamp, id)
        ) WITH CLUSTERING ORDER BY (forecast_timestamp DESC, id ASC)
    """


def _get_fs():
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY,
    )


def _load_station_options():
    stations = []
    try:
        fs = _get_fs()
        df = pd.read_parquet(f"{BUCKET_NAME}/gold/features.parquet", filesystem=fs, columns=["station"])
        if "station" in df.columns:
            stations = sorted({str(value) for value in df["station"].dropna().astype(str).tolist()})
    except Exception:
        stations = []

    if not stations and session is not None:
        try:
            rows = session.execute(f"SELECT station FROM {CASSANDRA_KEYSPACE}.{CASSANDRA_FORECAST_TABLE} LIMIT 1000")
            stations = sorted({str(row.station) for row in rows if row.station is not None})
        except Exception:
            stations = []

    return stations


def _load_metrics_payload():
    metrics_path = Path(MODEL_METRICS_PATH)
    if not metrics_path.exists():
        return {"generated_at": None, "models": []}

    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {"generated_at": None, "models": []}


def _load_metadata():
    metadata_path = Path(MODEL_METADATA_PATH)
    if not metadata_path.exists():
        return {}

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_gold_frame():
    fs = _get_fs()
    df = pd.read_parquet(f"{BUCKET_NAME}/gold/features.parquet", filesystem=fs)
    return sort_by_time(normalize_pm25_column(df))


def _select_station(frame, station):
    if "station" not in frame.columns:
        return None, frame.copy()

    stations = [str(value) for value in frame["station"].dropna().astype(str).tolist()]
    if not stations:
        return None, frame.copy()

    resolved_station = station.strip() if station else stations[0]
    if resolved_station not in stations:
        resolved_station = stations[0]

    station_frame = frame.loc[frame["station"].astype(str) == resolved_station].copy()
    return resolved_station, sort_by_time(station_frame)


def _timestamp_labels(frame):
    if all(column in frame.columns for column in ["year", "month", "day", "hour"]):
        return [
            pd.Timestamp(year=int(row.year), month=int(row.month), day=int(row.day), hour=int(row.hour)).isoformat()
            for row in frame[["year", "month", "day", "hour"]].itertuples(index=False)
        ]
    return [str(index) for index in range(len(frame))]


def _load_model(model_name):
    if model_name == "xgboost":
        booster = xgb.Booster()
        booster.load_model(str(MODEL_XGBOOST_PATH))
        return booster

    if model_name == "lightgbm":
        return lgb.Booster(model_file=str(MODEL_LIGHTGBM_PATH))

    return torch.load(str(MODEL_LSTM_PATH), map_location="cpu")


def _predict_model(model_name, frame, metadata, target_col):
    if model_name == "xgboost":
        model = _load_model(model_name)
        feature_names = resolve_inference_features(frame, metadata, model_name)
        inference_frame = prepare_inference_frame(frame, feature_names, model_name)
        return model.predict(xgb.DMatrix(inference_frame))

    if model_name == "lightgbm":
        model = _load_model(model_name)
        feature_names = resolve_inference_features(frame, metadata, model_name)
        inference_frame = prepare_inference_frame(frame, feature_names, model_name)
        return model.predict(inference_frame)

    checkpoint = _load_model(model_name)
    feature_cols = checkpoint["feature_cols"]
    seq_len = checkpoint.get("seq_length", 48)

    working = frame.copy()
    for column in feature_cols:
        if column not in working.columns:
            working[column] = 0

    data = working[feature_cols].values
    if len(data) <= seq_len:
        return None

    x_lstm = [data[index : index + seq_len] for index in range(len(data) - seq_len)]

    class LSTMModel(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=3, target_size=1):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
            self.fc = torch.nn.Linear(hidden_size * 2, target_size)

        def forward(self, x):
            _, (h_out, _) = self.lstm(x)
            out = torch.cat((h_out[-2, :, :], h_out[-1, :, :]), dim=1)
            return self.fc(out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMModel(
        checkpoint["input_size"],
        checkpoint["hidden_size"],
        num_layers=checkpoint.get("num_layers", 3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(x_lstm).float().to(device)
        return model(x_tensor).squeeze().cpu().numpy()


def _comparison_payload(station=None, test_ratio=0.2, max_points=180):
    frame = _load_gold_frame()
    resolved_station, station_frame = _select_station(frame, station)
    if not resolved_station:
        raise HTTPException(status_code=404, detail="No station found")

    metadata = _load_metadata()
    target_col = metadata.get("target_col", "Target_PM2.5_next_1h")
    station_frame = add_next_hour_target(station_frame, target_col).ffill().fillna(0)
    station_frame = station_frame.dropna(subset=[target_col]).reset_index(drop=True)
    if len(station_frame) < 10:
        raise HTTPException(status_code=404, detail="Not enough data for comparison")

    split_idx = max(int(len(station_frame) * (1 - test_ratio)), 1)
    eval_frame = station_frame.iloc[split_idx:].copy().reset_index(drop=True)
    eval_frame = eval_frame.ffill().fillna(0)
    timestamps = _timestamp_labels(eval_frame)

    models = []
    for model_name in SUPPORTED_MODELS:
        try:
            predictions = _predict_model(model_name, eval_frame.copy(), metadata, target_col)
            if predictions is None or len(predictions) == 0:
                continue

            actual = eval_frame[target_col].values
            common_len = min(len(actual), len(predictions), max_points)
            actual_tail = actual[-common_len:]
            predicted_tail = predictions[-common_len:]
            residuals = (actual_tail - predicted_tail).tolist()
            if model_name == "xgboost":
                model_label = "XGBoost"
            elif model_name == "lightgbm":
                model_label = "LightGBM"
            else:
                model_label = "LSTM"
            models.append({
                "Model": model_label,
                "MAE": float(abs(actual_tail - predicted_tail).mean()),
                "RMSE": float(math.sqrt(((actual_tail - predicted_tail) ** 2).mean())),
                "R2 Score": float(1 - (((actual_tail - predicted_tail) ** 2).sum() / (((actual_tail - actual_tail.mean()) ** 2).sum() or 1))),
                "scatter": {"actual": actual_tail.tolist(), "predicted": predicted_tail.tolist()},
                "residuals": residuals,
                "timeline": {
                    "timestamps": timestamps[-common_len:],
                    "actual": actual_tail.tolist(),
                    "predicted": predicted_tail.tolist(),
                },
            })
        except Exception:
            continue

    if not models:
        raise HTTPException(status_code=404, detail="No model available for comparison")

    models = sorted(models, key=lambda item: item["RMSE"])
    common_len = min(len(model["timeline"]["actual"]) for model in models)
    timeline_payload = {
        "timestamps": models[0]["timeline"]["timestamps"][-common_len:],
        "actual": models[0]["timeline"]["actual"][-common_len:],
        "models": {model["Model"]: model["timeline"]["predicted"][-common_len:] for model in models},
    }

    for model in models:
        model["timeline"]["timestamps"] = model["timeline"]["timestamps"][-common_len:]
        model["timeline"]["actual"] = model["timeline"]["actual"][-common_len:]
        model["timeline"]["predicted"] = model["timeline"]["predicted"][-common_len:]

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "station": resolved_station,
        "target_col": target_col,
        "models": models,
        "timeline": timeline_payload,
    }


@app.on_event("startup")
def startup():
    global cluster, session
    cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT)
    session = cluster.connect()
    session.execute(
        f"""
        CREATE KEYSPACE IF NOT EXISTS {CASSANDRA_KEYSPACE}
        WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '1'}}
        """
    )
    session.execute(_forecast_schema_sql())


@app.on_event("shutdown")
def shutdown():
    global cluster
    if cluster is not None:
        cluster.shutdown()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if not DASHBOARD_HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="Dashboard UI not found")
    return DASHBOARD_HTML_PATH.read_text(encoding="utf-8")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_alias():
    return dashboard()


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/stations")
def stations():
    return {"stations": _load_station_options()}


@app.get("/forecasts/latest")
def latest_forecast_alias(station: str = Query(default="")):
    return latest_forecast(station=station)


@app.get("/api/latest-forecast")
def latest_forecast(station: str = Query(default="")):
    stations = _load_station_options()
    resolved_station = station.strip() if station else (stations[0] if stations else "")
    if not resolved_station:
        raise HTTPException(status_code=404, detail="No station found")

    try:
        row = session.execute(
            f"""
            SELECT station, forecast_timestamp, observed_timestamp, predicted, model_name
            FROM {CASSANDRA_KEYSPACE}.{CASSANDRA_FORECAST_TABLE}
            WHERE station = %s
            LIMIT 1
            """,
            [resolved_station],
        ).one()
        if row is None:
            raise HTTPException(status_code=404, detail="No forecast found for the requested station")
        return {
            "station": row.station,
            "forecast_timestamp": row.forecast_timestamp.isoformat() if row.forecast_timestamp else None,
            "observed_timestamp": row.observed_timestamp.isoformat() if row.observed_timestamp else None,
            "predicted": row.predicted,
            "model_name": row.model_name,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/forecasts/history")
def forecast_history_alias(station: str = Query(default=""), limit: int = Query(default=24, ge=1, le=500)):
    return forecast_history(station=station, limit=limit)


@app.get("/api/forecast-history")
def forecast_history(station: str = Query(default=""), limit: int = Query(default=24, ge=1, le=500)):
    stations = _load_station_options()
    resolved_station = station.strip() if station else (stations[0] if stations else "")
    if not resolved_station:
        raise HTTPException(status_code=404, detail="No station found")

    try:
        rows = session.execute(
            f"""
            SELECT station, forecast_timestamp, observed_timestamp, predicted, model_name
            FROM {CASSANDRA_KEYSPACE}.{CASSANDRA_FORECAST_TABLE}
            WHERE station = %s
            LIMIT {limit}
            """,
            [resolved_station],
        )
        return [
            {
                "station": row.station,
                "forecast_timestamp": row.forecast_timestamp.isoformat() if row.forecast_timestamp else None,
                "observed_timestamp": row.observed_timestamp.isoformat() if row.observed_timestamp else None,
                "predicted": row.predicted,
                "model_name": row.model_name,
            }
            for row in rows
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model-comparison")
def model_comparison(station: str = Query(default="")):
    try:
        return _comparison_payload(station=station)
    except HTTPException:
        raise
    except Exception as exc:
        fallback = _load_metrics_payload()
        if fallback.get("models"):
            return fallback
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
