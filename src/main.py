import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import s3fs
from cassandra.cluster import Cluster
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

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
    MODEL_METRICS_PATH,
)


app = FastAPI(title="PM2.5 Forecast Dashboard", version="1.1.0")
cluster = None
session = None
DASHBOARD_HTML_PATH = Path(__file__).with_name("dashboard.html")


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
def model_comparison():
    return _load_metrics_payload()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
