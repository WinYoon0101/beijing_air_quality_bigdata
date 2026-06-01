import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
import requests

from feature_schema import normalize_pm25_column, sort_by_time
from config import (
    AQI_API_URL,
    BUCKET_NAME,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    RAW_DIR,
    WEATHER_API_URL,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
HISTORICAL_CSV = DATA_DIR / "airquality_data.csv"
BRONZE_HIST_OBJECT = "bronze/airquality_data.csv"
BRONZE_RT_OBJECT = "bronze/realtime/latest.json"
MOCK_MODE_NAMES = {"mock", "api_mock", "api-mock"}


def _create_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


def _ensure_bucket(s3):
    bucket_names = [bucket["Name"] for bucket in s3.list_buckets().get("Buckets", [])]
    if BUCKET_NAME not in bucket_names:
        s3.create_bucket(Bucket=BUCKET_NAME)


def _latest_raw_sample():
    raw_files = sorted(RAW_DIR.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong {RAW_DIR}")

    frames = [pd.read_csv(path) for path in raw_files]
    merged = pd.concat(frames, axis=0, ignore_index=True)
    return merged.tail(1).iloc[0].to_dict()


def _load_mock_source_frame():
    if HISTORICAL_CSV.exists():
        return pd.read_csv(HISTORICAL_CSV)

    raw_files = sorted(RAW_DIR.glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong {RAW_DIR}")

    frames = [pd.read_csv(path) for path in raw_files]
    return pd.concat(frames, axis=0, ignore_index=True)


def _build_mock_realtime_payload(requested_station=None):
    frame = normalize_pm25_column(_load_mock_source_frame())
    frame = sort_by_time(frame)

    if "station" in frame.columns:
        stations = sorted({str(value) for value in frame["station"].dropna().astype(str).tolist()})
    else:
        stations = []

    if not stations:
        sample = _latest_raw_sample()
        sample["source"] = "mock_raw_sample"
        sample["ingested_at"] = datetime.now(timezone.utc).isoformat()
        return sample

    resolved_station = requested_station if requested_station in stations else stations[0]
    station_frame = frame.loc[frame["station"].astype(str) == str(resolved_station)].copy()
    if station_frame.empty:
        station_frame = frame.copy()
        resolved_station = str(station_frame["station"].dropna().astype(str).iloc[-1]) if "station" in station_frame.columns and not station_frame["station"].dropna().empty else resolved_station

    latest_row = station_frame.tail(1).iloc[0].to_dict()
    rng = random.Random(f"{resolved_station}-{latest_row.get('year', '')}-{latest_row.get('month', '')}-{latest_row.get('day', '')}-{latest_row.get('hour', '')}")

    payload = {
        "source": "mock_api",
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "station": resolved_station,
        "year": int(latest_row.get("year", datetime.now().year)),
        "month": int(latest_row.get("month", datetime.now().month)),
        "day": int(latest_row.get("day", datetime.now().day)),
        "hour": int((latest_row.get("hour", datetime.now().hour) + 1) % 24),
        "No": int(latest_row.get("No", 0)) + 1 if pd.notna(latest_row.get("No", 0)) else 1,
        "PM2_5": float(latest_row.get("PM2_5", latest_row.get("PM2.5", 0.0)) or 0.0) * (1 + rng.uniform(-0.04, 0.06)),
    }

    weather_fields = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    air_quality_fields = ["PM2_5", "PM10", "SO2", "NO2", "CO", "O3"]
    weather_payload = {}
    air_quality_payload = {}

    for field in weather_fields:
        if field in latest_row:
            value = pd.to_numeric(latest_row.get(field), errors="coerce")
            if pd.notna(value):
                jitter = rng.uniform(-0.8, 0.8) if field != "RAIN" else max(0.0, rng.uniform(-0.2, 0.6))
                weather_payload[field] = round(float(value + jitter), 3)

    for field in air_quality_fields:
        if field in latest_row:
            value = pd.to_numeric(latest_row.get(field), errors="coerce")
            if pd.notna(value):
                jitter = rng.uniform(-1.5, 1.8)
                adjusted = max(0.0, float(value) + jitter)
                air_quality_payload[field] = round(adjusted, 3)

    if weather_payload:
        payload["weather"] = weather_payload
    if air_quality_payload:
        payload["air_quality"] = air_quality_payload

    for field in ["wd"]:
        if field in latest_row and pd.notna(latest_row.get(field)):
            payload[field] = str(latest_row.get(field))

    return payload


def _fetch_json(url):
    if not url:
        return None
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.json()


def _build_realtime_payload():
    payload = {
        "source": "api",
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    weather = _fetch_json(WEATHER_API_URL)
    aqi = _fetch_json(AQI_API_URL)

    if weather:
        payload["weather"] = weather
    if aqi:
        payload["air_quality"] = aqi

    if not weather and not aqi:
        payload["source"] = "fallback_raw_sample"
        payload.update(_latest_raw_sample())

    return payload


def upload_to_datalake(mode=None):
    ingest_mode = (mode or os.getenv("INGEST_MODE", "historical")).lower()
    s3 = _create_client()
    _ensure_bucket(s3)

    if ingest_mode in {"api", "api_mock", "api-mock", "mock"}:
        print("🔄 [INGESTION] Đang ghi payload realtime vào Bronze...")
        if ingest_mode in MOCK_MODE_NAMES:
            requested_station = os.getenv("MOCK_STATION", "").strip() or None
            seed_value = os.getenv("MOCK_SEED", "").strip()
            seed = int(seed_value) if seed_value.isdigit() else None
            payload = _build_mock_realtime_payload(requested_station=requested_station, seed=seed)
        else:
            payload = _build_realtime_payload()
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=BRONZE_RT_OBJECT,
            Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"✅ Đã lưu payload realtime tại s3://{BUCKET_NAME}/{BRONZE_RT_OBJECT}")
        return

    print("🔄 [INGESTION] Đang đẩy dữ liệu lịch sử lên MinIO Bronze...")
    if not HISTORICAL_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy file lịch sử: {HISTORICAL_CSV}")

    s3.upload_file(str(HISTORICAL_CSV), BUCKET_NAME, BRONZE_HIST_OBJECT)
    print(f"✅ Đã tải lên Data Lake thành công: s3://{BUCKET_NAME}/{BRONZE_HIST_OBJECT}")


if __name__ == "__main__":
    upload_to_datalake()