import json
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
import requests

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

    if ingest_mode == "api":
        print("🔄 [INGESTION] Đang ghi payload realtime vào Bronze...")
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