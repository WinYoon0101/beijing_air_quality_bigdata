import json

import pandas as pd
import s3fs

from feature_schema import add_realtime_feature_history, build_realtime_feature_frame, normalize_pm25_column
from config import BRONZE_RT_PATH, GOLD_PATH, GOLD_RT_PATH, MINIO_ACCESS_KEY, MINIO_ENDPOINT, MINIO_SECRET_KEY


WINDOW_SIZE = 48


def _read_parquet(fs, path):
    parquet_path = path.replace("s3a://", "")
    return pd.read_parquet(parquet_path, filesystem=fs)


def _read_realtime_payload(fs):
    try:
        realtime_path = BRONZE_RT_PATH.replace("s3a://", "")
        with fs.open(realtime_path, "r") as handle:
            return json.load(handle)
    except Exception:
        return None


def _flatten_payload(payload):
    if not isinstance(payload, dict):
        return {}

    flattened = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            flattened.update({nested_key: nested_value for nested_key, nested_value in value.items() if not isinstance(nested_value, dict)})

    for key, value in payload.items():
        if not isinstance(value, dict):
            flattened[key] = value

    return flattened


def run_hourly_etl():
    print("🔄 [HOURLY ETL] Đang tạo snapshot features với cửa sổ 48 giờ gần nhất...")
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY,
    )

    historical_df = normalize_pm25_column(_read_parquet(fs, GOLD_PATH))
    payload = _read_realtime_payload(fs)
    feature_frame = build_realtime_feature_frame(historical_df, _flatten_payload(payload), window_size=WINDOW_SIZE)
    feature_frame = add_realtime_feature_history(feature_frame).dropna().tail(WINDOW_SIZE).reset_index(drop=True)

    if len(feature_frame) < WINDOW_SIZE:
        raise RuntimeError(f"Không thể tạo đủ {WINDOW_SIZE} hàng feature cho snapshot realtime")

    output_path = GOLD_RT_PATH.replace("s3a://", "")
    feature_frame.to_parquet(output_path, filesystem=fs, index=False)
    print(f"✅ [HOURLY ETL] Đã lưu snapshot {WINDOW_SIZE} giờ tại {GOLD_RT_PATH}")


if __name__ == "__main__":
    run_hourly_etl()
