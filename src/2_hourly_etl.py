import json

import pandas as pd
import s3fs

from config import BUCKET_NAME, GOLD_PATH, GOLD_RT_PATH, MINIO_ACCESS_KEY, MINIO_ENDPOINT, MINIO_SECRET_KEY, BRONZE_RT_PATH


def _read_realtime_payload(fs):
    try:
        realtime_path = BRONZE_RT_PATH.replace("s3a://", "")
        with fs.open(realtime_path, "r") as handle:
            return json.load(handle)
    except Exception:
        return None


def run_hourly_etl():
    print("🔄 [HOURLY ETL] Đang tạo snapshot features mới nhất...")
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY,
    )

    historical_path = f"{BUCKET_NAME}/gold/features.parquet"
    df = pd.read_parquet(historical_path, filesystem=fs)
    latest = df.tail(1).copy().reset_index(drop=True)

    payload = _read_realtime_payload(fs)
    if isinstance(payload, dict):
        for column in ["station", "wd", "year", "month", "day", "hour", "PM2_5", "PM2.5"]:
            if column in payload:
                target_column = "PM2_5" if column == "PM2.5" else column
                latest.loc[0, target_column] = payload[column]

    output_path = GOLD_RT_PATH.replace("s3a://", "")
    latest.to_parquet(output_path, filesystem=fs, index=False)
    print(f"✅ [HOURLY ETL] Đã lưu snapshot tại {GOLD_RT_PATH}")


if __name__ == "__main__":
    run_hourly_etl()
