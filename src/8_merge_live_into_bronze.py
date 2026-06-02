from pathlib import Path

import boto3
import pandas as pd
import s3fs

from config import (
    BRONZE_LIVE_PATH,
    BUCKET_NAME,
    DATA_DIR,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
)


BRONZE_HIST_OBJECT = "bronze/airquality_data.csv"
LOCAL_MERGED_CSV = DATA_DIR / "airquality_data_merged.csv"


def _create_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )


def _read_historical_frame(client):
    temp_file = Path(__file__).resolve().parents[1] / "data" / "_tmp_hist.csv"
    try:
        client.download_file(BUCKET_NAME, BRONZE_HIST_OBJECT, str(temp_file))
        return pd.read_csv(temp_file)
    except Exception:
        fallback = DATA_DIR / "airquality_data.csv"
        if not fallback.exists():
            raise FileNotFoundError(
                "Không có dữ liệu historical cả trên MinIO lẫn local data/airquality_data.csv"
            )
        return pd.read_csv(fallback)
    finally:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)


def _read_live_frame():
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY,
    )
    live_path = BRONZE_LIVE_PATH.replace("s3a://", "")
    try:
        files = fs.glob(f"{live_path}/**/*.parquet")
        if not files:
            return pd.DataFrame()
        return pd.read_parquet(f"s3://{live_path}", filesystem=fs)
    except Exception:
        return pd.DataFrame()


def merge_live_into_bronze():
    s3_client = _create_s3_client()
    hist_df = _read_historical_frame(s3_client)
    live_df = _read_live_frame()

    if live_df.empty:
        print("[MERGE] Không có dữ liệu live mới, giữ nguyên Bronze historical.")
        return

    if "PM2_5" in live_df.columns and "PM2.5" not in live_df.columns:
        live_df = live_df.rename(columns={"PM2_5": "PM2.5"})

    shared_cols = [c for c in hist_df.columns if c in live_df.columns]
    if not shared_cols:
        print("[MERGE] Live data không cùng schema với historical, bỏ qua merge.")
        return

    merged = pd.concat([hist_df[shared_cols], live_df[shared_cols]], ignore_index=True)
    dedup_keys = [c for c in ["station", "year", "month", "day", "hour"] if c in merged.columns]
    if dedup_keys:
        merged = merged.drop_duplicates(subset=dedup_keys, keep="last")
    else:
        merged = merged.drop_duplicates()

    sort_keys = [c for c in ["station", "year", "month", "day", "hour"] if c in merged.columns]
    if sort_keys:
        merged = merged.sort_values(sort_keys)
    LOCAL_MERGED_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(LOCAL_MERGED_CSV, index=False)

    s3_client.upload_file(str(LOCAL_MERGED_CSV), BUCKET_NAME, BRONZE_HIST_OBJECT)
    print(
        f"[MERGE] Đã gộp live ({len(live_df)}) + historical ({len(hist_df)}) -> "
        f"{len(merged)} dòng vào s3://{BUCKET_NAME}/{BRONZE_HIST_OBJECT}"
    )


if __name__ == "__main__":
    merge_live_into_bronze()
