from pathlib import Path

import boto3

from config import (
    BUCKET_NAME,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
HISTORICAL_CSV = DATA_DIR / "airquality_data.csv"
BRONZE_HIST_OBJECT = "bronze/airquality_data.csv"


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


def upload_to_datalake():
    print("🔄 [INGESTION] Đang đẩy dữ liệu lịch sử lên MinIO Bronze...")
    if not HISTORICAL_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy file lịch sử: {HISTORICAL_CSV}")

    s3 = _create_client()
    _ensure_bucket(s3)
    s3.upload_file(str(HISTORICAL_CSV), BUCKET_NAME, BRONZE_HIST_OBJECT)
    print(f"✅ Đã tải lên Data Lake thành công: s3://{BUCKET_NAME}/{BRONZE_HIST_OBJECT}")


if __name__ == "__main__":
    upload_to_datalake()
