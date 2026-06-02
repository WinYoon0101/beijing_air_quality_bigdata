import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


# Cấu hình MinIO (S3)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")
BUCKET_NAME = os.getenv("BUCKET_NAME", "air-quality-lake")


# Đường dẫn Data Lake
BRONZE_PATH = f"s3a://{BUCKET_NAME}/bronze/airquality_data.csv"
SILVER_PATH = f"s3a://{BUCKET_NAME}/silver/cleaned_data.parquet"
GOLD_PATH = f"s3a://{BUCKET_NAME}/gold/features.parquet"


# Đường dẫn model và metadata
MODEL_DIR = PROJECT_ROOT / "src"
MODEL_XGBOOST_PATH = MODEL_DIR / "model_xgboost_pm25.json"
MODEL_LIGHTGBM_PATH = MODEL_DIR / "model_lightgbm_pm25.txt"
MODEL_LSTM_PATH = MODEL_DIR / "model_lstm_pm25.pth"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
MODEL_METRICS_PATH = MODEL_DIR / "model_metrics.json"
