import json
import warnings
from pathlib import Path

import pandas as pd
import s3fs
import xgboost as xgb
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score as r2
from sklearn.model_selection import train_test_split

from feature_schema import add_next_hour_target, select_numeric_features
from config import (
    BUCKET_NAME,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MODEL_METADATA_PATH,
    MODEL_XGBOOST_PATH,
)

warnings.filterwarnings('ignore')


def _save_metadata(feature_columns, target_col):
    metadata = {
        "model_name": "xgboost",
        "target_col": target_col,
        "feature_columns": feature_columns,
        "model_path": str(MODEL_XGBOOST_PATH.name),
    }
    Path(MODEL_METADATA_PATH).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_training_frame(df):
    target_col = "Target_PM2.5_next_1h"
    data = add_next_hour_target(df, target_col)
    data = data.dropna(subset=[target_col]).reset_index(drop=True)

    return data, target_col


def train_xgboost():
    print("🚀 [XGBOOST] Bắt đầu huấn luyện mô hình FINAL (Production Mode)...")

    # 1. Kết nối MinIO
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY,
    )

    path = f"{BUCKET_NAME}/gold/features.parquet"
    print(f"📂 Đang đọc dữ liệu từ: {path}")
    df = pd.read_parquet(path, filesystem=fs)

    # 2. Xử lý đặc trưng
    data, target_col = _prepare_training_frame(df)

    features_columns = select_numeric_features(data, target_col)

    if not features_columns:
        raise RuntimeError("Không tìm thấy cột số nào trong Gold layer để huấn luyện XGBoost")

    # 3. Chia dữ liệu (90/10)
    train_data, test = train_test_split(data, test_size=0.1, shuffle=False)
    
    print("\n Áp dụng BỘ THAM SỐ TỐI ƯU đã tìm được")
    best_p = {
        'learning_rate': 0.03247404578069104,
        'max_depth': 6,
        'min_child_weight': 59,
        'reg_alpha': 0.015530594579354432,
        'reg_lambda': 0.004702051943399315,
        'objective': 'reg:gamma',
        'eval_metric': 'rmse',
        'tree_method': 'hist', # Tối ưu hóa chạy trên CPU Local (rất nhẹ)
        'seed': 42,
    }

    print("\n⏳ Đang huấn luyện mô hình FINAL duy nhất...")
    # Tách Validation ra từ Train (10% của tập Train) để giám sát Early Stopping
    train_final, val_final = train_test_split(train_data, test_size=0.1, shuffle=False)

    dtrain_final = xgb.DMatrix(train_final[features_columns], label=train_final[target_col])