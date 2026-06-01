import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score as r2

from feature_schema import add_next_hour_target
from config import (
    BUCKET_NAME,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MODEL_METADATA_PATH,
    MODEL_LIGHTGBM_PATH,
)


warnings.filterwarnings('ignore')


def _save_metadata(feature_columns, target_col):
    metadata = {
        "model_name": "lightgbm",
        "target_col": target_col,
        "feature_columns": feature_columns,
        "model_path": str(MODEL_LIGHTGBM_PATH.name),
    }
    Path(MODEL_METADATA_PATH).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_training_frame(df):
    target_col = "Target_PM2.5_next_1h"
    data = add_next_hour_target(df, target_col)
    data = data.dropna(subset=[target_col]).reset_index(drop=True)

    return data, target_col


def train_lightgbm():
    print("🚀 [LIGHTGBM] Bắt đầu huấn luyện mô hình FINAL (Production Mode)...")

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

    for column in ['wd', 'station']:
        if column in data.columns:
            data[column] = data[column].astype('category')

    excluded_columns = {'No', target_col, 'year', 'month', 'day', 'hour', 'PM2.5', 'PM2_5'}
    features_columns = [col for col in data.columns if col not in excluded_columns]

    if not features_columns:
        raise RuntimeError("Không tìm thấy cột nào trong Gold layer để huấn luyện LightGBM")

    # 3. Chia dữ liệu (90/10)
    train_data, test = train_test_split(data, test_size=0.1, shuffle=False)
    
    print("\n🏆 Áp dụng BỘ THAM SỐ VÀNG từ Kaggle...")
    best_p = {
        'learning_rate': 0.03900319833456263,
        'max_depth': 9,
        'num_leaves': 34,
        'min_child_samples': 94,
        'reg_alpha': 2.0747102853705024,
        'reg_lambda': 0.0014561946328301816,
        'objective': 'gamma',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device_type': 'cpu',  # Chạy local an toàn hơn trên CPU
        'verbose': -1,
        'seed': 42,
    }

    print("\n⏳ Đang huấn luyện mô hình FINAL duy nhất...")
    # Tách Validation ra từ Train (10% của tập Train) để giám sát Early Stopping
    train_final, val_final = train_test_split(train_data, test_size=0.1, shuffle=False)

    dtrain_final = lgb.Dataset(train_final[features_columns], label=train_final[target_col])
    dval_final   = lgb.Dataset(val_final[features_columns], label=val_final[target_col], reference=dtrain_final)

    # 4. Huấn luyện (Early Stopping = 50)
    final_model = lgb.train(
        best_p,
        dtrain_final,
        valid_sets=[dtrain_final, dval_final],
        valid_names=['train', 'valid'],
        num_boost_round=1500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    # 5. Đánh giá tập Test
    preds = final_model.predict(test[features_columns])

    rmse_score = np.sqrt(mse(test[target_col], preds))
    mae_score = mae(test[target_col], preds)
    r2_score = r2(test[target_col], preds)

    print("\n" + "=" * 55)
    print("🏆 BÁO CÁO KẾT QUẢ ĐỒ ÁN (LIGHTGBM - LOCAL PIPELINE)")
    print("=" * 55)
    print(f"📊 RMSE : {rmse_score:.4f}")
    print(f"📊 MAE  : {mae_score:.4f}")
    print(f"📊 R2   : {r2_score:.4f}")
    print("-" * 55)

    # 6. Lưu mô hình và Metadata
    MODEL_LIGHTGBM_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(MODEL_LIGHTGBM_PATH))
    _save_metadata(features_columns, target_col)
    
    print(f"✅ Đã huấn luyện xong và lưu model tại: {MODEL_LIGHTGBM_PATH.parent}")


if __name__ == "__main__":
    train_lightgbm()