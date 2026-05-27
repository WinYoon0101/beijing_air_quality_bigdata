import pandas as pd
import xgboost as xgb
import s3fs
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import (
    BUCKET_NAME,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MODEL_METADATA_PATH,
    MODEL_XGBOOST_PATH,
)


def _save_metadata(feature_columns, target_col):
    metadata = {
        "model_name": "xgboost",
        "target_col": target_col,
        "feature_columns": feature_columns,
        "model_path": str(MODEL_XGBOOST_PATH.name),
    }
    Path(MODEL_METADATA_PATH).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

def train_xgboost():
    print("🚀 [XGBOOST] Bắt đầu huấn luyện...")
    
    # 1. Kết nối MinIO qua s3fs
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': MINIO_ENDPOINT}, 
        key=MINIO_ACCESS_KEY, 
        secret=MINIO_SECRET_KEY
    )
    
    # 2. SỬA LỖI: Bỏ tiền tố 's3://' trong đường dẫn khi dùng kèm filesystem=fs
    # Thay vì: f"s3://{BUCKET_NAME}/gold/features.parquet"
    # Hãy dùng: f"{BUCKET_NAME}/gold/features.parquet"
    path = f"{BUCKET_NAME}/gold/features.parquet"
    
    print(f"📂 Đang đọc dữ liệu từ: {path}")
    df = pd.read_parquet(path, filesystem=fs)
    df = df.ffill().fillna(0)
    
    # 3. Đổi tên cột mục tiêu (PM2_5 thay vì PM2.5 như đã sửa ở bước Spark)
    target_col = "PM2_5" 
    
    # Kiểm tra xem cột có tồn tại không để tránh lỗi
    if target_col not in df.columns:
        print(f"❌ Lỗi: Không tìm thấy cột {target_col}. Các cột hiện có: {df.columns.tolist()}")
        return

    features_columns = [col for col in df.columns if col not in ['No', 'station', 'wd', target_col]]
    features_columns = df[features_columns].select_dtypes(include=['number']).columns.tolist()
    
    # 4. Chia tập dữ liệu
    train_data, test = train_test_split(df, test_size=0.1, shuffle=False)
    train, valid = train_test_split(train_data, test_size=0.1, shuffle=False)
    
    # Chuẩn bị dữ liệu cho XGBoost
    d_train = xgb.DMatrix(train[features_columns], label=train[target_col])
    d_val = xgb.DMatrix(valid[features_columns], label=valid[target_col])
    
    XGB_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': ["mae", "rmse"], 
        'learning_rate': 0.05, 
        'max_depth': 6, 
        'tree_method': 'hist', 
        'seed': 42
    }
    
    # 5. Huấn luyện
    model = xgb.train(
        XGB_PARAMS, 
        d_train, 
        evals=[(d_val, "validation")], 
        num_boost_round=500, 
        verbose_eval=50, 
        early_stopping_rounds=20
    )
    
    model.save_model(str(MODEL_XGBOOST_PATH))
    _save_metadata(features_columns, target_col)
    print(f"✅ Đã huấn luyện xong và lưu model: {MODEL_XGBOOST_PATH.name}")

if __name__ == "__main__":
    train_xgboost()