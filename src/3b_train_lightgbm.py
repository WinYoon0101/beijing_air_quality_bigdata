import pandas as pd
import lightgbm as lgb
import s3fs
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import (
    BUCKET_NAME,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MODEL_LIGHTGBM_PATH,
    MODEL_METADATA_PATH,
)


def _save_metadata(feature_columns, target_col, categorical_columns):
    metadata = {
        "model_name": "lightgbm",
        "target_col": target_col,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "model_path": str(MODEL_LIGHTGBM_PATH.name),
    }
    Path(MODEL_METADATA_PATH).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

def train_lightgbm():
    print("🚀 [LIGHTGBM] Bắt đầu huấn luyện...")
    
    # 1. Kết nối MinIO
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': MINIO_ENDPOINT}, key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY)
    
    # 2. Đọc dữ liệu (Bỏ s3:// để tránh lỗi pyarrow)
    path = f"{BUCKET_NAME}/gold/features.parquet"
    df = pd.read_parquet(path, filesystem=fs)
    df = df.ffill().fillna(0)
    
    # 3. Định danh cột mục tiêu (PM2_5 thay vì PM2.5)
    target_col = "PM2_5"
    
    # Loại bỏ các cột không dùng làm feature
    features_columns = [col for col in df.columns if col not in ['No', target_col]]
    
    # 4. Chia tập dữ liệu (Giữ nguyên trình tự thời gian)
    train_data, test = train_test_split(df, test_size=0.1, shuffle=False)
    train, valid = train_test_split(train_data, test_size=0.1, shuffle=False)
    
    # 5. Xử lý dữ liệu phân loại (Categorical)
    # LightGBM xử lý cột category rất tốt, nhưng cần ép kiểu tường minh
    cat_features = ['wd', 'station']
    for col_name in cat_features:
        if col_name in train.columns:
            train[col_name] = train[col_name].astype('category')
            valid[col_name] = valid[col_name].astype('category')

    # 6. Tạo Dataset cho LightGBM
    d_train = lgb.Dataset(train[features_columns], label=train[target_col], 
                          categorical_feature=cat_features, free_raw_data=False)
    d_val = lgb.Dataset(valid[features_columns], label=valid[target_col], 
                        categorical_feature=cat_features, reference=d_train, free_raw_data=False)
    
    # 7. Cấu hình tham số
    LGB_PARAMS = {
        'objective': 'regression', # PM2.5 thường dùng regression hoặc huber
        'metric': ['mae', 'rmse'], 
        'boosting': 'gbdt', 
        'seed': 42, 
        'num_leaves': 100, 
        'learning_rate': 0.05, 
        'feature_fraction': 0.7, 
        'bagging_freq': 5, 
        'bagging_fraction': 0.8, 
        'n_jobs': -1, 
        'verbosity': -1
    }
    
    # 8. Huấn luyện với Callback để dừng sớm (Early Stopping)
    model = lgb.train(
        LGB_PARAMS, 
        train_set=d_train, 
        valid_sets=[d_train, d_val], 
        valid_names=['train', 'valid'],
        num_boost_round=1000, 
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # 9. Lưu model
    model.save_model(str(MODEL_LIGHTGBM_PATH))
    _save_metadata(features_columns, target_col, cat_features)
    print(f"✅ Đã huấn luyện xong và lưu model: {MODEL_LIGHTGBM_PATH.name}")

if __name__ == "__main__":
    train_lightgbm()