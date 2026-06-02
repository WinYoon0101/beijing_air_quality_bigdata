import json
import warnings
from pathlib import Path

import pandas as pd
import s3fs
import xgboost as xgb
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score as r2
import numpy as np
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
    # Tuyệt chiêu lưu tên file riêng biệt: metadata_xgboost.json
    save_path = Path(MODEL_METADATA_PATH).with_name("metadata_xgboost.json")
    save_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    
def _prepare_training_frame(df):
    target_col = "Target_PM2.5_next_1h"
    data = add_next_hour_target(df, target_col)
    data = data.dropna(subset=[target_col]).reset_index(drop=True)
    return data, target_col


def train_xgboost():
    print(" [HỆ THỐNG] Bắt đầu khởi động luồng huấn luyện XGBoost...")

    # 1. Kết nối MinIO
    print(" [BƯỚC 1/6] Đang kết nối với MinIO Data Lake...")
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': MINIO_ENDPOINT},
        key=MINIO_ACCESS_KEY,
        secret=MINIO_SECRET_KEY,
    )

    path = f"{BUCKET_NAME}/gold/features.parquet"
    print(f" [BƯỚC 2/6] Đang tải dữ liệu Gold từ: {path}")
    df = pd.read_parquet(path, filesystem=fs)

    # 2. Xử lý đặc trưng
    print(" [BƯỚC 3/6] Đang tiền xử lý và lọc các đặc trưng số học (Numeric Features)...")
    data, target_col = _prepare_training_frame(df)

    # Khác với LightGBM, XGBoost truyền thống chỉ thích chơi với số. 
    # Hàm select_numeric_features của bạn ở đây là rất chuẩn xác!
    features_columns = select_numeric_features(data, target_col)

    if not features_columns:
        raise RuntimeError("Không tìm thấy cột số nào trong Gold layer để huấn luyện XGBoost")

    # 3. Chia dữ liệu (90/10)
    print(" [BƯỚC 4/6] Đang chia tách tập dữ liệu Train/Validation/Test...")
    train_data, test = train_test_split(data, test_size=0.1, shuffle=False)
    
    print("\n🏆 Đã nạp BỘ THAM SỐ VÀNG từ thuật toán tối ưu.")
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

    print("🔥 [BƯỚC 5/6] ĐANG HUẤN LUYỆN MÔ HÌNH (Training in progress)...")
    
    # Tách Validation ra từ Train (10% của tập Train) để giám sát Early Stopping
    train_final, val_final = train_test_split(train_data, test_size=0.1, shuffle=False)

    dtrain_final = xgb.DMatrix(train_final[features_columns], label=train_final[target_col])
    dval_final = xgb.DMatrix(val_final[features_columns], label=val_final[target_col])
    dtest = xgb.DMatrix(test[features_columns])

    # 4. Huấn luyện (Early Stopping = 50)
    evallist = [(dtrain_final, 'train'), (dval_final, 'eval')]
    
    final_model = xgb.train(
        best_p,
        dtrain_final,
        num_boost_round=1500,
        evals=evallist,
        early_stopping_rounds=50,
        verbose_eval=100  # Cứ 100 vòng mới in log 1 lần cho đỡ rác màn hình
    )

    # 5. Đánh giá tập Test
    print("\n [BƯỚC 6/6] Đang kiểm thử sức mạnh mô hình trên tập dữ liệu ẩn (Test Set)...")
    preds = final_model.predict(dtest)

    rmse_score = np.sqrt(mse(test[target_col], preds))
    mae_score = mae(test[target_col], preds)
    r2_score = r2(test[target_col], preds)

    print("\n" + "=" * 55)
    print("🏆 BÁO CÁO KẾT QUẢ ĐỒ ÁN (XGBOOST - LOCAL PIPELINE)")
    print("=" * 55)
    print(f"📊 RMSE : {rmse_score:.4f}")
    print(f"📊 MAE  : {mae_score:.4f}")
    print(f"📊 R2   : {r2_score:.4f}")
    print("-" * 55)

    # 6. Lưu mô hình và Metadata
    print(" [HỆ THỐNG] Đang đóng gói Model và xuất file Metadata...")
    MODEL_XGBOOST_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(MODEL_XGBOOST_PATH))
    _save_metadata(features_columns, target_col)
    
    print(f"✅ HOÀN TẤT! Mô hình đã sẵn sàng lên sóng tại: {MODEL_XGBOOST_PATH.parent}")


if __name__ == "__main__":
    train_xgboost()