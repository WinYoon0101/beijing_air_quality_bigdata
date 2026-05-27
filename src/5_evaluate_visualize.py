import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import s3fs
import warnings
import json
from pathlib import Path
from datetime import datetime
from config import (
    BUCKET_NAME,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MODEL_LIGHTGBM_PATH,
    MODEL_LSTM_PATH,
    MODEL_METRICS_PATH,
    MODEL_XGBOOST_PATH,
)


def _configure_runtime():
    if not os.environ.get("JAVA_HOME"):
        if os.name == "nt":
            os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.31.11-hotspot"
        else:
            os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

    if os.name == "nt" and not os.environ.get("HADOOP_HOME"):
        os.environ["HADOOP_HOME"] = r"C:\hadoop"
        os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")


_configure_runtime()

warnings.filterwarnings('ignore')

# --- 2. ĐỊNH NGHĨA CẤU TRÚC LSTM ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, target_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, target_size)
        
    def forward(self, x):
        ula, _ = self.lstm(x)
        return self.fc(ula[:, -1, :])


def _predict_xgboost(test_df, target_col):
    if not Path(MODEL_XGBOOST_PATH).exists():
        return None

    model = xgb.Booster()
    model.load_model(str(MODEL_XGBOOST_PATH))
    feature_names = model.feature_names
    if not feature_names:
        feature_names = [c for c in test_df.columns if c not in [target_col, "No", "station", "wd"]]

    for col in feature_names:
        if col not in test_df.columns:
            test_df[col] = 0

    return model.predict(xgb.DMatrix(test_df[feature_names]))


def _predict_lightgbm(test_df, target_col):
    if not Path(MODEL_LIGHTGBM_PATH).exists():
        return None

    model = lgb.Booster(model_file=str(MODEL_LIGHTGBM_PATH))
    feature_names = model.feature_name()
    if not feature_names:
        feature_names = [c for c in test_df.columns if c not in ["No", target_col]]

    for col in feature_names:
        if col not in test_df.columns:
            test_df[col] = 0

    df_input = test_df[feature_names].copy()
    for col in ["station", "wd"]:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype("category")

    return model.predict(df_input)


def _predict_lstm(test_df, target_col):
    if not Path(MODEL_LSTM_PATH).exists():
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(str(MODEL_LSTM_PATH), map_location=device)
    saved_cols = checkpoint["feature_cols"]

    for col in saved_cols:
        if col not in test_df.columns:
            test_df[col] = 0

    data_test = test_df[saved_cols].values
    seq_len = 48
    if len(data_test) <= seq_len:
        return None

    x_lstm = [data_test[i : i + seq_len] for i in range(len(data_test) - seq_len)]

    model = LSTMModel(checkpoint["input_size"], checkpoint["hidden_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(np.array(x_lstm)).float().to(device)
        return model(x_tensor).squeeze().cpu().numpy()


def _align_predictions(y_true_all, predictions):
    available = {name: pred for name, pred in predictions.items() if pred is not None and len(pred) > 0}
    if not available:
        return None, {}

    min_len = min(len(pred) for pred in available.values())
    y_true = y_true_all[-min_len:]
    aligned = {name: pred[-min_len:] for name, pred in available.items()}
    return y_true, aligned

def evaluate_models():
    print("📊 [EVALUATION] Đang tải tập dữ liệu từ MinIO...")
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': MINIO_ENDPOINT}, 
                           key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY)
    
    # Đọc dữ liệu từ đường dẫn Gold
    df = pd.read_parquet(f"{BUCKET_NAME}/gold/features.parquet", filesystem=fs)
    
    target_col = "PM2_5"
    if target_col not in df.columns: 
        df = df.rename(columns={"PM2.5": target_col})

    # 1. Tách tập Test (10% cuối - Giữ nguyên trình tự thời gian)
    _, test_df = train_test_split(df, test_size=0.1, shuffle=False)
    
    # Làm sạch dữ liệu và sửa lỗi FutureWarning
    test_df = test_df.ffill().fillna(0)

    # 2. THỰC HIỆN DỰ BÁO
    print("🔮 Đang thực hiện dự báo trên 3 models...")
    preds_dict = {
        "XGBoost": _predict_xgboost(test_df.copy(), target_col),
        "LightGBM": _predict_lightgbm(test_df.copy(), target_col),
        "LSTM": _predict_lstm(test_df.copy(), target_col),
    }

    y_true, aligned_preds = _align_predictions(test_df[target_col].values, preds_dict)
    if y_true is None:
        raise RuntimeError("Không có model nào sẵn sàng để đánh giá. Hãy train ít nhất một model.")

    # 3. TÍNH TOÁN VÀ HIỂN THỊ CHỈ SỐ ĐÁNH GIÁ
    results = []
    for name, y_pred in aligned_preds.items():
        results.append({
            "Model": name, 
            "MAE": mean_absolute_error(y_true, y_pred), 
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)), 
            "R2 Score": r2_score(y_true, y_pred)
        })

    results = sorted(results, key=lambda row: row["RMSE"])
    metrics_payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "target_col": target_col,
        "sample_size": int(len(y_true)),
        "models": results,
    }
    Path(MODEL_METRICS_PATH).write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n✅ KẾT QUẢ ĐÁNH GIÁ CHI TIẾT:")
    print(pd.DataFrame(results))

    # 4. TRỰC QUAN HÓA KẾT QUẢ
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 8))
    
    # Vẽ 150 giờ cuối cùng để dễ quan sát sự khác biệt
    plot_len = 150 
    plt.plot(y_true[-plot_len:], label="Giá trị thực tế", color='black', linewidth=2.5, zorder=1)
    if "LSTM" in aligned_preds:
        plt.plot(aligned_preds["LSTM"][-plot_len:], label="Dự báo LSTM", color='#e74c3c', alpha=0.9, linewidth=2)
    if "XGBoost" in aligned_preds:
        plt.plot(aligned_preds["XGBoost"][-plot_len:], label="Dự báo XGBoost", color='#3498db', alpha=0.7, linestyle='--')
    if "LightGBM" in aligned_preds:
        plt.plot(aligned_preds["LightGBM"][-plot_len:], label="Dự báo LightGBM", color='#2ecc71', alpha=0.7, linestyle='-.')
    
    plt.title(f"So sánh nồng độ PM2.5 dự báo và thực tế ({plot_len} giờ cuối)", fontsize=14)
    plt.xlabel("Thời gian (Giờ)", fontsize=12)
    plt.ylabel("Nồng độ PM2.5 (µg/m³)", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Lưu biểu đồ vào thư mục dự án
    plt.savefig("pm25_model_comparison.png", dpi=300)
    print("\n📈 Đã lưu biểu đồ so sánh: pm25_model_comparison.png")
    plt.show()

if __name__ == "__main__":
    evaluate_models()