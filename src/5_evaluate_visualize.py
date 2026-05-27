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
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, BUCKET_NAME, MODEL_METRICS_PATH


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

    # 2. CHUẨN BỊ FEATURES CHO TỪNG LOẠI MÔ HÌNH
    # A. LSTM: Lấy thông tin từ checkpoint đã lưu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load("model_lstm_pm25.pth", map_location=device)
    saved_cols_lstm = checkpoint['feature_cols']
    
    # Đảm bảo tập test có đủ các cột như lúc train LSTM (nếu thiếu thì bù 0)
    for col in saved_cols_lstm:
        if col not in test_df.columns: 
            test_df[col] = 0
            
    # B. XGBoost: Loại bỏ các cột không phải số và cột định danh
    # Dùng list comprehension để lấy đúng 53 cột số mà model yêu cầu
    feats_xgb = [c for c in saved_cols_lstm if c not in [target_col, 'No', 'station', 'wd']]
    
    # C. LightGBM: Cần ép kiểu Category cho các cột phân loại
    feats_lgb = [c for c in df.columns if c not in ['No', target_col]]
    cat_features = ['station', 'wd']
    for col in cat_features:
        if col in test_df.columns:
            test_df[col] = test_df[col].astype('category')

    # 3. THỰC HIỆN DỰ BÁO
    print("🔮 Đang thực hiện dự báo trên 3 models...")
    # Thực tế cho y_true (cắt 48 dòng đầu để khớp với cửa sổ trượt của LSTM)
    y_true = test_df[target_col].values[48:]
    test_tree = test_df.iloc[48:]

    # --- Dự báo XGBoost ---
    xgb_m = xgb.Booster()
    xgb_m.load_model("model_xgboost_pm25.json")
    y_pred_xgb = xgb_m.predict(xgb.DMatrix(test_tree[feats_xgb]))

    # --- Dự báo LightGBM ---
    lgb_m = lgb.Booster(model_file="model_lightgbm_pm25.txt")
    y_pred_lgb = lgb_m.predict(test_tree[feats_lgb])

    # --- Dự báo LSTM ---
    # Chuẩn bị dữ liệu đầu vào dạng sequence (48h)
    data_test_lstm = test_df[saved_cols_lstm].values
    x_lstm = []
    for i in range(len(data_test_lstm) - 48):
        x_lstm.append(data_test_lstm[i : i + 48])
    
    lstm_model = LSTMModel(checkpoint['input_size'], checkpoint['hidden_size']).to(device)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()
    
    with torch.no_grad():
        x_tensor = torch.tensor(np.array(x_lstm)).float().to(device)
        y_pred_lstm = lstm_model(x_tensor).squeeze().cpu().numpy()

    # 4. TÍNH TOÁN VÀ HIỂN THỊ CHỈ SỐ ĐÁNH GIÁ
    results = []
    preds_dict = {"XGBoost": y_pred_xgb, "LightGBM": y_pred_lgb, "LSTM": y_pred_lstm}
    
    for name, y_pred in preds_dict.items():
        results.append({
            "Model": name, 
            "MAE": mean_absolute_error(y_true, y_pred), 
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)), 
            "R2 Score": r2_score(y_true, y_pred)
        })

    metrics_payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "target_col": target_col,
        "models": results,
    }
    Path(MODEL_METRICS_PATH).write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n✅ KẾT QUẢ ĐÁNH GIÁ CHI TIẾT:")
    print(pd.DataFrame(results))

    # 5. TRỰC QUAN HÓA KẾT QUẢ
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 8))
    
    # Vẽ 150 giờ cuối cùng để dễ quan sát sự khác biệt
    plot_len = 150 
    plt.plot(y_true[-plot_len:], label="Giá trị thực tế", color='black', linewidth=2.5, zorder=1)
    plt.plot(y_pred_lstm[-plot_len:], label="Dự báo LSTM", color='#e74c3c', alpha=0.9, linewidth=2)
    plt.plot(y_pred_xgb[-plot_len:], label="Dự báo XGBoost", color='#3498db', alpha=0.7, linestyle='--')
    plt.plot(y_pred_lgb[-plot_len:], label="Dự báo LightGBM", color='#2ecc71', alpha=0.7, linestyle='-.')
    
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