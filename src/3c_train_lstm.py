import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import s3fs
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from joblib import dump
from torch.utils.data import Dataset, DataLoader

from feature_schema import add_next_hour_target, sort_by_time
from config import (
    MINIO_ENDPOINT, 
    MINIO_ACCESS_KEY, 
    MINIO_SECRET_KEY, 
    BUCKET_NAME, 
    MODEL_LSTM_PATH, 
    MODEL_METADATA_PATH
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _save_metadata(feature_cols, target_col):
    metadata = {
        "model_name": "lstm",
        "target_col": target_col,
        "feature_columns": feature_cols,
        "model_path": str(MODEL_LSTM_PATH.name),
        "seq_length": 48,
        "num_layers": 3,
    }
    Path(MODEL_METADATA_PATH).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=3, target_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, 
                            dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, target_size)
        
    def forward(self, x):
        _, (h_out, _) = self.lstm(x)
        out = torch.cat((h_out[-2, :, :], h_out[-1, :, :]), dim=1)
        return self.fc(out)

def create_sequences(df, feature_cols, target_column, seq_length=48):
    sequences = []
    if "station" in df.columns:
        for _, group in tqdm(df.groupby("station", sort=False), desc="Tạo chuỗi theo station"):
            arr = group[feature_cols].values
            targets = group[target_column].values
            for i in range(len(arr) - seq_length):
                seq = arr[i : i + seq_length]
                label = targets[i + seq_length]
                sequences.append((seq, label))
    else:
        arr = df[feature_cols].values
        targets = df[target_column].values
        for i in tqdm(range(len(arr) - seq_length), desc="Tạo chuỗi"):
            seq = arr[i : i + seq_length]
            label = targets[i + seq_length]
            sequences.append((seq, label))
    return sequences

class AirQualityDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq).float(), torch.tensor(target).float()

def train_lstm():
    print(f"🚀 [LSTM] Thiết bị: {DEVICE}")
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': MINIO_ENDPOINT}, 
                           key=MINIO_ACCESS_KEY, secret=MINIO_SECRET_KEY)
    df = pd.read_parquet(f"{BUCKET_NAME}/gold/features.parquet", filesystem=fs)
    target_col = "Target_PM2.5_next_1h"
    df = add_next_hour_target(df, target_col)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # 2. Time-based split
    train, temp = train_test_split(df, test_size=0.2, shuffle=False)
    val, test = train_test_split(temp, test_size=0.5, shuffle=False)
    train = train.ffill().fillna(0)
    val = val.ffill().fillna(0)
    test = test.ffill().fillna(0)

    # 3. Chọn feature đã có sẵn trong Gold layer
    excluded_cols = {"No", "year", "month", "day", "hour", target_col, "PM2_5", "PM2.5"}
    feature_cols = [col for col in train.select_dtypes(include=[np.number]).columns if col not in excluded_cols]
    input_size = len(feature_cols)
    hidden_size = 100
    print(f"📋 Model học trên {input_size} cột đặc trưng.")

    # 4. Scaling
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_X = scaler_X.fit_transform(train[feature_cols])
    val_X = scaler_X.transform(val[feature_cols])
    test_X = scaler_X.transform(test[feature_cols])

    train_y = scaler_y.fit_transform(train[[target_col]])
    val_y = scaler_y.transform(val[[target_col]])
    test_y = scaler_y.transform(test[[target_col]])

    train_scaled = train.reset_index(drop=True).copy()
    val_scaled = val.reset_index(drop=True).copy()
    test_scaled = test.reset_index(drop=True).copy()

    train_scaled[feature_cols] = train_X
    val_scaled[feature_cols] = val_X
    test_scaled[feature_cols] = test_X

    train_scaled[target_col] = train_y.ravel()
    val_scaled[target_col] = val_y.ravel()
    test_scaled[target_col] = test_y.ravel()

    # 5. Tạo sequences và DataLoader
    train_seq = create_sequences(train_scaled, feature_cols, target_col)
    val_seq = create_sequences(val_scaled, feature_cols, target_col)
    test_seq = create_sequences(test_scaled, feature_cols, target_col)

    train_loader = DataLoader(AirQualityDataset(train_seq), batch_size=64, shuffle=True)
    val_loader = DataLoader(AirQualityDataset(val_seq), batch_size=64, shuffle=False)
    test_loader = DataLoader(AirQualityDataset(test_seq), batch_size=64, shuffle=False)

    model = LSTMModel(input_size=input_size, hidden_size=hidden_size).to(DEVICE)
    
    #  6. Áp dụng AdamW (weight_decay 1e-4) + CosineAnnealingLR + HuberLoss
    epochs = 60
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.HuberLoss()
    
    # 7. Huấn luyện với validation và checkpoint
    best_val_rmse = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for seq, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            seq, labels = seq.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            preds = model(seq).squeeze(-1) 
            loss = criterion(preds, labels)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * seq.size(0)
            
        train_loss_epoch = epoch_loss / len(train_loader.dataset)
        
        #  Cập nhật tốc độ học
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                preds = model(X_batch).cpu().numpy()
                targets = y_batch.numpy()
                val_preds.append(preds)
                val_targets.append(targets)

        if len(val_preds) == 0:
            val_rmse_real = float('nan')
        else:
            val_preds = np.vstack(val_preds)
            val_targets = np.concatenate([arr.reshape(-1, 1) for arr in val_targets], axis=0)
            val_preds_inv = scaler_y.inverse_transform(val_preds.reshape(-1, 1))
            val_targets_inv = scaler_y.inverse_transform(val_targets)
            val_rmse_real = float(np.sqrt(mse(val_targets_inv, val_preds_inv)))

        print(f"Epoch {epoch+1}/{epochs} | Train HuberLoss: {train_loss_epoch:.6f} | Val RMSE: {val_rmse_real:.4f}")

        # Lưu checkpoint và scalers nếu tốt hơn
        if not np.isnan(val_rmse_real) and val_rmse_real < best_val_rmse:
            best_val_rmse = val_rmse_real
            model_dict = {
                'model_state_dict': model.state_dict(),
                'feature_cols': feature_cols,
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': 3,
                'seq_length': 48,
                'target_col': target_col,
            }
            model_path = Path(MODEL_LSTM_PATH)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model_dict, str(model_path))
            
            scaler_x_path = model_path.parent / f"{model_path.stem}_scaler_X.joblib"
            scaler_y_path = model_path.parent / f"{model_path.stem}_scaler_y.joblib"
            
            dump(scaler_X, str(scaler_x_path))
            dump(scaler_y, str(scaler_y_path))
            print(f"   => Đã lưu checkpoint & scalers mới (Val RMSE: {best_val_rmse:.4f})")
            
    model.eval() 
    _save_metadata(feature_cols, target_col)
    print(f"✅ Đã xong! Best checkpoint và scalers được lưu tại thư mục: {model_path.parent}")

if __name__ == "__main__":
    train_lstm()