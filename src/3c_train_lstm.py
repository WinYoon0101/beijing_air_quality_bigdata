import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import s3fs
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, BUCKET_NAME, MODEL_LSTM_PATH


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

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, target_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, 
                            dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, target_size)
        
    def forward(self, x):
        ula, (h_out, _) = self.lstm(x)
        return self.fc(ula[:, -1, :])

def create_sequences(df, target_column, feature_cols, seq_length=48):
    np_data = df[feature_cols].values
    target_idx = feature_cols.index(target_column)
    sequences = []
    for i in tqdm(range(len(np_data) - seq_length), desc="Tạo chuỗi"):
        seq = np_data[i : i + seq_length]
        label = np_data[i + seq_length, target_idx]
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
    
    target_col = "PM2_5"
    if target_col not in df.columns:
        df = df.rename(columns={"PM2.5": target_col})

    # 1. Chia Train/Test (90/10) - Không shuffle để giữ thứ tự thời gian
    train_df, _ = train_test_split(df, test_size=0.1, shuffle=False)
    train_df = train_df.ffill().fillna(0)

    # 2. Xác định danh sách cột số cố định
    feature_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    input_size = len(feature_cols)
    hidden_size = 128
    print(f"📋 Model học trên {input_size} cột.")

    # 3. Chuẩn bị DataLoader
    train_seq = create_sequences(train_df, target_col, feature_cols)
    train_loader = DataLoader(AirQualityDataset(train_seq), batch_size=64, shuffle=True)
    
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.MSELoss()
    
    # 4. Huấn luyện
    model.train()
    for epoch in range(10):
        epoch_loss = 0
        for seq, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
            seq, labels = seq.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(seq).squeeze(), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"📊 Avg Loss: {epoch_loss/len(train_loader):.4f}")
        
    # 5. Lưu Model kèm Metadata để chống lỗi mismatch
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_cols': feature_cols,
        'input_size': input_size,
        'hidden_size': hidden_size
    }, str(MODEL_LSTM_PATH))
    print(f"✅ Đã lưu model và cấu hình cột tại: {MODEL_LSTM_PATH.name}")

if __name__ == "__main__":
    train_lstm()