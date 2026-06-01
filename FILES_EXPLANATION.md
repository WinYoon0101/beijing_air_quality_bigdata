# Giải thích các file trong dự án

Dưới đây là tóm tắt vai trò của từng file/folder chính trong workspace.

- `docker-compose.yml`: cấu hình Docker Compose khởi tạo MinIO, Spark master/worker, Cassandra, Grafana và Airflow.
- `requirements.txt`: danh sách thư viện Python cần cài (pandas, pyspark, lightgbm, torch,...).
- `run_pipeline.bat`: batch script Windows cung cấp menu tương tác để khởi động Docker, chạy pipeline, huấn luyện và inference.
- `src/dashboard.html`: giao diện web dashboard hiện đại với tab forecast và tab so sánh model.

Folder `data/`:
- `data/raw/`: chứa các file CSV thô (ví dụ PRSA_Data_*.csv).
- `data/airquality_data.csv`: file đầu ra sau bước tiền xử lý nằm ở level local, đã chuẩn hóa cột `PM2_5` và thứ tự thời gian trước khi được đẩy lên Bronze.

- `4_batch_inference.py`: đọc Gold, load model LightGBM/XGBoost, dự báo trên các bản ghi gần nhất và ghi kết quả vào Cassandra, dùng chung helper schema với phần train/realtime.
- `main.py`: FastAPI backend cho dashboard, hiện hỗ trợ cả route cũ `/api/latest-forecast`, `/api/forecast-history` và route chuẩn hóa `/forecasts/latest`, `/forecasts/history`.

File model:
- `model_lightgbm_pm25.txt`, `model_xgboost_pm25.json`, `model_lstm_pm25.pth`: các file model đã được lưu sẵn (nếu có) trong repo.

Ghi chú quan trọng:
- Một số file (`config.py`, `docker-compose.yml`) hiện chứa credentials mẫu. Nên thay bằng biến môi trường hoặc file `.env` trước khi đưa lên repo công khai.
- PySpark trên Windows yêu cầu cài đặt JDK 11 và cấu hình `HADOOP_HOME` (đã có cấu hình tạm trong `2_spark_etl.py`).
- Gold batch, Gold realtime và train/eval hiện dùng chung helper trong `src/feature_schema.py` để chuẩn hóa cột và target.

**File map (quick reference)**

| File | Role | Input | Output |
|---|---|---|---|
| `src/0_data_preprocessing.py` | Local preprocessing of raw CSVs | `data/raw/*.csv` | `data/airquality_data.csv` (cleaned CSV) |
| `src/1_ingestion_minio.py` | Upload historical or realtime payloads to Bronze | `data/airquality_data.csv` or API payloads | `s3://<bucket>/bronze/airquality_data.csv` or `bronze/realtime/latest.json` |
| `src/1_ingest_api.py` | Helper to run ingestion in API mode (alias) | N/A (invokes `1_ingestion_minio.py`) | Writes Bronze realtime payload |
| `src/2_spark_etl.py` | Batch Spark ETL: clean + feature engineering for batch | `bronze/airquality_data.csv` | `silver/cleaned_data.parquet`, `gold/features.parquet` |
| `src/2_hourly_etl.py` | Hourly realtime ETL: build 48-hour snapshot features | `gold/features.parquet`, `bronze/realtime/latest.json` | `gold/realtime/latest_features.parquet` (48-row snapshot) |
| `src/feature_schema.py` | Centralized helpers: normalize, target, feature selection, realtime builder | DataFrames | Helper-returned DataFrames/feature lists (used by train & inference) |
| `src/3a_train_xgboost.py` | Train XGBoost from Gold | `gold/features.parquet` | `src/model_xgboost_pm25.json`, metadata |
| `src/3b_train_lightgbm.py` | Train LightGBM from Gold | `gold/features.parquet` | `src/model_lightgbm_pm25.txt`, metadata |
| `src/3c_train_lstm.py` | Train LSTM (sequence model) | `gold/features.parquet` | `src/model_lstm_pm25.pth`, metadata |
| `src/4_batch_inference.py` | Batch inference over Gold | `gold/features.parquet` | Writes forecasts to Cassandra (and/or logs) |
| `src/4_realtime_inference.py` | Realtime inference using snapshot | `gold/realtime/latest_features.parquet` (fallback: gold/features.parquet) | Insert T+1 forecasts into Cassandra |
| `src/5_evaluate_visualize.py` | Evaluate models and create metrics/plots | `gold/features.parquet` | `src/model_metrics.json`, evaluation plots |
| `src/main.py` | FastAPI serving forecasts & dashboard | Cassandra forecasts | HTTP endpoints (`/forecasts/*`, `/api/*`) used by UI |
| `src/config.py` | Central config (paths, credentials) | Environment vars / defaults | Constants used by scripts (paths, endpoints) |

Refer to these files when tracing a record from raw → Bronze → Gold → prediction.
