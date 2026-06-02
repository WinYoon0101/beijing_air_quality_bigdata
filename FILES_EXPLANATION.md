# Giải thích các file trong dự án

Dưới đây là tóm tắt vai trò của từng file/folder chính trong workspace.

- `docker-compose.yml`: cấu hình Docker Compose khởi tạo MinIO, Spark master/worker, Cassandra, Grafana, Airflow **và Kafka/Zookeeper** (phục vụ luồng realtime).
- `requirements.txt`: danh sách thư viện Python cần cài (pandas, pyspark, lightgbm, torch,...).
- `run_pipeline.bat`: batch script Windows cung cấp menu tương tác để khởi động Docker, chạy pipeline, huấn luyện và inference.
- `src/dashboard.html`: giao diện web dashboard hiện đại: **so sánh model** + **realtime prediction** (WebSocket + mini-chart + realtime log).

Folder `data/`:
- `data/raw/`: chứa các file CSV thô (ví dụ PRSA_Data_*.csv).
- `data/airquality_data.csv`: file đầu ra sau bước tiền xử lý nằm ở level local, đã chuẩn hóa cột `PM2_5` và thứ tự thời gian trước khi được đẩy lên Bronze.

- `evaluation_data.py`: logic đánh giá model dùng chung cho API và script visualize.
- `main.py`: FastAPI dashboard so sánh model + realtime endpoints (`/api/model-comparison`, `WS /ws/predictions`).

File model:
- `model_lightgbm_pm25.txt`, `model_xgboost_pm25.json`, `model_lstm_pm25.pth`: các file model đã được lưu sẵn (nếu có) trong repo.

Ghi chú quan trọng:
- Một số file (`config.py`, `docker-compose.yml`) hiện chứa credentials mẫu. Nên thay bằng biến môi trường hoặc file `.env` trước khi đưa lên repo công khai.
- PySpark trên Windows yêu cầu cài đặt JDK 11 và cấu hình `HADOOP_HOME` (đã có cấu hình tạm trong `2_spark_etl.py`).
- Gold batch và train/eval dùng chung helper trong `src/feature_schema.py` để chuẩn hóa cột và target.

**File map (quick reference)**

| File | Role | Input | Output |
|---|---|---|---|
| `src/0_data_preprocessing.py` | Local preprocessing of raw CSVs | `data/raw/*.csv` | `data/airquality_data.csv` (cleaned CSV) |
| `src/1_ingestion_minio.py` | Upload historical data to Bronze | `data/airquality_data.csv` | `s3://<bucket>/bronze/airquality_data.csv` |
| `src/2_spark_etl.py` | Batch Spark ETL: clean + feature engineering for batch | `bronze/airquality_data.csv` | `silver/cleaned_data.parquet`, `gold/features.parquet` |
| `src/feature_schema.py` | Centralized helpers: normalize, target, feature selection | DataFrames | Helper-returned DataFrames/feature lists (used by train & inference) |
| `airflow/dags/pm25_orchestration_dag.py` | Airflow DAG batch: preprocess → ingest → ETL → train (LightGBM default) → evaluate | N/A | MinIO Gold, model files, metrics |
| `src/3a_train_xgboost.py` | Train XGBoost from Gold | `gold/features.parquet` | `src/model_xgboost_pm25.json`, metadata |
| `src/3b_train_lightgbm.py` | Train LightGBM from Gold | `gold/features.parquet` | `src/model_lightgbm_pm25.txt`, metadata |
| `src/3c_train_lstm.py` | Train LSTM (sequence model) | `gold/features.parquet` | `src/model_lstm_pm25.pth`, metadata |
| `src/evaluation_data.py` | Build evaluation payload for dashboard | `gold/features.parquet`, model files | JSON metrics + chart data |
| `src/5_evaluate_visualize.py` | Evaluate models and create metrics/plots | `gold/features.parquet` | `src/model_metrics.json`, PNG plots in `src/` |
| `src/main.py` | FastAPI model comparison dashboard | Gold + models via `evaluation_data` | `/api/model-comparison`, serves `dashboard.html` |
| `src/config.py` | Central config (paths, credentials) | Environment vars / defaults | Constants used by scripts (paths, endpoints) |
| `src/realtime_model.py` | Load model + metadata for realtime inference | `metadata_*.json` + model files | Predictor dùng trong Spark streaming |
| `src/6_kafka_producer_simulator.py` | Kafka producer giả lập 1s/dòng | `data/airquality_data.csv` | Kafka topic realtime |
| `src/7_spark_streaming_online_predict.py` | Spark Structured Streaming consumer: predict + ghi Bronze live | Kafka topic + model | `bronze/live_stream` (parquet) + `src/live_predictions.jsonl` |
| `src/8_merge_live_into_bronze.py` | Gộp live Bronze vào historical Bronze để retrain | `bronze/live_stream` + `bronze/airquality_data.csv` | cập nhật `bronze/airquality_data.csv` |
| `airflow/dags/pm25_daily_retraining_dag.py` | Airflow DAG retrain daily từ historical + live | N/A | Gold + model files + metrics |

Refer to these files when tracing a record from raw → Bronze → Gold → prediction.
