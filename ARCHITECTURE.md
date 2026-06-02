# Kiến trúc hệ thống — Hệ dự báo PM2.5 Bắc Kinh

Kiến trúc tập trung vào luồng batch lịch sử: xây lakehouse, huấn luyện model, đánh giá và phục vụ dashboard. Helper trong `feature_schema.py` chuẩn hóa cột, target và feature cho train / eval / inference batch.

## Thành phần chính

- `data/raw/`: nguồn CSV lịch sử.
- MinIO: Bronze, Silver, Gold.
- Spark / batch ETL: xử lý dữ liệu lịch sử.
- Kafka: ingest dữ liệu realtime (producer/consumer).
- Spark Structured Streaming: dự báo realtime + ghi Bronze live.
- XGBoost / LightGBM / LSTM: huấn luyện và so sánh model.
- FastAPI: dashboard so sánh model + realtime WebSocket.
- Airflow: điều phối DAG `pm25_historical_training` và DAG daily retraining.

## Luồng dữ liệu

1. Raw CSV được tiền xử lý và nạp vào Bronze.
2. `src/0_batch_etl.py` / `src/2_spark_etl.py` tạo Silver và Gold.
3. `src/3_train_model.py` huấn luyện model (mặc định LightGBM nếu không set `TRAIN_MODEL`).
4. `src/5_evaluate_visualize.py` đánh giá và xuất metrics/biểu đồ.
5. `src/main.py` phục vụ dashboard so sánh model qua `evaluation_data.py`.
6. `airflow/dags/pm25_orchestration_dag.py` chạy chuỗi preprocess → ingest → ETL → train → evaluate.

## Mermaid

```mermaid
flowchart LR

    SRC["Dữ liệu lịch sử CSV<br>(data/raw/)"]

    subgraph LAKE["Data Lakehouse (MinIO)"]
        direction TB
        BRONZE["Bronze<br/>Historical Data"]
        SILVER["Silver<br/>Cleaned Data"]
        GOLD["Gold<br/>features.parquet"]
    end

    subgraph ETL["Batch ETL"]
        direction TB
        PRE["0_data_preprocessing.py"]
        INGEST["1_ingestion_minio.py"]
        BATCH_ETL["0_batch_etl.py"]
    end

    subgraph ML["AI Pipeline"]
        direction TB
        TRAIN["3_train_model.py<br/>(mặc định LightGBM)"]
        EVAL["5_evaluate_visualize.py"]
    end

    subgraph APP["Serving"]
        API["FastAPI dashboard"]
    end

    subgraph ORCH["Airflow"]
        DAG["pm25_historical_training"]
    end

    SRC --> PRE --> INGEST --> BRONZE
    BRONZE --> BATCH_ETL --> SILVER --> GOLD
    GOLD --> TRAIN --> EVAL --> API
    DAG -.-> PRE
    DAG -.-> INGEST
    DAG -.-> BATCH_ETL
    DAG -.-> TRAIN
    DAG -.-> EVAL
```

## Online Predicting + Retraining (Batch + Speed Layer)

```mermaid
flowchart TD
    %% Batch Layer (Retraining)
    subgraph Batch Layer [Airflow Orchestrated - Daily Retraining]
        HIST[Historical CSV] --> B_BRONZE[Bronze]
        B_BRONZE --> B_SILVER[Silver]
        B_SILVER --> B_GOLD[Gold]
        B_GOLD --> TRAIN[Retrain XGBoost/LightGBM]
        TRAIN --> MINIO_MODEL[(MinIO Model Registry)]
    end

    %% Speed/Online Layer (Streaming)
    subgraph Speed Layer [Real-time Streaming]
        SIM[Kafka Producer<br/>Gia lap 1s/dong] --> KAFKA[Kafka Topic]
        KAFKA --> SPARK_STREAM[Spark Streaming<br/>Consumer]
        MINIO_MODEL -.-> |Load model| SPARK_STREAM
        SPARK_STREAM --> |Du bao Realtime| API[FastAPI WebSockets]
        SPARK_STREAM --> |Luu data live| B_BRONZE
    end
```
