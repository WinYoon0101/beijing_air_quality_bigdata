import json
import os
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import s3fs
import xgboost as xgb
from cassandra.cluster import Cluster

from config import (
    BUCKET_NAME,
    CASSANDRA_FORECAST_TABLE,
    CASSANDRA_HOST,
    CASSANDRA_KEYSPACE,
    CASSANDRA_PORT,
    GOLD_PATH,
    GOLD_RT_PATH,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MODEL_LIGHTGBM_PATH,
    MODEL_METADATA_PATH,
    MODEL_XGBOOST_PATH,
)


warnings.filterwarnings('ignore')


def _load_metadata():
    metadata_path = Path(MODEL_METADATA_PATH)
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def _load_model(model_name):
    if model_name.lower() == "xgboost":
        booster = xgb.Booster()
        booster.load_model(str(MODEL_XGBOOST_PATH))
        return booster

    return lgb.Booster(model_file=str(MODEL_LIGHTGBM_PATH))


def _read_latest_features(fs):
    for path in [GOLD_RT_PATH, GOLD_PATH]:
        try:
            parquet_path = path.replace("s3a://", "")
            df = pd.read_parquet(parquet_path, filesystem=fs)
            if not df.empty:
                return df.tail(1).copy()
        except Exception:
            continue

    raise RuntimeError("Không thể đọc feature snapshot từ Gold layer")


def _observed_timestamp(row):
    if all(key in row for key in ["year", "month", "day", "hour"]):
        return pd.Timestamp(
            year=int(row["year"]),
            month=int(row["month"]),
            day=int(row["day"]),
            hour=int(row["hour"]),
        ).to_pydatetime()
    return datetime.utcnow()


def _resolve_features(model, metadata, df):
    feature_columns = metadata.get("feature_columns")
    if feature_columns:
        return feature_columns

    if hasattr(model, "feature_name"):
        try:
            feature_columns = list(model.feature_name())
            if feature_columns:
                return feature_columns
        except TypeError:
            pass

    if hasattr(model, "feature_names") and model.feature_names:
        return list(model.feature_names)

    excluded = {"No", "PM2_5", "PM2.5"}
    return [column for column in df.columns if column not in excluded]


def run_realtime_inference():
    print("🚀 [REALTIME INFERENCE] Khởi động quy trình dự báo 1 giờ tới...")

    try:
        cluster = Cluster([CASSANDRA_HOST], port=CASSANDRA_PORT)
        session = cluster.connect()
        session.execute(f"""
            CREATE KEYSPACE IF NOT EXISTS {CASSANDRA_KEYSPACE}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': '1'}}
        """)
        session.execute(f"""
            CREATE TABLE IF NOT EXISTS {CASSANDRA_KEYSPACE}.{CASSANDRA_FORECAST_TABLE} (
                station text,
                forecast_timestamp timestamp,
                id UUID,
                observed_timestamp timestamp,
                predicted float,
                model_name text,
                PRIMARY KEY ((station), forecast_timestamp, id)
            ) WITH CLUSTERING ORDER BY (forecast_timestamp DESC, id ASC)
        """)
        print("✅ Cassandra đã sẵn sàng.")
    except Exception as exc:
        print(f"❌ Thất bại khi kết nối Cassandra: {exc}")
        return

    try:
        fs = s3fs.S3FileSystem(
            client_kwargs={'endpoint_url': MINIO_ENDPOINT},
            key=MINIO_ACCESS_KEY,
            secret=MINIO_SECRET_KEY,
        )
        df = _read_latest_features(fs)

        target_col = "PM2_5"
        if target_col not in df.columns and "PM2.5" in df.columns:
            df = df.rename(columns={"PM2.5": target_col})

        df["observed_timestamp"] = df.apply(_observed_timestamp, axis=1)
        print(f"📅 Đã lấy snapshot mới nhất tại {df['observed_timestamp'].iloc[0]}")
    except Exception as exc:
        print(f"❌ Lỗi đọc dữ liệu: {exc}")
        return

    metadata = _load_metadata()
    model_name = os.getenv("FORECAST_MODEL", metadata.get("model_name", "lightgbm"))

    try:
        model = _load_model(model_name)
        expected_features = _resolve_features(model, metadata, df)
        print(f"📊 Model yêu cầu {len(expected_features)} features. Đang chuẩn bị dữ liệu...")

        for column in expected_features:
            if column not in df.columns:
                df[column] = 0

        df_inference = df[expected_features].copy()
        for column in ["station", "wd"]:
            if column in df_inference.columns:
                df_inference[column] = df_inference[column].astype("category")

        if model_name.lower() == "xgboost":
            df["predicted"] = model.predict(xgb.DMatrix(df_inference))
        else:
            df["predicted"] = model.predict(df_inference)
        print("✅ Dự báo thành công.")
    except Exception as exc:
        print(f"❌ Lỗi dự báo: {exc}")
        return

    insert_stmt = session.prepare(f"""
        INSERT INTO {CASSANDRA_KEYSPACE}.{CASSANDRA_FORECAST_TABLE} (
            station,
            forecast_timestamp,
            id,
            observed_timestamp,
            predicted,
            model_name
        ) VALUES (?, ?, ?, ?, ?, ?)
    """)

    count = 0
    for _, row in df.iterrows():
        try:
            station_value = row["station"] if "station" in row and pd.notna(row["station"]) else "all"
            observed_timestamp = row["observed_timestamp"]
            forecast_timestamp = observed_timestamp + timedelta(hours=1)
            session.execute(insert_stmt, [
                str(station_value),
                forecast_timestamp,
                uuid.uuid4(),
                observed_timestamp,
                float(row["predicted"]),
                model_name.lower(),
            ])
            count += 1
        except Exception:
            continue

    print(f"🔥 HOÀN THÀNH! Đã nạp {count} dự báo vào Cassandra.")
    cluster.shutdown()


if __name__ == "__main__":
    run_realtime_inference()