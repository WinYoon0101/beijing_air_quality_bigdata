import json
import os
from datetime import datetime, timezone

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from config import (
    BRONZE_LIVE_CHECKPOINT,
    BRONZE_LIVE_PATH,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    LIVE_PREDICTIONS_PATH,
    MINIO_ACCESS_KEY,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    STREAM_BATCH_SECONDS,
)
from realtime_model import RealtimePredictor


def _configure_runtime():
    if not os.environ.get("JAVA_HOME"):
        if os.name == "nt":
            os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.31.11-hotspot"
        else:
            os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

    if os.name == "nt" and not os.environ.get("HADOOP_HOME"):
        hadoop_dir = r"C:\hadoop"
        os.environ["HADOOP_HOME"] = hadoop_dir
        os.environ["PATH"] = hadoop_dir + r"\bin;" + os.environ.get("PATH", "")
        os.environ["HADOOP_CONF_DIR"] = hadoop_dir + r"\bin"


def _build_spark_session():
    packages = ",".join(
        [
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
            "org.apache.hadoop:hadoop-aws:3.3.4",
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        ]
    )
    return (
        SparkSession.builder.appName("PM25_Online_Predicting")
        .master("local[*]")
        .config("spark.jars.packages", packages)
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )


def _event_schema():
    return StructType(
        [
            StructField("No", IntegerType(), True),
            StructField("year", IntegerType(), True),
            StructField("month", IntegerType(), True),
            StructField("day", IntegerType(), True),
            StructField("hour", IntegerType(), True),
            StructField("PM2.5", DoubleType(), True),
            StructField("PM10", DoubleType(), True),
            StructField("SO2", DoubleType(), True),
            StructField("NO2", DoubleType(), True),
            StructField("CO", DoubleType(), True),
            StructField("O3", DoubleType(), True),
            StructField("TEMP", DoubleType(), True),
            StructField("PRES", DoubleType(), True),
            StructField("DEWP", DoubleType(), True),
            StructField("RAIN", DoubleType(), True),
            StructField("wd", StringType(), True),
            StructField("WSPM", DoubleType(), True),
            StructField("station", StringType(), True),
        ]
    )


def run_streaming():
    _configure_runtime()
    predictor = RealtimePredictor()
    spark = _build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    kafka_raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = (
        kafka_raw.selectExpr("CAST(value AS STRING) AS raw_json")
        .select(from_json(col("raw_json"), _event_schema()).alias("data"))
        .select("data.*")
    )

    # Query 1: lưu dữ liệu live vào Bronze để cuối ngày retrain.
    bronze_query = (
        parsed.writeStream.format("parquet")
        .outputMode("append")
        .option("path", BRONZE_LIVE_PATH)
        .option("checkpointLocation", BRONZE_LIVE_CHECKPOINT)
        .trigger(processingTime=f"{STREAM_BATCH_SECONDS} seconds")
        .start()
    )

    # Query 2: dự báo realtime cho từng micro-batch, lưu về file để FastAPI WebSocket phát.
    def predict_and_export(batch_df, batch_id):
        if batch_df.rdd.isEmpty():
            return
        pdf = batch_df.toPandas()
        if pdf.empty:
            return

        preds = predictor.predict(pdf)
        now_iso = datetime.now(timezone.utc).isoformat()
        payload_rows = []
        for idx, pred in enumerate(preds):
            row = pdf.iloc[idx].to_dict()
            payload_rows.append(
                {
                    "timestamp": now_iso,
                    "batch_id": int(batch_id),
                    "station": str(row.get("station", "")),
                    "year": row.get("year"),
                    "month": row.get("month"),
                    "day": row.get("day"),
                    "hour": row.get("hour"),
                    "actual_pm25": row.get("PM2.5", row.get("PM2_5")),
                    "pred_pm25_next_1h": float(pred),
                    "model_name": predictor.model_name,
                }
            )

        LIVE_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LIVE_PREDICTIONS_PATH.open("a", encoding="utf-8") as fp:
            for payload in payload_rows:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

        print(
            f"[ONLINE] Batch {batch_id}: dự báo {len(payload_rows)} dòng "
            f"({predictor.model_name})"
        )

    predict_query = (
        parsed.writeStream.foreachBatch(predict_and_export)
        .outputMode("append")
        .trigger(processingTime=f"{STREAM_BATCH_SECONDS} seconds")
        .start()
    )

    print(
        f"[ONLINE] Streaming đã chạy. Topic={KAFKA_TOPIC}, "
        f"Model={predictor.model_name}, BronzeLive={BRONZE_LIVE_PATH}"
    )
    spark.streams.awaitAnyTermination()
    bronze_query.stop()
    predict_query.stop()


if __name__ == "__main__":
    run_streaming()
