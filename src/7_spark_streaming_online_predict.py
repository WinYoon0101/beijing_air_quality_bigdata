import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd
import pyspark
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
from realtime_model import get_realtime_predictor


def _configure_runtime():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

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
    spark_version = pyspark.__version__
    major_version = int(spark_version.split('.')[0])
    scala_version = "2.13" if major_version >= 4 else "2.12"
    
    packages = ",".join(
        [
            f"org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}",
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
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000")
        .config("spark.hadoop.fs.s3a.connection.request.timeout", "60000")
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60")
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400")
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
    predictor = get_realtime_predictor()
    spark = _build_spark_session()
    
    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.streaming.metricsEnabled", "false")

    kafka_raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("includeHeaders", "false")
        .load()
    )

    parsed = (
        kafka_raw.selectExpr("CAST(value AS STRING) AS raw_json")
        .select(from_json(col("raw_json"), _event_schema()).alias("data"))
        .select("data.*")
    )

    # bronze_query = (
    #     parsed.writeStream.format("parquet")
    #     .outputMode("append")
    #     .option("path", BRONZE_LIVE_PATH)
    #     .option("checkpointLocation", BRONZE_LIVE_CHECKPOINT)
    #     .trigger(processingTime=f"{STREAM_BATCH_SECONDS} seconds")
    #     .start()
    # )

    def predict_and_export(batch_df, batch_id):
        if batch_df.rdd.isEmpty():
            print(f"💤 [ONLINE] Batch {batch_id}: Trống (Đang lắng nghe Kafka...)")
            return
            
        pdf = batch_df.toPandas()
        if pdf.empty:
            print(f"💤 [ONLINE] Batch {batch_id}: Trống (Đang lắng nghe Kafka...)")
            return

        print(f"🚀 [ONLINE] Batch {batch_id}: Bắt đầu xử lý {len(pdf)} dòng...")

        try:
            preds = predictor.predict(pdf)
        except Exception as exc:
            print(f"❌ [ONLINE] Batch {batch_id}: predict lỗi: {exc}")
            return
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

        print(f"✅ [ONLINE] Batch {batch_id}: Đã dự báo và lưu thành công {len(payload_rows)} dòng ({predictor.model_name})")

    predict_query = (
        parsed.writeStream.foreachBatch(predict_and_export)
        .outputMode("append")
        .trigger(processingTime=f"{STREAM_BATCH_SECONDS} seconds")
        .start()
    )

    print(f"🔥 [HỆ THỐNG] Đã khởi động Spark Streaming!")
    print(f"📡 Topic: {KAFKA_TOPIC} | Model: {predictor.model_name}")
    print(f"--------------------------------------------------")
    
    spark.streams.awaitAnyTermination()
    # bronze_query.stop()
    predict_query.stop()


if __name__ == "__main__":
    run_streaming()