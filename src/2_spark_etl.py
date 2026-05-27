import os
import warnings


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


_configure_runtime()

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, avg, sum as spark_sum
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, BRONZE_PATH, GOLD_PATH, SILVER_PATH

warnings.filterwarnings('ignore')

def run_etl():
    spark = SparkSession.builder.appName("AirQuality_ETL").master("local[*]") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT) \
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY) \
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY) \
        .config("spark.hadoop.fs.s3a.path.style.access", True) \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem").getOrCreate()

    print("🔄 [SPARK ETL] Đang tính Lag/Rolling độc lập theo từng Trạm...")
    df = spark.read.csv(BRONZE_PATH, header=True, inferSchema=True)
    
    # --- SỬA LỖI: ĐỔI TÊN CỘT PM2.5 THÀNH PM2_5 ---
    df = df.withColumnRenamed("PM2.5", "PM2_5")
    clean_df = df.dropna()
    clean_df.write.mode("overwrite").parquet(SILVER_PATH)
    df = clean_df
    
    time_cols = ["year", "month", "day", "hour"]
    win_spec = Window.partitionBy("station").orderBy(*time_cols)
    win_roll = Window.partitionBy("station").orderBy(*time_cols).rowsBetween(-1, 0)
    
    lag_cols = [c for c in df.columns if c not in ['No', 'year', 'month', 'day', 'hour']]
    # Chú ý: Đã đổi tên PM2.5 thành PM2_5 ở danh sách loại trừ bên dưới
    roll_cols = [c for c in df.columns if c not in ['No', 'station', 'wd', 'year', 'month', 'day', 'hour', 'PM2_5']]
    
    # Bọc col(c) để Spark hiểu đây là tên cột một cách an toàn nhất
    for c in lag_cols:
        df = df.withColumn(f"{c}_lag_1", lag(col(c), 1).over(win_spec))
        df = df.withColumn(f"{c}_lag_2", lag(col(c), 2).over(win_spec))
        
    for c in roll_cols:
        df = df.withColumn(f"{c}_rolling_2", avg(col(c)).over(win_roll))
        
    df = df.withColumn("cum_wspm", spark_sum("WSPM").over(win_spec))
    df = df.withColumn("saturated_vapor_pressure", 61.1 * ((7.5 * col("TEMP")) / (237.3 + col("TEMP"))))
    df = df.withColumn("actual_vapor_pressure", 61.1 * ((7.5 * col("DEWP")) / (237.3 + col("DEWP"))))
    
    df.dropna().write.mode("overwrite").parquet(GOLD_PATH)
    print("✅ [SPARK ETL] Đã lưu dữ liệu Gold (Parquet) thành công rực rỡ!")

    spark.stop() 
    print("🧹 [SPARK] Đã giải phóng tài nguyên hệ thống.")

if __name__ == "__main__":
    run_etl()