import os
import warnings
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, avg, sum as spark_sum, sin, cos, lit
from config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, BRONZE_PATH, GOLD_PATH, SILVER_PATH

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
        
        # Bịt mắt file cấu hình XML bị lỗi ở ổ C
        os.environ["HADOOP_CONF_DIR"] = hadoop_dir + r"\bin"

_configure_runtime()
warnings.filterwarnings('ignore')

def run_etl():
    print("🚀 [HỆ THỐNG] Đang khởi động Spark Engine... Vui lòng đợi vài giây!")
    
    # Bản 3.3.4 bọc thép 100% + Bật thanh tiến trình (Console Progress)
    spark = SparkSession.builder.appName("AirQuality_ETL").master("local[*]") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT) \
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY) \
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY) \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000") \
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "5000") \
        .config("spark.hadoop.fs.s3a.connection.request.timeout", "60000") \
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60") \
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400") \
        .config("spark.ui.showConsoleProgress", "true") \
        .getOrCreate()

    # TẮT LOG RÁC: Chỉ hiện lỗi nghiêm trọng (ERROR), ẩn các log INFO/WARN
    spark.sparkContext.setLogLevel("ERROR")

    print("✅ [HỆ THỐNG] Khởi động Spark thành công! Bắt đầu luồng ETL...")
    
    # Đọc từ Bronze
    print("📥 [BƯỚC 1/3] Đang kéo dữ liệu thô (Bronze) từ MinIO Data Lake...")
    df = spark.read.csv(BRONZE_PATH, header=True, inferSchema=True)
    
    # Lưu xuống Silver
    print("🧹 [BƯỚC 2/3] Đang làm sạch dữ liệu và lưu vào phân khu Silver...")
    df = df.withColumnRenamed("PM2.5", "PM2_5")
    clean_df = df.dropna()
    clean_df.write.mode("overwrite").parquet(SILVER_PATH)
    df = clean_df

    # Biến đổi dữ liệu
    print("⚙️ [BƯỚC 3/3] Đang tính toán đặc trưng phức tạp (Lag, Rolling, Sin/Cos)...")
    df = df.withColumn("hour_sin", sin(2 * lit(3.141592653589793) * col("hour") / 24))
    df = df.withColumn("hour_cos", cos(2 * lit(3.141592653589793) * col("hour") / 24))
    df = df.withColumn("month_sin", sin(2 * lit(3.141592653589793) * col("month") / 12))
    df = df.withColumn("month_cos", cos(2 * lit(3.141592653589793) * col("month") / 12))
    
    time_cols = ["year", "month", "day", "hour"]
    win_spec = Window.partitionBy("station").orderBy(*time_cols)
    win_roll = Window.partitionBy("station").orderBy(*time_cols).rowsBetween(-1, 0)
    
    lag_cols = [c for c in df.columns if c not in ['No', 'year', 'month', 'day', 'hour']]
    roll_cols = [c for c in df.columns if c not in ['No', 'station', 'wd', 'year', 'month', 'day', 'hour', 'PM2_5']]
    
    for c in lag_cols:
        df = df.withColumn(f"{c}_lag_1", lag(col(c), 1).over(win_spec))
        df = df.withColumn(f"{c}_lag_2", lag(col(c), 2).over(win_spec))
        
    for c in roll_cols:
        df = df.withColumn(f"{c}_rolling_2", avg(col(c)).over(win_roll))
        
    df = df.withColumn("cum_wspm", spark_sum("WSPM").over(win_spec))
    df = df.withColumn("saturated_vapor_pressure", 61.1 * ((7.5 * col("TEMP")) / (237.3 + col("TEMP"))))
    df = df.withColumn("actual_vapor_pressure", 61.1 * ((7.5 * col("DEWP")) / (237.3 + col("DEWP"))))
    
    # Lưu xuống Gold
    print("💾 [ĐANG GHI DỮ LIỆU] Tiến hành ghi file Parquet siêu nén xuống Gold...")
    df.dropna().write.mode("overwrite").parquet(GOLD_PATH)
    
    print("🎉 [HOÀN THÀNH] Đã lưu dữ liệu Gold thành công rực rỡ! Data Lake đã sẵn sàng.")

    spark.stop() 
    print("🧹 [HỆ THỐNG] Đã giải phóng tài nguyên RAM/CPU.")

if __name__ == "__main__":
    run_etl()