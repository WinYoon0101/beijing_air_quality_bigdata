@echo off
color 0B
echo ========================================================
echo   HE THONG MLOPS: DU BAO PM2.5 BAC KINH
echo ========================================================
echo [1] Khoi dong Ha Tang Docker (MinIO, Spark, Cassandra, Grafana, Airflow)
echo [2] Batch historical flow (Preprocess + Bronze + Silver + Gold)
echo [3] Ingest API realtime vao Bronze
echo [4] Hourly ETL tao snapshot features
echo [5] Huan luyen mo hinh AI
echo [6] Du bao T+1 va luu Cassandra
echo [7] Danh gia va Ve Bieu do (Evaluation)
echo [8] FastAPI backend cho dashboard
echo ========================================================
set /p action="Chon chuc nang (1-8): "

if "%action%"=="1" (
    docker-compose up -d
    echo Doi 45 giay de he thong khoi dong hoan toan...
    pause
)
if "%action%"=="2" (
    cd src
    echo Dang tien xu ly... & python 0_data_preprocessing.py
    echo Dang day len MinIO Bronze... & python 1_ingestion_minio.py
    echo PySpark dang chay... & python 0_batch_etl.py
    pause
)
if "%action%"=="3" (
    cd src
    python 1_ingest_api.py
    pause
)
if "%action%"=="4" (
    cd src
    python 2_hourly_etl.py
    pause
)
if "%action%"=="5" (
    cd src
    echo A. XGBoost
    echo B. LightGBM
    set /p model="Chon mo hinh can huan luyen (A/B): "
    if /I "%model%"=="A" set TRAIN_MODEL=xgboost
    if /I "%model%"=="B" set TRAIN_MODEL=lightgbm
    python 3_train_model.py
    set TRAIN_MODEL=
    pause
)
if "%action%"=="6" (
    cd src
    python 4_realtime_inference.py
    pause
)
if "%action%"=="7" (
    cd src
    python 5_evaluate_visualize.py
    pause
)
if "%action%"=="8" (
    cd src
    python main.py
    pause
)