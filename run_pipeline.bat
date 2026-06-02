@echo off
color 0B
echo ========================================================
echo   HE THONG MLOPS: DU BAO PM2.5 BAC KINH
echo ========================================================
echo [1] Khoi dong Ha Tang Docker (MinIO, Spark, Airflow)
echo [2] Batch historical flow (Preprocess + Bronze + Silver + Gold)
echo [3] Huan luyen mo hinh AI
echo [4] Danh gia va Ve Bieu do (Evaluation)
echo [5] Dashboard so sanh model (FastAPI)
echo ========================================================
set /p action="Chon chuc nang (1-5): "

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
    echo A. XGBoost
    echo B. LightGBM
    echo C. LSTM
    set /p model="Chon mo hinh can huan luyen (A/B/C): "
    if /I "%model%"=="A" set TRAIN_MODEL=xgboost
    if /I "%model%"=="B" set TRAIN_MODEL=lightgbm
    if /I "%model%"=="C" set TRAIN_MODEL=lstm
    python 3_train_model.py
    set TRAIN_MODEL=
    pause
)
if "%action%"=="4" (
    cd src
    python 5_evaluate_visualize.py
    pause
)
if "%action%"=="5" (
    cd src
    python main.py
    pause
)
