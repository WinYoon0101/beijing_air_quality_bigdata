from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/airflow/project/src"


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="pm25_historical_training",
    default_args=default_args,
    description="Batch historical pipeline for PM2.5 lakehouse training",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["pm25", "historical", "training"],
) as historical_dag:
    preprocess = BashOperator(
        task_id="preprocess_historical_data",
        bash_command=f"cd {PROJECT_DIR} && python 0_data_preprocessing.py",
    )

    ingest_bronze = BashOperator(
        task_id="ingest_bronze_historical",
        bash_command=f"cd {PROJECT_DIR} && python 1_ingestion_minio.py",
    )

    batch_etl = BashOperator(
        task_id="run_batch_etl",
        bash_command=f"cd {PROJECT_DIR} && python 0_batch_etl.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT_DIR} && python 3_train_model.py",
    )

    evaluate = BashOperator(
        task_id="evaluate_models",
        bash_command=f"cd {PROJECT_DIR} && python 5_evaluate_visualize.py",
    )

    preprocess >> ingest_bronze >> batch_etl >> train_model >> evaluate


with DAG(
    dag_id="pm25_hourly_forecast",
    default_args=default_args,
    description="Hourly realtime forecast pipeline for PM2.5 T+1",
    start_date=datetime(2026, 1, 1),
    schedule_interval="0 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["pm25", "realtime", "forecast"],
) as hourly_dag:
    ingest_api = BashOperator(
        task_id="ingest_api_payload",
        bash_command=f"cd {PROJECT_DIR} && python 1_ingest_api.py",
    )

    hourly_etl = BashOperator(
        task_id="run_hourly_etl",
        bash_command=f"cd {PROJECT_DIR} && python 2_hourly_etl.py",
    )

    realtime_inference = BashOperator(
        task_id="run_realtime_inference",
        bash_command=f"cd {PROJECT_DIR} && python 4_realtime_inference.py",
    )

    ingest_api >> hourly_etl >> realtime_inference