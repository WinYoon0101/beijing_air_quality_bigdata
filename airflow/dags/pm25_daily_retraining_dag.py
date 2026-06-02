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
    dag_id="pm25_daily_retraining",
    default_args=default_args,
    description="Daily retraining pipeline with historical + live Bronze data",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["pm25", "retraining", "daily"],
) as daily_retrain_dag:
    merge_live = BashOperator(
        task_id="merge_live_into_bronze",
        bash_command=f"cd {PROJECT_DIR} && python 8_merge_live_into_bronze.py",
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

    merge_live >> batch_etl >> train_model >> evaluate
