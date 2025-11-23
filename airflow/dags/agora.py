from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id='agora_dbt_run',
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    run_stg_agora_products = BashOperator(
        task_id='run_stg_agora_products_model',
        bash_command='dbt run --project-dir /opt/airflow/dbt/agora --profile agora --select stg_agora_products',
    )