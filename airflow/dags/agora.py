from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

DBT_COMMAND_BASE = 'dbt run --project-dir /opt/airflow/dbt/agora --profile agora --select'
MODELS_TO_RUN = [
    'stg_agora_products',
    'stg_agora_brands',
    'stg_agora_suppliers',
    'stg_agora_stores',
    'stg_agora_warehouses',
    'stg_agora_orders',
    'stg_agora_order_log',
    'stg_agora_transport_log',
    'stg_agora_returns',
]

with DAG(
    dag_id='agora_dbt_run_staging',
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    staging_tasks = []
    for model in MODELS_TO_RUN:
        task = BashOperator(
            task_id=f'run_{model}_model',
            bash_command=f'{DBT_COMMAND_BASE} {model}',
        )
        staging_tasks.append(task)

    staging_tasks