from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
import os

# --- Configuration Constants (Inside the Docker Container) ---
# NOTE: This path must match where your dbt project is mounted inside the Docker container.
# Assuming you mounted the 'dbt' folder to '/opt/airflow/dbt' (a very common practice).
DBT_ROOT_PATH = '/opt/airflow/dbt'
DBT_PROJECT_DIR = f'{DBT_ROOT_PATH}/agora' # Your project root inside the container
DBT_PROFILE_TARGET = 'dev' # Your target profile name (as defined in profiles.yml)

with DAG(
    dag_id="dbt_run_stg_products",
    schedule=None,
    start_date=pendulum.datetime(2025, 11, 23, tz="UTC"),
    catchup=False,
    tags=["dbt", "staging", "products"],
) as dag:
    
    # Define the dbt command to run a single model.
    # We use 'dbt run' and the '--select' argument to target only one model.
    dbt_run_command = f"""
    # Navigate to the dbt project directory
    cd {DBT_PROJECT_DIR} && 
    
    # Execute the dbt run command, selecting only the staging product model
    # We use 'stg_agora_products' as the file name stem.
    dbt run --project-dir . --target {DBT_PROFILE_TARGET} --select stg_agora_products
    """

    run_single_staging_model = BashOperator(
        task_id="run_stg_agora_products_model",
        bash_command=dbt_run_command,
        # Ensure dbt knows where to find its credentials file
        # NOTE: If profiles.yml is mounted to /root/.dbt, remove --profiles-dir.
        env={
            "DBT_PROFILES_DIR": "/root/.dbt" 
        }
    )