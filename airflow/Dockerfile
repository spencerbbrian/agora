# Start from your desired Airflow image
ARG AIRFLOW_VERSION=3.1.3
FROM apache/airflow:${AIRFLOW_VERSION}

# Switch to the non-root 'airflow' user to run pip install
# The base image defaults to this user, but it's safer to ensure it here.
USER airflow

# Install dbt-snowflake and its dependencies
RUN pip install --no-cache-dir dbt-snowflake

# Switch back to 'root' if you had any other system-level commands,
# but for simple package installation, we can omit it for now if we ensure
# the next step is handled safely.
# If you don't need system-level commands, you can remove the USER root/USER airflow
# lines from the original suggestion, but adding them makes the intention clear.

# Re-install Airflow dependencies (optional, but good for dependency hygiene)
RUN pip install --upgrade --no-cache-dir "apache-airflow[celery,postgres,redis]"

# Switch to the final non-root user that the entrypoint expects
USER airflow