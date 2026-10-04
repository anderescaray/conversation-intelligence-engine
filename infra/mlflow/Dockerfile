# The official image ships no Postgres driver; the backend store needs psycopg2.
FROM ghcr.io/mlflow/mlflow:v3.16.1
RUN pip install --no-cache-dir psycopg2-binary==2.9.11
