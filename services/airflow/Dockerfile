FROM apache/airflow:3.0.1-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Install Airflow providers with constraints
RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.0.1/constraints-3.12.txt" \
    apache-airflow-providers-postgres \
    apache-airflow-providers-redis \
    apache-airflow-providers-http \
    psycopg2-binary \
    pyarrow \
    scikit-learn \
    numpy \
    pandas

# Install Polars and Feast separately without constraints (for latest features)
RUN pip install --no-cache-dir \
    polars[pyarrow] \
    feast[postgres,redis]==0.42.0

# Set environment variables
ENV AIRFLOW_VERSION=3.0.1
ENV PYTHON_VERSION=3.12