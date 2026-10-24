# Custom Airflow image for the VibeSpace data pipeline.
# Multi-arch base image (amd64 + arm64 / Apple Silicon). CPU-only PyTorch keeps it small.
FROM apache/airflow:2.10.5-python3.11

ARG AIRFLOW_VERSION=2.10.5
ARG PYTHON_VERSION=3.11
ARG CONSTRAINTS_URL=https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt

USER root
# git is used by DVC to find the repository root.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /tmp/requirements.txt
# The PyTorch CPU index serves torch==2.7.1+cpu; Airflow's constraints keep shared
# packages (pandas, numpy, sympy, ...) at the versions Airflow was tested with.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "apache-airflow==${AIRFLOW_VERSION}" \
        -r /tmp/requirements.txt \
        --constraint "${CONSTRAINTS_URL}" \
    && git config --global --add safe.directory '*'

ENV PYTHONPATH=/opt/vibespace/Data-Pipeline \
    GX_ANALYTICS_ENABLED=false \
    TOKENIZERS_PARALLELISM=false
