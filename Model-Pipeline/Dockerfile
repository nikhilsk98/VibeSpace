# VibeSpace Model-Pipeline trainer image (CPU only).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet \
    MPLBACKEND=Agg

# libgomp1: OpenMP runtime used by LightGBM (PyTorch and scikit-learn share it on Linux)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/Model-Pipeline

# CPU-only PyTorch first (same version as requirements.txt), then the pinned requirements
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0" \
    && pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-deps . \
    && useradd --create-home --uid 1000 vibespace \
    && mkdir -p docker-out \
    && chown -R vibespace:vibespace /app

USER vibespace

ENTRYPOINT ["python", "-m", "vibespace_model.cli"]
CMD ["train", "--config", "configs/model_config.yaml"]
