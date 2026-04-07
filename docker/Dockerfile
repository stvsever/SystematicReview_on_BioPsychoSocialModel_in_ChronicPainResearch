FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV BPS_WORKSPACE_ROOT=/workspace/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md LICENSE .env.example /workspace/
COPY src /workspace/src
COPY config /workspace/config
COPY paper /workspace/paper

RUN mkdir -p /workspace/src/data/manual_imports

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

CMD ["python", "-m", "bps_review", "run-all"]
