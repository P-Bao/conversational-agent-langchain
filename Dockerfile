FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src ./src

RUN mkdir -p /app/.cache/huggingface

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8005/healthz || exit 1

CMD ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8005"]
