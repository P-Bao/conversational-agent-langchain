FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Enable bytecode compilation (faster startup)
ENV UV_COMPILE_BYTECODE=1

# Copy from cache instead of linking (required for Docker layer caching)
ENV UV_LINK_MODE=copy

# Increase UV HTTP timeout (default 30s is too short for large CUDA wheels
# like nvidia-cudnn-cu13 ~349MB when network is slow).
ENV UV_HTTP_TIMEOUT=300

# Set PYTHONPATH so agent module is importable
ENV PYTHONPATH=/src

# Set HF cache directory
ENV HF_HOME=/root/.cache/huggingface

# Copy python installation files
COPY ./pyproject.toml ./pyproject.toml
COPY ./README.md ./README.md
COPY ./uv.lock ./uv.lock

# Install python dependencies
RUN uv sync --frozen --no-install-project

# Copy source code
COPY ./src /src

# Sync project
RUN uv sync --frozen

EXPOSE 8001

CMD ["uv", "run", "uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8001"]
