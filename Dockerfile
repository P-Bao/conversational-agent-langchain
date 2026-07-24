FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Enable bytecode compilation (faster startup)
ENV UV_COMPILE_BYTECODE=1

# Copy from cache instead of linking (required for Docker layer caching)
ENV UV_LINK_MODE=copy

# Set PYTHONPATH so agent module is importable
ENV PYTHONPATH=/src

# Copy python installation files
COPY ./pyproject.toml ./pyproject.toml
COPY ./README.md ./README.md
COPY ./uv.lock ./uv.lock

# Install python dependencies (no CUDA/torch wheels — embedding is remote)
RUN uv sync --frozen --no-install-project

# Copy source code
COPY ./src /src

# Sync project
RUN uv sync --frozen

EXPOSE 8001

CMD ["uv", "run", "uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8001"]
