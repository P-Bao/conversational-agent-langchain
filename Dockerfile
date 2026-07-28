FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Enable bytecode compilation (faster startup)
ENV UV_COMPILE_BYTECODE=1

# Copy from cache instead of linking (required for Docker layer caching)
ENV UV_LINK_MODE=copy

# Set PYTHONPATH so agent module is importable
ENV PYTHONPATH=/src

# Copy python installation files and dependencies
COPY ./pyproject.toml ./README.md ./uv.lock .

# Install python dependencies (CPU-only torch + FlagEmbedding via uv pip)
RUN uv sync --frozen --no-install-project

# Install torch (CPU) and FlagEmbedding before final sync to avoid wheel rebuilds
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu --no-deps
RUN uv pip install FlagEmbedding

# Copy source code
COPY ./src /src

# Final sync
RUN uv sync --frozen

EXPOSE 8001

CMD ["uv", "run", "uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8001"]
