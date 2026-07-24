"""Loading the Settings via Pydantic."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Loading the settings with pydantic.

    Retrieval-only configuration. Embedding + rerank được delegate tới remote
    HTTP endpoint (Colab ngrok / server GPU riêng) — không chạy local BGE trong
    Docker. Eval LLM (DeepEval) tách config riêng trong test scope.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # === Dense embedding (remote BGE-m3) ===
    embedding_provider: str = "remote"
    embedding_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("embedding_base_url", "EMBEDDING_BASE_URL", "AU_EMBED_BASE_URL"),
    )

    # === Qdrant Collection ===
    qdrant_collection_name: str = Field(
        default="documents",
        validation_alias=AliasChoices("qdrant_collection", "QDRANT_COLLECTION", "qdrant_collection_name"),
    )

    # === Reranker (remote BGE-reranker v2-m3, optional — default 'none' = passthrough) ===
    rerank_provider: str = "none"
    rerank_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("rerank_base_url", "RERANK_BASE_URL", "AU_RERANK_BASE_URL"),
    )
    rerank_top_k: int = 5

    # === Retrieval Configuration ===
    retrieval_k: int = 40
    retrieval_k_retry: int = 100

    # === QDRANT ===
    qdrant_url: str = "http://localhost"
    qdrant_api_key: str | None = Field(default=None, validation_alias=AliasChoices("qdrant_api_key", "qdrant_cloud_api_key"))
    qdrant_port: int = 6333
    qdrant_prefer_grpc: bool = False


config = Config()
