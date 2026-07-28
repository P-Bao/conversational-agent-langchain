"""Loading the Settings via Pydantic."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Loading the settings with pydantic.

    Retrieval-only configuration. Embedding is delegated to remote HTTP
    endpoint (embedding-server). Reranker runs locally using BGE-reranker-v2-m3
    (FlagEmbedding). Eval LLM (DeepEval) has separate config in test scope.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # === Dense embedding (remote BGE-m3 via embedding-server) ===
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

    # === Reranker (local BGE-reranker-v2-m3 via FlagEmbedding) ===
    rerank_provider: str = "bge"
    rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        validation_alias=AliasChoices("rerank_model", "RERANK_MODEL", "AU_RERANK_MODEL_NAME"),
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
