"""Loading the Settings via Pydantic."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Loading the settings with pydantic.

    Retrieval-only configuration: BGE-m3 cho dense + sparse (env tách), BGE reranker v2-m3
    (optional). Eval LLM (DeepEval) tách config riêng trong test scope.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # === Dense embedding (BGE-m3) ===
    embedding_provider: str = "bge"
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        validation_alias=AliasChoices("embedding_model", "EMBEDDING_MODEL", "AU_EMBED_MODEL_NAME", "AU_EMBED_MODEL"),
    )
    embedding_size: int = Field(
        default=1024,
        validation_alias=AliasChoices("embedding_size", "EMBEDDING_SIZE", "AU_EMBED_DIMENSION"),
    )

    # === Sparse embedding (BGE-m3 — cùng model nhưng env tách để swap linh hoạt) ===
    sparse_model: str = Field(
        default="BAAI/bge-m3",
        validation_alias=AliasChoices("sparse_model", "SPARSE_MODEL", "AU_SPARSE_MODEL_NAME", "AU_SPARSE_MODEL"),
    )

    # === Qdrant Collection ===
    qdrant_collection_name: str = Field(
        default="default",
        validation_alias=AliasChoices("qdrant_collection", "QDRANT_COLLECTION", "qdrant_collection_name"),
    )

    # === Reranker (BGE-reranker v2-m3, optional — default 'none' = passthrough) ===
    rerank_provider: str = "none"
    rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        validation_alias=AliasChoices("rerank_model", "RERANK_MODEL", "AU_RERANK_MODEL_NAME", "AU_RERANK_MODEL"),
    )
    rerank_top_k: int = 5

    # === Retrieval Configuration ===
    retrieval_k: int = 40
    retrieval_k_retry: int = 100

    # === Hybrid Fusion Configuration (Qdrant RRF / DBSF) ===
    fusion_algorithm: str = Field(
        default="rrf",
        validation_alias=AliasChoices("fusion_algorithm", "FUSION_ALGORITHM", "hybrid_fusion"),
    )

    # === QDRANT ===
    qdrant_url: str = "http://localhost"
    qdrant_api_key: str | None = Field(default=None, validation_alias=AliasChoices("qdrant_api_key", "qdrant_cloud_api_key"))
    qdrant_port: int = 6333
    qdrant_prefer_grpc: bool = False

    # === Sparse vector name (đổi từ fast-sparse-bm25 → bge-m3-sparse) ===
    sparse_vector_name: str = "bge-m3-sparse"


config = Config()
