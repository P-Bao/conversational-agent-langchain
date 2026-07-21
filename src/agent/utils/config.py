"""Loading the Settings via Pydantic."""

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Loading the settings with pydantic."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # === Gemini API ===
    gemini_api_key: str = ""

    # === Generation Model (Gemini via LiteLLM) ===
    generation_model: str = "gemini/gemini-2.5-flash"

    # === Embedding Model (Gemini embedding) ===
    embedding_model: str = "gemini-embedding-002"
    embedding_size: int = 3072

    # === Generic OpenAI-compatible LLM config (optional, used by LiteLLM) ===
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model_name: str = ""

    # === Embedding provider config (optional) ===
    embedding_provider: str = "google"
    embedding_base_url: str = ""
    embedding_api_key: str = ""

    # === Qdrant Collection ===
    qdrant_collection_name: str = Field(
        default="default",
        validation_alias=AliasChoices("qdrant_collection", "QDRANT_COLLECTION", "qdrant_collection_name"),
    )

    # === Unused but kept for backward-compat ===
    openai_api_type: str = "openai"
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    openai_api_key: str = ""
    openai_api_version: str = "2024-02-15-preview"
    cohere_api_key: str = ""
    cohere_model_name: str = "command-r-plus"

    # === Reranker Configuration (DISABLED - using RRF only) ===
    rerank_provider: str = "none"
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = ""
    rerank_top_k: int = 5

    # === Retrieval Configuration ===
    retrieval_k: int = 40
    retrieval_k_retry: int = 100

    # === Hybrid Fusion Configuration (Qdrant RRF / DBSF) ===
    fusion_algorithm: str = Field(
        default="rrf",
        validation_alias=AliasChoices("fusion_algorithm", "FUSION_ALGORITHM", "hybrid_fusion"),
    )
    rrf_k: int = Field(default=60, validation_alias=AliasChoices("rrf_k", "RRF_K"))
    dbsf_window: int = Field(default=1000, validation_alias=AliasChoices("dbsf_window", "DBSF_WINDOW"))

    # === QDRANT (no auth, matches qdrant_docker default config) ===
    qdrant_url: str = "http://localhost"
    qdrant_api_key: str | None = Field(default=None, validation_alias=AliasChoices("qdrant_api_key", "qdrant_cloud_api_key"))
    qdrant_port: int = 6333
    qdrant_prefer_grpc: bool = False

    # === Tracing ===
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"

    # === Backward-compat aliases ===
    model_name: str = Field(default="gemini/gemini-2.5-flash")
    embedding_model_name: str = Field(default="gemini-embedding-002")


config = Config()
