from __future__ import annotations

from agent.utils.config import Config


def _defaults(cls: type[Config]) -> dict[str, object]:
    return {name: f.default for name, f in cls.model_fields.items() if f.default is not None}


def test_field_defaults_embedding() -> None:
    d = _defaults(Config)
    assert d["embedding_provider"] == "remote"
    assert d["embedding_base_url"] == ""


def test_field_defaults_embedding_api_key() -> None:
    assert Config.model_fields["embedding_api_key"].default is None


def test_field_defaults_rerank() -> None:
    d = _defaults(Config)
    assert d["rerank_provider"] == "bge"
    assert d["rerank_model"] == "BAAI/bge-reranker-v2-m3"
    assert d["rerank_top_k"] == 5


def test_field_defaults_qdrant() -> None:
    d = _defaults(Config)
    assert d["qdrant_collection_name"] == "documents"
    assert d["qdrant_url"] == "http://localhost"
    assert d["qdrant_port"] == 6333
    assert d["qdrant_prefer_grpc"] is False


def test_field_defaults_retrieval_k() -> None:
    d = _defaults(Config)
    assert d["retrieval_k"] == 40
    assert d["retrieval_k_retry"] == 100


def test_overrides() -> None:
    cfg = Config(
        rerank_provider="bge",
        rerank_top_k=7,
        embedding_base_url="https://embed.ngrok-free.app",
        embedding_api_key="secret-key",
    )
    assert cfg.rerank_top_k == 7
    assert cfg.rerank_provider == "bge"
    assert cfg.embedding_base_url == "https://embed.ngrok-free.app"
    assert cfg.embedding_api_key == "secret-key"
