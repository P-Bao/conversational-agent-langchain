from __future__ import annotations

import importlib
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


ALLOWED_TEST_HOSTS: set[str] = {"localhost", "127.0.0.1", "::1", "testserver"}
VCR_REDACTED_HEADERS: set[str] = {
    "authorization",
    "api-key",
    "x-api-key",
    "x-goog-api-key",
    "cookie",
    "set-cookie",
    "openai-organization",
    "openai-project",
    "x-request-id",
    "cf-ray",
    "x-debug-trace-id",
}


def _is_allowed_host(url: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname
    if host is None:
        return True
    return host in ALLOWED_TEST_HOSTS


@pytest.fixture(scope="session")
def anyio_backend() -> Literal["asyncio"]:
    return "asyncio"


@pytest.fixture(autouse=True, scope="session")
def test_env_defaults() -> None:
    # Load .env trước (cho eval/Qwen test cần giá trị thật). Unit test vẫn patch
    # everything nên không bị ảnh hưởng. Nếu .env thiếu thì giữ default rỗng —
    # test DeepEval sẽ fail-fast thay vì dùng endpoint rởm.
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass

    os.environ["EMBEDDING_PROVIDER"] = "remote"
    # Base URL embedding/reranker lấy từ .env (Colab ngrok / server Docker thật).
    # Override qua shell env nếu cần (CI). Không hardcode default.
    os.environ.setdefault("EMBEDDING_BASE_URL", "")
    os.environ["RERANK_PROVIDER"] = "none"
    os.environ.setdefault("RERANK_BASE_URL", "")
    os.environ["QDRANT_URL"] = os.environ.get("TEST_QDRANT_URL") or "http://localhost"
    os.environ["QDRANT_PORT"] = os.environ.get("TEST_QDRANT_PORT") or "6333"
    os.environ["QDRANT_COLLECTION_NAME"] = os.environ.get("TEST_QDRANT_COLLECTION_NAME") or "documents"
    os.environ.setdefault("QDRANT_API_KEY", "test_api_key")


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": sorted(VCR_REDACTED_HEADERS),
        "filter_query_parameters": ["key", "api_key"],
        "before_record_request": _sanitize_vcr_request,
        "before_record_response": _sanitize_vcr_response,
        "decode_compressed_response": True,
        "record_mode": os.getenv("VCR_RECORD_MODE", "once"),
    }


def _sanitize_vcr_request(request: Any) -> Any:
    request.headers = _strip_sensitive_headers(dict(request.headers))
    return request


def _sanitize_vcr_response(response: dict[str, Any]) -> dict[str, Any]:
    response["headers"] = _strip_sensitive_headers(dict(response.get("headers", {})))
    return response


def _strip_sensitive_headers(headers: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in headers.items() if key.lower() not in VCR_REDACTED_HEADERS}


@pytest.fixture(autouse=True)
def block_external_http(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Block outgoing HTTP by default, except localhost/testserver.

    Set `ALLOW_NETWORK_TESTS=1` to bypass this guard.
    """
    if (
        os.getenv("ALLOW_NETWORK_TESTS") == "1"
        or request.node.get_closest_marker("vcr")
        or request.node.get_closest_marker("qwen")
    ):
        return

    import httpx
    import requests

    original_sync_request = httpx.Client.request
    original_async_request = httpx.AsyncClient.request
    original_requests_request = requests.sessions.Session.request

    def guarded_sync_request(self: httpx.Client, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        if not _is_allowed_host(str(url)):
            raise RuntimeError(f"External HTTP blocked in tests: {url}")
        return original_sync_request(self, method, url, *args, **kwargs)

    async def guarded_async_request(self: httpx.AsyncClient, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        if not _is_allowed_host(str(url)):
            raise RuntimeError(f"External HTTP blocked in tests: {url}")
        return await original_async_request(self, method, url, *args, **kwargs)

    def guarded_requests_request(
        self: requests.sessions.Session, method: str, url: str, *args: Any, **kwargs: Any
    ) -> Any:
        if not _is_allowed_host(str(url)):
            raise RuntimeError(f"External HTTP blocked in tests: {url}")
        return original_requests_request(self, method, url, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "request", guarded_sync_request)
    monkeypatch.setattr(httpx.AsyncClient, "request", guarded_async_request)
    monkeypatch.setattr(requests.sessions.Session, "request", guarded_requests_request)


@pytest.fixture(scope="session")
def app() -> Iterator[FastAPI]:
    """Import the FastAPI app (no startup side effects in v7)."""
    module = importlib.import_module("agent.api")
    yield module.app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def resources_path() -> Path:
    return Path("tests/resources")
