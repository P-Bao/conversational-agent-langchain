"""Health check routes.

- ``/healthz`` is a lightweight liveness probe — the process is up and serving.
- ``/readyz`` verifies that the configured Qdrant backend responds and that
  the configured collection is reachable.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger
from qdrant_client.http.exceptions import UnexpectedResponse

from agent.utils.config import config
from agent.utils.vdb import get_async_qdrant_client


router = APIRouter()


@router.get("/healthz", tags=["health"])
def healthz() -> dict[str, str]:
    """Liveness probe: returns 200 as long as the process serves HTTP."""
    return {"status": "ok"}


@router.get("/readyz", tags=["health"])
async def readyz() -> JSONResponse:
    """Readiness probe: verifies Qdrant connectivity and that the configured
    collection exists.

    Returns 200 if the collection is reachable, 503 otherwise.
    """
    client = get_async_qdrant_client()
    collection = config.qdrant_collection_name
    try:
        exists = await client.collection_exists(collection_name=collection)
    except UnexpectedResponse as exc:
        logger.warning(f"readiness: Qdrant returned non-2xx for '{collection}': {exc}")
        return JSONResponse(
            status_code=503,
            content={"status": "fail", "reason": "qdrant_error", "details": str(exc)},
        )
    except Exception as exc:  # noqa: BLE001 — surface transport-level errors
        logger.warning(f"readiness: cannot reach Qdrant: {exc}")
        return JSONResponse(
            status_code=503,
            content={"status": "fail", "reason": "qdrant_unreachable", "details": str(exc)},
        )

    if not exists:
        logger.info(f"readiness: collection '{collection}' missing on Qdrant")
        return JSONResponse(
            status_code=503,
            content={
                "status": "fail",
                "reason": "collection_missing",
                "collection": collection,
            },
        )

    return JSONResponse(
        status_code=200,
        content={"status": "ready", "collection": collection},
    )


def _summarize(response: Any) -> dict[str, Any]:  # pragma: no cover - helper
    return {"status_code": getattr(response, "status_code", None)}
