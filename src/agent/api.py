"""Main API — retrieval & search only."""

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from agent.routes import health, rag, search
from agent.utils.config import Config

load_dotenv(override=True)
config = Config()
logger.info("Startup: Retrieval & Search API v7.0.0")


app = FastAPI(
    title="Retrieval & Search API",
    version="7.0.0",
    description="Retrieval-only API: returns relevant document context from external Qdrant using BGE-m3 (dense+sparse) and optional BGE reranker.",
)


def my_schema() -> dict:
    """Generate the OpenAPI Schema."""
    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title="Retrieval & Search API",
        version="7.0.0",
        description="Retrieval-only API.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = my_schema


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "details": str(exc)},
    )


logger.info("Loading REST API Finished.")

app.include_router(router=rag.router, prefix="/rag")
app.include_router(router=search.router, prefix="/semantic")
app.include_router(router=health.router, tags=["health"])


@app.get(path="/", tags=["root"])
def read_root() -> str:
    """Returning the root."""
    return "Welcome to the RAG Backend. Please navigate to /docs for the OpenAPI!"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
