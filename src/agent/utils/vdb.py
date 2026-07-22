"""Vector Database Utilities."""

import warnings

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from loguru import logger
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from agent.utils.config import Config
from agent.utils.embeddings import get_sparse_embedding

settings = Config()
sparse_embeddings = get_sparse_embedding(settings)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="Api key is used with an insecure connection")
    qdrant_client = QdrantClient(
        location=settings.qdrant_url,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="Api key is used with an insecure connection")
    async_qdrant_client = AsyncQdrantClient(
        location=settings.qdrant_url,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )


def init_vdb(collection_name: str, embedding: Embeddings) -> QdrantVectorStore:
    """Establish a connection to the Qdrant DB."""
    logger.info(f"USING COLLECTION: {collection_name}")

    vector_db = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embedding,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name=settings.sparse_vector_name,
    )
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db


def load_vec_db_conn() -> QdrantClient:
    """Return the module-level synchronous QdrantClient singleton."""
    return qdrant_client


def get_async_qdrant_client() -> AsyncQdrantClient:
    """Return the module-level asynchronous QdrantClient singleton."""
    return async_qdrant_client


def initialize_vector_db(collection_name: str, embeddings_size: int) -> None:
    """Initializes the vector db for a given backend."""
    client = load_vec_db_conn()
    if client.collection_exists(collection_name=collection_name):
        logger.info(f"SUCCESS: Collection {collection_name} already exists.")
    else:
        generate_collection(collection_name=collection_name, embeddings_size=embeddings_size)


def generate_collection(collection_name: str, embeddings_size: int) -> None:
    """Generate a collection for a given backend."""
    client = load_vec_db_conn()
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
        sparse_vectors_config={
            settings.sparse_vector_name: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


async def initialize_vector_db_async(collection_name: str, embeddings_size: int) -> None:
    """Initializes the vector db for a given backend asynchronously."""
    client = get_async_qdrant_client()
    if await client.collection_exists(collection_name=collection_name):
        logger.info(f"SUCCESS: Collection {collection_name} already exists.")
    else:
        await generate_collection_async(collection_name=collection_name, embeddings_size=embeddings_size)


async def generate_collection_async(collection_name: str, embeddings_size: int) -> None:
    """Generate a collection for a given backend asynchronously."""
    client = get_async_qdrant_client()
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
        sparse_vectors_config={
            settings.sparse_vector_name: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")


def initialize_all_vector_dbs(config: Config) -> None:
    """Initializes all vector dbs."""
    initialize_vector_db(collection_name=config.qdrant_collection_name, embeddings_size=config.embedding_size)
