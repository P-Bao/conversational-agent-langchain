#!/usr/bin/env python3
"""
migrate_dump_to_qdrant.py
-------------------------
Script one-off: đọc Mongo dump từ input/ -> chunk -> embed (dense + sparse BM25) -> upsert vào Qdrant.

Tính năng:
- Giới hạn số document xử lý qua MIGRATE_MAX_DOCUMENTS (.env) hoặc --limit N (CLI, ưu tiên cao hơn)
- Checkpoint/resume qua JSONL (bỏ qua global_id đã xử lý)
- Dùng ĐÚNG cơ chế hybrid search của repo: named vectors dense + sparse (fast-sparse-bm25),
  retrieval_mode=RetrievalMode.HYBRID, FastEmbedSparse(model_name="Qdrant/bm25")
- Payload đầy đủ: document_id, organization_unit_id, unit_type, campus, document_type,
  title, status, created_at, updated_at, chunk_index, text, global_id
- Log rõ ràng: số document sẽ xử lý / tổng số document, số chunk, tiến độ upsert

CHỈ VIẾT CODE — KHÔNG TỰ CHẠY (theo ràng buộc agent.md mục 7).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from loguru import logger
from qdrant_client import QdrantClient, models
from tqdm import tqdm

# Import local modules
sys.path.insert(0, str(Path(__file__).parent))
from dump_reader import denormalize_all_documents, get_input_dir
from chunking import (
    process_documents,
    filter_final_chunks,
    load_config as load_chunking_config,
    get_encoder,
    load_checkpoint as load_chunk_checkpoint,
)


# ============================================================
# 0. Config & Constants
# ============================================================

def load_migration_config() -> Dict[str, Any]:
    """Nạp config migration từ .env."""
    load_dotenv()

    # Input dir
    input_dir = Path(os.environ.get("INPUT_DIR", "../input")).resolve()

    # Collection name
    collection_name = os.environ.get("QDRANT_COLLECTION_NAME", "documents")

    # Qdrant connection
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    qdrant_prefer_grpc = os.environ.get("QDRANT_PREFER_GRPC", "false").lower() == "true"

    # Embedding config (BGE-m3)
    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "bge")
    embedding_model = os.environ.get("AU_EMBED_MODEL_NAME", os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3"))
    embedding_size = int(os.environ.get("AU_EMBED_DIMENSION", os.environ.get("EMBEDDING_SIZE", "1024")))
    sparse_model = os.environ.get("AU_SPARSE_MODEL_NAME", os.environ.get("SPARSE_MODEL", "BAAI/bge-m3"))
    sparse_vector_name = "bge-m3-sparse"

    # Checkpoint file
    checkpoint_file = Path(os.environ.get("MIGRATE_CHECKPOINT_FILE", "./migration_checkpoint.jsonl"))

    # Batch size for upsert
    upsert_batch_size = int(os.environ.get("MIGRATE_UPSERT_BATCH_SIZE", "50"))

    return {
        "input_dir": input_dir,
        "collection_name": collection_name,
        "qdrant_url": qdrant_url,
        "qdrant_port": qdrant_port,
        "qdrant_api_key": qdrant_api_key,
        "qdrant_prefer_grpc": qdrant_prefer_grpc,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_size": embedding_size,
        "sparse_model": sparse_model,
        "sparse_vector_name": sparse_vector_name,
        "checkpoint_file": checkpoint_file,
        "upsert_batch_size": upsert_batch_size,
    }


# ============================================================
# 1. Qdrant Collection Management
# ============================================================

def get_qdrant_client(cfg: Dict) -> QdrantClient:
    """Tạo Qdrant client từ config."""
    return QdrantClient(
        url=cfg["qdrant_url"],
        port=cfg["qdrant_port"],
        api_key=cfg["qdrant_api_key"],
        prefer_grpc=cfg["qdrant_prefer_grpc"],
    )


def ensure_collection_exists(client: QdrantClient, collection_name: str, dense_vector_size: int, sparse_vector_name: str, recreate: bool = False) -> None:
    """
    Tạo collection:
    - Nếu recreate=True và collection đã tồn tại: xóa collection cũ để ghi đè.
    - Dense vector: size=dense_vector_size, COSINE
    - Sparse vector: name=bge-m3-sparse
    """
    if client.collection_exists(collection_name=collection_name):
        if recreate:
            logger.info(f"Xóa collection cũ '{collection_name}' theo yêu cầu ghi đè (recreate=True)...")
            client.delete_collection(collection_name=collection_name)
        else:
            logger.info(f"Collection '{collection_name}' đã tồn tại.")
            return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dense_vector_size,
            distance=models.Distance.COSINE,
        ),
        sparse_vectors_config={
            sparse_vector_name: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )
    logger.info(f"Đã tạo collection '{collection_name}' với dense({dense_vector_size}) + sparse('{sparse_vector_name}').")


# ============================================================
# 2. Migration Checkpoint (resume)
# ============================================================

def load_migration_checkpoint(path: Path) -> Dict[str, Dict]:
    """Đọc migration checkpoint: {global_id: record}."""
    done = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[rec["global_id"]] = rec
                except Exception:
                    pass
    return done


def append_migration_checkpoint(path: Path, records: List[Dict]) -> None:
    """Ghi thêm records vào migration checkpoint."""
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# 3. Payload Builder
# ============================================================

def build_payload(chunk_record: Dict, doc_meta: Dict) -> Dict[str, Any]:
    """Xây dựng payload cho Qdrant từ chunk record và document metadata."""
    org_unit_id = doc_meta.get("organization_unit_id", {})
    org_unit_oid = org_unit_id.get("$oid") if isinstance(org_unit_id, dict) else None

    created_at = doc_meta.get("created_at", {})
    created_at_str = created_at.get("$date") if isinstance(created_at, dict) else str(created_at)

    updated_at = doc_meta.get("updated_at", {})
    updated_at_str = updated_at.get("$date") if isinstance(updated_at, dict) else str(updated_at)

    return {
        "document_id": chunk_record["document_id"],
        "organization_unit_id": org_unit_oid,
        "unit_type": doc_meta.get("unit_type"),
        "campus": doc_meta.get("campus"),
        "document_type": doc_meta.get("document_type"),
        "title": chunk_record.get("title", doc_meta.get("title", "")),
        "status": doc_meta.get("status"),
        "created_at": created_at_str,
        "updated_at": updated_at_str,
        "chunk_index": chunk_record["chunk_index"],
        "global_id": chunk_record["global_id"],
        "text": chunk_record["text"],
        "keywords": chunk_record.get("keywords", []),
        "unit_name": doc_meta.get("unit_name"),
        "unit_code": doc_meta.get("unit_code"),
    }


# ============================================================
# 4. Main Migration Function
# ============================================================

def run_migration(limit: Optional[int] = None, recreate: bool = False) -> None:
    """Chạy migration đầy đủ."""
    start_time = time.time()

    # 1. Load config
    cfg = load_migration_config()
    chunking_cfg = load_chunking_config()

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU MIGRATION: Mongo dump -> Qdrant Hybrid (BGE-m3)")
    logger.info("=" * 60)

    # 2. Đếm tổng số document trong dump (để log)
    from dump_reader import read_collection
    all_docs = list(read_collection("organization_db.documents"))
    total_docs_in_dump = len(all_docs)
    logger.info(f"Tổng số document trong dump: {total_docs_in_dump}")

    # 3. Xác định limit thực tế
    env_limit = os.environ.get("MIGRATE_MAX_DOCUMENTS")
    env_limit = int(env_limit) if env_limit and env_limit.isdigit() else 0

    effective_limit = limit if limit is not None else (env_limit if env_limit > 0 else None)

    if effective_limit:
        logger.info(f"Giới hạn test: sẽ xử lý {effective_limit}/{total_docs_in_dump} document đầu tiên.")
    else:
        logger.info(f"Không giới hạn: sẽ xử lý toàn bộ {total_docs_in_dump} document.")

    # 4. Embeddings (dense + sparse từ BGE-m3)
    from agent.utils.config import Config
    from agent.utils.embeddings import get_embedding_model, get_sparse_embedding

    app_config = Config()
    dense_embedding = get_embedding_model(app_config)
    sparse_embedding = get_sparse_embedding(app_config)
    dense_vector_size = cfg["embedding_size"]

    logger.info(f"Dense embedding model: {cfg['embedding_model']}, size: {dense_vector_size}")
    logger.info(f"Sparse embedding model: {cfg['sparse_model']}, name: {cfg['sparse_vector_name']}")

    # 5. Qdrant client & collection
    qdrant_client = get_qdrant_client(cfg)
    ensure_collection_exists(qdrant_client, cfg["collection_name"], dense_vector_size, cfg["sparse_vector_name"], recreate=recreate)

    # 6. Tạo QdrantVectorStore với hybrid config
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=cfg["collection_name"],
        embedding=dense_embedding,
        sparse_embedding=sparse_embedding,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name=cfg["sparse_vector_name"],
    )
    logger.info(f"QdrantVectorStore initialized: collection={cfg['collection_name']}, "
                f"retrieval_mode=HYBRID, sparse_vector_name={cfg['sparse_vector_name']}")

    # 7. Load checkpoints
    migration_checkpoint = load_migration_checkpoint(cfg["checkpoint_file"])
    logger.info(f"Đã load migration checkpoint: {len(migration_checkpoint)} chunks đã upsert.")

    chunking_checkpoint = load_chunk_checkpoint(Path(chunking_cfg["checkpoint_file"]))
    logger.info(f"Đã load chunking checkpoint: {len(chunking_checkpoint)} chunks đã enrich.")

    # 8. Process documents -> chunks
    encoder = get_encoder()
    docs_iter = denormalize_all_documents(limit=effective_limit)

    chunk_records = process_documents(docs_iter, chunking_cfg, encoder, limit=effective_limit)
    final_chunks = filter_final_chunks(chunk_records, chunking_cfg["min_chunk_tokens"])
    logger.info(f"Sau chunking + lọc: {len(final_chunks)} chunks từ {len(chunk_records)} chunks thô.")

    # 9. Filter chunks chưa upsert
    pending_chunks = [c for c in final_chunks if c["global_id"] not in migration_checkpoint]
    logger.info(f"Chunks cần upsert mới: {len(pending_chunks)} (đã có: {len(final_chunks) - len(pending_chunks)})")

    if not pending_chunks:
        logger.info("Không có chunk mới cần upsert. Migration hoàn tất.")
        return

    # 10. Upsert theo batch (rate limit disabled cho local model)
    batch_size = cfg["upsert_batch_size"]
    n_batches = (len(pending_chunks) + batch_size - 1) // batch_size
    logger.info(f"Bắt đầu upsert {len(pending_chunks)} chunks theo {n_batches} batch (size={batch_size})...")

    # rate-limit disabled (local model)
    upserted_count = 0
    for batch_idx in tqdm(range(n_batches), desc="Upsert batches"):
        batch = pending_chunks[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        texts = [c["text"] for c in batch]
        payloads = [build_payload(c, c["metadata"]) for c in batch]
        ids = [c["global_id"] for c in batch]

        try:
            vector_store.add_texts(texts=texts, metadatas=payloads, ids=ids)
            upserted_count += len(batch)

            checkpoint_records = [
                {"global_id": c["global_id"], "upserted_at": time.time(), "batch": batch_idx}
                for c in batch
            ]
            append_migration_checkpoint(cfg["checkpoint_file"], checkpoint_records)

        except Exception as e:
            logger.error(f"Lỗi upsert batch {batch_idx}: {e}")
            continue

    # 13. Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("MIGRATION HOÀN TẤT")
    logger.info("=" * 60)
    logger.info(f"Thời gian: {elapsed:.1f}s")
    logger.info(f"Document xử lý: {effective_limit or total_docs_in_dump}/{total_docs_in_dump}")
    logger.info(f"Tổng chunks tạo ra: {len(final_chunks)}")
    logger.info(f"Chunks mới upsert: {upserted_count}")
    logger.info(f"Chunks đã có (bỏ qua): {len(final_chunks) - upserted_count}")
    logger.info(f"Collection: {cfg['collection_name']}")
    logger.info(f"Checkpoint: {cfg['checkpoint_file']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migration Mongo dump -> Qdrant Hybrid (dense + sparse BGE-m3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m agent.scripts.migrate_dump_to_qdrant --limit 20             # Test 20 document đầu
  python -m agent.scripts.migrate_dump_to_qdrant --recreate             # Xóa collection cũ & tạo mới để ghi đè
  python -m agent.scripts.migrate_dump_to_qdrant                        # Xử lý tất cả (bỏ qua chunk đã có)
        """,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số document xử lý (ưu tiên cao hơn MIGRATE_MAX_DOCUMENTS trong .env). 0 = không giới hạn.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Nếu set, xóa collection cũ trên Qdrant và tạo mới hoàn toàn để ghi đè dữ liệu.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    limit = args.limit if args.limit and args.limit > 0 else None
    env_recreate = os.environ.get("RECREATE_COLLECTION", "false").lower() == "true"
    recreate = args.recreate or env_recreate
    run_migration(limit=limit, recreate=recreate)