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

    # Embedding config (dense)
    embedding_base_url = os.environ.get("EMBEDDING_BASE_URL")
    embedding_api_key = os.environ.get("EMBEDDING_API_KEY")
    embedding_model = os.environ.get("EMBEDDING_MODEL")

    # Sparse vector name (phải khớp với repo)
    sparse_vector_name = "fast-sparse-bm25"

    # Checkpoint file
    checkpoint_file = Path(os.environ.get("MIGRATE_CHECKPOINT_FILE", "./migration_checkpoint.jsonl"))

    # Rate limit cho Embeddings
    embedding_rpm = int(os.environ.get("EMBEDDING_RPM", "100"))
    embedding_tpm = int(os.environ.get("EMBEDDING_TPM", "30000"))
    embedding_rpd = int(os.environ.get("EMBEDDING_RPD", "1000"))

    # Batch size for upsert
    upsert_batch_size = int(os.environ.get("MIGRATE_UPSERT_BATCH_SIZE", "50"))

    return {
        "input_dir": input_dir,
        "collection_name": collection_name,
        "qdrant_url": qdrant_url,
        "qdrant_port": qdrant_port,
        "qdrant_api_key": qdrant_api_key,
        "qdrant_prefer_grpc": qdrant_prefer_grpc,
        "embedding_base_url": embedding_base_url,
        "embedding_api_key": embedding_api_key,
        "embedding_model": embedding_model,
        "embedding_rpm": embedding_rpm,
        "embedding_tpm": embedding_tpm,
        "embedding_rpd": embedding_rpd,
        "sparse_vector_name": sparse_vector_name,
        "checkpoint_file": checkpoint_file,
        "upsert_batch_size": upsert_batch_size,
    }


# ============================================================
# 1. Dense Embedding Client (OpenAI-compatible API)
# ============================================================

class OpenAICompatibleEmbeddings(Embeddings):
    """
    Embedding client tương thích OpenAI API (base_url + api_key + model).
    Dùng cho bất kỳ provider nào: OpenRouter, OpenAI, local vLLM, etc.
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        from langchain_openai import OpenAIEmbeddings
        self._client = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)


def get_dense_embedding(cfg: Dict) -> Embeddings:
    """Tạo dense embedding client từ config."""
    if not (cfg["embedding_base_url"] and cfg["embedding_api_key"] and cfg["embedding_model"]):
        raise ValueError(
            "Thiếu cấu hình dense embedding: EMBEDDING_BASE_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL "
            "phải được set trong .env"
        )
    
    # Fallback cho Gemini API (vì OpenAI compatibility layer của Gemini hiện chưa hỗ trợ /embeddings)
    if "generativelanguage.googleapis" in cfg["embedding_base_url"]:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=cfg["embedding_model"],
            google_api_key=cfg["embedding_api_key"],
        )

    return OpenAICompatibleEmbeddings(
        base_url=cfg["embedding_base_url"],
        api_key=cfg["embedding_api_key"],
        model=cfg["embedding_model"],
    )


# ============================================================
# 2. Qdrant Collection Management
# ============================================================

def get_qdrant_client(cfg: Dict) -> QdrantClient:
    """Tạo Qdrant client từ config."""
    return QdrantClient(
        url=cfg["qdrant_url"],
        port=cfg["qdrant_port"],
        api_key=cfg["qdrant_api_key"],
        prefer_grpc=cfg["qdrant_prefer_grpc"],
    )


def ensure_collection_exists(client: QdrantClient, collection_name: str, dense_vector_size: int, sparse_vector_name: str) -> None:
    """
    Tạo collection nếu chưa tồn tại, với cấu trúc named vectors:
    - Dense vector: mặc định (name=""), size=dense_vector_size, COSINE
    - Sparse vector: name=sparse_vector_name, dùng FastEmbed BM25
    """
    if client.collection_exists(collection_name=collection_name):
        logger.info(f"Collection '{collection_name}' đã tồn tại.")
        return

    # Cần set sparse model trước khi tạo collection
    client.set_sparse_model(embedding_model_name="Qdrant/bm25")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=dense_vector_size,
            distance=models.Distance.COSINE,
        ),
        sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
    )
    logger.info(f"Đã tạo collection '{collection_name}' với dense({dense_vector_size}) + sparse('{sparse_vector_name}').")


# ============================================================
# 3. Migration Checkpoint (resume)
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
# 4. Payload Builder
# ============================================================

def build_payload(chunk_record: Dict, doc_meta: Dict) -> Dict[str, Any]:
    """Xây dựng payload cho Qdrant từ chunk record và document metadata."""
    # Lấy các trường quan trọng từ doc_meta
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
# 5. Main Migration Function
# ============================================================

def run_migration(limit: Optional[int] = None) -> None:
    """
    Chạy migration đầy đủ.
    Args:
        limit: Giới hạn số document xử lý (None = không giới hạn, xử lý hết).
               Ưu tiên: CLI argument > MIGRATE_MAX_DOCUMENTS env > None.
    """
    start_time = time.time()

    # 1. Load config
    cfg = load_migration_config()
    chunking_cfg = load_chunking_config()

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU MIGRATION: Mongo dump -> Qdrant Hybrid")
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

    # 4. Dense embedding
    dense_embedding = get_dense_embedding(cfg)
    # Lấy vector size bằng cách embed 1 text mẫu
    test_embedding = dense_embedding.embed_query("test")
    dense_vector_size = len(test_embedding)
    logger.info(f"Dense embedding model: {cfg['embedding_model']}, vector size: {dense_vector_size}")

    # 5. Sparse embedding (FastEmbed BM25 - khớp repo)
    sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
    logger.info("Sparse embedding: FastEmbedSparse(model_name='Qdrant/bm25')")

    # 6. Qdrant client & collection
    qdrant_client = get_qdrant_client(cfg)
    ensure_collection_exists(qdrant_client, cfg["collection_name"], dense_vector_size, cfg["sparse_vector_name"])

    # 7. Tạo QdrantVectorStore với hybrid config (y hệt repo)
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

    # 8. Load migration checkpoint
    migration_checkpoint = load_migration_checkpoint(cfg["checkpoint_file"])
    logger.info(f"Đã load migration checkpoint: {len(migration_checkpoint)} chunks đã upsert.")

    # 9. Load chunking checkpoint (để biết chunk nào đã enrich)
    chunking_checkpoint = load_chunk_checkpoint(Path(chunking_cfg["checkpoint_file"]))
    logger.info(f"Đã load chunking checkpoint: {len(chunking_checkpoint)} chunks đã enrich.")

    # 10. Process documents -> chunks
    encoder = get_encoder()
    docs_iter = denormalize_all_documents(limit=effective_limit)

    # Pipeline chunking (sử dụng chunking.py đã viết)
    chunk_records = process_documents(docs_iter, chunking_cfg, encoder, limit=effective_limit)
    final_chunks = filter_final_chunks(chunk_records, chunking_cfg["min_chunk_tokens"])
    logger.info(f"Sau chunking + lọc: {len(final_chunks)} chunks từ {len(chunk_records)} chunks thô.")

    # 11. Filter chunks chưa upsert
    pending_chunks = [c for c in final_chunks if c["global_id"] not in migration_checkpoint]
    logger.info(f"Chunks cần upsert mới: {len(pending_chunks)} (đã có: {len(final_chunks) - len(pending_chunks)})")

    if not pending_chunks:
        logger.info("Không có chunk mới cần upsert. Migration hoàn tất.")
        return

    # 12. Upsert theo batch
    batch_size = cfg["upsert_batch_size"]
    n_batches = (len(pending_chunks) + batch_size - 1) // batch_size

    logger.info(f"Bắt đầu upsert {len(pending_chunks)} chunks theo {n_batches} batch (size={batch_size})...")

    # Rate Limit Configuration
    rpm_limit = cfg["embedding_rpm"]
    tpm_limit = cfg["embedding_tpm"]
    rpd_limit = cfg["embedding_rpd"]

    rpm_delay = 60.0 / rpm_limit if rpm_limit > 0 else 0
    rpd_delay = 86400.0 / rpd_limit if rpd_limit > 0 else 0
    base_delay = max(rpm_delay, rpd_delay)
    
    if base_delay > 0 or tpm_limit > 0:
        logger.info(f"Đã bật Rate Limit: RPM={rpm_limit}, TPM={tpm_limit}, RPD={rpd_limit}. Base delay = {base_delay:.2f}s")

    upserted_count = 0
    for batch_idx in tqdm(range(n_batches), desc="Upsert batches"):
        batch = pending_chunks[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        # Tính toán thời gian cần nghỉ dựa trên lượng token của batch
        batch_tokens = sum(c.get("token_count", len(c.get("text", "")) // 4) for c in batch)
        tpm_delay = (batch_tokens / tpm_limit) * 60.0 if tpm_limit > 0 else 0
        sleep_time = max(base_delay, tpm_delay)

        if sleep_time > 0 and batch_idx > 0:
            time.sleep(sleep_time)

        texts = [c["text"] for c in batch]
        payloads = [build_payload(c, c["metadata"]) for c in batch]
        ids = [c["global_id"] for c in batch]

        try:
            vector_store.add_texts(texts=texts, metadatas=payloads, ids=ids)
            upserted_count += len(batch)

            # Ghi checkpoint migration
            checkpoint_records = [
                {"global_id": c["global_id"], "upserted_at": time.time(), "batch": batch_idx}
                for c in batch
            ]
            append_migration_checkpoint(cfg["checkpoint_file"], checkpoint_records)

        except Exception as e:
            logger.error(f"Lỗi upsert batch {batch_idx}: {e}")
            # Tiếp tục batch tiếp theo thay vì dừng hẳn
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
        description="Migration Mongo dump -> Qdrant Hybrid (dense + sparse BM25)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/migrate_dump_to_qdrant.py --limit 20     # Test 20 document đầu
  python scripts/migrate_dump_to_qdrant.py                 # Xử lý tất cả (hoặc theo MIGRATE_MAX_DOCUMENTS)
  MIGRATE_MAX_DOCUMENTS=50 python scripts/migrate_dump_to_qdrant.py  # Giới hạn qua .env
        """,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số document xử lý (ưu tiên cao hơn MIGRATE_MAX_DOCUMENTS trong .env). 0 = không giới hạn.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    limit = args.limit if args.limit and args.limit > 0 else None
    run_migration(limit=limit)