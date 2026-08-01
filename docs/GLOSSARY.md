# Thuật Ngữ — Glossary (v7.1.0)

> **Mới v7.1:** Query Transformation, hybrid retrieval, local reranker, NIM-only DeepEval, port 8005/ami-network.

## A — M

| Thuật ngữ | Giải thích | Liên quan |
|---|---|---|
| **BGE-m3** | Multi-lingual embedding model BAAI. Repo không chạy local — gọi qua HTTP endpoint `embedding-server` (`EMBEDDING_BASE_URL`). Trả cả dense 1024-dim **và** sparse vectors (BM25-like). Dùng cho hybrid retrieval. | [ARCHITECTURE.md](ARCHITECTURE.md), [CONFIGURATION.md](CONFIGURATION.md), `src/agent/utils/embeddings.py` |
| **BGE Reranker** | Cross-encoder `BAAI/bge-reranker-v2-m3`. **v7.1 chạy local** qua FlagEmbedding (`RERANK_PROVIDER=bge` default). So sánh query vs từng document pairs → relevance score. Legacy remote `RERANK_PROVIDER=remote` vẫn hỗ trợ. | [ARCHITECTURE.md](ARCHITECTURE.md), `src/agent/utils/reranker.py`, [CONFIGURATION.md](CONFIGURATION.md) |
| **Dense Embedding** | Vector float 1024-dim biểu diễn ngữ nghĩa. BGE-m3 dense dùng COSINE similarity. | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Sparse Embedding** | Vector sparse (indices + values) giống BM25 từ BGE-m3. Qdrant HYBRID search kết hợp dense + sparse. | [ARCHITECTURE.md](ARCHITECTURE.md), `src/agent/utils/retriever.py` |
| **FlagEmbedding** | Thư viện BAAI cho BGE models. **v7.1 chỉ dùng local reranker** (`BAAI/bge-reranker-v2-m3`) trong API container. Embedding dùng remote server (repo ngoài). | `rag_test_bge_m3_reranker_ngrok.ipynb` (repo ngoài) |
| **Hybrid Retrieval** | Qdrant `RetrievalMode.HYBRID` — tìm kiếm song song dense + sparse vectors, merge score. Mặc định từ v7.1. | [ARCHITECTURE.md](ARCHITECTURE.md), `src/agent/utils/retriever.py` |
| **LangGraph** | Framework LangChain xây dựng agent pipeline dạng graph. v7.1: conditional graph `query_transform? -> retriever -> END`. | `src/agent/backend/graph.py` |
| **Liveness probe** | Endpoint `/healthz` — chỉ xác nhận process sống (luôn 200). | [DEPLOYMENT.md](DEPLOYMENT.md), [OPERATIONS.md](OPERATIONS.md) |

## N — Z

| Thuật ngữ | Giải thích | Liên quan |
|---|---|---|
| **Query Transformation** | Node tùy chọn (bật `QUERY_TRANSFORM_ENABLED=true`) dùng Qwen self-host LLM làm 3 việc: **rewrite** (phân giải câu hỏi), **step-back** (câu hỏi tổng quát hơn), **decompose** (chia 2-4 sub-queries). Chạy song song qua `RunnableParallel`. | [ARCHITECTURE.md](ARCHITECTURE.md), `src/agent/backend/nodes/query_transform.py`, [CONFIGURATION.md](CONFIGURATION.md) |
| **Readiness probe** | Endpoint `/readyz` — xác nhận Qdrant kết nối OK + collection tồn tại. Trả 200 + ready hoặc 503 + reason. | [DEPLOYMENT.md](DEPLOYMENT.md), [OPERATIONS.md](OPERATIONS.md) |
| **Reranking** | Bước sau search: dùng cross-encoder (local BGE-reranker-v2-m3) sắp xếp lại top-K theo relevance. **v7.1 default = bge (local)** — không có latency mạng remote. `RERANK_PROVIDER=none` để tắt. | `src/agent/utils/reranker.py`, [CONFIGURATION.md](CONFIGURATION.md) |
| **DeepEval** | Framework đánh giá RAG quality. **v7.1: NVIDIA NIM only** (không còn Qwen backend). 5 metrics: GEval Correctness, Faithfulness, ContextualRelevancy, ContextualPrecision, ContextualRecall. Dùng `evaluate()` batch. | [EVALUATION.md](EVALUATION.md), `tests/test_rag_deepeval_nim.py` |
| **Qdrant** | Vector DB hỗ trợ dense + sparse search + filter + payload. v7.1 dùng `RetrievalMode.HYBRID` (dense + sparse). Giao tiếp qua HTTP `:6333`. | [DEPLOYMENT.md](DEPLOYMENT.md), `src/agent/utils/vdb.py` |
| **Retrieval-Only** | Repo chỉ trả về documents, không sinh answer. Downstream LLM chịu trách nhiệm sinh câu trả lời từ context này. | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **External Management System** | Hệ thống ingestion quản lý Qdrant riêng biệt — tạo collection, ingest data, xoá document. Repo này chỉ **đọc** Qdrant. | [DATA_INGESTION.md](DATA_INGESTION.md) |
| **ami-network** | Docker external network chia sẻ giữa Qdrant + embedding-server + API + (optional Qwen). Tạo 1 lần: `docker network create ami-network`. | [SETUP.md](SETUP.md), [DEPLOYMENT.md](DEPLOYMENT.md) |