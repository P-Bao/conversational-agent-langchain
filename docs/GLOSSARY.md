# Thuật Ngữ — Glossary (v7.0.0)

## A — M

| Thuật ngữ | Giải thích | Liên quan |
|---|---|---|
| **BGE-m3** | Multi-lingual embedding model của BAAI. Cùng lúc output dense vector (ngữ nghĩa), sparse vector (từ vựng), và ColBERT (late interaction). Dùng 1024-dim dense + lexical weights sparse cho hybrid search. | [ARCHITECTURE.md](ARCHITECTURE.md), [CONFIGURATION.md](CONFIGURATION.md) |
| **BGE Reranker** | Cross-encoder model `BAAI/bge-reranker-v2-m3`. So sánh query vs từng document pairs → ra relevance score. Đa ngữ, tối ưu cho tiếng Việt. | [ARCHITECTURE.md](ARCHITECTURE.md), `src/agent/utils/reranker.py` |
| **Dense Embedding** | Vector số thực (float) biểu diễn ngữ nghĩa của văn bản. BGE-m3 dense dim = 1024. Dùng similarity COSINE. | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **DBSF** | Distribution-Based Score Fusion — cách kết hợp dense + sparse search thay thế RRF. | [CONFIGURATION.md](CONFIGURATION.md) |
| **FlagEmbedding** | Thư viện chính thức của BAAI cho BGE models. Cung cấp `BGEM3FlagModel` (embed) và `FlagReranker` (rerank). | `src/agent/utils/embeddings.py`, `src/agent/utils/reranker.py` |
| **Hybrid Search** | Kết hợp kết quả từ dense search (ngữ nghĩa) và sparse search (từ vựng chính xác) để tăng recall. Qdrant dùng RRF hoặc DBSF để fusion. | `src/agent/utils/retriever.py` |
| **LangGraph** | Framework của LangChain để xây dựng agent pipeline dạng graph. V7 giữ nguyên kiến trúc: 1 node `retriever` → END. | `src/agent/backend/graph.py` |
| **Liveness probe** | Endpoint `/healthz` chỉ xác nhận process còn sống (luôn trả 200 nếu process chạy). Không gọi dependency. | [DEPLOYMENT.md](DEPLOYMENT.md), [OPERATIONS.md](OPERATIONS.md) |

## N — Z

| Thuật ngữ | Giải thích | Liên quan |
|---|---|---|
| **Named Vector** | Qdrant cho phép nhiều vector trong 1 collection, gọi bằng tên. V7 dùng dense (default) + sparse `bge-m3-sparse`. | `src/agent/utils/vdb.py` |
| **Retrieval-Only** | Repo chỉ trả về documents, không sinh answer. Downstream LLM chịu trách nhiệm sinh câu trả lời từ context này. | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Readiness probe** | Endpoint `/readyz` xác nhận Qdrant có kết nối + collection tồn tại. Trả 200 + `ready` hoặc 503 + `reason`. | [DEPLOYMENT.md](DEPLOYMENT.md) |
| **RRF** | Reciprocal Rank Fusion — phương pháp kết hợp ranking từ nhiều search. Weight theo reciprocal rank. Default fusion algorithm. | [CONFIGURATION.md](CONFIGURATION.md) |
| **Reranking** | Bước thứ 2 sau search: dùng cross-encoder để sắp xếp lại top-K documents theo relevance. BGE-reranker-v2-m3 cho kết quả chính xác hơn raw search score. Mặc định ở v7 = TẮT (`RERANK_PROVIDER=none`) để tiết kiệm RAM. | `src/agent/utils/reranker.py`, [CONFIGURATION.md](CONFIGURATION.md) |
| **Sparse Embedding** | Vector thưa (lexical weights) biểu diễn tần suất từ vựng. BGE-m3 sparse trả về indices + values tương ứng từng token trong câu. Dùng exact match + gần gũi với BM25. | [ARCHITECTURE.md](ARCHITECTURE.md), `src/agent/utils/embeddings.py` |
| **DeepEval** | Framework đánh giá quality của retrieval: ContextualPrecision (độ chính xác) và ContextualRecall (độ bao phủ). Dùng LLM (Qwen/NVIDIA) làm judge. Ở v7 chạy qua TestClient (`/rag/`) chứ không gọi Graph() trực tiếp. | [EVALUATION.md](EVALUATION.md), `tests/test_rag_deepeval_qwen.py` |
| **QDrant** | Vector database hỗ trợ hybrid search: dense vector + sparse vector + filter + payload. V7 giao tiếp qua HTTP (`:6333`). | [DEPLOYMENT.md](DEPLOYMENT.md) |
| **External Management System** | Hệ thống quản lý Qdrant riêng biệt — chịu trách nhiệm tạo collection, ingest data, xoá document. Repo này chỉ **đọc** Qdrant. | [DATA_INGESTION.md](DATA_INGESTION.md) |
