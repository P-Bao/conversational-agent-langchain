# Thuật Ngữ — Glossary (v7.0.0)

## A — M

| Thuật ngữ | Giải thích | Liên quan |
|---|---|---|
| **BGE-m3** | Multi-lingual embedding model của BAAI. Repo này **không chạy local** — gọi model qua HTTP endpoint ngoài (`EMBEDDING_BASE_URL`, Colab ngrok / server GPU). Chỉ dùng dense vector 1024-dim (sparse đã bỏ). | [ARCHITECTURE.md](ARCHITECTURE.md), [CONFIGURATION.md](CONFIGURATION.md), `src/agent/utils/embeddings.py` |
| **BGE Reranker** | Cross-encoder model `BAAI/bge-reranker-v2-m3`. Repo này gọi qua HTTP endpoint ngoài (`RERANK_BASE_URL`) khi `RERANK_PROVIDER=remote`. So sánh query vs từng document pairs → ra relevance score. | [ARCHITECTURE.md](ARCHITECTURE.md), `src/agent/utils/reranker.py` |
| **Dense Embedding** | Vector số thực (float) biểu diễn ngữ nghĩa của văn bản. BGE-m3 dense dim = 1024. Dùng similarity COSINE. Repo này dense-only (gọi remote endpoint). | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **FlagEmbedding** | Thư viện chính thức của BAAI cho BGE models. **Không còn là dependency của repo này** từ v7.1 — model chạy trên server HTTP ngoài (Colab notebook `rag_test_bge_m3_reranker_ngrok.ipynb`). | `rag_test_bge_m3_reranker_ngrok.ipynb` (repo ngoài) |
| **LangGraph** | Framework của LangChain để xây dựng agent pipeline dạng graph. V7 giữ nguyên kiến trúc: 1 node `retriever` → END. | `src/agent/backend/graph.py` |
| **Liveness probe** | Endpoint `/healthz` chỉ xác nhận process còn sống (luôn trả 200 nếu process chạy). Không gọi dependency. | [DEPLOYMENT.md](DEPLOYMENT.md), [OPERATIONS.md](OPERATIONS.md) |

## N — Z

| Thuật ngữ | Giải thích | Liên quan |
|---|---|---|
| **Retrieval-Only** | Repo chỉ trả về documents, không sinh answer. Downstream LLM chịu trách nhiệm sinh câu trả lời từ context này. | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Readiness probe** | Endpoint `/readyz` xác nhận Qdrant có kết nối + collection tồn tại. Trả 200 + `ready` hoặc 503 + `reason`. | [DEPLOYMENT.md](DEPLOYMENT.md) |
| **Reranking** | Bước thứ 2 sau search: dùng cross-encoder (remote) để sắp xếp lại top-K documents theo relevance. BGE-reranker-v2-m3 (qua HTTP) cho kết quả chính xác hơn raw search score. Mặc định ở v7 = TẮT (`RERANK_PROVIDER=none`) vì model sống trên server ngoài. | `src/agent/utils/reranker.py`, [CONFIGURATION.md](CONFIGURATION.md) |
| **DeepEval** | Framework đánh giá quality của retrieval: ContextualPrecision (độ chính xác) và ContextualRecall (độ bao phủ). Dùng LLM (Qwen/NVIDIA) làm judge. Ở v7 chạy qua TestClient (`/rag/`) chứ không gọi Graph() trực tiếp. | [EVALUATION.md](EVALUATION.md), `tests/test_rag_deepeval_qwen.py` |
| **QDrant** | Vector database hỗ trợ dense vector search + filter + payload. V7 giao tiếp qua HTTP (`:6333`). | [DEPLOYMENT.md](DEPLOYMENT.md) |
| **External Management System** | Hệ thống quản lý Qdrant riêng biệt — chịu trách nhiệm tạo collection, ingest data, xoá document. Repo này chỉ **đọc** Qdrant. | [DATA_INGESTION.md](DATA_INGESTION.md) |
