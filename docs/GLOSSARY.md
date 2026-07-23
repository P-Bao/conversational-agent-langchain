# Thuật Ngữ — Glossary

## A — M

| Thuật ngữ | Giải thích | Liên quan |
|---|---|---|
| **BGE-m3** | Multi-lingual embedding model của BAAI. Cùng lúc output dense vector (ngữ nghĩa), sparse vector (từ vựng), và ColBERT (late interaction). Dùng 1024-dim dense + lexical weights sparse cho hybrid search. | [ARCHITECTURE.md](ARCHITECTURE.md), [CONFIGURATION.md](CONFIGURATION.md) |
| **BGE Reranker** | Cross-encoder model `BAAI/bge-reranker-v2-m3`. So sánh query vs từng document pairs → ra relevance score. Đa ngữ, tối ưu cho tiếng Việt. | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Chunking** | Chia văn bản dài thành đoạn nhỏ (chunk). Dùng Markdown-aware splitter + RecursiveCharacterTextSplitter. Kích thước mặc định 1500 ký tự, overlap 100. Gộp chunk ngắn dưới 100 token. | [DATA_INGESTION.md](DATA_INGESTION.md), `src/agent/scripts/chunking.py` |
| **Dense Embedding** | Vector số thực (float) biểu diễn ngữ nghĩa của văn bản. BGE-m3 dense dim = 1024. Dùng similarity COSINE. | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **DBSF** | Distribution-Based Score Fusion — cách kết hợp dense + sparse search thay thế RRF. Dùng cho fusion algorithm env. | [CONFIGURATION.md](CONFIGURATION.md) |
| **FlagEmbedding** | Thư viện chính thức của BAAI cho BGE models. Cung cấp `BGEM3FlagModel` (embed) và `FlagReranker` (rerank). | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Hybrid Search** | Kết hợp kết quả từ dense search (ngữ nghĩa) và sparse search (từ vựng chính xác) để tăng recall. Qdrant dùng RRF hoặc DBSF để fusion. | `src/agent/utils/retriever.py` |
| **LangGraph** | Framework của LangChain để xây dựng agent pipeline dạng graph. V6 dùng graph 1 node: entry → retriever → END. | `src/agent/backend/graph.py` |
| **Mongo Dump** | File JSON export từ MongoDB (Extended JSON format). Chứa collection `organization_db.documents`, `organization_units`, `users`. Migration script đọc từ `input/`. | `src/agent/scripts/dump_reader.py` |

## N — Z

| Thuật ngữ | Giải thích | Liên quan |
|---|---|---|
| **Named Vector** | Qdrant cho phép nhiều vector trong 1 collection, gọi bằng tên. V6 dùng dense (mặc định) + sparse `bge-m3-sparse`. | `src/agent/utils/vdb.py` |
| **Retrieval-Only** | Backend chỉ trả về documents, không sinh answer. Downstream LLM chịu trách nhiệm sinh câu trả lời từ context này. | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **RRF** | Reciprocal Rank Fusion — phương pháp kết hợp ranking từ nhiều search. Weight theo reciprocal rank. Default fusion algorithm. | [CONFIGURATION.md](CONFIGURATION.md) |
| **Reranking** | Bước thứ 2 sau search: dùng cross-encoder để sắp xếp lại top-K documents theo relevance. BGE-reranker-v2-m3 cho kết quả chính xác hơn raw search score. | `src/agent/utils/reranker.py` |
| **Sparse Embedding** | Vector thưa (lexical weights) biểu diễn tần suất từ vựng. BGE-m3 sparse trả về indices + values tương ứng từng token trong câu. Dùng exact match + gần gũi với BM25. | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **DeepEval** | Framework đánh giá quality của retrieval: ContextualPrecision (độ chính xác) và ContextualRecall (độ bao phủ). Dùng LLM (Qwen/NVIDIA) làm judge. | [EVALUATION.md](EVALUATION.md), `tests/test_rag_deepeval_qwen.py` |
| **Global ID** | Định danh duy nhất cho mỗi chunk: `MD5(doc_id + "::" + chunk_index)` → UUID. Dùng để Qdrant upsert/dedup. | `src/agent/scripts/chunking.py` |
| **QDrant** | Vector database hỗ trợ hybrid search: dense vector + sparse vector + filter + payload. V6 dùng Qdrant local qua HTTP (:6333). | [DEPLOYMENT.md](DEPLOYMENT.md) |
| **Checkpoint / Resume** | Migration có checkpoint via JSONL file: ghi lại global_id đã xử lý. Nếu script bị gián đoạn, chạy lại sẽ skip những chunk đã có. | `src/agent/scripts/migrate_dump_to_qdrant.py` |
| **Fixed ID** | UUID dạng chuẩn từ MD5 hash — deterministic. Cùng global_id luôn cho 1 chunk. Dùng thay cho auto-increment. | `src/agent/scripts/chunking.py` |
