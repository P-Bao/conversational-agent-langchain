# Nhập Dữ Liệu — Data Ingestion Guide

Có 2 cách nạp dữ liệu vào Qdrant:

## Cách 1: Upload qua REST API

Phù hợp: nạp nhanh 1-20 file, dev/testing, frontend upload.

### Quy trình:

```mermaid
Upload File → tmp_dir → DocumentLoader → TextSplitter → QdrantVectorStore.add_texts()
```

### Parameters:

| Parameter | Loại | Default | Mô tả |
|---|---|---|---|
| `collection_name` | query | required | Tên collection Qdrant đích |
| `file_ending` | query | `.pdf` | `.pdf` hoặc `.txt` |
| `files` | form-data | required | Multipart files |

### Các bước:

1. Gọi API upload
2. Backend lưu file vào tmp folder
3. Load file bằng `PyPDFium2Loader` (PDF) hoặc `TextLoader` (txt)
4. Chunk bằng `RecursiveCharacterTextSplitter` (chunk_size=750, overlap=200)
5. BGE-m3 dense embed toàn bộ chunks
6. Upsert vào Qdrant collection

### Splitting config (hardcoded trong `EmbeddingManagement`):

- `chunk_size = 750` characters
- `chunk_overlap = 200` characters
- Separators: `["\n\n", "\n", ".", "!"]`

So với chunking trong migration script (có Markdown-aware + merge short), API upload
dùng splitter đơn giản hơn.

## Cách 2: Migration từ Mongo Dump

Phù hợp: nạp dataset lớn (1000+ documents), production, có input từ hệ thống cũ.

### Prerequisites:

- Thư mục `INPUT_DIR` (mặc định `../input/`) chứa file JSON dạng mongoexport (Extended JSON)
- File required: `organization_db.documents.json`
- File optional: `organization_db.organization_units.json`, `organization_db.users.json`

### Script:

```powershell
uv run python -m agent.scripts.migrate_dump_to_qdrant [--help] [--limit N] [--recreate]
```

### CLI Flags:

| Flag | Môi trường | Mô tả |
|---|---|---|
| `--limit N` | CLI | Chỉ xử lý N documents đầu (test) |
| `--recreate` | CLI | Xóa collection cũ, tạo mới |
| (không set) | — | Resume từ checkpoint, skip global_id đã upsert |
| `MIGRATE_MAX_DOCUMENTS` | env | Limit thay thế (ưu tiên thấp hơn `--limit`) |
| `MIGRATE_UPSERT_BATCH_SIZE` | env | Batch upsert (default 50) |

### Migration Flow:

```
Input JSON → dump_reader → chunking (Markdown-aware split + merge short) → BGE-m3 dense embed + sparse embed → upsert batch vào Qdrant → checkpoint
```

### Chunking stage details:

- `chunk_size=1500`, `chunk_overlap=100` (default, config qua env)
- Tự động detect Markdown headers → MarkdownHeaderTextSplitter; nếu không → RecursiveCharacterTextSplitter
- Gộp chunk dưới `MIN_CHUNK_TOKENS` (100 token)
- `global_id`: `MD5(doc_id + "::" + chunk_index)` → deterministic UUID
- Optional LLM enrich (title/keywords) nếu `ENABLE_LLM_ENRICH=true`

### Checkpoint / Resume:

Migration dùng 2 checkpoint files:

| File | Mô tả |
|---|---|
| `chunk_checkpoint.jsonl` | Ghi lại global_id của chunk đã tạo (từ chunking stage) |
| `migration_checkpoint.jsonl` | Ghi lại global_id của record đã upsert vào Qdrant |

Nếu script bị gián đoạn:
1. Chạy lại không flag → automatic resume (skip đã upsert)
2. Nếu muốn re-migration từ đầu → xóa checkpoint files + `--recreate`

### Data Format Examples:

**Input JSON** (trong `input/organization_db.documents.json`):

```json
[
  {
    "_id": {"$oid": "abc123"},
    "title": "Introduction to AI",
    "content": "# AI Introduction\n\nArtificial intelligence...",
    "organization_unit_id": {"$oid": "unit001"},
    "document_type": "slide",
    "status": "published",
    "campus": "Hanoi",
    "created_at": {"$date": "2024-01-15T00:00:00Z"},
    "updated_at": {"$date": "2024-06-01T00:00:00Z"},
    "unit_name": "AI Department",
    "unit_code": "AI101"
  }
]
```

**Output payload** (Qdrant point):

```json
{
  "document_id": "abc123",
  "organization_unit_id": "unit001",
  "unit_type": "department",
  "campus": "Hanoi",
  "document_type": "slide",
  "title": "Introduction to AI",
  "status": "published",
  "chunk_index": 0,
  "text": "# AI Introduction\n\nArtificial intelligence...",
  "global_id": "550e8400-e29b-41d4-a716-446655400000",
  "keywords": ["AI", "introduction"],
  "unit_name": "AI Department",
  "unit_code": "AI101"
}
```

## Comparison

| Tiêu chí | API Upload | Migration Script |
|---|---|---|
| Nguồn | User upload (multipart) | Mongo dump JSON |
| Chunking | Simple (750/200) | Markdown-aware + merge short (1500/100) |
| Checkpoint | Không | Có (JSONL resume) |
| LLM Enrich | Không | Optional |
| Metadata | Minimal (source, page) | Full (org_unit, campus, type, title, ...) |
| Performance | 1 request/file | Batch upsert 50/turn |
| Use case | Dev, testing, bổ sung ít docs | Production migration |
