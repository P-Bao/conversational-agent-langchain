#!/usr/bin/env python3
"""
chunking.py
-----------
Chunking documents từ MongoDB dump theo kỹ thuật:
1. MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter (nếu nội dung có Markdown header)
2. Hoặc chỉ RecursiveCharacterTextSplitter (nếu nội dung plain text)
3. Gộp chunk quá ngắn theo MIN_CHUNK_TOKENS (đếm bằng tiktoken, fallback ước lượng)
4. global_id ổn định: "{doc_id}::{chunk_index:05d}"
5. Checkpoint/resume JSONL (đọc/ghi file checkpoint để tiếp tục từ chỗ dừng)
6. Enrich title/keywords bằng LLM (tùy chọn, có thể tắt qua flag)

Tham khảo kỹ thuật từ build_learning_contexts.py.
"""

import json
import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any

from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ============================================================
# 0. Config & Encoder
# ============================================================

class _ApproxEncoder:
    """Fallback khi không tải được tiktoken: ước lượng ~4 ký tự/token."""
    def encode(self, text: str) -> List[int]:
        return [0] * max(1, len(text) // 4)


def get_encoder():
    """Lấy tiktoken encoder cl100k_base, fallback _ApproxEncoder."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"[warn] Không tải được tiktoken ({e}); dùng bộ đếm token xấp xỉ (~4 ký tự/token).")
        return _ApproxEncoder()


def load_config() -> Dict[str, Any]:
    """Nạp config từ biến môi trường."""
    load_dotenv()

    cfg = {
        "chunk_size": int(os.environ.get("CHUNK_SIZE", "1500")),
        "chunk_overlap": int(os.environ.get("CHUNK_OVERLAP", "100")),
        "min_chunk_tokens": int(os.environ.get("MIN_CHUNK_TOKENS", "100")),
        "checkpoint_file": Path(os.environ.get("CHUNK_CHECKPOINT_FILE", "./chunk_checkpoint.jsonl")),
        "enable_llm_enrich": os.environ.get("ENABLE_LLM_ENRICH", "false").lower() == "true",
        "llm_base_url": os.environ.get("LLM_BASE_URL"),
        "llm_api_key": os.environ.get("LLM_API_KEY"),
        "llm_model": os.environ.get("LLM_MODEL"),
    }
    return cfg


# ============================================================
# 1. Markdown-aware splitting
# ============================================================

def split_markdown_structural(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Cắt theo cấu trúc Markdown header, sau đó cắt đệ quy theo kích thước ký tự."""
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    header_docs = md_splitter.split_text(text)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return char_splitter.split_documents(header_docs)


def split_plain_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Cắt plain text bằng RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.create_documents([text])


def split_document_content(content: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Tự động chọn chiến lược cắt:
    - Nếu content có dấu '# ' (Markdown header) -> dùng Markdown-aware
    - Ngược lại -> plain text splitter
    """
    if "\n# " in content or content.startswith("# "):
        return split_markdown_structural(content, chunk_size, chunk_overlap)
    return split_plain_text(content, chunk_size, chunk_overlap)


# ============================================================
# 2. Gộp chunk quá ngắn
# ============================================================

def merge_short_chunks(chunks: List[Document], encoder, min_tokens: int) -> List[Document]:
    """
    Gộp chunk liền kề nếu chunk hiện tại có token < min_tokens.
    Giữ metadata của chunk đầu tiên trong cặp.
    """
    if not chunks:
        return []

    merged = [chunks[0]]
    for doc in chunks[1:]:
        last = merged[-1]
        n_tok = len(encoder.encode(doc.page_content))
        if n_tok < min_tokens:
            # Gộp vào chunk trước
            last.page_content = last.page_content.rstrip() + "\n\n" + doc.page_content
            # Gộp metadata nếu cần (giữ metadata của chunk đầu)
        else:
            merged.append(doc)
    return merged


# ============================================================
# 3. Checkpoint (resume)
# ============================================================

def load_checkpoint(path: Path) -> Dict[str, Dict]:
    """Đọc checkpoint JSONL: {global_id: record}."""
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


def append_checkpoint(path: Path, records: List[Dict]) -> None:
    """Ghi thêm records vào checkpoint file."""
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# 4. LLM Enrich (tùy chọn)
# ============================================================

ENRICH_PROMPT = """Bạn nhận một đoạn văn bản từ tài liệu đào tạo/giáo vụ. Hãy trích xuất:
1. title: tiêu đề ngắn gọn (< 12 từ), tiếng Việt nếu nội dung tiếng Việt
2. keywords: 3-6 từ khóa/chuyên thuật chính (giữ nguyên thuật ngữ tiếng Anh nếu có)
3. is_coherent: true nếu đoạn đủ ngữ nghĩa độc lập, false nếu là đoạn nối/tiếp nối

CHỈ trả về JSON array hợp lệ, không thêm markdown, không giải thích.
Mỗi phần tử: {{"id": <local_id>, "title": "...", "keywords": ["...", "..."], "is_coherent": true|false}}

Các đoạn:
{items}
"""


def build_llm(cfg: Dict):
    """Tạo LangChain ChatOpenAI client trỏ về base_url/api_key/model chung."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=cfg["llm_model"],
        api_key=cfg["llm_api_key"],
        base_url=cfg["llm_base_url"],
        temperature=0,
        max_retries=2,
        timeout=90,
    )


def extract_json_array(raw: str) -> Optional[List[Dict]]:
    """Trích xuất JSON array từ output LLM (có thể bị bọc trong markdown fence)."""
    import re
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


def enrich_batch(llm, batch: List[Dict]) -> Dict[int, Dict]:
    """
    Gọi LLM cho một batch các chunk, trả về map local_id -> {title, keywords, is_coherent}.
    """
    items_text = "\n\n".join(
        f'--- id={item["local_id"]} ---\n{item["text"][:3000]}'
        for item in batch
    )
    prompt = ENRICH_PROMPT.format(n=len(batch), items=items_text)
    result_map = {}

    try:
        resp = llm.invoke(prompt)
        parsed = extract_json_array(resp.content)
        if parsed:
            for entry in parsed:
                try:
                    lid = int(entry["id"])
                    result_map[lid] = {
                        "title": str(entry.get("title", ""))[:200],
                        "keywords": list(entry.get("keywords", []))[:8],
                        "is_coherent": bool(entry.get("is_coherent", True)),
                    }
                except Exception:
                    continue
    except Exception as e:
        print(f"  [warn] LLM enrich error: {e}")

    # Fallback cho item bị thiếu
    for item in batch:
        if item["local_id"] not in result_map:
            result_map[item["local_id"]] = {
                "title": item.get("fallback_title", "Untitled"),
                "keywords": [],
                "is_coherent": True,
            }
    return result_map


# ============================================================
# 5. Main chunking pipeline
# ============================================================

import uuid

def make_global_id(doc_id: str, chunk_idx: int) -> str:
    """Tạo global_id ổn định định dạng UUID (bắt buộc cho Qdrant)."""
    raw_id = f"{doc_id}::{chunk_idx}"
    # Băm chuỗi gốc ra 32 kí tự hex, rồi parse thành chuẩn UUID
    md5_hash = hashlib.md5(raw_id.encode("utf-8")).hexdigest()
    return str(uuid.UUID(md5_hash))


def process_documents(
    documents: Iterator[Dict[str, Any]],
    cfg: Dict,
    encoder,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Pipeline chunking đầy đủ:
    1. Đọc document -> split -> merge short -> gán global_id
    2. Load checkpoint, bỏ qua đã có
    3. (Tuỳ chọn) LLM enrich batch
    4. Ghi checkpoint, trả về list records
    """
    checkpoint_path = cfg["checkpoint_file"]
    done_map = load_checkpoint(checkpoint_path)

    llm = None
    if cfg["enable_llm_enrich"]:
        if not (cfg["llm_base_url"] and cfg["llm_api_key"] and cfg["llm_model"]):
            print("[warn] ENABLE_LLM_ENRICH=true nhưng thiếu LLM_BASE_URL/API_KEY/MODEL — tắt enrich.")
            cfg["enable_llm_enrich"] = False
        else:
            llm = build_llm(cfg)

    all_records = []
    pending_batch = []
    batch_size = 6  # số chunk / 1 lần gọi LLM

    doc_count = 0
    for doc in documents:
        if limit and doc_count >= limit:
            break

        doc_id = doc.get("_id", {}).get("$oid", f"doc_{doc_count}")
        content = doc.get("content", "") or ""
        title = doc.get("title", "") or ""

        if not content.strip():
            doc_count += 1
            continue

        # 1. Split
        raw_docs = split_document_content(content, cfg["chunk_size"], cfg["chunk_overlap"])
        # 2. Merge short
        merged_docs = merge_short_chunks(raw_docs, encoder, cfg["min_chunk_tokens"])

        # 3. Gán global_id, chuẩn bị record
        for idx, chunk_doc in enumerate(merged_docs):
            gid = make_global_id(doc_id, idx)
            if gid in done_map:
                all_records.append(done_map[gid])
                continue

            record = {
                "global_id": gid,
                "document_id": doc_id,
                "chunk_index": idx,
                "text": chunk_doc.page_content.strip(),
                "fallback_title": title or chunk_doc.metadata.get("h3") or chunk_doc.metadata.get("h2") or chunk_doc.metadata.get("h1") or "Untitled",
                "metadata": {
                    **doc,
                    **chunk_doc.metadata,
                    "document_id": doc_id,
                    "chunk_index": idx,
                },
            }
            pending_batch.append(record)

            # Khi đủ batch -> enrich (nếu bật) -> flush checkpoint
            if cfg["enable_llm_enrich"] and len(pending_batch) >= batch_size:
                for i, item in enumerate(pending_batch):
                    item["local_id"] = i
                enrich_map = enrich_batch(llm, pending_batch)
                for item in pending_batch:
                    meta = enrich_map[item["local_id"]]
                    item["title"] = meta["title"]
                    item["keywords"] = meta["keywords"]
                    item["is_coherent"] = meta["is_coherent"]
                    item["token_count"] = len(encoder.encode(item["text"]))
                append_checkpoint(checkpoint_path, pending_batch)
                all_records.extend(pending_batch)
                pending_batch = []

        doc_count += 1

    # Flush batch còn lại
    if pending_batch:
        if cfg["enable_llm_enrich"]:
            for i, item in enumerate(pending_batch):
                item["local_id"] = i
            enrich_map = enrich_batch(llm, pending_batch)
            for item in pending_batch:
                meta = enrich_map[item["local_id"]]
                item["title"] = meta["title"]
                item["keywords"] = meta["keywords"]
                item["is_coherent"] = meta["is_coherent"]
                item["token_count"] = len(encoder.encode(item["text"]))
        else:
            for item in pending_batch:
                item["title"] = item["fallback_title"]
                item["keywords"] = []
                item["is_coherent"] = True
                item["token_count"] = len(encoder.encode(item["text"]))
        append_checkpoint(checkpoint_path, pending_batch)
        all_records.extend(pending_batch)

    print(f"Chunking xong: {len(all_records)} chunks từ {doc_count} documents.")
    return all_records


def filter_final_chunks(records: List[Dict], min_tokens: int) -> List[Dict]:
    """Lọc chunk cuối cùng: chỉ giữ chunk có token >= min_tokens."""
    return [r for r in records if r.get("token_count", 0) >= min_tokens]


if __name__ == "__main__":
    import sys

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(f"=== Chunking test (limit={limit or 'all'}) ===")

    cfg = load_config()
    encoder = get_encoder()

    # Import dump_reader từ sibling module
    sys.path.insert(0, str(Path(__file__).parent))
    from dump_reader import denormalize_all_documents

    docs_iter = denormalize_all_documents(limit=limit)
    records = process_documents(docs_iter, cfg, encoder, limit=limit)
    final = filter_final_chunks(records, cfg["min_chunk_tokens"])
    print(f"Sau lọc min_tokens: {len(final)} chunks.")

    # In mẫu
    for r in final[:3]:
        print(f"\n  global_id: {r['global_id']}")
        print(f"  tokens: {r['token_count']}")
        print(f"  title: {r['title']}")
        print(f"  keywords: {r['keywords']}")
        print(f"  text[:200]: {r['text'][:200]}...")