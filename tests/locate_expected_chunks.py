"""locate_expected_chunks.py
--------------------------
Script dev (Task 10a): quét Qdrant collection local để điền expected_chunk_locators
vào tests/golden_questions_v2.json sau khi migration xong.

Không tốn quota eval LLM (chỉ đọc Qdrant local).
"""

import json
from pathlib import Path

from agent.utils.config import Config
from agent.utils.vdb import load_vec_db_conn


def locate_chunks() -> None:
    cfg = Config()
    client = load_vec_db_conn()
    golden_path = Path("tests/golden_questions_v2.json")

    if not golden_path.exists():
        print(f"File {golden_path} không tồn tại.")
        return

    with open(golden_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not client.collection_exists(collection_name=cfg.qdrant_collection_name):
        print(f"Collection {cfg.qdrant_collection_name} chưa tồn tại trên Qdrant.")
        return

    # Scroll tất cả points trong Qdrant collection
    all_points = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=cfg.qdrant_collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(records)
        if next_offset is None:
            break
        offset = next_offset

    print(f"Đã nạp {len(all_points)} points từ Qdrant collection '{cfg.qdrant_collection_name}'.")

    updated_count = 0
    for item in questions:
        locators = []
        expected_contexts = item.get("expected_context", [])

        for fragment in expected_contexts:
            frag_lower = fragment.strip().lower()
            for pt in all_points:
                payload = pt.payload or {}
                text = (payload.get("text") or payload.get("page_content") or "").lower()

                if frag_lower in text:
                    locators.append({
                        "global_id": payload.get("global_id") or str(pt.id),
                        "document_id": payload.get("document_id"),
                        "chunk_index": payload.get("chunk_index"),
                        "source": payload.get("source") or payload.get("title"),
                        "unit_type": payload.get("unit_type"),
                    })

        # Deduplicate locators by global_id
        seen = set()
        unique_locators = []
        for loc in locators:
            gid = loc["global_id"]
            if gid not in seen:
                seen.add(gid)
                unique_locators.append(loc)

        item["expected_chunk_locators"] = unique_locators
        if unique_locators:
            updated_count += 1

    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"Đã cập nhật expected_chunk_locators cho {updated_count}/{len(questions)} câu hỏi.")


if __name__ == "__main__":
    locate_chunks()
