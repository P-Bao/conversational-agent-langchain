#!/usr/bin/env python3
"""
dump_reader.py
--------------
Đọc bản dump MongoDB (Extended JSON của mongoexport) từ thư mục input/.
Hỗ trợ 3 collection: organization_units, documents, users.

Định dạng file: JSON array của các document, mỗi document có trường _id dạng
{"$oid": "..."} — dùng bson.json_util.loads để parse đúng kiểu ObjectId, datetime, v.v.
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any



def get_input_dir() -> Path:
    """Lấy thư mục input từ biến môi trường INPUT_DIR, mặc định ../input."""
    input_dir = os.environ.get("INPUT_DIR", "../input")
    path = Path(input_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Thư mục input không tồn tại: {path}")
    return path


def read_collection(name: str) -> Iterator[Dict[str, Any]]:
    """
    Đọc collection từ file JSON trong input/.
    Tự nhận diện file: input/{name}.json (Extended JSON của mongoexport).
    Trả về iterator các dict đã parse đúng kiểu bson (ObjectId, datetime, etc.).
    """
    input_dir = get_input_dir()
    file_path = input_dir / f"{name}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dump: {file_path}")

    # mongoexport xuất ra JSON array (mở đầu bằng [, kết thúc bằng ])
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.loads(f.read())

    if not isinstance(data, list):
        raise ValueError(f"File {file_path} không phải JSON array như mong đợi.")

    for doc in data:
        yield doc


def read_all_documents() -> List[Dict[str, Any]]:
    """Đọc tất cả documents từ collection 'documents'."""
    return list(read_collection("organization_db.documents"))


def read_all_organization_units() -> List[Dict[str, Any]]:
    """Đọc tất cả organization_units."""
    return list(read_collection("organization_db.organization_units"))


def read_all_users() -> List[Dict[str, Any]]:
    """Đọc tất cả users."""
    return list(read_collection("organization_db.users"))


def build_unit_lookup(units: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Tạo lookup dict: unit_id (string) -> {unit_type, campus, name}.
    unit_id lấy từ _id.$oid.
    """
    lookup = {}
    for unit in units:
        oid = unit.get("_id", {}).get("$oid")
        if oid:
            lookup[oid] = {
                "unit_type": unit.get("unit_type"),
                "campus": unit.get("campus"),
                "name": unit.get("name"),
                "code": unit.get("code"),
            }
    return lookup


def denormalize_document(doc: Dict[str, Any], unit_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ghép thông tin organization_unit vào document.
    - doc['organization_unit_id'] là dict {"$oid": "..."}
    - Trả về dict mới chứa thêm unit_type, campus, unit_name, unit_code.
    """
    oid_dict = doc.get("organization_unit_id", {})
    unit_id = oid_dict.get("$oid") if isinstance(oid_dict, dict) else None

    unit_info = unit_lookup.get(unit_id, {}) if unit_id else {}

    return {
        **doc,
        "unit_type": unit_info.get("unit_type"),
        "campus": unit_info.get("campus"),
        "unit_name": unit_info.get("name"),
        "unit_code": unit_info.get("code"),
    }


def denormalize_all_documents(
    limit: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """
    Đọc documents + organization_units, denormalize, trả về iterator.
    Nếu limit > 0: chỉ yield limit document đầu tiên (dùng cho test migration).
    """
    units = read_all_organization_units()
    unit_lookup = build_unit_lookup(units)
    print(f"Đã đọc {len(units)} organization_units, xây dựng lookup xong.")

    count = 0
    for doc in read_collection("organization_db.documents"):
        yield denormalize_document(doc, unit_lookup)
        count += 1
        if limit and count >= limit:
            break

    print(f"Đã denormalize {count} document(s).")


if __name__ == "__main__":
    # Test nhanh: đọc 3 document đầu tiên
    import sys

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"=== Test dump_reader: đọc {limit} document đầu ===")
    for i, doc in enumerate(denormalize_all_documents(limit=limit)):
        print(f"\n--- Document {i+1} ---")
        print(f"_id: {doc.get('_id', {}).get('$oid')}")
        print(f"title: {doc.get('title')}")
        print(f"unit_type: {doc.get('unit_type')}")
        print(f"campus: {doc.get('campus')}")
        print(f"unit_name: {doc.get('unit_name')}")
        print(f"content (200 chars): {doc.get('content', '')[:200]}...")