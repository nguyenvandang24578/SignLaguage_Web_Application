#!/usr/bin/env python3
"""
Script đọc sign_language_data.json và xóa trường "Loại" khỏi tất cả các mục.
"""

import json
import os

FILE_PATH = os.path.join(os.path.dirname(__file__), "sign_language_data.json")

# Đọc file JSON
with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Tổng số mục: {len(data)}")

# Xóa trường "Loại" khỏi từng mục
removed_count = 0
for item in data:
    if "Loại" in item:
        del item["Loại"]
        removed_count += 1

print(f"Đã xóa trường 'Loại' khỏi {removed_count} mục.")

# Ghi lại file JSON (pretty print để giữ định dạng dễ đọc)
with open(FILE_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Đã ghi lại file: {FILE_PATH}")
