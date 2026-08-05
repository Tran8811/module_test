import json
import os


def export_chunks(chunks, output_file):
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(chunks)} chunks to {output_file}")