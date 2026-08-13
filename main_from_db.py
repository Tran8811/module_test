import json
import os

from test_generator.pg_reader import fetch_chunks_for_documents, list_documents
from test_generator.exporter import export_chunks

from test_generator.question_generator import generate_questions
from test_generator.candidate_retriever import retrieve_candidates
from test_generator.label_generator import generate_labels
from test_generator.answer_generator import generate_answer


# ĐIỀN document_id thật muốn gen test vào đây (lấy từ list_documents() bên
# dưới, hoặc để rỗng [] để tool tự in ra 20 document đầu cho bạn chọn).
DOCUMENT_IDS: list[str] = ["248e2b74-233d-4041-a841-f640aea97346"]

# Đặt True để lấy TẤT CẢ document trong DB thay vì chỉ DOCUMENT_IDS ở trên.
ALL_DOCUMENTS = False

# Đặt True để lấy MỘT NỬA số document trong DB (194 -> ~97)
HALF_DOCUMENTS = True


def _index_by_id(chunks):
    return {c["chunk_id"]: c for c in chunks}


def _save_progress(retrieval_dataset, qa_dataset, failed_items):
    os.makedirs("output", exist_ok=True)

    with open("output/retrieval_test_db.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_dataset, f, ensure_ascii=False, indent=2)

    with open("output/qa_test_db.json", "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

    with open("output/failed_items_db.json", "w", encoding="utf-8") as f:
        json.dump(failed_items, f, ensure_ascii=False, indent=2)


def main():

    # ==========================
    # 1. Lấy chunk từ DB (thay cho load_pdf + chunk_documents)
    # ==========================

    document_ids = DOCUMENT_IDS

    if ALL_DOCUMENTS:
        docs = list_documents(limit=None)
        document_ids = [d["id"] for d in docs]
        print(f"ALL_DOCUMENTS=True -> lấy toàn bộ {len(document_ids)} document trong DB")
    elif HALF_DOCUMENTS:
        docs = list_documents(limit=None)
        half = len(docs) // 2
        document_ids = [d["id"] for d in docs[:half]]
        print(f"HALF_DOCUMENTS=True -> lấy {len(document_ids)}/{len(docs)} document đầu (một nửa)")

    if not document_ids:
        docs = list_documents(limit=20)
        print("Chưa điền DOCUMENT_IDS. 20 document đầu trong DB:")
        for d in docs:
            print(f"  {d['id']}  |  {d['name']}")
        print("\nSửa DOCUMENT_IDS = [\"<id>\", ...] ở đầu file rồi chạy lại.")
        return

    chunks = fetch_chunks_for_documents(document_ids)
    print(f"Lấy được {len(chunks)} chunk từ {len(document_ids)} document trong DB")

    if not chunks:
        print("0 chunk -- kiểm tra lại document_id, hoặc document đó chưa có node nào có embedding.")
        return

    export_chunks(chunks, "output/chunks_from_db.json")

    # Lookup theo chunk_id thật (đặc biệt quan trọng ở luồng DB, vì
    # chunk_id là global id, không đảm bảo trùng vị trí trong subset này).
    chunks_by_id = _index_by_id(chunks)

    retrieval_dataset = []
    qa_dataset = []
    failed_items = []

    # ==========================
    # 2. Sinh dataset -- Y HỆT main.py, không đổi gì phần này
    # ==========================

    for chunk_index, chunk in enumerate(chunks):

        try:
            questions = generate_questions(chunk, all_chunks=chunks)
        except Exception as exc:
            print(f"[SKIP-CHUNK] chunk_id={chunk.get('chunk_id')} question generation failed: {exc}")
            failed_items.append({
                "stage": "generate_questions",
                "chunk_id": chunk.get("chunk_id"),
                "error": str(exc),
            })
            continue

        for question in questions:
            question_text = question["question"]
            print("=" * 80)
            print(question)

            try:
                candidate_ids = retrieve_candidates(question, chunks, top_k=5)
                print("Candidate IDs:", candidate_ids)

                labels = generate_labels(question, candidate_ids, chunks)
                print("Labels:", labels)

                answer = generate_answer(question_text, labels, chunks)
                print("Answer:", answer)

            except Exception as exc:
                print(f"[SKIP-QUESTION] {question_text!r} failed: {exc}")
                failed_items.append({
                    "stage": "retrieve/label/answer",
                    "question": question,
                    "error": str(exc),
                })
                _save_progress(retrieval_dataset, qa_dataset, failed_items)
                continue

            def _chunk_info(item):
                cid = item.get("chunk_id")
                chunk_ = chunks_by_id.get(cid)
                if chunk_ is None:
                    return {
                        "chunk_id": cid,
                        "relevance": item.get("relevance", 0),
                        "source": None,
                        "document_id": None,
                        "text_snippet": None,
                        "error": "unknown chunk_id",
                    }
                meta = chunk_.get("metadata", {})
                return {
                    "chunk_id": cid,
                    "relevance": item.get("relevance", 0),
                    "source": meta.get("source"),
                    "document_id": meta.get("document_id"),
                    "text_snippet": chunk_["text"][:200].strip(),
                }

            retrieval_dataset.append({
                "question": question,
                "ground_truth_chunks": [_chunk_info(it) for it in labels],
            })

            qa_dataset.append({
                "question": question,
                "ground_truth_answer": answer,
                "supporting_chunks": [_chunk_info(it) for it in labels],
            })

            _save_progress(retrieval_dataset, qa_dataset, failed_items)

    _save_progress(retrieval_dataset, qa_dataset, failed_items)

    print()
    print("=" * 80)
    print("Done!")
    print(f"Retrieval samples : {len(retrieval_dataset)}")
    print(f"QA samples        : {len(qa_dataset)}")
    print(f"Failed items      : {len(failed_items)} (see output/failed_items_db.json)")


if __name__ == "__main__":
    main()