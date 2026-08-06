import json
import os

from test_generator.parser import load_pdf
from test_generator.hierarchical_chunker import split_documents_hierarchical
from test_generator.exporter import export_chunks

from test_generator.question_generator import generate_questions
from test_generator.candidate_retriever import retrieve_candidates
from test_generator.label_generator import generate_labels
from test_generator.answer_generator import generate_answer


def _save_progress(retrieval_dataset, qa_dataset, failed_items):
    """Write current progress to disk. Safe to call repeatedly -- overwrites
    each time, so a crash never loses more than the current in-flight item."""
    os.makedirs("output", exist_ok=True)

    with open("output/retrieval_test.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_dataset, f, ensure_ascii=False, indent=2)

    with open("output/qa_test.json", "w", encoding="utf-8") as f:
        json.dump(qa_dataset, f, ensure_ascii=False, indent=2)

    with open("output/failed_items.json", "w", encoding="utf-8") as f:
        json.dump(failed_items, f, ensure_ascii=False, indent=2)


def main():

    # ==========================
    # 1. Load & Chunk document
    # ==========================

    documents = []
    for file_path in ["data/Diem-CK_HP.pdf", "data/OOP_2013.pdf"]:
        documents.extend(load_pdf(file_path))

    chunks = split_documents_hierarchical(documents)

    export_chunks(chunks, "output/chunks.json")

    retrieval_dataset = []
    qa_dataset = []
    failed_items = []  # every skipped chunk/question is recorded here, with the error, so nothing silently disappears

    # ==========================
    # 2. Generate Dataset
    # ==========================

    for chunk_index, chunk in enumerate(chunks):

        # --------------------------
        # Question Generation (also an LLM call -- can fail the same way)
        # --------------------------
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
                # --------------------------
                # Candidate Retrieval
                # --------------------------
                candidate_ids = retrieve_candidates(
                    question,
                    chunks,
                    top_k=5
                )

                print("Candidate IDs:", candidate_ids)

                def _print_chunk_refs(ids):
                    refs = []
                    for cid in ids:
                        c = chunks[cid]
                        m = c.get("metadata", {})
                        refs.append({
                            "chunk_id": cid,
                            "source": m.get("source"),
                            "file_name": m.get("file_name"),
                            "snippet": c["text"][:120].strip()
                        })
                    print("Candidate refs:", refs)

                _print_chunk_refs(candidate_ids)

                # --------------------------
                # Label Generation
                # --------------------------
                labels = generate_labels(
                    question_text,
                    candidate_ids,
                    chunks
                )

                print("Labels:", labels)
                if labels:
                    labelled_refs = []
                    for it in labels:
                        cid = it.get("chunk_id")
                        c = chunks[cid]
                        m = c.get("metadata", {})
                        labelled_refs.append({
                            "chunk_id": cid,
                            "relevance": it.get("relevance"),
                            "source": m.get("source"),
                            "file_name": m.get("file_name"),
                            "snippet": c["text"][:120].strip()
                        })
                    print("Labelled refs:", labelled_refs)

                # --------------------------
                # Answer Generation
                # --------------------------
                answer = generate_answer(
                    question_text,
                    labels,
                    chunks
                )

                print("Answer:", answer)

            except Exception as exc:
                # Any LLM call above (retrieval / labeling / answer) can
                # fail even after llm.py's own retries are exhausted (e.g.
                # the server pod stayed down longer than the retry budget).
                # Skip just this question instead of losing the whole run.
                print(f"[SKIP-QUESTION] {question_text!r} failed: {exc}")
                failed_items.append({
                    "stage": "retrieve/label/answer",
                    "question": question,
                    "error": str(exc),
                })
                _save_progress(retrieval_dataset, qa_dataset, failed_items)
                continue

            # --------------------------
            # Retrieval Sample (include chunk origin metadata)
            # --------------------------

            def _chunk_info(item):
                cid = item.get("chunk_id")
                chunk_ = chunks[cid]
                meta = chunk_.get("metadata", {})
                snippet = chunk_["text"][:200].strip()
                return {
                    "chunk_id": cid,
                    "relevance": item.get("relevance", 0),
                    "source": meta.get("source"),
                    "file_name": meta.get("file_name"),
                    "text_snippet": snippet,
                }

            retrieval_dataset.append(
                {
                    "question": question,
                    "ground_truth_chunks": [_chunk_info(it) for it in labels]
                }
            )

            # --------------------------
            # QA Sample (include supporting chunk metadata)
            # --------------------------

            qa_dataset.append(
                {
                    "question": question,
                    "ground_truth_answer": answer,
                    "supporting_chunks": [_chunk_info(it) for it in labels]
                }
            )

            # Save after every successful question too, so a crash never
            # rolls back further than the item currently in flight.
            _save_progress(retrieval_dataset, qa_dataset, failed_items)

    # ==========================
    # 3. Final export (progress was already saved incrementally above,
    #    this just guarantees the final on-disk state matches memory)
    # ==========================

    _save_progress(retrieval_dataset, qa_dataset, failed_items)

    print()
    print("=" * 80)
    print("Done!")
    print(f"Retrieval samples : {len(retrieval_dataset)}")
    print(f"QA samples        : {len(qa_dataset)}")
    print(f"Failed items      : {len(failed_items)} (see output/failed_items.json)")


if __name__ == "__main__":
    main()