import json

from parser import load_pdf
from chunker import split_documents
from exporter import export_chunks

from question_generator import generate_questions
from candidate_retriever import retrieve_candidates
from label_generator import generate_labels
from answer_generator import generate_answer


def main():

    # ==========================
    # 1. Load & Chunk document
    # ==========================

    documents = load_pdf("data/Diem-CK_HP.pdf")

    chunks = split_documents(documents)

    export_chunks(chunks, "output/chunks.json")

    retrieval_dataset = []
    qa_dataset = []

    # ==========================
    # 2. Generate Dataset
    # ==========================

    for chunk in chunks:

        questions = generate_questions(chunk)

        for question in questions:

            print("=" * 80)
            print(question)

            # --------------------------
            # Candidate Retrieval
            # --------------------------

            candidate_ids = retrieve_candidates(
                question,
                chunks,
                top_k=5
            )

            print("Candidate IDs:", candidate_ids)

            # --------------------------
            # Label Generation
            # --------------------------

            labels = generate_labels(
                question,
                candidate_ids,
                chunks
            )

            print("Labels:", labels)

            # --------------------------
            # Answer Generation
            # --------------------------

            answer = generate_answer(
                question,
                labels,
                chunks
            )

            print("Answer:", answer)

            # --------------------------
            # Retrieval Sample
            # --------------------------

            retrieval_dataset.append(
                {
                    "question": question,
                    "ground_truth_chunks": labels
                }
            )

            # --------------------------
            # QA Sample
            # --------------------------

            qa_dataset.append(
                {
                    "question": question,
                    "ground_truth_answer": answer,
                    "supporting_chunks": [
                        item["chunk_id"]
                        for item in labels
                    ]
                }
            )

    # ==========================
    # 3. Export
    # ==========================

    with open(
        "output/retrieval_test.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            retrieval_dataset,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        "output/qa_test.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            qa_dataset,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("=" * 80)
    print("Done!")
    print(f"Retrieval samples : {len(retrieval_dataset)}")
    print(f"QA samples        : {len(qa_dataset)}")


if __name__ == "__main__":
    main()