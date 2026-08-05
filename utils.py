# utils.py

def assign_chunk_ids(chunks):
    """
    Gán ID duy nhất cho mỗi chunk.
    """
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks