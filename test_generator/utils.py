# utils.py

def assign_chunk_ids(chunks):
    """
    Gán ID duy nhất cho mỗi chunk.
    """
    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            metadata = chunk.setdefault("metadata", {})
            metadata["chunk_id"] = idx
            chunk["chunk_id"] = idx
        else:
            setattr(chunk, "chunk_id", idx)
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                chunk.metadata["chunk_id"] = idx
    return chunks