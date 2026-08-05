import re

from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils import assign_chunk_ids

DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


def _split_text_with_regex(text: str, separator: str, keep_separator: bool = True) -> list[str]:
    if separator == "":
        return [text]

    sep_pattern = re.escape(separator)
    if keep_separator:
        splits_ = re.split(f"({sep_pattern})", text)
        if keep_separator == "end":
            splits = [splits_[i] + splits_[i + 1] for i in range(0, len(splits_) - 1, 2)]
        else:
            splits = [splits_[i] + splits_[i + 1] for i in range(1, len(splits_), 2)]
        if len(splits_) % 2 == 0:
            splits += splits_[-1:]
        return [s for s in splits if s]

    return [s for s in re.split(sep_pattern, text) if s]


def _merge_splits(splits: list[str], separator: str) -> list[str]:
    if not splits:
        return []

    merged = [splits[0]]
    for split in splits[1:]:
        if len(merged[-1]) + len(separator) + len(split) <= CHUNK_SIZE:
            merged[-1] = merged[-1] + separator + split
        else:
            merged.append(split)
    return merged


def _tla_split_text(text: str, separators: list[str]) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]

    separator = separators[-1]
    new_separators = []
    for i, s_ in enumerate(separators):
        if not s_ or re.search(re.escape(s_), text):
            separator = s_
            new_separators = separators[i + 1 :]
            break

    splits = _split_text_with_regex(text, separator, keep_separator=True)
    final_chunks: list[str] = []
    good_splits: list[str] = []
    merge_sep = "" if separator == "" else separator

    for split in splits:
        if len(split) <= CHUNK_SIZE:
            good_splits.append(split)
        else:
            if good_splits:
                final_chunks.extend(_merge_splits(good_splits, merge_sep))
                good_splits = []
            if not new_separators:
                final_chunks.append(split)
            else:
                final_chunks.extend(_tla_split_text(split, new_separators))

    if good_splits:
        final_chunks.extend(_merge_splits(good_splits, merge_sep))

    return final_chunks


def split_documents(documents):
    """Split documents into chunks using the TLA chunking mechanism.

    This implementation no longer depends directly on langchain's
    RecursiveCharacterTextSplitter. It recursively splits content by
    paragraph, line, and space, then applies a final character fallback.
    """
    chunks = []
    for doc in documents:
        texts = _tla_split_text(doc.page_content, DEFAULT_SEPARATORS)
        for text in texts:
            chunks.append(
                {
                    "chunk_id": None,
                    "text": text,
                    "metadata": {**doc.metadata}
                }
            )

    chunks = assign_chunk_ids(chunks)
    for chunk in chunks:
        chunk["metadata"]["chunk_id"] = chunk["chunk_id"]

    return chunks