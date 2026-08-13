import re
from typing import Iterable

from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
)
from .utils import assign_chunk_ids

DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]

# PDF table extraction (PyPDFLoader) usually loses row boundaries for
# tabular score sheets like "STT. MSSV Họ tên ... scores ...", producing
# one long blob of text with no real newlines between student rows. The
# word-count-based splitter then cuts wherever it hits the token budget,
# which can (and does) slice a student's name in half across two chunks
# -- e.g. one chunk ending in "...Ngô" and the next starting with "Đức
# Thịnh...". Once that happens, no amount of retrieval-scoring fixes can
# find that student, because the full name never appears intact in any
# single chunk.
#
# Fix: before running the token-aware splitter, insert an explicit
# newline right before every detected row start ("<STT>. <8-digit MSSV>
# <Name...>"), so "\n\n"/"\n" separators can then split cleanly on row
# boundaries and a row is never torn in the middle.
_TABLE_ROW_START_RE = re.compile(r"(?<!^)(?<!\n)(\d{1,3}\.\s+\d{8}\s+[A-ZÀ-Ỵ])")


def _normalize_table_row_boundaries(text: str) -> str:
    return _TABLE_ROW_START_RE.sub(r"\n\1", text)


def _token_count(text: str) -> int:
    """Approximate token count using word-splitting.

    This is an approximation; for more accurate counts replace with a
    proper tokenizer (e.g. tiktoken) tuned to your model.
    """
    if not text:
        return 0
    return len(text.split())


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


def _merge_splits_token_aware(splits: list[str], separator: str) -> list[str]:
    """Merge adjacent splits while respecting token limits."""
    if not splits:
        return []

    merged = [splits[0]]
    for split in splits[1:]:
        sep = "" if separator == "" else separator
        if _token_count(merged[-1] + sep + split) <= CHUNK_MAX_TOKENS:
            merged[-1] = merged[-1] + sep + split
        else:
            merged.append(split)
    return merged


def _tla_split_text(text: str, separators: list[str]) -> list[str]:
    """Token-aware recursive TLA split.

    Uses approximate token counts to decide chunk boundaries. Falls back
    to character-length checks for very short texts if token settings
    aren't changed.
    """
    if _token_count(text) <= CHUNK_MAX_TOKENS:
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
        if _token_count(split) <= CHUNK_MAX_TOKENS:
            good_splits.append(split)
        else:
            if good_splits:
                final_chunks.extend(_merge_splits_token_aware(good_splits, merge_sep))
                good_splits = []
            if not new_separators:
                final_chunks.append(split)
            else:
                final_chunks.extend(_tla_split_text(split, new_separators))

    if good_splits:
        final_chunks.extend(_merge_splits_token_aware(good_splits, merge_sep))

    return final_chunks


def _extract_media_blocks(text: str) -> Iterable[tuple[str, bool]]:
    """Yield (segment, is_media) where media are HTML tables or images.

    This keeps tables and images as separate segments so they won't be
    broken across chunks.
    """
    # Pattern matches <table ...>...</table> or <img .../> blocks (case-insensitive)
    table_re = re.compile(r"(<table[\s\S]*?</table>)", re.IGNORECASE)
    img_re = re.compile(r"(<img[^>]*>)", re.IGNORECASE)

    idx = 0
    # First split by tables, then by images inside non-table parts.
    for m in table_re.finditer(text):
        if m.start() > idx:
            pre = text[idx : m.start()]
            # split images out of pre
            last = 0
            for im in img_re.finditer(pre):
                if im.start() > last:
                    yield pre[last:im.start()], False
                yield im.group(1), True
                last = im.end()
            if last < len(pre):
                yield pre[last:], False
        yield m.group(1), True
        idx = m.end()

    if idx < len(text):
        tail = text[idx:]
        last = 0
        for im in img_re.finditer(tail):
            if im.start() > last:
                yield tail[last:im.start()], False
            yield im.group(1), True
            last = im.end()
        if last < len(tail):
            yield tail[last:], False


def _apply_overlap(chunks: list[dict]) -> None:
    """Mutate chunk texts to include overlap from previous chunk (word-based)."""
    if not chunks or CHUNK_OVERLAP_TOKENS <= 0:
        return

    for i in range(1, len(chunks)):
        prev = chunks[i - 1]["text"]
        cur = chunks[i]["text"]
        prev_tokens = prev.split()
        overlap_n = min(CHUNK_OVERLAP_TOKENS, len(prev_tokens))
        if overlap_n > 0:
            tail = " ".join(prev_tokens[-overlap_n:])
            # avoid duplicating if overlap already present
            if not cur.startswith(tail):
                chunks[i]["text"] = tail + "\n" + cur


def split_documents(documents):
    chunks = []
    for doc in documents:
        content = doc.page_content or ""
        base_meta = {**(getattr(doc, "metadata", {}) or {})}
        source = base_meta.get("source") or base_meta.get("file_name") or base_meta.get("path")
        if not source and hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            # try common langchain loader keys
            source = doc.metadata.get("source") or doc.metadata.get("file_name")

        if source:
            base_meta["source"] = source
        else:
            base_meta.setdefault("source", "unknown")

        # Keep any page information if present
        if "page" in base_meta:
            base_meta["page"] = base_meta.get("page")

        for segment, is_media in _extract_media_blocks(content):
            if is_media:
                # keep media blocks as single chunks
                chunks.append({"chunk_id": None, "text": segment, "metadata": {**base_meta}})
            else:
                segment = _normalize_table_row_boundaries(segment)
                texts = _tla_split_text(segment, DEFAULT_SEPARATORS)
                for text in texts:
                    chunks.append({"chunk_id": None, "text": text, "metadata": {**base_meta}})

    chunks = assign_chunk_ids(chunks)
    for chunk in chunks:
        # Ensure chunk metadata includes chunk_id and preserve source/file_name/page
        chunk_meta = chunk.get("metadata", {}) or {}
        chunk_meta["chunk_id"] = chunk["chunk_id"]
        # Normalize file_name key if present in source path
        if "file_name" not in chunk_meta and chunk_meta.get("source") and "/" in chunk_meta.get("source"):
            chunk_meta["file_name"] = chunk_meta["source"].split("/")[-1]
        chunk["metadata"] = chunk_meta

    _apply_overlap(chunks)
    return chunks


def split_text_items(text_items):
    """Split a list of plain text items into TLA-style token-aware chunks.

    Accepts a list of dicts: {"text": str, "metadata": { ... }} and
    returns a list of chunk dicts similar to `split_documents`.
    """
    chunks = []
    for item in text_items:
        text = item.get("text", "") or ""
        metadata = item.get("metadata", {}) or {}

        # Ensure metadata contains a `source` field when possible
        base_meta = {**metadata}
        source = base_meta.get("source") or base_meta.get("file_name") or base_meta.get("path")
        if source:
            base_meta["source"] = source
        else:
            base_meta.setdefault("source", "unknown")

        for segment, is_media in _extract_media_blocks(text):
            if is_media:
                chunks.append({"chunk_id": None, "text": segment, "metadata": {**base_meta}})
            else:
                segment = _normalize_table_row_boundaries(segment)
                texts = _tla_split_text(segment, DEFAULT_SEPARATORS)
                for t in texts:
                    chunks.append({"chunk_id": None, "text": t, "metadata": {**base_meta}})

    chunks = assign_chunk_ids(chunks)
    for chunk in chunks:
        chunk_meta = chunk.get("metadata", {}) or {}
        chunk_meta["chunk_id"] = chunk["chunk_id"]
        if "file_name" not in chunk_meta and chunk_meta.get("source") and "/" in chunk_meta.get("source"):
            chunk_meta["file_name"] = chunk_meta["source"].split("/")[-1]
        chunk["metadata"] = chunk_meta

    _apply_overlap(chunks)
    return chunks