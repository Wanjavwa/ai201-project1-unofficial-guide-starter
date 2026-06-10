"""Chunking strategy.

Approach: paragraph-aware packing with character overlap.

Rather than blindly cutting every N characters (which slices sentences in half),
we first split on paragraph/blank-line boundaries, then greedily pack paragraphs
into chunks up to CHUNK_SIZE. When a single paragraph is larger than CHUNK_SIZE
(common in long-form housing guides), we fall back to a sentence-aware split so
we never exceed the embedding model's input window.

Consecutive chunks share CHUNK_OVERLAP characters of tail text so a fact that
lands near a boundary still appears whole in at least one chunk.

See planning.md / README.md "Chunking Strategy" for the reasoning.
"""

import re

import config


def _split_paragraphs(text: str) -> list[str]:
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter: break after . ! ? followed by whitespace.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _hard_wrap(unit: str, size: int) -> list[str]:
    """Last resort: a single unit longer than `size` is cut on char boundaries."""
    return [unit[i:i + size] for i in range(0, len(unit), size)]


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    min_chunk_size: int = config.MIN_CHUNK_SIZE,
) -> list[str]:
    """Split one document's text into overlapping, paragraph-aware chunks."""
    # Break text into the smallest natural units we'll pack: paragraphs, and
    # sentences for oversized paragraphs.
    units: list[str] = []
    for para in _split_paragraphs(text):
        if len(para) <= chunk_size:
            units.append(para)
        else:
            for sent in _split_sentences(para):
                units.extend(
                    _hard_wrap(sent, chunk_size) if len(sent) > chunk_size else [sent]
                )

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        # Candidate would overflow: flush current chunk, then start a new one
        # seeded with the overlap tail of what we just flushed.
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{unit}".strip() if tail else unit
        else:
            current = unit

    if current and (len(current) >= min_chunk_size or not chunks):
        chunks.append(current)

    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Turn loaded documents into chunk records ready for embedding.

    Each chunk record:
      {"id": "dorm_reviews_reddit::3",
       "text": "...",
       "doc_id": "dorm_reviews_reddit",
       "source": "dorm_reviews_reddit.txt",
       "chunk_index": 3}
    """
    records = []
    for doc in documents:
        pieces = chunk_text(doc["text"])
        for i, piece in enumerate(pieces):
            records.append({
                "id": f"{doc['doc_id']}::{i}",
                "text": piece,
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "chunk_index": i,
            })
    return records


if __name__ == "__main__":
    import ingest

    docs = ingest.load_documents()
    records = chunk_documents(docs)
    sizes = [len(r["text"]) for r in records]
    print(f"\n{len(records)} chunks from {len(docs)} documents.")
    if sizes:
        print(f"  chunk size: min={min(sizes)} max={max(sizes)} "
              f"avg={sum(sizes) // len(sizes)} chars")
