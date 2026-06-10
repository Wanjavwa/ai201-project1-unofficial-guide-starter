"""Embedding + vector store + semantic retrieval.

Uses sentence-transformers (all-MiniLM-L6-v2) to embed chunks locally, stores
them in a persistent ChromaDB collection, and exposes a semantic `search()`.

We compute embeddings ourselves (rather than letting Chroma's default embedder
do it) so the exact same model is used at index time and query time, and so the
embedding model is an explicit, documented choice.
"""

from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts into normalized vectors."""
    vectors = _model().encode(
        texts,
        normalize_embeddings=True,   # cosine distance behaves well on unit vectors
        show_progress_bar=len(texts) > 64,
    )
    return vectors.tolist()


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def build_collection(chunks: list[dict]) -> int:
    """(Re)build the Chroma collection from chunk records. Returns chunk count."""
    client = _client()
    # Start clean so re-running ingestion never leaves stale chunks behind.
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    embeddings = embed(texts)
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "doc_id": c["doc_id"],
                "source": c["source"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ],
    )
    return collection.count()


def get_collection():
    """Return the existing collection, or raise a helpful error if missing."""
    client = _client()
    try:
        return client.get_collection(config.COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Vector store not found. Run `python build_index.py` first "
            "(after adding documents to documents/)."
        ) from exc


def search(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """Semantic search. Returns up to top_k chunk hits sorted by relevance.

    Each hit:
      {"text": "...", "source": "...", "doc_id": "...",
       "chunk_index": 2, "distance": 0.31}

    Hits with cosine distance above config.MAX_DISTANCE are filtered out.
    """
    collection = get_collection()
    query_embedding = embed([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for text, meta, dist in zip(docs, metas, dists):
        if dist > config.MAX_DISTANCE:
            continue
        hits.append({
            "text": text,
            "source": meta["source"],
            "doc_id": meta["doc_id"],
            "chunk_index": meta["chunk_index"],
            "distance": round(float(dist), 4),
        })
    return hits


if __name__ == "__main__":
    # Quick retrieval smoke test against an already-built index.
    import sys

    q = " ".join(sys.argv[1:]) or "Is the housing lottery actually random?"
    print(f"Query: {q}\n")
    for i, hit in enumerate(search(q), 1):
        print(f"[{i}] {hit['source']} (chunk {hit['chunk_index']}, "
              f"dist {hit['distance']})")
        print(f"    {hit['text'][:200]}...\n")
