"""Pipeline runner: ingest -> chunk -> embed -> store.

Run this once after adding documents to documents/ (and again whenever you
change the documents or chunking settings):

    python build_index.py
"""

import ingest
import chunk
import index
import config


def main() -> None:
    print(f"1. Ingesting documents from {config.DOCUMENTS_DIR} ...")
    docs = ingest.load_documents()
    if not docs:
        print("\nNo documents found. Add .txt/.md/.pdf/.html files to "
              f"{config.DOCUMENTS_DIR} and re-run.")
        return
    print(f"   -> {len(docs)} documents loaded.\n")

    print(f"2. Chunking (size={config.CHUNK_SIZE}, "
          f"overlap={config.CHUNK_OVERLAP}) ...")
    chunks = chunk.chunk_documents(docs)
    sizes = [len(c["text"]) for c in chunks]
    print(f"   -> {len(chunks)} chunks "
          f"(avg {sum(sizes) // len(sizes)} chars).\n")

    print(f"3. Embedding with {config.EMBEDDING_MODEL} and storing in ChromaDB ...")
    count = index.build_collection(chunks)
    print(f"   -> {count} chunks embedded and stored in {config.CHROMA_DIR}\n")

    print("Done. Query it with:  python app.py")


if __name__ == "__main__":
    main()
