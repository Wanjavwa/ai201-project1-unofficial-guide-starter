"""CLI query interface for The Unofficial Guide.

Usage:
    python app.py                      # interactive prompt loop
    python app.py "your question"      # one-shot question
    python app.py --show-chunks ...    # also print the retrieved chunks

Type a campus-housing question and get a grounded, cited answer.
"""

import sys

import generate


def _print_result(result: dict, show_chunks: bool) -> None:
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])

    print("\n" + "-" * 70)
    print("SOURCES")
    print("-" * 70)
    if result["sources"]:
        for src in result["sources"]:
            print(f"  - {src}")
    else:
        print("  (no sufficiently relevant sources found)")

    if show_chunks:
        print("\n" + "-" * 70)
        print("RETRIEVED CHUNKS")
        print("-" * 70)
        for i, hit in enumerate(result["hits"], 1):
            print(f"\n[{i}] {hit['source']} "
                  f"(chunk {hit['chunk_index']}, distance {hit['distance']})")
            print(f"    {hit['text']}")
    print()


def _ask(query: str, show_chunks: bool) -> None:
    print("\nSearching the collected documents ...")
    result = generate.answer_query(query)
    _print_result(result, show_chunks)


def main() -> None:
    args = sys.argv[1:]
    show_chunks = False
    if "--show-chunks" in args:
        show_chunks = True
        args = [a for a in args if a != "--show-chunks"]

    if args:  # one-shot mode
        _ask(" ".join(args), show_chunks)
        return

    # Interactive mode
    print("=" * 70)
    print("THE UNOFFICIAL GUIDE — campus housing Q&A")
    print("Ask a question, or type 'quit' to exit. "
          "Add '--show-chunks' below for retrieval detail.")
    print("=" * 70)
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        _ask(query, show_chunks)


if __name__ == "__main__":
    main()
