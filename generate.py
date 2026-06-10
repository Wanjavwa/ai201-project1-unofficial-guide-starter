"""Grounded answer generation via Groq (llama-3.3-70b-versatile).

The model is instructed to answer ONLY from the retrieved chunks and to cite the
source of each claim. If the chunks don't contain the answer, it must say so
rather than fall back on its own knowledge.
"""

from groq import Groq

import config

SYSTEM_PROMPT = """You are The Unofficial Guide, a question-answering assistant \
for student-shared knowledge about campus housing.

Rules you MUST follow:
1. Answer ONLY using the information in the CONTEXT below. Do not use outside or \
prior knowledge, and do not guess.
2. If the context does not contain enough information to answer, reply exactly: \
"I don't have enough information in the collected documents to answer that." \
Do not invent details.
3. Cite your sources inline using the bracketed source label that precedes each \
context block, e.g. [dorm_reviews.txt]. Every claim must be traceable to a source.
4. Be concise and concrete. Prefer what students actually reported over vague \
generalities. If sources disagree, say so and cite both.
"""


def _format_context(hits: list[dict]) -> str:
    """Render retrieved chunks into a labeled context block for the prompt."""
    blocks = []
    for hit in hits:
        label = f"[{hit['source']}]"
        blocks.append(f"{label}\n{hit['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, hits: list[dict]) -> str:
    """Generate a grounded answer from retrieved chunks."""
    if not hits:
        return ("I don't have enough information in the collected documents to "
                "answer that.")

    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one free at https://console.groq.com)."
        )

    client = Groq(api_key=config.GROQ_API_KEY)
    context = _format_context(hits)
    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer using only the context above, with inline [source] citations."
    )

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=config.GENERATION_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


def answer_query(query: str, top_k: int = config.TOP_K) -> dict:
    """End-to-end: retrieve, generate, and return answer + sources + hits."""
    import index  # local import to avoid loading the embedder unless needed

    hits = index.search(query, top_k=top_k)
    answer = generate_answer(query, hits)
    sources = sorted({h["source"] for h in hits})
    return {"query": query, "answer": answer, "sources": sources, "hits": hits}
