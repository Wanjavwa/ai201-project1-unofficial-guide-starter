"""Document ingestion pipeline.

Loads every supported file from documents/, extracts plain text, and cleans it
(strips HTML, collapses whitespace, removes obvious navigation/boilerplate).

Supported formats:
  .txt / .md   -> read directly
  .pdf         -> pdfplumber text extraction (digital PDFs only, no OCR)
  .html / .htm -> BeautifulSoup, scripts/styles/nav removed

Each document becomes a dict:
  {"doc_id": "dorm_reviews_reddit", "source": "dorm_reviews_reddit.txt", "text": "..."}
"""

import re
from pathlib import Path

import config

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".htm"}

# Lines that are almost always navigation/boilerplate noise rather than content.
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(home|menu|search|log\s?in|sign\s?up|subscribe)\s*$", re.I),
    re.compile(r"^\s*(share|reply|report|save|upvote|downvote)\s*$", re.I),
    re.compile(r"^\s*\d+\s*comments?\s*$", re.I),
    re.compile(r"^\s*(cookie|privacy)\s+policy", re.I),
    re.compile(r"^\s*©.*$"),
]


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    import pdfplumber  # imported lazily so non-PDF users don't need it

    with pdfplumber.open(path) as pdf:
        pages = [p.extract_text() for p in pdf.pages]
    return "\n\n".join(t for t in pages if t)


def _read_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_read_text_file(path), "html.parser")
    # Remove non-content elements before extracting text.
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _clean_text(text: str) -> str:
    """Normalize whitespace and drop boilerplate lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            cleaned_lines.append("")  # keep blank line as paragraph separator
            continue
        if any(p.match(line) for p in _BOILERPLATE_PATTERNS):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    # Collapse 3+ newlines into a clean paragraph break, and runs of spaces.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _extract(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return _read_text_file(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext in {".html", ".htm"}:
        return _read_html(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def load_documents(documents_dir: Path = config.DOCUMENTS_DIR) -> list[dict]:
    """Return a list of cleaned document dicts from the documents directory."""
    documents = []
    files = sorted(
        p for p in documents_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for path in files:
        raw = _extract(path)
        text = _clean_text(raw)
        if not text:
            print(f"  [skip] {path.name}: no extractable text "
                  f"(scanned/image-only PDF?)")
            continue
        documents.append({
            "doc_id": path.stem,
            "source": path.name,
            "text": text,
        })
        print(f"  [load] {path.name}: {len(text)} chars")

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents "
          f"({sum(len(d['text']) for d in docs)} total chars).")
