"""Central configuration for The Unofficial Guide RAG pipeline.

Every tunable knob lives here so the rest of the code reads cleanly and so the
choices documented in planning.md / README.md map to a single place.
"""

from pathlib import Path

from dotenv import load_dotenv
import os

# Load GROQ_API_KEY (and anything else) from .env in the repo root.
load_dotenv()

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).parent
DOCUMENTS_DIR = ROOT / "documents"          # raw documents you collect
CHROMA_DIR = ROOT / "chroma_db"             # persistent vector store (gitignored)
COLLECTION_NAME = "unofficial_guide"

# --- Chunking ----------------------------------------------------------------
# Sizes are in CHARACTERS (not tokens). See planning.md "Chunking Strategy" for
# the reasoning behind these numbers.
CHUNK_SIZE = 700            # target characters per chunk
CHUNK_OVERLAP = 150         # characters of overlap between consecutive chunks
MIN_CHUNK_SIZE = 80         # discard trailing fragments smaller than this

# --- Embeddings --------------------------------------------------------------
# all-MiniLM-L6-v2: 384-dim, runs locally on CPU, ~256 word-piece input cap.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval ---------------------------------------------------------------
TOP_K = 4                  # how many chunks to retrieve per query
# Chroma returns cosine *distance* (0 = identical). We drop anything above this
# so the generator isn't fed near-irrelevant context.
MAX_DISTANCE = 1.2

# --- Generation (Groq) -------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
GENERATION_TEMPERATURE = 0.1   # low temp = stay close to the retrieved text
