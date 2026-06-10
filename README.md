# The Unofficial Guide — Project 1

A Retrieval-Augmented Generation (RAG) system that answers plain-language
questions about **campus housing at Maple State University** using real
student-generated knowledge — dorm reviews, off-campus apartment warnings,
lottery threads, and crowd-sourced cost data — and cites its sources.

```
You: Is the housing lottery actually random?
Guide: No. The selection time ("time ticket") is weighted by earned credit
       hours, so juniors and seniors pick before sophomores; randomness only
       breaks ties between students with the same credits. Honors students get
       an earlier priority window. [reddit_housing_lottery.txt]
Sources: reddit_housing_lottery.txt, lottery_appeals_and_waitlist.txt
```

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; use source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt

cp .env.example .env              # then put your free Groq key in .env (console.groq.com)

python build_index.py            # ingest -> chunk -> embed -> store (run once)
python app.py                    # interactive Q&A
python app.py "which dorm has no AC?" --show-chunks   # one-shot, with retrieval detail
python evaluate.py               # run the 5 test questions
```

**Project layout**

| File | Role |
|------|------|
| `config.py` | All tunable settings (chunk size, model names, top-k, thresholds) |
| `ingest.py` | Stage 1 — load + clean documents (`.txt/.md/.pdf/.html`) |
| `chunk.py` | Stage 2 — paragraph-aware chunking |
| `index.py` | Stages 3–4 — embed, store in ChromaDB, semantic `search()` |
| `generate.py` | Stage 5 — grounded answer generation via Groq |
| `build_index.py` | Runs stages 1–3 end to end |
| `app.py` | CLI query interface |
| `evaluate.py` | Runs the 5 test questions and writes `eval_results.md` |

> **Corpus note:** the files in `documents/` are a realistic hand-written sample
> corpus for a fictional school so the system is runnable and demoable out of the
> box. Swap in documents collected from your own school — the pipeline ingests
> anything you drop into `documents/`.

---

## Domain

Campus housing knowledge — which dorm has AC, whether the lottery is really
random, which landlords ignore mold, how the meal-plan rule actually works. This
is valuable because the official housing portal describes rooms and policies in
neutral marketing language: it will never tell you North Tower has no AC and is
brutal in early September, or that Maple Court paints over mold and stonewalls
maintenance tickets. That knowledge lives in subreddits, GroupMe/Discord chats,
and crowd-sourced spreadsheets — scattered, anonymous, and contradictory, which
is exactly what retrieval is good at consolidating.

---

## Document Sources

| #  | Source | Type | URL or file path |
|----|--------|------|------------------|
| 1  | r/MapleState — "Is the lottery random?" | Forum thread | `documents/reddit_housing_lottery.txt` |
| 2  | r/MapleState — selection part 2 | Forum thread | `documents/lottery_appeals_and_waitlist.txt` |
| 3  | North Tower review | Dorm review | `documents/dorm_review_north_tower.txt` |
| 4  | Westfield Suites review | Dorm review | `documents/dorm_review_westfield_suites.txt` |
| 5  | Riverside Hall review | Dorm review | `documents/dorm_review_riverside.txt` |
| 6  | Maple Court Apartments | Off-campus review | `documents/offcampus_maple_court_apartments.txt` |
| 7  | College View Flats | Off-campus review | `documents/offcampus_college_view.txt` |
| 8  | Meal plan requirement Q&A | New-student Q&A | `documents/meal_plan_requirement.txt` |
| 9  | Roommate matching | Forum thread | `documents/roommate_matching_tips.txt` |
| 10 | Move-in day tips | Discord pinned post | `documents/move_in_day_tips.txt` |
| 11 | Amenities cheat sheet | Community wiki | `documents/laundry_and_amenities.txt` |
| 12 | Cost comparison | Crowd-sourced sheet | `documents/cost_comparison.txt` |

**Ingestion pipeline** (`ingest.py`): each file is loaded by type (`.txt/.md`
read directly, `.pdf` via pdfplumber, `.html` via BeautifulSoup with
scripts/nav/footer stripped), then cleaned — boilerplate lines (login/share/
upvote/cookie-policy/etc.) are dropped, Windows line endings normalized, runs of
blank lines collapsed to single paragraph breaks, and repeated spaces collapsed.
Output is one clean text blob per document, ready for chunking. Documents that
yield no text (e.g. scanned image-only PDFs) are skipped with a warning.

---

## Chunking Strategy

**Chunk size:** 700 characters (target/max)

**Overlap:** 150 characters (~20%)

**Why these choices fit your documents:**
The corpus is short reviews and forum threads, not long-form guides. The chunker
is paragraph-aware: it splits on blank-line boundaries, then greedily packs whole
paragraphs up to 700 chars, only falling back to sentence/character splitting
when one paragraph exceeds the limit. 700 chars keeps every chunk under
all-MiniLM-L6-v2's ~256-token input window (so nothing is silently truncated at
embed time) while being large enough that a whole short review or a complete
forum comment usually lands in one chunk — one coherent opinion per vector. The
150-char overlap means a fact straddling a boundary (a price followed by "card
only," say) still appears whole in at least one chunk. Preprocessing before
chunking is the cleaning described above (strip HTML/nav, normalize whitespace).

**Final chunk count:** 36 chunks across 12 documents (avg ~543 chars/chunk).

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`, run locally
(384-dim, no API key, no rate limits, fast on CPU). Embeddings are L2-normalized
and stored in ChromaDB using cosine distance.

**Production tradeoff reflection:**
If cost weren't a constraint and this served real students, I'd weigh:
- **Domain accuracy:** all-MiniLM is small and general. A larger model
  (`text-embedding-3-large`, or an open MTEB leader like `bge-large`) would
  better separate insider near-synonyms that matter here — "time ticket" vs
  "lottery," "window AC unit" vs "central AC."
- **Context length:** MiniLM truncates ~256 tokens, which forces small chunks. A
  long-context embedder could embed a full review as one unit and keep more
  surrounding context per vector.
- **Privacy / local vs API:** these are anonymous student complaints about
  landlords and dorms. Local embedding keeps that text off third-party servers —
  a real reason to accept lower accuracy. An API model adds latency and a
  data-sharing question.
- **Multilingual:** if students post in multiple languages, MiniLM's
  English-centric training hurts; `bge-m3` or a multilingual MiniLM would help.

---

## Grounded Generation

**System prompt grounding instruction** (`generate.py`, abridged):

> You are The Unofficial Guide... **Answer ONLY using the information in the
> CONTEXT below. Do not use outside or prior knowledge, and do not guess. If the
> context does not contain enough information to answer, reply exactly: "I don't
> have enough information in the collected documents to answer that."** Cite your
> sources inline using the bracketed source label that precedes each context
> block, e.g. `[dorm_reviews.txt]`. Every claim must be traceable to a source. If
> sources disagree, say so and cite both.

Grounding is enforced by three mechanisms, not just the instruction:
1. **Structural:** retrieved chunks are the only content in the user message,
   each prefixed with a `[source]` label, and the question is appended after.
2. **Relevance filtering:** chunks above the cosine-distance cutoff
   (`MAX_DISTANCE = 1.2`) are dropped before generation, so the model isn't fed
   weakly-related text that invites drift. If *nothing* passes, the system
   returns the refusal string without even calling the LLM.
3. **Low temperature (0.1):** keeps the output close to the retrieved wording.

**How source attribution is surfaced in the response:** inline `[source.txt]`
citations in the answer text, plus a deduplicated `SOURCES` list printed under
every answer by the CLI. `--show-chunks` additionally prints the exact retrieved
chunks with their distances for inspection.

---

## Evaluation Report

Run with `python evaluate.py` (model `llama-3.3-70b-versatile`, embeddings
`all-MiniLM-L6-v2`, top_k=4). Full transcript with retrieved chunks is in
[`eval_results.md`](eval_results.md). Summary:

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Is the housing lottery actually random? | No — weighted by earned credits; random only as a tiebreaker | "Not entirely random… weighted by earned credit hours… random tiebreaker within a credit band," cited the thread + RA confirmation | Relevant | Accurate |
| 2 | Which dorm has no AC? | North Tower (Riverside & Westfield have AC) | "North Tower does not have air conditioning" with citation | Relevant | Accurate |
| 3 | Laundry cost in North Tower? | $1.50 wash + $1.50 dry, student card | "$1.50 per wash and $1.50 per dry, charged to your student card," two sources | Relevant | Accurate |
| 4 | Can a Riverside first-year take a smaller meal plan? | Yes — Block 80, because Riverside has floor kitchens | "Yes… Block 80… because Riverside has a shared kitchen on every floor" | Relevant | Accurate |
| 5 | Full-year cost: Maple Court vs North Tower? | North Tower ≈ $11,900/yr (room + required meal plan); Maple Court = $7,800 rent + **unlisted utilities, no meal plan**; clean comparison not possible | Computed North Tower $11,900 correctly, but **added a $3,800 meal plan to the off-campus Maple Court total** → concluded Maple Court ($11,600) is cheaper | Partially relevant | **Inaccurate** |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

**Result: 4 / 5 accurate.** Q1–Q4 retrieved the correct chunks and answered
correctly with citations. Q5 failed — analyzed below.

---

## Failure Case Analysis

**Question that failed:** Q5 — "Over a full academic year, is it cheaper to live
off-campus at Maple Court or in North Tower? Give a total for each."

**What the system returned:** It correctly computed North Tower at ~$11,900/year
(room ~$3,200/sem + required first-year meal plan ~$2,750/sem). But for the
off-campus Maple Court option it **added a meal plan** it should not have —
"$1,900/sem Block 80 × 2 = $3,800" — reaching $11,600/year and concluding Maple
Court is cheaper. That conclusion is wrong: the documents explicitly state that
off-campus students don't need a meal plan at all, so the off-campus figure
should be ~$7,800 rent + unlisted utilities + groceries.

**Root cause (tied to a specific pipeline stage): retrieval.** The fact that
disarms this question — "If you move off campus you don't need a meal plan at
all" — lives in `meal_plan_requirement.txt`, but **none of that document's chunks
were retrieved** for this query. The retrieved set was `offcampus_college_view`
(chunk 0), `cost_comparison` (chunks 0 and 1), and `offcampus_maple_court`
(chunk 0). The query's wording ("off-campus", "Maple Court", "total") pulled the
embedding toward the off-campus and cost documents, crowding the meal-plan
document out of the top-4. The generator then saw "Block 80 ~$1,900" sitting in
the retrieved `cost_comparison` chunk and, lacking the exemption fact, plugged it
into the off-campus total. So a correct grounding mechanism (only answer from
retrieved text) produced a wrong answer because a *necessary* chunk wasn't
retrieved — a classic retrieval-recall failure feeding a confident mis-synthesis,
made worse by an arithmetic/multi-hop question that spans documents.

**What you would change to fix it:**
1. **Raise recall for multi-hop queries** — increase top-k (e.g. 8) and/or add
   query decomposition so "is off-campus cheaper" also retrieves on "off-campus
   meal plan requirement," surfacing the exemption chunk.
2. **Hybrid retrieval (stretch)** — add BM25 keyword search; the literal token
   "meal plan" would rank `meal_plan_requirement.txt` highly even when the dense
   embedding doesn't.
3. **Prompt guardrail for arithmetic** — instruct the model that if a comparison
   requires a figure not present in the context (here, off-campus utilities), it
   should flag the gap and refuse to declare a winner rather than substitute a
   plausible-looking number.

---

## Spec Reflection

**One way the spec helped you during implementation:**
Writing the Chunking Strategy section *before* coding forced the decision that
the corpus is review/forum text, not long guides — so chunks should be small and
paragraph-aligned rather than a fixed 500/1000-char cut. That single upfront
decision is why a whole short review usually lands in one chunk, and it's why
retrieval for Q1–Q4 returns clean, single-opinion chunks instead of fragments.

**One way your implementation diverged from the spec, and why:**
The plan didn't anticipate weakly-relevant chunks polluting generation, so during
implementation I added a cosine-distance relevance cutoff (`MAX_DISTANCE`) and an
early refusal path when nothing passes. This wasn't in the original spec but it
turned out to be the cleanest way to enforce grounding — the LLM never sees junk
context, so it has less to hallucinate from.

---

## AI Usage

**Instance 1**

- *What I gave the AI:* The Chunking Strategy section of planning.md (700 chars /
  150 overlap, "paragraph-aware, fits short reviews") and the requirement that
  chunks stay under the embedding model's input window.
- *What it produced:* A `chunk_text()` that split purely on a fixed character
  count with overlap.
- *What I changed or overrode:* I rejected the naive fixed-cut version and
  directed it to split on paragraph/blank-line boundaries first and only fall
  back to sentence/char splitting for oversized paragraphs, so a single forum
  comment isn't sliced in half.

**Instance 2**

- *What I gave the AI:* The Grounded Generation requirement and a note that the
  system must refuse when the answer isn't in the documents.
- *What it produced:* A working Groq call with a "use the context" system prompt.
- *What I changed or overrode:* The first prompt still let the model soften
  refusals. I tightened it to an exact refusal string, added the `[source]`
  labeling in the context block, and added a structural relevance filter so the
  model never receives low-relevance chunks in the first place — moving grounding
  out of "please behave" and into the pipeline.
