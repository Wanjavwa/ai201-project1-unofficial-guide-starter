# Project 1 Planning: The Unofficial Guide

> Spec written before the pipeline code. The Chunking Strategy and Retrieval
> Approach sections were revisited during implementation and reflect the final
> choices.

---

## Domain

**Campus housing at Maple State University (MSU)** — the student-generated
knowledge about dorms and off-campus apartments that you can't get from the
official housing portal: which dorm actually has air conditioning, whether the
"lottery" is really random, which landlord paints over mold, how the meal-plan
requirement really works, and what move-in day is actually like.

This knowledge is valuable because the official housing website lists rooms,
rates, and policies in neutral marketing language — it will never tell you that
North Tower has no AC and is brutal for the first three weeks, or that a specific
off-campus complex has a recurring mold problem and stonewalls maintenance
requests. That information lives in subreddit threads, GroupMe chats, Discord
servers, and crowd-sourced spreadsheets. It's scattered, anonymous, and
contradictory, which is exactly what a retrieval system is good at consolidating.

> **Note on the corpus:** the documents in `documents/` are a realistic,
> hand-written sample corpus for a fictional "Maple State University" so the
> system is fully runnable and demoable. Replace or augment them with documents
> you collect from your own school; the pipeline reads any `.txt/.md/.pdf/.html`
> dropped into `documents/`.

---

## Documents

12 documents covering different housing subtopics and perspectives (dorm
reviews, off-campus reviews, logistics, policy Q&A, cost data). Variety is
deliberate: review-style posts, forum threads, and reference cheat-sheets chunk
and retrieve differently.

| #  | Source | Description | Location |
|----|--------|-------------|----------|
| 1  | r/MapleState thread | Is the housing lottery actually random? | `documents/reddit_housing_lottery.txt` |
| 2  | Lottery follow-up thread | Appeals, waitlist, medical AC accommodations, swaps | `documents/lottery_appeals_and_waitlist.txt` |
| 3  | Dorm review | North Tower (oldest, no AC, cheap, central) | `documents/dorm_review_north_tower.txt` |
| 4  | Dorm review | Westfield Suites (AC, private bath, expensive, far) | `documents/dorm_review_westfield_suites.txt` |
| 5  | Dorm review | Riverside Hall (newest, AC, social, thin walls) | `documents/dorm_review_riverside.txt` |
| 6  | Off-campus review | Maple Court Apartments (cheap, mold, bad landlord) | `documents/offcampus_maple_court_apartments.txt` |
| 7  | Off-campus review | College View Flats (pricey, well-managed, far) | `documents/offcampus_college_view.txt` |
| 8  | New-student Q&A | Meal plan requirement + Riverside Block 80 exception | `documents/meal_plan_requirement.txt` |
| 9  | Reddit thread | Roommate matching (mutual request, questionnaire) | `documents/roommate_matching_tips.txt` |
| 10 | Discord pinned post | Move-in day survival tips | `documents/move_in_day_tips.txt` |
| 11 | Community wiki | Laundry / AC / kitchen amenities cheat sheet | `documents/laundry_and_amenities.txt` |
| 12 | Crowd-sourced sheet | Per-semester cost comparison by dorm | `documents/cost_comparison.txt` |

---

## Chunking Strategy

**Chunk size:** 700 characters (target/maximum)

**Overlap:** 150 characters between consecutive chunks

**Reasoning:**
The corpus is a mix of short review posts (~1,200–1,500 chars) and forum threads
made of distinct comments. A naive "split every N characters" would slice a
review mid-sentence or cut a single Reddit comment in half. Instead the chunker
(`chunk.py`) is **paragraph-aware**: it splits on blank-line boundaries first,
then greedily packs whole paragraphs into chunks up to 700 chars, only falling
back to sentence- and character-splitting when a single paragraph is itself too
big.

- **700 chars** keeps each chunk comfortably under the embedding model's input
  window (all-MiniLM-L6-v2 truncates around 256 word-pieces, ~1,000 chars), so
  no text is silently dropped at embed time. It's also large enough that a whole
  short review or a complete forum comment usually lands in one chunk, which
  keeps a single opinion intact rather than fragmenting it.
- **150-char overlap (~20%)** means a fact that lands near a boundary — e.g. a
  laundry price followed by "card only" on the next paragraph — still appears
  whole in at least one chunk, so retrieval doesn't return half a fact.
- These are smaller than you'd use for long-form guides precisely because
  review/forum text packs one distinct claim per short paragraph; bigger chunks
  would dilute the embedding with multiple unrelated opinions.

**Final chunk count:** 36 chunks across the 12 documents (avg ~543 chars/chunk).

---

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`, run locally.
384-dimensional, no API key, no rate limits, fast on CPU. Vectors are L2-
normalized and stored in ChromaDB with cosine distance.

**Top-k:** 4. A low-relevance cutoff (`MAX_DISTANCE = 1.2` cosine distance) drops
chunks that are only weakly related so the generator isn't fed near-irrelevant
context that invites hallucination.

**Production tradeoff reflection:**
If I were deploying this for real students and cost weren't a constraint, I'd
weigh:
- **Accuracy on domain text:** all-MiniLM is a small general model. A larger
  instruction-tuned embedding model (e.g. an OpenAI `text-embedding-3-large` or
  a top MTEB open model like `bge-large`) would better distinguish near-
  synonyms that matter here ("window AC unit" vs "central AC", "time ticket" vs
  "lottery").
- **Context length:** MiniLM truncates ~256 tokens, forcing small chunks. A
  long-context embedder would let me embed a whole review as one unit and
  preserve more context per vector.
- **Latency vs. local control:** local MiniLM has zero network latency and keeps
  potentially sensitive student posts off third-party servers — a real privacy
  consideration for anonymous housing complaints. An API model adds latency and
  a data-sharing question in exchange for accuracy.
- **Multilingual:** if the student body posts in multiple languages, MiniLM's
  English-centric training would hurt; a multilingual model (e.g.
  `paraphrase-multilingual-MiniLM` or `bge-m3`) would matter.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | Is the MSU housing lottery actually random? | No — the time ticket is weighted by earned credit hours; randomness is only a tiebreaker within the same credit band. Honors students get a priority window. |
| 2 | Which dorm does not have air conditioning? | North Tower (bring a fan). Riverside and Westfield have central AC. |
| 3 | How much does laundry cost in North Tower? | $1.50 per wash + $1.50 per dry, charged to the student card; basement only. |
| 4 | Can a first-year in Riverside choose a smaller meal plan? | Yes — Riverside has floor kitchens, so its first-years may take the Block 80 plan instead of Unlimited; North Tower/Westfield first-years cannot. |
| 5 | (Hard) Over a full academic year, is it cheaper to live off-campus at Maple Court or in North Tower? Give a total for each. | North Tower ≈ $11,900/yr (room ~$3,200/sem + required first-year meal plan ~$2,750/sem). Maple Court = $7,800 rent (12-mo lease) + **unlisted utilities, no meal plan**. A clean head-to-head isn't possible since utility/food figures aren't in the corpus. |

Q5 is intentionally a multi-hop, cross-document arithmetic question designed to
surface a failure, per the project hints. (An earlier candidate — "cheapest AC
option incl. meal plan" — was rejected during evaluation because the cost-
comparison document pre-stated the answer, so the system passed it too easily and
no failure surfaced.) Q5 did fail: see the Failure Case Analysis in README.md.

---

## Anticipated Challenges

1. **Cross-document reasoning (Q5).** Each fact needed for Q5 (which dorms have
   AC, the room costs, who can take Block 80) lives in a *different* document.
   With top-k = 4, retrieval may not pull all three relevant chunks at once, and
   even if it does, the LLM has to do arithmetic correctly. This is the most
   likely failure point.

2. **Synonym / vocabulary mismatch.** Students use insider terms ("time ticket",
   "Block 80", "window units"). A small embedding model may not place a query
   phrased in everyday words ("when do I get to pick my room?") near the chunk
   that uses the insider term, causing off-target retrieval.

3. **Contradictory/overlapping sources.** "Quiet" appears as a positive for both
   North Tower (thick walls) and Westfield (study lounges); the system must
   attribute correctly rather than blend them.

---

## Architecture

```
                    documents/  (.txt .md .pdf .html)
                          │
                          ▼
  ┌───────────────────────────────────────────────┐
  │ 1. INGESTION            ingest.py               │
  │    load + extract text (pdfplumber / bs4)       │
  │    clean: strip nav/boilerplate, normalize ws   │
  └───────────────────────────────────────────────┘
                          │  [{doc_id, source, text}]
                          ▼
  ┌───────────────────────────────────────────────┐
  │ 2. CHUNKING             chunk.py                │
  │    paragraph-aware packing, 700 chars / 150 ovl │
  └───────────────────────────────────────────────┘
                          │  [{id, text, source, ...}]
                          ▼
  ┌───────────────────────────────────────────────┐
  │ 3. EMBED + STORE        index.py                │
  │    sentence-transformers all-MiniLM-L6-v2       │
  │    -> ChromaDB persistent collection (cosine)   │
  └───────────────────────────────────────────────┘
                          │
        user query ──────▶│
                          ▼
  ┌───────────────────────────────────────────────┐
  │ 4. RETRIEVAL            index.search()          │
  │    embed query, top-k=4, distance cutoff        │
  └───────────────────────────────────────────────┘
                          │  retrieved chunks + sources
                          ▼
  ┌───────────────────────────────────────────────┐
  │ 5. GENERATION           generate.py             │
  │    Groq llama-3.3-70b-versatile, temp 0.1       │
  │    grounded system prompt + inline [citations]  │
  └───────────────────────────────────────────────┘
                          │
                          ▼
                 CLI answer + sources   (app.py)
```

---

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**
Give Claude this planning.md's Documents + Chunking Strategy sections and ask it
to implement `ingest.load_documents()` (multi-format load + clean) and
`chunk.chunk_text()` with the 700/150 paragraph-aware spec. Verify by running
`python chunk.py` and checking the printed min/max/avg chunk sizes stay under
700 and that no chunk splits mid-sentence on a spot check.

**Milestone 4 — Embedding and retrieval:**
Give Claude the Retrieval Approach section and ask it to implement
`index.build_collection()` and `index.search()` using all-MiniLM-L6-v2 + ChromaDB
with cosine distance and a relevance cutoff. Verify by running
`python index.py "<query>"` for each test question and confirming the top chunks
come from the document I know contains the answer.

**Milestone 5 — Generation and interface:**
Give Claude the Grounded Generation requirement and ask it to write the system
prompt and `generate.answer_query()` so the model answers only from retrieved
context and cites `[source]` inline, plus a CLI in `app.py`. Verify by asking a
question whose answer is NOT in the corpus and confirming the system refuses
("I don't have enough information...") instead of hallucinating.
