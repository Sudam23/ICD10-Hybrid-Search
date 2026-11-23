# Architecture

## The problem

ICD-10-CM is the diagnosis code set used for every clinical encounter and every
claim in US healthcare. The 2026 release has **98,186 codes**, of which 74,719
are billable on their own. Finding the right one means knowing the exact
official wording: the code book calls a ringing in the ears *"Tinnitus"* and
high blood pressure *"Essential (primary) hypertension"*. Plain keyword search
over the code titles therefore fails on exactly the phrasing a non-expert uses.

This project makes the code set searchable by meaning, and measures how well
that works.

## Data flow

```
 CMS ICD-10-CM 2026 release (public domain)
   ├── icd10cm_order_2026.txt      98,186 rows: code, billable flag, titles
   ├── icd10cm_tabular_2026.xml    hierarchy: chapters, blocks, synonyms
   └── icd10cm_index_2026.xml      Alphabetic Index  ──► gold evaluation labels
                                                          (never indexed - see below)
            │
            ▼  data/parse.py + data/goldset.py
   corpus.jsonl.gz (one record per code)  +  gold sets (dev / test / lay)
            │
            ▼  indexing/build.py
   ┌────────────────────────┐   ┌──────────────────────┐
   │ dense_int8.npy  36 MB  │   │ lexical_bm25.npz     │
   │ 98,186 x 384 int8      │   │ inverted index       │
   │ + per-row scales       │   │ 11k terms, 944k post │
   └────────────────────────┘   └──────────────────────┘
            │                              │
            ▼                              ▼
         search/service.py  ──►  FastAPI  ──►  static web UI
```

## A query, step by step

1. **Validate.** Length and range checks happen in the Pydantic layer, before
   anything reaches NumPy.
2. **Code shortcut.** If the query looks like a code (`E119`, `E11.9`, `E11`),
   it is answered by lookup and those hits are pinned to the top. Embedding a
   code string as if it were English returns nonsense.
3. **Semantic arm.** The query is embedded with `BAAI/bge-small-en-v1.5` via
   FastEmbed/ONNX Runtime, carrying the bge instruction prefix that the model
   expects on queries but not on documents. The vector is scored against all
   98,186 stored vectors.
4. **Keyword arm.** BM25 over the inverted index, scoring the same corpus.
5. **Filters.** Billable-only and chapter filters are applied to the score
   vectors *before* the top-k cut, so a filtered search still returns a full
   page rather than a short one.
6. **Fusion.** The top 100 from each arm are merged with Reciprocal Rank
   Fusion.
7. **Hydrate.** Each hit is returned with its chapter, block, official
   synonyms, both arms' ranks and scores, and the query terms it matched - so
   the UI can show *why* a result is there.

## Design decisions worth defending

### Exact search, not an approximate index

98,186 x 384 is about 38 MFLOP per query, which NumPy does in ~15 ms. A
FAISS/HNSW index would add a dependency, a build step, a tuning surface and an
accuracy loss, in exchange for latency that is already imperceptible. Brute
force is the right answer at this corpus size; the moment it is not, the
`DenseIndex` interface is the only thing that has to change.

### int8 quantisation

Float32 vectors are 151 MB - too large to keep in the repository and wasteful
inside a 512 MB container. Symmetric per-row int8 quantisation brings this to
36 MB. The accuracy this costs is measured, not assumed: see `docs/EVALUATION.md`.

### Rank fusion, not score fusion

Cosine similarity lives in [-1, 1]; BM25 is unbounded and depends on corpus
statistics. Normalising them onto a shared scale needs per-query calibration
that breaks in the tails. RRF combines *ranks*, which sidesteps the problem.
The damping constant `k` is the parameter that matters, and it was swept on a
dev split rather than guessed.

### Two retrievers instead of one

They fail differently. The keyword arm cannot find "Tinnitus" from "ringing in
the ears"; the semantic arm blurs the distinction between "with" and "without
complications" that separates two real codes. `docs/EVALUATION.md` reports how
many queries each arm gets that the other misses.

### The Alphabetic Index is a label source, not a feature

The CMS Alphabetic Index maps lay and clinical terms onto codes - exactly the
synonyms that would most improve retrieval. It is deliberately **not** indexed,
because it is where the evaluation labels come from. Using it on both sides
would be training on the test set, and the reported numbers would mean nothing.
Using it as a feature *and* finding a different label source is the obvious
next step, and is listed as such in the README.

### String pooling in the corpus loader

Five fields repeat across tens of thousands of records: 22 distinct chapter
titles, 285 block titles, and a parent description shared by every sibling
code. The loader interns them, so one string object is held per distinct value
rather than one per record, and the synonym lists are tuples (the ~52k codes
with no synonyms share a single empty tuple). That is the difference between a
125 MB corpus and a 63 MB one, which is what lets the whole service - corpus,
int8 vectors and BM25 index - fit inside a 512 MB free-tier container.

### Silenced NumPy warnings in dense scoring

`score_all` wraps its blocked matmul in `np.errstate(divide="ignore",
over="ignore", invalid="ignore")`. NumPy 2.x on macOS, backed by Accelerate,
raises divide-by-zero, overflow and invalid-value `RuntimeWarning`s from
`matmul` even when every input is finite (numpy/numpy#24097); left alone they
flood the server log on every single query. The suppression is not taken on
trust: `tests/test_quantize.py` checks the scores against a float64 reference
(max absolute difference 1.9e-8), and `score_all` raises `FloatingPointError`
if any score comes back non-finite, so a *real* numerical fault still stops the
request instead of being swallowed.

## Layout

```
src/icd10_search/
  config.py              every tunable, in one place
  data/
    download.py          fetch + verify the CMS archives
    parse.py             fixed-width + XML -> one record per code
    text.py              what actually gets embedded vs. what BM25 sees
    goldset.py           evaluation labels, dev/test split
    pipeline.py          download -> parse -> gold sets
  indexing/
    embedder.py          FastEmbed wrapper, query/document asymmetry
    quantize.py          int8 quantisation and blocked scoring
    build.py             builds both indexes + a manifest
  search/
    dense.py             memory-mapped int8 cosine search
    lexical.py           BM25 on a NumPy inverted index
    fusion.py            reciprocal rank fusion
    service.py           the orchestrator
  evaluation/
    metrics.py           recall@k, MRR@10, latency percentiles
    tune.py              fusion parameter sweep (dev split only)
    run.py               the reported evaluation (test splits only)
    compare.py           document-text variant comparison
  api/
    schemas.py           request/response contracts
    main.py              FastAPI app, rate limiting, static hosting
web/                     three static files, no build step
tests/                   89 unit tests + an integration suite
```

## The manifest

`data/index/manifest.json` records what was built, from which corpus (by
SHA-256 prefix), with which model and text variant. The service refuses to
start if the manifest disagrees with the corpus or with the configured model.
This is what prevents a stale index from silently serving wrong results after a
model change - the failure is loud, at startup, instead of quiet and in
production.
