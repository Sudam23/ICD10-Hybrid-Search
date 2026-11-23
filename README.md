# ICD-10-CM Hybrid Code Finder

Search all **98,186 ICD-10-CM 2026 diagnosis codes** by describing a condition in
plain English. Type *"ringing in the ears"* and get **H93.19 Tinnitus** — a code
whose official title does not contain a single word of the query.

**Live demo:** <https://icd10-hybrid-search.onrender.com> · **API docs:** <https://icd10-hybrid-search.onrender.com/docs>

> The demo runs on a free instance that sleeps after about 15 minutes of
> inactivity, so the first request after an idle period waits for the container
> to wake. Queries take roughly a second there against 17 ms on a normal
> machine: the free instance is CPU-throttled, and embedding plus scoring
> 98,186 vectors is arithmetic. Keyword-only mode stays sub-millisecond even
> there, which is a fair illustration of what each arm costs.

[![CI](https://github.com/Sudam23/icd10-hybrid-search/actions/workflows/ci.yml/badge.svg)](https://github.com/Sudam23/icd10-hybrid-search/actions/workflows/ci.yml)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Sudam23/icd10-hybrid-search)

> **Demonstration project.** Results are retrieval suggestions from public CMS
> reference data. This is not a coding tool and not medical advice.

---

## Why this is not just "embeddings over a CSV"

Three things make the difference between a notebook demo and something you can
put in front of a user:

1. **It is measured.** 45,856 query/code pairs were derived from the **CMS
   Alphabetic Index** — so the labels come from CMS, not from me — and split
   into a 1,000-query dev set and a disjoint 1,000-query test set. Every
   parameter was chosen on dev; every number below is from test.
2. **Two retrievers, because one is not enough.** A semantic arm finds
   *tinnitus* from *"ringing in the ears"*; a BM25 arm keeps the distinction
   between *"with"* and *"without complications"* that separates two real
   codes. They are fused by rank, and the numbers below show how much each one
   contributes.
3. **It fits the free tier.** Measured inside a container capped at 512 MB:
   236 MB at rest, ~320 MB under load, ready to answer 5 seconds after start.
   37.7 MB of int8 vectors plus a 1.7 MB inverted index — no vector database,
   no GPU, no API keys.

---

## Measured retrieval quality

<!-- RESULTS:START -->
### CMS Alphabetic Index — held-out test split — 1,000 queries

| Retrieval mode | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Median latency |
|---|---|---|---|---|---|
| Keyword only (BM25) | 38.9% | 56.9% | 63.7% | 0.467 | 0 ms |
| Semantic only | 40.5% | 62.6% | 69.1% | 0.500 | 15 ms |
| **Hybrid (shipped)** | 44.1% | 66.2% | 73.0% | 0.533 | 16 ms |

Why both arms: of these 1,000 queries, **118 were found only by the semantic arm** and **64 only by the keyword arm** (both: 573; neither: 245). Neither arm dominates, which is exactly the case for fusing them.

### Hand-written lay queries — 54 queries

| Retrieval mode | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Median latency |
|---|---|---|---|---|---|
| Keyword only (BM25) | 22.2% | 51.8% | 68.5% | 0.345 | 0 ms |
| Semantic only | 24.1% | 64.8% | 74.1% | 0.412 | 16 ms |
| **Hybrid (shipped)** | 35.2% | 72.2% | 77.8% | 0.493 | 16 ms |

**What int8 quantisation costs:** scoring the same queries against the un-quantised float32 vectors gives recall@10 of 69.4% versus 69.1% for int8 — the 4x smaller index costs 0.30 percentage points.

_Measured on 2026-11-23 with `BAAI/bge-small-en-v1.5`, 98,186 codes indexed. Regenerate with `make eval && python scripts/render_results.py`._
<!-- RESULTS:END -->

---

## How it works

```
 plain-English query
        │
        ├──► semantic arm   bge-small-en-v1.5 (ONNX, CPU) ──► cosine vs 98,186 int8 vectors
        │
        └──► keyword arm    BM25 over an inverted index (11,108 terms, 1.08M postings)
                                        │
                     filters applied before the top-k cut (billable / chapter)
                                        │
                            Reciprocal Rank Fusion (k=10)
                                        │
                      ranked codes + the evidence for each one
```

A query that *looks like a code* (`E119`, `E11.9`, `E11`) skips all of that and
is answered by lookup, because embedding a code string as if it were English
returns nonsense.

Each code is indexed together with its official **inclusion terms** — the
synonyms CMS publishes in the XML tabular list. That is how *"high blood
pressure"* reaches **I10 Essential (primary) hypertension**: the phrase appears
nowhere in the title, but CMS files it as a synonym.

Full design notes, including why there is no FAISS index and why ranks are fused
rather than scores: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Quick start

```bash
git clone https://github.com/Sudam23/icd10-hybrid-search.git && cd icd10-hybrid-search
make venv          # uv venv + install
make serve         # http://127.0.0.1:8000
```

The parsed corpus and both indexes are committed, so this works on a fresh
clone with no build step. The build is deterministic — rebuilding from the CMS
source produces an index that is byte-for-byte identical to the committed one
(verified by SHA-256), so the shipped artifact can be audited rather than
trusted. To rebuild it yourself:

```bash
make data          # ~1 min    download + parse the CMS release, build gold sets
make index         # ~26 min   embed 98,186 descriptions on CPU
make eval          # ~6 min    reproduce every number in this README
```

With Docker:

```bash
make docker-build && make docker-run
```

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/search?q=&k=&mode=&billable_only=&chapter=` | Search. `mode` is `hybrid`, `dense` or `lexical`. |
| `GET /api/code/{code}` | Full detail for one code, dotted or undotted. |
| `GET /api/stats` | Corpus and index metadata. |
| `GET /api/chapters` | The 22 chapters, for the filter. |
| `GET /api/evaluation` | The machine-readable evaluation report. |
| `GET /api/health` | Liveness and index readiness, separately. |
| `GET /docs` | OpenAPI UI. |

```bash
curl "http://127.0.0.1:8000/api/search?q=ringing+in+the+ears&k=3"
```

Every result carries its evidence — the rank and score from each arm, and which
query terms matched:

```json
{
  "rank": 1,
  "code_dotted": "H93.19",
  "description": "Tinnitus, unspecified ear",
  "billable": true,
  "chapter_title": "Diseases of the ear and mastoid process",
  "dense_rank": 1, "dense_score": 0.781,
  "lexical_rank": null, "lexical_score": 6.95,
  "matched_terms": ["ear"]
}
```

---

## How quality is measured

Two gold sets, for two different questions:

**1. CMS Alphabetic Index (45,856 pairs, 1,000 dev / 1,000 test).** CMS ships
the back-of-the-book index that coders look terms up in — *Deficiency →
anti-hemophilic → factor → B → D67*. Flattening those paths gives real clinical
phrasing that is **not** the code's own title, with a label CMS assigned. Ambiguous
phrases (one phrase, several codes) are dropped.

**2. Hand-written lay queries (54).** Plain English a patient would use
(*"trouble sleeping"*, *"nosebleed"*). I wrote these, which is a real limitation:
they are a smoke test of lay phrasing, not a benchmark.

**The split is load-bearing.** Fusion weights, the RRF constant and the document
text variant were all chosen on the dev split; the test split and the lay set
were never used to select anything. Full method, the parameter sweep and the
failure analysis: **[docs/EVALUATION.md](docs/EVALUATION.md)**.

**A deliberate omission.** The Alphabetic Index would be the single biggest
quality win as a *synonym source* — it is a hand-curated map from lay terms to
codes. It is not indexed, because it is where the labels come from. Using it on
both sides would be training on the test set.

---

## Limitations

- **Not a coding tool.** It retrieves candidate codes from reference data. It
  does not read clinical notes, does not assign codes, and has no notion of
  coding guidelines, sequencing or medical necessity.
- **The lay gold set is small and self-written** (54 queries). Treat its numbers
  as indicative.
- **Near-duplicate results.** Codes that differ only by 7th character (initial /
  subsequent / sequela encounter) can fill a page. They are genuinely distinct
  codes, but collapsing them behind one row would be a better experience.
- **The rate limiter is in-process**, so it is correct for a single instance —
  which is what the free tier runs — and would need shared state behind more.
- **English only**, and only the 2026 US release.

## What I would do next

1. Add a cross-encoder re-ranker over the fused top 50 and measure whether the
   latency cost buys enough MRR to be worth it.
2. Find a second label source so the Alphabetic Index can be used as a feature.
3. Collapse 7th-character variants into one expandable result.
4. Replace brute-force scoring with an ANN index — but only once the corpus is
   large enough for that to be the bottleneck, which at 98k it is not.

---

## Data and licence

Source data is the **ICD-10-CM FY2026 release** published by the
[U.S. Centers for Medicare & Medicaid Services](https://www.cms.gov/medicare/coding-billing/icd-10-codes),
a work of the U.S. federal government and in the public domain. No patient data
of any kind is used anywhere in this project.

Code is MIT licensed. See [LICENSE](LICENSE).
