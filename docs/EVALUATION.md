# Evaluation

Every number in this file and in the README is produced by
`make eval` (`src/icd10_search/evaluation/run.py`) and stored verbatim in
[`evaluation_report.json`](evaluation_report.json). None of them is typed by
hand.

## What is being measured

Each gold query has exactly one correct code, so the metrics are the
single-relevant-document forms:

* **Recall@k** — the fraction of queries whose correct code appears in the top
  *k*. With one relevant document this is also accuracy@k, which is why
  precision@k is not reported separately: it would carry no extra information.
* **MRR@10** — the mean of 1/rank of the correct code, counting 0 when it is
  not in the top 10. Recall@10 asks *"is the answer on the page?"*; MRR asks
  *"is it at the top of the page?"*. A code finder needs both.
* **Latency** — median and p95, not mean. The mean hides the cold-cache
  outliers a user actually notices.

## Where the labels come from

### Gold set 1 — the CMS Alphabetic Index (the real benchmark)

CMS publishes the *Index to Diseases and Injuries*: the back-of-the-book index
a human coder looks terms up in. It is a tree of terms ending in codes:

```
Deficiency, deficient
  └── anti-hemophilic
        └── factor
              └── B  ──►  D67
```

Flattening a path gives `"anti-hemophilic factor B Deficiency"` with the label
`D67` — **assigned by CMS, not by me**, and phrased in words that are not the
code's own title (`D67` is *"Hereditary factor IX deficiency"*). That makes it a
genuine retrieval test rather than a string-matching test.

Construction rules, all enforced in code and tested in `tests/test_goldset.py`:

| Rule | Why |
|---|---|
| Sub-terms keep their order, the head term moves to the end | The index is written head-first (`Pain > abdominal`); English says `abdominal pain`. |
| Paths deeper than 4 levels are dropped | Flattening them produces word salad, which would measure the flattener, not the engine. |
| Cross-reference entries (`Abandonment — see Maltreatment`) are dropped | They carry no code. |
| Codes absent from the corpus are dropped | The index points at some categories that need further characters. |
| A phrase mapping to more than one code is dropped | There is no single right answer to score against. |
| Entries of fewer than two words are dropped | Single words are mostly disease eponyms, a different task. |

That leaves **45,856 usable pairs**, from which a seeded random draw takes
**1,000 dev** and **1,000 test**, disjoint.

### Gold set 2 — hand-written lay queries (54)

Plain English a patient would type: *"trouble sleeping"*, *"nosebleed"*,
*"high blood pressure"*. I wrote these myself and chose each target as the
default "unspecified" code, cross-checked against the Alphabetic Index where it
has an entry. **This is the weakest part of the evaluation** and is reported as
such: 54 queries written by the same person who built the engine is a smoke
test of lay phrasing, not a benchmark. Its value is that it probes a different
register than gold set 1, where the queries use clinical vocabulary.

Every code in it is validated against the live corpus at build time, so the set
fails loudly rather than rotting silently when CMS retires a code.

## The dev/test discipline

Three things were *chosen* rather than assumed: the document text variant, the
fusion weights and the RRF constant. All three were selected on the **dev
split**, which is disjoint from everything reported.

Nothing was ever tuned on the test split or on the lay set.

<!-- RESULTS:START -->
## Results

### CMS Alphabetic Index — held-out test split — 1,000 queries

| Retrieval mode | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Median latency |
|---|---|---|---|---|---|
| Keyword only (BM25) | 38.9% | 56.9% | 63.7% | 0.467 | 0 ms |
| Semantic only | 40.5% | 62.6% | 69.1% | 0.500 | 15 ms |
| **Hybrid (shipped)** | 44.1% | 66.2% | 73.0% | 0.533 | 16 ms |

Agreement at k=10 — both arms found it: 573; semantic only: 118; keyword only: 64; neither: 245.

int8 vs float32 (semantic arm): recall@10 69.1% vs 69.4%, MRR@10 0.500 vs 0.503.

### Hand-written lay queries — 54 queries

| Retrieval mode | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Median latency |
|---|---|---|---|---|---|
| Keyword only (BM25) | 22.2% | 51.8% | 68.5% | 0.345 | 0 ms |
| Semantic only | 24.1% | 64.8% | 74.1% | 0.412 | 16 ms |
| **Hybrid (shipped)** | 35.2% | 72.2% | 77.8% | 0.493 | 16 ms |

Agreement at k=10 — both arms found it: 32; semantic only: 8; keyword only: 5; neither: 9.

int8 vs float32 (semantic arm): recall@10 74.1% vs 74.1%, MRR@10 0.412 vs 0.421.

## What is embedded matters (dev split, semantic arm only)

| Text indexed per code | Recall@1 | Recall@10 | MRR@10 |
|---|---|---|---|
| Official title only | 37.2% | 65.5% | 0.458 |
| + synonyms + parent title | 39.9% | 69.1% | 0.492 |
| **+ block title (shipped)** | 44.6% | 70.4% | 0.532 |

Attaching the official CMS synonyms and the surrounding hierarchy to each code description is worth more than any parameter tuning done here.

## Fusion parameter sweep (dev split)

36 configurations, 1,000 dev queries, selected on mrr_at_10.

| Configuration | Recall@1 | Recall@10 | MRR@10 |
|---|---|---|---|
| Semantic arm alone (baseline) | 44.6% | 70.4% | 0.532 |
| **Chosen: RRF k=10, keyword weight 0.75** | 47.2% | 74.1% | 0.557 |
| Worst configuration in the sweep (k=10, weight 0.0) | 44.6% | 70.4% | 0.532 |

The full grid is in [`tuning_report.json`](tuning_report.json).

## Where it fails

Queries whose correct code is not in the top 10, taken straight from the run:

| Query | Expected | Returned at rank 1 |
|---|---|---|
| Acanthocheilonema Infestation | B74.4 Mansonelliasis | B88.2 Other arthropod infestations |
| Barrel chest | M95.4 Acquired deformity of chest and rib | S28.0XXS Crushed chest, sequela |
| Bayle's disease | A52.17 General paresis | E74.04 McArdle disease |
| Bloch-Sulzberger disease or syndrome | Q82.3 Incontinentia pigmenti | E78.72 Smith-Lemli-Opitz syndrome |
| Brenner Tumor | D27.9 Benign neoplasm of unspecified ovary | D48.114 Desmoid tumor, intraabdominal |

<!-- RESULTS:END -->
