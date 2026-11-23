# Demo script

A five-minute walkthrough for an interview, and the questions that tend to
follow. Nothing here is a claim the repository does not support - every number
comes from `make eval`.

## 1. Open with the problem (30 seconds)

> "ICD-10-CM has 98,186 diagnosis codes. The official title for ringing in the
> ears is 'Tinnitus', and for high blood pressure it's 'Essential (primary)
> hypertension'. So keyword search over the code titles fails on exactly the
> words a non-expert types. I built a search engine that works by meaning, and
> then measured how much that actually helps."

## 2. Show it working (90 seconds)

In the live app:

1. Type **ringing in the ears** → Tinnitus comes back first. Point out that the
   word "tinnitus" is nowhere in the query.
2. Switch the mode toggle to **Keyword** and search again → it returns contusions
   of the ear. "This is what the semantic arm is buying."
3. Switch back to **Hybrid**, type **E11.9** → the exact code is pinned at the
   top. "A code-shaped query is a lookup, not a description; embedding it as
   English returns nonsense, so it is detected and handled separately."
4. Point at the evidence row under any result: the semantic rank, the BM25 rank
   and the matched terms. "Every result can explain itself."
5. Point at the timing line: the whole query is answered in tens of
   milliseconds against all 98k codes.

## 3. Show the numbers (90 seconds)

Scroll to the measured-quality section, or open `docs/EVALUATION.md`.

> "The gold set is not something I wrote. CMS publishes the Alphabetic Index -
> the back-of-the-book index a coder looks terms up in - so I derived 45,856
> query/code pairs from it, where the label comes from CMS and the query wording
> is not the code's own title. I held out two disjoint splits: one to tune the
> fusion parameters, one that is only ever reported on."

Then the headline: hybrid beats either arm on its own, and by how much.

## 4. Expect these questions

**"Why not FAISS or a vector database?"**
At 98k x 384, exact scoring is ~38 MFLOP, about 15 ms in NumPy. An approximate
index would add a dependency, a build step and an accuracy loss to save latency
nobody can perceive. The `DenseIndex` class is the seam where that changes if
the corpus grows.

**"Why int8?"**
Float32 vectors are 144 MB; int8 is 36 MB, which fits in a repository and in a
512 MB container. The accuracy cost is measured and reported, not assumed.

**"Why rank fusion instead of combining the scores?"**
Cosine similarity is bounded, BM25 is not, and their scales depend on the
query. RRF combines ranks, which removes the calibration problem entirely.

**"How did you choose the fusion constant?"**
By sweeping it on a dev split that is disjoint from everything reported.
`docs/tuning_report.json` holds the whole grid. Equal-weight fusion with the
default constant of 60 was *worse* than the semantic arm alone on lay-language
queries; a sharper constant fixed it.

**"What would you do next?"**
The Alphabetic Index would be the single biggest quality win as a synonym
source - but it is where my labels come from, so using it would be training on
the test set. I would need a different gold source first. After that: a
cross-encoder re-ranker over the top 50, and collapsing the near-duplicate
7th-character variants that currently fill a page of results.

**"What is it not?"**
It is not a coding tool. It retrieves candidate codes from public reference
data; it does not read a clinical note, it does not assign a code, and it
carries that disclaimer on every page and in every API response.
