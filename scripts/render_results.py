#!/usr/bin/env python3
"""Write the measured numbers into README.md and docs/EVALUATION.md.

Run after `make eval`. Everything between the RESULTS:START and RESULTS:END
markers in each file is regenerated from the JSON reports, so no number in the
documentation is ever typed by hand - if a table and the report disagree, the
table is stale and this script fixes it.

    python scripts/render_results.py          # rewrite the generated regions
    python scripts/render_results.py --check  # fail if any of them is stale (CI)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_REPORT = REPO_ROOT / "docs" / "evaluation_report.json"
TUNING_REPORT = REPO_ROOT / "docs" / "tuning_report.json"
VARIANT_REPORT = REPO_ROOT / "docs" / "variant_comparison.json"

sys.path.insert(0, str(REPO_ROOT / "src"))
from icd10_search.config import ICD10_RELEASE_YEAR, RRF_K  # noqa: E402

MODE_LABELS = {
    "lexical": "Keyword only (BM25)",
    "dense": "Semantic only",
    "hybrid": "**Hybrid (shipped)**",
}
SET_LABELS = {
    "cms_index_test": "CMS Alphabetic Index — held-out test split",
    "hand_written_lay": "Hand-written lay queries",
}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def mode_table(entry: dict) -> str:
    rows = [
        "| Retrieval mode | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Median latency |",
        "|---|---|---|---|---|---|",
    ]
    for mode in ("lexical", "dense", "hybrid"):
        metrics = entry["modes"].get(mode)
        if not metrics:
            continue
        recall = metrics["recall_at"]
        rows.append(
            f"| {MODE_LABELS[mode]} | {pct(recall['1'])} | {pct(recall['5'])} | "
            f"{pct(recall['10'])} | {metrics['mrr_at_10']:.3f} | "
            f"{metrics['median_latency_ms']:.0f} ms |"
        )
    return "\n".join(rows)


def readme_section(report: dict) -> str:
    out: list[str] = []
    index = report.get("index", {})
    for set_name, entry in report["gold_sets"].items():
        out.append(f"### {SET_LABELS.get(set_name, set_name)} — {entry['num_queries']:,} queries\n")
        out.append(mode_table(entry))
        out.append("")

        comp = entry.get("complementarity_at_10")
        if comp and set_name == "cms_index_test":
            out.append(
                f"Why both arms: of these {comp['total']:,} queries, "
                f"**{comp['only_first']:,} were found only by the semantic arm** and "
                f"**{comp['only_second']:,} only by the keyword arm** "
                f"(both: {comp['both']:,}; neither: {comp['neither']:,}). "
                "Neither arm dominates, which is exactly the case for fusing them.\n"
            )

    delta = report["gold_sets"].get("cms_index_test", {}).get("int8_recall_delta_at_10")
    if delta is not None:
        float32 = report["gold_sets"]["cms_index_test"]["dense_float32"]["recall_at"]["10"]
        int8 = report["gold_sets"]["cms_index_test"]["modes"]["dense"]["recall_at"]["10"]
        direction = "costs" if delta < 0 else "gains"
        out.append(
            f"**What int8 quantisation costs:** scoring the same queries against the "
            f"un-quantised float32 vectors gives recall@10 of {pct(float32)} versus "
            f"{pct(int8)} for int8 — the 4x smaller index {direction} "
            f"{abs(delta) * 100:.2f} percentage points.\n"
        )

    out.append(
        f"_Measured on {report['generated_at'][:10]} with "
        f"`{index.get('embedding_model')}`, {index.get('num_documents', 0):,} codes indexed. "
        "Regenerate with `make eval && python scripts/render_results.py`._"
    )
    return "\n".join(out)


def evaluation_section(report: dict, tuning: dict | None, variants: dict | None) -> str:
    out: list[str] = ["## Results\n"]
    for set_name, entry in report["gold_sets"].items():
        out.append(f"### {SET_LABELS.get(set_name, set_name)} — {entry['num_queries']:,} queries\n")
        out.append(mode_table(entry))
        out.append("")
        comp = entry.get("complementarity_at_10")
        if comp:
            out.append(
                f"Agreement at k=10 — both arms found it: {comp['both']:,}; "
                f"semantic only: {comp['only_first']:,}; keyword only: {comp['only_second']:,}; "
                f"neither: {comp['neither']:,}.\n"
            )
        if "dense_float32" in entry:
            float32 = entry["dense_float32"]
            out.append(
                f"int8 vs float32 (semantic arm): recall@10 "
                f"{pct(entry['modes']['dense']['recall_at']['10'])} vs "
                f"{pct(float32['recall_at']['10'])}, MRR@10 "
                f"{entry['modes']['dense']['mrr_at_10']:.3f} vs {float32['mrr_at_10']:.3f}.\n"
            )

    if variants:
        out.append("## What is embedded matters (dev split, semantic arm only)\n")
        out.append("| Text indexed per code | Recall@1 | Recall@10 | MRR@10 |")
        out.append("|---|---|---|---|")
        names = {
            "desc": "Official title only",
            "desc_incl_parent": "+ synonyms + parent title",
            "desc_incl_parent_block": "**+ block title (shipped)**",
        }
        for row in sorted(variants["rows"], key=lambda r: r["mrr_at_10"]):
            out.append(
                f"| {names.get(row['variant'], row['variant'])} | {pct(row['recall_at_1'])} | "
                f"{pct(row['recall_at_10'])} | {row['mrr_at_10']:.3f} |"
            )
        out.append(
            "\nAttaching the official CMS synonyms and the surrounding hierarchy to each "
            "code description is worth more than any parameter tuning done here.\n"
        )

    if tuning:
        best, baseline = tuning["best"], tuning["dense_only_baseline"]
        out.append("## Fusion parameter sweep (dev split)\n")
        out.append(
            f"{len(tuning['grid'])} configurations, {tuning['num_queries']:,} dev queries, "
            f"selected on {tuning['selection_metric']}.\n"
        )
        out.append("| Configuration | Recall@1 | Recall@10 | MRR@10 |")
        out.append("|---|---|---|---|")
        out.append(
            f"| Semantic arm alone (baseline) | {pct(baseline['recall_at_1'])} | "
            f"{pct(baseline['recall_at_10'])} | {baseline['mrr_at_10']:.3f} |"
        )
        out.append(
            f"| **Chosen: RRF k={best['rrf_k']}, keyword weight {best['weight_lexical']}** | "
            f"{pct(best['recall_at_1'])} | {pct(best['recall_at_10'])} | {best['mrr_at_10']:.3f} |"
        )
        worst = min(tuning["grid"], key=lambda r: r["mrr_at_10"])
        out.append(
            f"| Worst configuration in the sweep (k={worst['rrf_k']}, "
            f"weight {worst['weight_lexical']}) | {pct(worst['recall_at_1'])} | "
            f"{pct(worst['recall_at_10'])} | {worst['mrr_at_10']:.3f} |"
        )
        out.append("\nThe full grid is in [`tuning_report.json`](tuning_report.json).\n")

    failures = report["gold_sets"].get("cms_index_test", {}).get("examples", {})
    misses = failures.get("not_found_in_top_10", [])
    if misses:
        out.append("## Where it fails\n")
        out.append(
            "Queries whose correct code is not in the top 10, taken straight from the run:\n"
        )
        out.append("| Query | Expected | Returned at rank 1 |")
        out.append("|---|---|---|")
        for miss in misses[:5]:
            out.append(
                f"| {miss['query']} | {miss['expected_code']} {miss['expected_description'][:48]} | "
                f"{miss['top_result_code']} {str(miss['top_result_description'])[:48]} |"
            )
        out.append("")
    return "\n".join(out)


def architecture_diagrams(report: dict) -> str:
    """The two big-picture diagrams, drawn from the real index metadata.

    Drawn as two separate diagrams on purpose. A single diagram with one
    subgraph per phase was tried and rejected: Mermaid ignores a subgraph's own
    ``direction`` once edges cross the subgraph boundary, so the serve half laid
    itself out above the build half and the prose describing a "left half" and a
    "right half" no longer matched what a reader saw.
    """
    idx = report["index"]
    hybrid = report["gold_sets"]["cms_index_test"]["modes"]["hybrid"]
    weights = report["fusion_weights"]
    int8_mb = idx["artifact_bytes"]["dense_int8.npy"] / 1_000_000
    float_mb = int8_mb * 4  # int8 is one byte per dimension where float32 is four
    minutes = idx["build_seconds"] / 60

    return f"""**Step 1 — build the index.** This runs once, offline. It takes about
{minutes:.0f} minutes on a laptop and nothing here happens while a user is waiting.

```mermaid
flowchart LR
    CMS["<b>CMS ICD-10-CM {ICD10_RELEASE_YEAR}</b><br/>the official public release files"]
    PARSE["<b>Parse and merge</b><br/>code titles, chapter ranges,<br/>index synonyms"]
    CORPUS[("<b>corpus.jsonl.gz</b><br/>{idx["num_documents"]:,} codes")]
    EMB["<b>Embed every code</b><br/>{idx["embedding_model"]}<br/>{idx["embedding_dim"]} numbers per code"]
    QUANT["<b>Shrink to int8</b><br/>{idx["quantization"]}<br/>{int8_mb:.0f} MB instead of {float_mb:.0f} MB"]
    BM["<b>Build the word index</b><br/>BM25 over {idx["bm25"]["vocabulary_terms"]:,} terms<br/>k1={idx["bm25"]["k1"]}, b={idx["bm25"]["b"]}"]
    DIDX[("<b>Meaning index</b><br/>dense_int8.npy")]
    LIDX[("<b>Word index</b><br/>lexical_bm25.npz")]

    CMS --> PARSE --> CORPUS
    CORPUS --> EMB --> QUANT --> DIDX
    CORPUS --> BM --> LIDX

    classDef build fill:#dbe7fb,stroke:#2f5fa8,color:#1a1a1a
    classDef store fill:#ffe9a8,stroke:#9a7b25,stroke-width:2px,color:#1a1a1a
    class CMS,PARSE,EMB,QUANT,BM build
    class CORPUS,DIDX,LIDX store
```

**Step 2 — answer a search.** This runs on every query and takes about
{hybrid["median_latency_ms"]:.0f} ms on a laptop. It reads the two indexes built in step 1 and
writes nothing. The free instance the demo is deployed to has about a tenth of a CPU
core, where the same query is measured at roughly 1.9 s — see
[deployed speed](#deployed-speed).

```mermaid
flowchart LR
    Q["<b>Someone types</b><br/><i>ringing in the ears</i>"]
    DENSE["<b>Meaning arm</b><br/>cosine against all<br/>{idx["num_documents"]:,} int8 vectors"]
    LEX["<b>Word arm</b><br/>BM25 over the same codes"]
    DIDX2[("<b>Meaning index</b><br/>held in memory")]
    LIDX2[("<b>Word index</b><br/>held in memory")]
    FUSE["<b>Combine the two rankings</b><br/>weighted reciprocal rank fusion<br/>meaning {weights["dense"]}, words {weights["lexical"]}"]
    OUT["<b>Ranked codes</b><br/>each with the evidence for it:<br/>rank and score from each arm,<br/>and which words matched"]

    Q --> DENSE
    Q --> LEX
    DIDX2 -. "loaded at startup" .-> DENSE
    LIDX2 -. "loaded at startup" .-> LEX
    DENSE --> FUSE
    LEX --> FUSE
    FUSE --> OUT

    classDef serve fill:#d8f0dd,stroke:#1e7e34,color:#1a1a1a
    classDef store fill:#ffe9a8,stroke:#9a7b25,stroke-width:2px,color:#1a1a1a
    class Q,DENSE,LEX,FUSE,OUT serve
    class DIDX2,LIDX2 store
```

**How to read them.** Blue boxes run **once**, when the index is built. Green boxes run
**on every search**. The yellow cylinders are the files on disk — the only thing the two
halves share, and the reason a search is fast: all the slow work already happened in
step 1."""


def query_diagram(report: dict) -> str:
    """The detailed path of one query, with its constants read from the report."""
    idx = report["index"]
    weights = report["fusion_weights"]
    return f"""```mermaid
flowchart TD
    A["<b>query text</b>"] --> B{{"does it look like a code?<br/>e.g. E11.9"}}
    B -- "yes" --> C["<b>look the code up directly</b><br/>pin it and its siblings on top"]
    B -- "no" --> D(["run both arms"])

    D --> E1["<b>Meaning arm</b><br/>add the bge query prefix"]
    E1 --> E2["embed to {idx["embedding_dim"]} numbers<br/>ONNX Runtime, CPU"]
    E2 --> E3["cosine against all<br/>{idx["num_documents"]:,} int8 vectors"]

    D --> F1["<b>Word arm</b><br/>lowercase, strip plurals,<br/>drop stop-words"]
    F1 --> F2["BM25 score<br/>k1={idx["bm25"]["k1"]}, b={idx["bm25"]["b"]}"]

    E3 --> G["<b>apply the filters</b><br/>billable / chapter"]
    F2 --> G
    G --> H["take the top 100 from each arm"]
    H --> I["<b>weighted reciprocal rank fusion</b><br/>k={RRF_K}, meaning {weights["dense"]}, words {weights["lexical"]}"]
    C --> J["<b>final ranked list</b>"]
    I --> J
    J --> K["<b>attach the evidence</b><br/>rank and score from each arm,<br/>and the words that matched"]

    classDef serve fill:#d8f0dd,stroke:#1e7e34,color:#1a1a1a
    classDef pin fill:#fde2e2,stroke:#b03a3a,color:#1a1a1a
    class A,D,E1,E2,E3,F1,F2,G,H,I,J,K serve
    class C pin
```"""


#: Set by --check: report staleness instead of rewriting.
STALE: list[str] = []


def splice(path: Path, body: str, marker: str = "RESULTS", check: bool = False) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"(<!-- {marker}:START -->\n)(.*?)(<!-- {marker}:END -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise SystemExit(f"{path} has no {marker}:START / {marker}:END markers")
    updated = pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(3), text)
    name = f"{path.relative_to(REPO_ROOT)}  [{marker}]"
    if updated == text:
        print(f"up to date  {name}")
    elif check:
        STALE.append(name)
        print(f"STALE       {name}")
    else:
        path.write_text(updated, "utf-8")
        print(f"updated     {name}")


def main() -> int:
    check = "--check" in sys.argv[1:]
    if not EVAL_REPORT.exists():
        raise SystemExit("no evaluation report; run `make eval` first")
    report = json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
    tuning = json.loads(TUNING_REPORT.read_text("utf-8")) if TUNING_REPORT.exists() else None
    variants = json.loads(VARIANT_REPORT.read_text("utf-8")) if VARIANT_REPORT.exists() else None

    splice(REPO_ROOT / "README.md", readme_section(report), check=check)
    splice(REPO_ROOT / "README.md", architecture_diagrams(report), "DIAGRAM-ARCH", check)
    splice(REPO_ROOT / "README.md", query_diagram(report), "DIAGRAM-QUERY", check)
    splice(
        REPO_ROOT / "docs" / "EVALUATION.md",
        evaluation_section(report, tuning, variants),
        check=check,
    )
    if STALE:
        print(
            "\nThese regions disagree with docs/evaluation_report.json."
            "\nRun `python scripts/render_results.py` and commit the result.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
