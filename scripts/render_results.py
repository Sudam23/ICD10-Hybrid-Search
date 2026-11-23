#!/usr/bin/env python3
"""Write the measured numbers into README.md and docs/EVALUATION.md.

Run after `make eval`. Everything between the RESULTS:START and RESULTS:END
markers in each file is regenerated from the JSON reports, so no number in the
documentation is ever typed by hand - if a table and the report disagree, the
table is stale and this script fixes it.

    python scripts/render_results.py
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


def splice(path: Path, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- RESULTS:START -->\n).*?(<!-- RESULTS:END -->)", re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"{path} has no RESULTS:START / RESULTS:END markers")
    path.write_text(pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text), "utf-8")
    print(f"updated {path.relative_to(REPO_ROOT)}")


def main() -> int:
    if not EVAL_REPORT.exists():
        raise SystemExit("no evaluation report; run `make eval` first")
    report = json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
    tuning = json.loads(TUNING_REPORT.read_text("utf-8")) if TUNING_REPORT.exists() else None
    variants = json.loads(VARIANT_REPORT.read_text("utf-8")) if VARIANT_REPORT.exists() else None

    splice(REPO_ROOT / "README.md", readme_section(report))
    splice(REPO_ROOT / "docs" / "EVALUATION.md", evaluation_section(report, tuning, variants))
    return 0


if __name__ == "__main__":
    sys.exit(main())
