"""Run the evaluation and write ``docs/evaluation_report.json``.

What gets measured
------------------
1. Each retrieval mode (lexical / dense / hybrid) on each gold set.
2. How the two arms complement each other - the evidence for fusing them.
3. The accuracy cost of int8 quantisation, by re-scoring the dense arm against
   the un-quantised float32 vectors when they are present on disk.

Everything written here is produced by this script; no number in the README is
typed by hand.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from icd10_search import __version__
from icd10_search.config import (
    EVAL_REPORT_PATH,
    GOLD_INDEX_TEST_PATH,
    GOLD_LAY_PATH,
    INDEX_DIR,
    RRF_WEIGHT_DENSE,
    RRF_WEIGHT_LEXICAL,
)
from icd10_search.data.goldset import GoldQuery, load_gold_set
from icd10_search.evaluation.metrics import (
    DEFAULT_KS,
    QueryOutcome,
    complementarity,
    summarise,
)
from icd10_search.indexing.build import FLOAT32_VECTORS_PATH
from icd10_search.indexing.quantize import top_k
from icd10_search.search.service import SearchService

MODES = ("lexical", "dense", "hybrid")


def run_mode(
    service: SearchService,
    queries: list[GoldQuery],
    mode: str,
    k: int = 10,
) -> list[QueryOutcome]:
    outcomes: list[QueryOutcome] = []
    for gold in queries:
        started = time.perf_counter()
        response = service.search(gold.query, k=k, mode=mode)
        elapsed = (time.perf_counter() - started) * 1000
        outcomes.append(
            QueryOutcome(
                query=gold.query,
                expected_code=gold.expected_code,
                returned_codes=[result.code for result in response.results],
                latency_ms=elapsed,
            )
        )
    return outcomes


def run_float32_dense(
    service: SearchService,
    queries: list[GoldQuery],
    vectors_path: Path,
    k: int = 10,
) -> list[QueryOutcome] | None:
    """Score the dense arm against un-quantised vectors, for an int8 cost estimate."""
    if not vectors_path.exists():
        return None
    vectors = np.load(vectors_path, mmap_mode="r")
    outcomes: list[QueryOutcome] = []
    for gold in queries:
        started = time.perf_counter()
        query_vector = service.embedder.embed_queries([gold.query])[0]
        scores = np.empty(vectors.shape[0], dtype=np.float32)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for start in range(0, vectors.shape[0], 8192):
                stop = min(start + 8192, vectors.shape[0])
                scores[start:stop] = np.asarray(vectors[start:stop]) @ query_vector
        doc_ids, _ = top_k(scores, k)
        elapsed = (time.perf_counter() - started) * 1000
        outcomes.append(
            QueryOutcome(
                query=gold.query,
                expected_code=gold.expected_code,
                returned_codes=[service.records[int(i)].code for i in doc_ids],
                latency_ms=elapsed,
            )
        )
    return outcomes


def evaluate(
    *,
    index_dir: Path = INDEX_DIR,
    limit: int | None = None,
    include_float32: bool = True,
) -> dict:
    service = SearchService.load(index_dir)

    gold_sets: dict[str, list[GoldQuery]] = {
        "cms_index_test": load_gold_set(GOLD_INDEX_TEST_PATH),
        "hand_written_lay": load_gold_set(GOLD_LAY_PATH),
    }
    if limit:
        gold_sets = {name: queries[:limit] for name, queries in gold_sets.items()}

    report: dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_version": __version__,
        "index": service.manifest,
        "fusion_weights": {"dense": RRF_WEIGHT_DENSE, "lexical": RRF_WEIGHT_LEXICAL},
        "ks": list(DEFAULT_KS),
        "gold_sets": {},
    }

    for set_name, queries in gold_sets.items():
        print(f"\n[eval] {set_name}: {len(queries):,} queries")
        per_mode_outcomes: dict[str, list[QueryOutcome]] = {}
        metrics: dict[str, dict] = {}

        for mode in MODES:
            started = time.perf_counter()
            outcomes = run_mode(service, queries, mode)
            per_mode_outcomes[mode] = outcomes
            metrics[mode] = summarise(mode, outcomes).to_dict()
            print(
                f"  {mode:8} recall@1={metrics[mode]['recall_at']['1']:.3f} "
                f"recall@5={metrics[mode]['recall_at']['5']:.3f} "
                f"recall@10={metrics[mode]['recall_at']['10']:.3f} "
                f"mrr@10={metrics[mode]['mrr_at_10']:.3f} "
                f"({time.perf_counter() - started:.1f}s)"
            )

        entry: dict[str, object] = {
            "num_queries": len(queries),
            "modes": metrics,
            "complementarity_at_10": complementarity(
                per_mode_outcomes["dense"], per_mode_outcomes["lexical"], k=10
            ),
            "examples": _example_failures(per_mode_outcomes["hybrid"], service),
        }

        if include_float32:
            float32_outcomes = run_float32_dense(
                service, queries, index_dir / FLOAT32_VECTORS_PATH.name
            )
            if float32_outcomes is not None:
                entry["dense_float32"] = summarise("dense_float32", float32_outcomes).to_dict()
                int8 = metrics["dense"]["recall_at"]["10"]
                full = entry["dense_float32"]["recall_at"]["10"]
                entry["int8_recall_delta_at_10"] = round(int8 - full, 4)
                print(
                    f"  float32  recall@10={full:.3f} "
                    f"(int8 costs {abs(int8 - full) * 100:.2f} points)"
                )

        report["gold_sets"][set_name] = entry  # type: ignore[index]

    return report


def _example_failures(outcomes: list[QueryOutcome], service: SearchService, n: int = 5) -> dict:
    """A few concrete wins and misses, so the report can be sanity-checked by eye."""
    misses = [o for o in outcomes if o.rank is None][:n]
    wins = [o for o in outcomes if o.rank == 1][:n]

    def describe(outcome: QueryOutcome) -> dict:
        expected = service.get_code(outcome.expected_code)
        top = service.get_code(outcome.returned_codes[0]) if outcome.returned_codes else None
        return {
            "query": outcome.query,
            "expected_code": expected.code_dotted if expected else outcome.expected_code,
            "expected_description": expected.long_desc if expected else "",
            "top_result_code": top.code_dotted if top else None,
            "top_result_description": top.long_desc if top else None,
            "rank_of_expected": outcome.rank,
        }

    return {
        "correct_at_rank_1": [describe(o) for o in wins],
        "not_found_in_top_10": [describe(o) for o in misses],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ICD-10 retrieval quality")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N queries")
    parser.add_argument("--no-float32", action="store_true", help="skip the int8-vs-float32 check")
    parser.add_argument("--out", type=Path, default=EVAL_REPORT_PATH)
    args = parser.parse_args()

    report = evaluate(
        index_dir=args.index_dir,
        limit=args.limit,
        include_float32=not args.no_float32,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[eval] report written to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
