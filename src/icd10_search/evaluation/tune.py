"""Choose the fusion parameters on the dev split.

Equal-weight Reciprocal Rank Fusion turned out to be *worse* than the dense
arm alone: BM25 always returns a full pool of candidates, so for a lay-language
query it contributes a hundred weak matches that the fusion then rewards for
appearing in both lists. The fix is to weight the arms - but the weight has to
be measured, not guessed.

Two things make this cheap and honest:

* The expensive part of a query (embedding it and scoring it against 98k
  vectors) does not depend on the fusion parameters, so each dev query is run
  through both retrievers exactly once and the ranked candidate lists are
  cached. The sweep itself then costs milliseconds per configuration.
* The sweep only ever sees the dev split. The test split and the hand-written
  lay set are untouched here, so the numbers reported in the README are not
  the numbers this script optimised.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from icd10_search.config import (
    CANDIDATE_POOL,
    GOLD_INDEX_DEV_PATH,
    INDEX_DIR,
    TUNING_REPORT_PATH,
)
from icd10_search.data.goldset import GoldQuery, load_gold_set
from icd10_search.evaluation.metrics import QueryOutcome, summarise
from icd10_search.indexing.quantize import top_k
from icd10_search.search.fusion import RankedList, reciprocal_rank_fusion
from icd10_search.search.service import SearchService

WEIGHT_GRID = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.75, 1.0)
RRF_K_GRID = (10, 30, 60, 100)


@dataclass(slots=True)
class CachedQuery:
    """Both arms' candidate lists for one query, computed once."""

    gold: GoldQuery
    dense_ids: np.ndarray
    dense_scores: np.ndarray
    lexical_ids: np.ndarray
    lexical_scores: np.ndarray


def collect_candidates(
    service: SearchService,
    queries: list[GoldQuery],
    pool: int = CANDIDATE_POOL,
) -> list[CachedQuery]:
    cached: list[CachedQuery] = []
    started = time.perf_counter()
    for position, gold in enumerate(queries, start=1):
        query_vector = service.embedder.embed_queries([gold.query])[0]
        dense_ids, dense_scores = top_k(service.dense.score(query_vector), pool)
        lexical_all = service.lexical.score(gold.query)
        lexical_ids, lexical_scores = top_k(lexical_all, pool)
        keep = lexical_scores > 0
        cached.append(
            CachedQuery(
                gold=gold,
                dense_ids=dense_ids,
                dense_scores=dense_scores,
                lexical_ids=lexical_ids[keep],
                lexical_scores=lexical_scores[keep],
            )
        )
        if position % 200 == 0:
            rate = position / (time.perf_counter() - started)
            print(f"  cached {position:,}/{len(queries):,} ({rate:.0f} q/s)", flush=True)
    return cached


def score_configuration(
    service: SearchService,
    cached: list[CachedQuery],
    *,
    weight_dense: float,
    weight_lexical: float,
    rrf_k: int,
    k: int = 10,
) -> list[QueryOutcome]:
    outcomes: list[QueryOutcome] = []
    for item in cached:
        lists = [
            RankedList("dense", item.dense_ids, item.dense_scores, weight=weight_dense),
            RankedList("lexical", item.lexical_ids, item.lexical_scores, weight=weight_lexical),
        ]
        fused = reciprocal_rank_fusion(lists, k=rrf_k, top_n=k)
        outcomes.append(
            QueryOutcome(
                query=item.gold.query,
                expected_code=item.gold.expected_code,
                returned_codes=[service.records[doc_id].code for doc_id, _, _ in fused],
                latency_ms=0.0,
            )
        )
    return outcomes


def tune(
    *,
    index_dir: Path = INDEX_DIR,
    limit: int | None = None,
    select_on: str = "mrr_at_10",
) -> dict:
    service = SearchService.load(index_dir)
    queries = load_gold_set(GOLD_INDEX_DEV_PATH)
    if limit:
        queries = queries[:limit]

    print(f"[tune] caching candidates for {len(queries):,} dev queries ...")
    cached = collect_candidates(service, queries)

    grid: list[dict] = []
    print(f"[tune] sweeping {len(WEIGHT_GRID) * len(RRF_K_GRID)} configurations ...")
    for rrf_k in RRF_K_GRID:
        for weight_lexical in WEIGHT_GRID:
            outcomes = score_configuration(
                service,
                cached,
                weight_dense=1.0,
                weight_lexical=weight_lexical,
                rrf_k=rrf_k,
            )
            metrics = summarise(f"w={weight_lexical} k={rrf_k}", outcomes).to_dict()
            grid.append(
                {
                    "weight_dense": 1.0,
                    "weight_lexical": weight_lexical,
                    "rrf_k": rrf_k,
                    "recall_at_1": metrics["recall_at"]["1"],
                    "recall_at_5": metrics["recall_at"]["5"],
                    "recall_at_10": metrics["recall_at"]["10"],
                    "mrr_at_10": metrics["mrr_at_10"],
                }
            )

    best = max(grid, key=lambda row: (row[select_on], row["recall_at_10"]))
    dense_only = next(row for row in grid if row["weight_lexical"] == 0.0 and row["rrf_k"] == 60)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": "cms_index_dev",
        "num_queries": len(queries),
        "selection_metric": select_on,
        "note": (
            "Tuned on the dev split only. weight_lexical=0 is the dense arm on its own, "
            "which is the baseline any hybrid configuration has to beat."
        ),
        "dense_only_baseline": dense_only,
        "best": best,
        "grid": grid,
    }

    print("\n[tune] dense-only baseline:", json.dumps(dense_only))
    print("[tune] best configuration: ", json.dumps(best))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the fusion parameters on the dev split")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--select-on", default="mrr_at_10", choices=["mrr_at_10", "recall_at_10"])
    parser.add_argument("--out", type=Path, default=TUNING_REPORT_PATH)
    args = parser.parse_args()

    report = tune(index_dir=args.index_dir, limit=args.limit, select_on=args.select_on)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[tune] report written to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
