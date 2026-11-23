"""Retrieval metrics.

Each gold query has exactly one correct code, so the metrics are the simple
single-relevant-document forms:

``recall@k``   fraction of queries whose correct code appears in the top k.
               With one relevant document this equals accuracy@k and equals
               precision@k * k, so only one of them is worth reporting.
``MRR@k``      mean of 1/rank of the correct code, 0 if it is not in the top k.
               Rewards putting the right answer first, not merely somewhere.

Latency is reported as median and p95 rather than mean, because a mean hides
the cold-cache outliers that a user actually notices.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

DEFAULT_KS = (1, 3, 5, 10)


@dataclass(slots=True)
class QueryOutcome:
    query: str
    expected_code: str
    returned_codes: list[str]
    latency_ms: float

    @property
    def rank(self) -> int | None:
        """1-based rank of the correct code, or None if absent."""
        try:
            return self.returned_codes.index(self.expected_code) + 1
        except ValueError:
            return None


@dataclass(slots=True)
class MetricSet:
    name: str
    num_queries: int
    recall_at: dict[int, float] = field(default_factory=dict)
    mrr_at_10: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "num_queries": self.num_queries,
            "recall_at": {str(k): round(v, 4) for k, v in self.recall_at.items()},
            "mrr_at_10": round(self.mrr_at_10, 4),
            "median_latency_ms": round(self.median_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; avoids a SciPy dependency for one number."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, int(round(fraction * len(ordered) + 0.5)) - 1))
    return ordered[position]


def summarise(
    name: str, outcomes: list[QueryOutcome], ks: tuple[int, ...] = DEFAULT_KS
) -> MetricSet:
    total = len(outcomes)
    if total == 0:
        return MetricSet(name=name, num_queries=0)

    ranks = [outcome.rank for outcome in outcomes]
    latencies = [outcome.latency_ms for outcome in outcomes]

    recall = {k: sum(1 for rank in ranks if rank is not None and rank <= k) / total for k in ks}
    mrr = sum(1.0 / rank for rank in ranks if rank is not None and rank <= 10) / total

    return MetricSet(
        name=name,
        num_queries=total,
        recall_at=recall,
        mrr_at_10=mrr,
        median_latency_ms=statistics.median(latencies),
        p95_latency_ms=percentile(latencies, 0.95),
    )


def complementarity(
    outcomes_a: list[QueryOutcome],
    outcomes_b: list[QueryOutcome],
    k: int = 10,
) -> dict[str, object]:
    """How often two retrievers disagree - the case for fusing them.

    If one arm were strictly better than the other on every query, hybrid
    search would be pointless. The "only A" / "only B" counts are the direct
    evidence that it is not.
    """

    def hit(outcome: QueryOutcome) -> bool:
        return outcome.rank is not None and outcome.rank <= k

    pairs = list(zip(outcomes_a, outcomes_b, strict=True))
    both = sum(1 for a, b in pairs if hit(a) and hit(b))
    only_a = sum(1 for a, b in pairs if hit(a) and not hit(b))
    only_b = sum(1 for a, b in pairs if hit(b) and not hit(a))
    neither = sum(1 for a, b in pairs if not hit(a) and not hit(b))
    return {
        "k": k,
        "both": both,
        "only_first": only_a,
        "only_second": only_b,
        "neither": neither,
        "total": len(pairs),
    }
