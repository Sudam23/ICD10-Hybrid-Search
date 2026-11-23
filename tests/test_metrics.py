"""Metric arithmetic - if these are wrong, every number in the README is wrong."""

from __future__ import annotations

import pytest

from icd10_search.evaluation.metrics import (
    QueryOutcome,
    complementarity,
    percentile,
    summarise,
)


def outcome(expected, returned, latency=1.0):
    return QueryOutcome(
        query="q", expected_code=expected, returned_codes=returned, latency_ms=latency
    )


def test_rank_is_one_based_and_none_when_missing():
    assert outcome("B", ["A", "B", "C"]).rank == 2
    assert outcome("Z", ["A", "B"]).rank is None


def test_recall_counts_hits_within_each_cutoff():
    metrics = summarise("m", [outcome("A", ["A"]), outcome("B", ["X", "Y", "B"])])
    assert metrics.recall_at[1] == 0.5
    assert metrics.recall_at[3] == 1.0


def test_mrr_is_the_mean_reciprocal_rank():
    metrics = summarise("m", [outcome("A", ["A"]), outcome("B", ["X", "B"])])
    assert metrics.mrr_at_10 == pytest.approx((1.0 + 0.5) / 2)


def test_a_miss_contributes_zero_to_mrr():
    assert summarise("m", [outcome("Z", ["A"])]).mrr_at_10 == 0.0


def test_an_empty_run_does_not_divide_by_zero():
    assert summarise("m", []).num_queries == 0


def test_percentile_is_nearest_rank():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10
    assert percentile([], 0.5) == 0.0


def test_complementarity_partitions_every_query():
    first = [outcome("A", ["A"]), outcome("B", ["X"]), outcome("C", ["X"])]
    second = [outcome("A", ["X"]), outcome("B", ["B"]), outcome("C", ["X"])]
    report = complementarity(first, second)
    assert report["only_first"] == 1
    assert report["only_second"] == 1
    assert report["neither"] == 1
    assert report["both"] == 0
    assert (
        sum(report[key] for key in ("both", "only_first", "only_second", "neither"))
        == report["total"]
    )
