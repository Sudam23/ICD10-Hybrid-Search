"""End-to-end checks against the real 98k index.

Skipped automatically when the index has not been built, so a fresh clone can
still run `make test` before `make index`.
"""

from __future__ import annotations

import pytest

from icd10_search.config import DENSE_VECTORS_PATH, INDEX_MANIFEST_PATH

pytestmark = pytest.mark.skipif(
    not (DENSE_VECTORS_PATH.exists() and INDEX_MANIFEST_PATH.exists()),
    reason="real index not built; run `make data && make index`",
)


@pytest.fixture(scope="module")
def service():
    from icd10_search.search.service import SearchService

    return SearchService.load()


def test_index_covers_the_whole_code_set(service):
    assert service.stats()["total_codes"] == 98_186
    assert service.stats()["billable_codes"] == 74_719


@pytest.mark.parametrize(
    ("query", "expected_code"),
    [
        ("high blood pressure", "I10"),
        ("ringing in the ears", "H9319"),
        ("type 2 diabetes with no complications", "E119"),
        ("covid-19", "U071"),
        ("urinary tract infection", "N390"),
    ],
)
def test_known_queries_find_their_code_in_the_top_10(service, query, expected_code):
    response = service.search(query, k=10)
    assert expected_code in [
        result.code for result in response.results
    ], f"{query!r} returned {[r.code_dotted for r in response.results]}"


def test_a_real_code_lookup_is_pinned(service):
    assert service.search("E11.9").results[0].code == "E119"


def test_latency_is_interactive(service):
    service.search("warm up the caches")
    timings = service.search("crushing chest pain radiating to the left arm").timings_ms
    assert timings["total"] < 500, timings
