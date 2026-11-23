"""Search service behaviour, on the toy index from conftest."""

from __future__ import annotations

import pytest

from icd10_search.search.service import SearchService


def codes(response) -> list[str]:
    return [result.code for result in response.results]


def test_semantic_mode_finds_the_right_code(toy_service: SearchService):
    assert "I10" in codes(toy_service.search("high blood pressure", mode="dense"))


def test_every_mode_returns_results(toy_service: SearchService):
    for mode in ("hybrid", "dense", "lexical"):
        assert toy_service.search("diabetes", mode=mode).results


def test_unknown_mode_is_rejected(toy_service: SearchService):
    with pytest.raises(ValueError, match="unknown mode"):
        toy_service.search("diabetes", mode="magic")


def test_empty_query_is_rejected(toy_service: SearchService):
    with pytest.raises(ValueError, match="must not be empty"):
        toy_service.search("   ")


def test_overlong_query_is_rejected(toy_service: SearchService):
    with pytest.raises(ValueError, match="at most"):
        toy_service.search("x" * 5000)


def test_exact_code_query_is_pinned_first(toy_service: SearchService):
    response = toy_service.search("E119")
    assert response.results[0].code == "E119"
    assert response.results[0].match_reason == "exact_code"


def test_dotted_code_query_works_too(toy_service: SearchService):
    assert toy_service.search("E11.9").results[0].code == "E119"


def test_code_prefix_returns_the_children(toy_service: SearchService):
    returned = codes(toy_service.search("E11"))
    assert returned[0] == "E11"
    assert {"E118", "E119"} <= set(returned)
    assert toy_service.search("E11").results[1].match_reason == "code_prefix"


def test_fully_specified_code_offers_its_siblings(toy_service: SearchService):
    """E11.9 has no children, so the alternatives under E11 are offered instead."""
    response = toy_service.search("E11.9")
    returned = codes(response)
    assert returned[0] == "E119"
    assert "E118" in returned, "the sibling code a coder is choosing between is missing"
    sibling = next(r for r in response.results if r.code == "E118")
    assert sibling.match_reason == "code_sibling"
    assert sibling.score is None


def test_siblings_are_not_mislabelled_as_prefix_matches(toy_service: SearchService):
    response = toy_service.search("E11.9")
    for result in response.results:
        if result.match_reason == "code_prefix":
            assert result.code.startswith("E119")


def test_code_lookup_can_be_disabled(toy_service: SearchService):
    response = toy_service.search("E119", code_lookup=False)
    assert all(r.match_reason != "exact_code" for r in response.results)


def test_billable_filter_excludes_category_codes(toy_service: SearchService):
    returned = codes(toy_service.search("diabetes", billable_only=True, k=10))
    assert "E11" not in returned
    assert "E119" in returned


def test_chapter_filter_restricts_results(toy_service: SearchService):
    response = toy_service.search("fever", chapter=18, k=10)
    assert response.results
    assert all(result.chapter == 18 for result in response.results)


def test_filters_are_applied_before_the_cut_not_after(toy_service: SearchService):
    """A filtered search still fills the page, instead of returning a short list."""
    unfiltered = toy_service.search("diabetes", k=3)
    filtered = toy_service.search("diabetes", k=3, billable_only=True)
    assert len(filtered.results) == len(unfiltered.results) == 3


def test_k_is_clamped_to_the_maximum(toy_service: SearchService):
    assert len(toy_service.search("fever", k=10_000).results) <= 10


def test_hybrid_results_carry_per_arm_evidence(toy_service: SearchService):
    response = toy_service.search("typhoid fever", mode="hybrid")
    top = response.results[0]
    assert top.dense_rank or top.lexical_rank
    assert "fever" in top.matched_terms


def test_timings_are_reported(toy_service: SearchService):
    timings = toy_service.search("fever").timings_ms
    assert {"embed", "dense", "lexical", "fuse", "total"} <= set(timings)
    assert timings["total"] > 0


def test_lexical_mode_does_not_embed(toy_service: SearchService):
    assert "embed" not in toy_service.search("fever", mode="lexical").timings_ms


def test_get_code_accepts_both_spellings(toy_service: SearchService):
    assert toy_service.get_code("E11.9").code == "E119"
    assert toy_service.get_code("e119").code == "E119"
    assert toy_service.get_code("ZZZ999") is None


def test_stats_and_chapters(toy_service: SearchService):
    stats = toy_service.stats()
    assert stats["total_codes"] == 10
    assert stats["billable_codes"] == 9
    chapters = toy_service.chapters()
    assert {c["chapter"] for c in chapters} == {1, 4, 8, 9, 18, 19}


def test_pinned_results_have_no_fused_score(toy_service: SearchService):
    """A code lookup did not come from the ranking, so it reports no fused score."""
    top = toy_service.search("E119").results[0]
    assert top.score is None
    assert toy_service.search("fever").results[0].score is not None
