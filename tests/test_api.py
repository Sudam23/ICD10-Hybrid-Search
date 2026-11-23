"""HTTP contract, served from the toy index."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from icd10_search.api import main as api_main


@pytest.fixture()
def client(toy_service, monkeypatch):
    """A client whose app already has the toy service loaded, bypassing startup."""
    monkeypatch.setitem(api_main.STATE, "service", toy_service)
    monkeypatch.setitem(api_main.STATE, "error", None)
    monkeypatch.setattr(api_main, "rate_limiter", api_main.RateLimiter(10_000, 60))
    with TestClient(api_main.app) as test_client:
        monkeypatch.setitem(api_main.STATE, "service", toy_service)
        yield test_client


def test_health_reports_ready(client):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["index_ready"] is True


def test_search_returns_results_and_a_disclaimer(client):
    payload = client.get("/api/search", params={"q": "high blood pressure"}).json()
    assert payload["results"]
    assert "not clinical advice" in payload["disclaimer"] or "Demo" in payload["disclaimer"]
    assert payload["results"][0]["code_dotted"]


def test_search_requires_a_query(client):
    assert client.get("/api/search").status_code == 422


def test_search_rejects_an_overlong_query(client):
    assert client.get("/api/search", params={"q": "x" * 1000}).status_code == 422


def test_search_rejects_an_out_of_range_k(client):
    assert client.get("/api/search", params={"q": "fever", "k": 999}).status_code == 422


def test_search_rejects_an_unknown_mode(client):
    assert client.get("/api/search", params={"q": "fever", "mode": "psychic"}).status_code == 422


def test_search_honours_filters(client):
    payload = client.get(
        "/api/search", params={"q": "diabetes", "billable_only": "true", "k": 5}
    ).json()
    assert all(result["billable"] for result in payload["results"])


def test_code_detail(client):
    payload = client.get("/api/code/E11.9").json()
    assert payload["code"] == "E119"
    assert payload["billable"] is True


def test_unknown_code_is_404(client):
    assert client.get("/api/code/ZZZ999").status_code == 404


def test_stats_and_chapters_and_examples(client):
    assert client.get("/api/stats").json()["total_codes"] == 10
    assert client.get("/api/chapters").json()
    assert client.get("/api/examples").json()["examples"]


def test_rate_limiter_returns_429(client, monkeypatch):
    monkeypatch.setattr(api_main, "rate_limiter", api_main.RateLimiter(3, 60))
    statuses = [client.get("/api/health").status_code for _ in range(5)]
    assert 429 in statuses
    assert statuses.count(200) == 3


def test_service_unavailable_before_the_index_loads(monkeypatch):
    monkeypatch.setitem(api_main.STATE, "service", None)
    monkeypatch.setitem(api_main.STATE, "error", None)
    monkeypatch.setattr(api_main, "rate_limiter", api_main.RateLimiter(10_000, 60))
    client = TestClient(api_main.app)
    response = client.get("/api/search", params={"q": "fever"})
    assert response.status_code == 503


def test_search_reports_why_each_result_matched(client):
    payload = client.get("/api/search", params={"q": "high blood pressure"}).json()
    reasons = {result["match_reason"] for result in payload["results"]}
    assert reasons
    assert reasons <= {"hybrid", "dense", "lexical", "fused"}


def test_code_query_is_pinned_and_labelled_as_an_exact_match(client):
    payload = client.get("/api/search", params={"q": "E11.9"}).json()
    first = payload["results"][0]
    assert first["code_dotted"] == "E11.9"
    assert first["match_reason"] == "exact_code"
    assert first["score"] is None


def test_code_prefix_query_is_labelled_as_a_prefix_match(client):
    payload = client.get("/api/search", params={"q": "E11"}).json()
    reasons = [r["match_reason"] for r in payload["results"]]
    assert reasons[0] == "exact_code"
    assert "code_prefix" in reasons
