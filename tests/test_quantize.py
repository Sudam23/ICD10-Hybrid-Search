"""Int8 quantisation must not change which documents win."""

from __future__ import annotations

import numpy as np
import pytest

from icd10_search.indexing.embedder import l2_normalise
from icd10_search.indexing.quantize import dequantize, quantize, score_all, top_k


@pytest.fixture()
def vectors() -> np.ndarray:
    rng = np.random.default_rng(7)
    return l2_normalise(rng.standard_normal((500, 384)).astype(np.float32))


def test_quantize_round_trip_is_close(vectors):
    quantized, scales = quantize(vectors)
    assert quantized.dtype == np.int8
    assert scales.shape == (vectors.shape[0],)
    reconstructed = dequantize(quantized, scales)
    assert np.abs(reconstructed - vectors).max() < (np.abs(vectors).max() / 127) / 2 + 1e-6


def test_score_all_matches_a_float64_reference(vectors):
    """Guards the chunked int8 matmul, including the macOS Accelerate path."""
    quantized, scales = quantize(vectors)
    query = l2_normalise(np.random.default_rng(1).standard_normal((1, 384)).astype(np.float32))[0]

    got = score_all(quantized, scales, query, block=64)
    reference = (
        np.einsum("ij,j->i", quantized.astype(np.float64), query.astype(np.float64)) * scales
    )
    assert np.isfinite(got).all()
    assert np.abs(got - reference).max() < 1e-5


def test_score_all_is_close_to_exact_cosine(vectors):
    quantized, scales = quantize(vectors)
    query = vectors[11]
    approximate = score_all(quantized, scales, query)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        exact = vectors @ query
    assert np.abs(approximate - exact).max() < 0.01
    assert int(np.argmax(approximate)) == int(np.argmax(exact)) == 11


def test_score_all_rejects_a_dimension_mismatch(vectors):
    quantized, scales = quantize(vectors)
    with pytest.raises(ValueError, match="does not match index dim"):
        score_all(quantized, scales, np.zeros(16, dtype=np.float32))


def test_top_k_orders_by_descending_score():
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3], dtype=np.float32)
    indices, values = top_k(scores, 3)
    assert indices.tolist() == [1, 3, 2]
    assert values.tolist() == pytest.approx([0.9, 0.7, 0.5])


def test_top_k_handles_k_larger_than_the_corpus():
    scores = np.array([0.2, 0.4], dtype=np.float32)
    indices, _ = top_k(scores, 10)
    assert indices.tolist() == [1, 0]
