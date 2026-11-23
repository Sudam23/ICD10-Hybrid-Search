"""Int8 quantisation of the dense index.

98,186 vectors x 384 dims as float32 is 151 MB - too big to keep in a git
repository and wasteful to hold in a 512 MB container. Symmetric per-row int8
quantisation cuts that to 38 MB with no approximate-nearest-neighbour
structure, so search stays exact over the quantised vectors.

Per row i:  scale_i = max(|v_i|) / 127,  q_i = round(v_i / scale_i)

Scoring then costs one extra multiply: ``score_i = (q_i . query) * scale_i``.
The accuracy actually lost by this is measured, not assumed - see the
"int8 vs float32" row in docs/EVALUATION.md.
"""

from __future__ import annotations

import numpy as np

INT8_MAX = 127.0
SCORING_BLOCK = 8192


def quantize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(int8 matrix, float32 per-row scales)``."""
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {vectors.shape}")
    peaks = np.abs(vectors).max(axis=1)
    np.maximum(peaks, 1e-12, out=peaks)
    scales = (peaks / INT8_MAX).astype(np.float32)
    quantized = np.rint(vectors / scales[:, None]).clip(-127, 127).astype(np.int8)
    return quantized, scales


def dequantize(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Reconstruct the approximate float32 vectors (used by tests and the eval)."""
    return quantized.astype(np.float32) * scales[:, None]


def score_all(
    quantized: np.ndarray,
    scales: np.ndarray,
    query: np.ndarray,
    *,
    block: int = SCORING_BLOCK,
) -> np.ndarray:
    """Cosine similarity of ``query`` against every stored vector.

    Exact brute force, evaluated in blocks. At 98k x 384 this is ~38 MFLOP,
    which NumPy does in single-digit milliseconds - far simpler and more
    predictable than an ANN index at this corpus size.
    """
    query = np.ascontiguousarray(query, dtype=np.float32).reshape(-1)
    if query.shape[0] != quantized.shape[1]:
        raise ValueError(
            f"query dim {query.shape[0]} does not match index dim {quantized.shape[1]}"
        )
    scores = np.empty(quantized.shape[0], dtype=np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for start in range(0, quantized.shape[0], block):
            stop = min(start + block, quantized.shape[0])
            chunk = quantized[start:stop].astype(np.float32, copy=False)
            scores[start:stop] = chunk @ query
    if not np.isfinite(scores).all():
        raise FloatingPointError("dense scoring produced non-finite values")
    scores *= scales
    return scores


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, scores)`` of the k highest scores, best first."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    candidates = np.argpartition(-scores, k - 1)[:k]
    ordered = candidates[np.argsort(-scores[candidates], kind="stable")]
    return ordered, scores[ordered]
