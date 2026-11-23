"""Reciprocal rank fusion."""

from __future__ import annotations

import numpy as np

from icd10_search.search.fusion import RankedList, reciprocal_rank_fusion


def ranked(name, ids, weight=1.0):
    return RankedList(name, np.array(ids), np.linspace(1.0, 0.1, len(ids)), weight=weight)


def test_document_found_by_both_retrievers_outranks_one_found_by_one():
    fused = reciprocal_rank_fusion([ranked("a", [1, 2]), ranked("b", [2, 3])], k=60)
    assert fused[0][0] == 2


def test_score_matches_the_formula():
    fused = reciprocal_rank_fusion([ranked("a", [7])], k=60)
    doc_id, score, provenance = fused[0]
    assert doc_id == 7
    assert score == 1 / 61
    assert provenance == {"a": 1}


def test_weights_shift_the_outcome():
    """Down-weighting the noisy arm is what makes hybrid beat dense alone.

    Document 99 is the failure mode seen on real lay-language queries: BM25
    loves it (rank 1) while the embedding puts it 10th. With equal weights the
    keyword arm drags it above the semantic arm's own top hit; at weight 0.1 it
    no longer can.
    """
    dense = ranked("dense", [*range(1, 10), 99])
    equal = reciprocal_rank_fusion([dense, ranked("lexical", [99])], k=60)
    assert equal[0][0] == 99

    damped = reciprocal_rank_fusion([dense, ranked("lexical", [99], weight=0.1)], k=60)
    assert damped[0][0] == 1
    assert dict(((doc, score) for doc, score, _ in damped))[99] < damped[0][1]


def test_provenance_records_every_contributing_rank():
    fused = reciprocal_rank_fusion([ranked("a", [5, 6]), ranked("b", [6, 5])], k=10)
    provenance = {doc_id: ranks for doc_id, _, ranks in fused}
    assert provenance[5] == {"a": 1, "b": 2}
    assert provenance[6] == {"a": 2, "b": 1}


def test_top_n_truncates():
    fused = reciprocal_rank_fusion([ranked("a", list(range(20)))], k=60, top_n=5)
    assert len(fused) == 5


def test_smaller_k_sharpens_the_top_rank_advantage():
    lists = [ranked("a", [1]), ranked("b", [2])]
    assert reciprocal_rank_fusion(lists, k=1)[0][1] > reciprocal_rank_fusion(lists, k=100)[0][1]
