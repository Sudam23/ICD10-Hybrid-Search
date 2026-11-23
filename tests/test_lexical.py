"""Tokenisation and BM25."""

from __future__ import annotations

import numpy as np

from icd10_search.search.lexical import LexicalIndex, tokenize

DOCUMENTS = [
    "Type 2 diabetes mellitus without complications",
    "Type 2 diabetes mellitus with complications",
    "Essential primary hypertension high blood pressure",
    "Fracture of left wrist initial encounter",
    "Cholera due to Vibrio cholerae",
]


def test_clinical_qualifiers_survive_tokenisation():
    """'with' and 'without' separate two different codes, so they are not stopwords."""
    tokens = tokenize("Type 2 diabetes mellitus without complications")
    assert "without" in tokens
    assert "with" in tokenize("diabetes with complications")


def test_function_words_are_dropped():
    assert tokenize("Cholera due to the Vibrio") == ["cholera", "due", "vibrio"]


def test_s_stemmer_joins_singular_and_plural():
    assert tokenize("fractures") == tokenize("fracture")
    assert tokenize("abscess") == ["abscess"]


def test_digits_are_kept():
    assert "2" in tokenize("type 2 diabetes")


def test_bm25_ranks_the_matching_document_first():
    index = LexicalIndex.build(DOCUMENTS)
    scores = index.score("high blood pressure")
    assert int(np.argmax(scores)) == 2
    assert scores[0] == 0.0


def test_bm25_discriminates_with_from_without():
    index = LexicalIndex.build(DOCUMENTS)
    scores = index.score("diabetes without complications")
    assert scores[0] > scores[1]


def test_unknown_terms_score_zero():
    index = LexicalIndex.build(DOCUMENTS)
    assert index.score("xyzzy nonexistentterm").max() == 0.0


def test_idf_is_higher_for_rarer_terms():
    index = LexicalIndex.build(DOCUMENTS)
    rare = index.idf(index.vocab["cholera"])
    common = index.idf(index.vocab["diabete"])
    assert rare > common


def test_save_and_load_round_trip(tmp_path):
    index = LexicalIndex.build(DOCUMENTS)
    path = index.save(tmp_path / "bm25.npz")
    reloaded = LexicalIndex.load(path)

    assert reloaded.vocab == index.vocab
    assert reloaded.num_docs == index.num_docs
    assert reloaded.k1 == index.k1 and reloaded.b == index.b
    np.testing.assert_allclose(reloaded.score("diabetes"), index.score("diabetes"))
