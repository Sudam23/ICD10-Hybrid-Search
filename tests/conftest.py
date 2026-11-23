"""Shared fixtures.

The unit tests must run anywhere - including CI with no network and no built
index - so they construct a tiny in-memory corpus and a deterministic fake
embedder instead of loading the real 98k index. The real index is exercised
separately in ``test_integration.py``, which skips itself when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from icd10_search.data.parse import CodeRecord
from icd10_search.data.text import document_text, lexical_text
from icd10_search.indexing.quantize import quantize
from icd10_search.search.dense import DenseIndex
from icd10_search.search.lexical import LexicalIndex
from icd10_search.search.service import SearchService

VOCAB = [
    "cholera",
    "typhoid",
    "fever",
    "diabetes",
    "hypertension",
    "pressure",
    "blood",
    "fracture",
    "wrist",
    "left",
    "pain",
    "chest",
    "ear",
    "tinnitus",
]


def _toy_vector(text: str) -> np.ndarray:
    """A deterministic bag-of-words vector over a tiny vocabulary.

    This is a stand-in for a sentence transformer, not an imitation of one: it
    is enough to make 'similar text scores higher' true, which is all the
    service logic under test depends on.
    """
    lowered = text.lower()
    vector = np.array([float(lowered.count(term)) for term in VOCAB], dtype=np.float32)
    if not vector.any():
        vector[0] = 0.01
    return vector / np.linalg.norm(vector)


class FakeEmbedder:
    model_name = "fake/toy-model"

    def embed_documents(self, texts, **_kwargs) -> np.ndarray:
        return np.vstack([_toy_vector(text) for text in texts])

    def embed_queries(self, texts, **_kwargs) -> np.ndarray:
        return np.vstack([_toy_vector(text) for text in texts])

    def warm_up(self) -> None:
        return None


def _record(code, desc, *, billable=True, chapter=1, synonyms=(), parent=None) -> CodeRecord:
    return CodeRecord(
        code=code,
        code_dotted=code if len(code) <= 3 else f"{code[:3]}.{code[3:]}",
        billable=billable,
        short_desc=desc[:60],
        long_desc=desc,
        chapter=chapter,
        chapter_title=f"Chapter {chapter}",
        block=f"{code[:3]}-{code[:3]}",
        block_title="Test block",
        parent_code=parent,
        parent_desc=None,
        inclusion_terms=list(synonyms),
        inherited_terms=[],
    )


@pytest.fixture(scope="session")
def toy_records() -> list[CodeRecord]:
    return [
        _record("A009", "Cholera, unspecified"),
        _record("A0100", "Typhoid fever, unspecified", synonyms=["Salmonella typhi infection"]),
        _record("E11", "Type 2 diabetes mellitus", billable=False, chapter=4),
        _record("E119", "Type 2 diabetes mellitus without complications", chapter=4, parent="E11"),
        _record("E118", "Type 2 diabetes mellitus with complications", chapter=4, parent="E11"),
        _record("I10", "Essential hypertension", chapter=9, synonyms=["high blood pressure"]),
        _record("R079", "Chest pain, unspecified", chapter=18),
        _record("S52501A", "Fracture of left wrist, initial encounter", chapter=19),
        _record("H9319", "Tinnitus, unspecified ear", chapter=8),
        _record("R509", "Fever, unspecified", chapter=18),
    ]


@pytest.fixture(scope="session")
def toy_service(toy_records) -> SearchService:
    embedder = FakeEmbedder()
    vectors = embedder.embed_documents([document_text(r) for r in toy_records])
    quantized, scales = quantize(vectors)
    dense = DenseIndex(quantized, scales)
    lexical = LexicalIndex.build([lexical_text(r) for r in toy_records])
    manifest = {
        "embedding_model": "fake/toy-model",
        "doc_text_variant": "desc_incl_parent",
        "quantization": "int8-symmetric-per-row",
        "bm25": {"vocabulary_terms": len(lexical.vocab)},
        "built_at": "2026-01-01T00:00:00+00:00",
        "artifact_bytes": {},
    }
    return SearchService(toy_records, dense, lexical, embedder, manifest)
