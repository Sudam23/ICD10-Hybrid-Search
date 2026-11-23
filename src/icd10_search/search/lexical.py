"""BM25 over a compact NumPy inverted index.

Why hand-rolled rather than a library: the whole index is four flat arrays, it
loads in milliseconds, it adds no dependency to the deployed container, and
every number in the ranking can be traced by hand when explaining the project.
At 98k very short documents it scores a query in ~1 ms.

Layout (one ``.npz``):

    vocab_blob     utf-8 bytes of "\\n".join(terms), in term-id order
    term_offsets   int64[V+1]  slice bounds into the postings arrays
    postings_docs  int32[P]    document ids, ascending within a term
    postings_tfs   int32[P]    term frequency of that term in that document
    doc_lengths    int32[N]    token count per document

Text handling is deliberately minimal and symmetric between indexing and
querying:

* lower-case, split on anything that is not a letter or digit;
* a tiny stop list - note that "with" and "without" are **not** stopwords,
  because in ICD-10 they separate two different codes
  ("Type 2 diabetes mellitus *without* complications" is E11.9);
* an "s-stemmer" that strips one trailing "s" from longer tokens, so
  "fractures" and "fracture" meet. It is crude ("diabetes" becomes "diabete")
  but it is applied identically on both sides, so matching is unaffected.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from icd10_search.config import BM25_B, BM25_K1

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset({"the", "a", "an", "of", "and", "to", "in", "on", "at", "by", "or"})


def _s_stem(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case, split, drop stopwords, s-stem. Used for documents and queries."""
    return [_s_stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class LexicalIndex:
    """An immutable BM25 index."""

    def __init__(
        self,
        vocab: dict[str, int],
        term_offsets: np.ndarray,
        postings_docs: np.ndarray,
        postings_tfs: np.ndarray,
        doc_lengths: np.ndarray,
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        self.vocab = vocab
        self.term_offsets = term_offsets
        self.postings_docs = postings_docs
        self.postings_tfs = postings_tfs
        self.doc_lengths = doc_lengths.astype(np.float32)
        self.k1 = k1
        self.b = b

        self.num_docs = int(doc_lengths.shape[0])
        self.avg_doc_length = float(self.doc_lengths.mean()) if self.num_docs else 0.0
        denominator = self.avg_doc_length or 1.0
        self._length_norm = (k1 * (1.0 - b + b * self.doc_lengths / denominator)).astype(np.float32)

    @classmethod
    def build(cls, texts: Sequence[str], **kwargs) -> LexicalIndex:
        postings: dict[str, dict[int, int]] = defaultdict(dict)
        doc_lengths = np.zeros(len(texts), dtype=np.int32)

        for doc_id, text in enumerate(texts):
            tokens = tokenize(text)
            doc_lengths[doc_id] = len(tokens)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, frequency in counts.items():
                postings[token][doc_id] = frequency

        terms = sorted(postings)
        vocab = {term: term_id for term_id, term in enumerate(terms)}
        term_offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        docs_parts: list[np.ndarray] = []
        tfs_parts: list[np.ndarray] = []

        for term_id, term in enumerate(terms):
            entries = postings[term]
            doc_ids = np.fromiter(entries.keys(), dtype=np.int32, count=len(entries))
            frequencies = np.fromiter(entries.values(), dtype=np.int32, count=len(entries))
            order = np.argsort(doc_ids, kind="stable")
            docs_parts.append(doc_ids[order])
            tfs_parts.append(frequencies[order])
            term_offsets[term_id + 1] = term_offsets[term_id] + len(entries)

        empty32 = np.empty(0, dtype=np.int32)
        return cls(
            vocab=vocab,
            term_offsets=term_offsets,
            postings_docs=np.concatenate(docs_parts) if docs_parts else empty32,
            postings_tfs=np.concatenate(tfs_parts) if tfs_parts else empty32,
            doc_lengths=doc_lengths,
            **kwargs,
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        terms = sorted(self.vocab, key=self.vocab.get)
        blob = np.frombuffer("\n".join(terms).encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(
            path,
            vocab_blob=blob,
            term_offsets=self.term_offsets,
            postings_docs=self.postings_docs,
            postings_tfs=self.postings_tfs,
            doc_lengths=self.doc_lengths.astype(np.int32),
            params=np.array([self.k1, self.b], dtype=np.float32),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> LexicalIndex:
        with np.load(path) as payload:
            blob = payload["vocab_blob"].tobytes().decode("utf-8")
            terms = blob.split("\n") if blob else []
            k1, b = (float(v) for v in payload["params"])
            return cls(
                vocab={term: term_id for term_id, term in enumerate(terms)},
                term_offsets=payload["term_offsets"],
                postings_docs=payload["postings_docs"],
                postings_tfs=payload["postings_tfs"],
                doc_lengths=payload["doc_lengths"],
                k1=k1,
                b=b,
            )

    def idf(self, term_id: int) -> float:
        """Robertson/Sparck-Jones IDF with the +1 smoothing that keeps it positive."""
        document_frequency = int(self.term_offsets[term_id + 1] - self.term_offsets[term_id])
        return float(
            np.log(1.0 + (self.num_docs - document_frequency + 0.5) / (document_frequency + 0.5))
        )

    def score(self, query: str) -> np.ndarray:
        """BM25 score of the query against every document."""
        scores = np.zeros(self.num_docs, dtype=np.float32)
        seen: set[int] = set()
        for token in tokenize(query):
            term_id = self.vocab.get(token)
            if term_id is None or term_id in seen:
                continue
            seen.add(term_id)
            start, stop = self.term_offsets[term_id], self.term_offsets[term_id + 1]
            doc_ids = self.postings_docs[start:stop]
            frequencies = self.postings_tfs[start:stop].astype(np.float32)
            contribution = (
                self.idf(term_id)
                * frequencies
                * (self.k1 + 1.0)
                / (frequencies + self._length_norm[doc_ids])
            )
            scores[doc_ids] += contribution
        return scores
