"""Build every search artifact from the parsed corpus.

Outputs (all written to ``data/index/``):

    dense_int8.npy     int8[N, 384]  quantised document embeddings
    dense_scales.npy   float32[N]    per-row de-quantisation scale
    lexical_bm25.npz   the inverted index
    manifest.json      what was built, from what, with which settings

The manifest is the contract between build time and serve time: the API
refuses to start if the index was built from a different corpus or a different
embedding model than the one it is configured for. That is what stops a stale
index from silently serving wrong results after a model change.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from icd10_search.config import (
    CORPUS_PATH,
    DENSE_SCALES_PATH,
    DENSE_VECTORS_PATH,
    DOC_TEXT_VARIANT,
    EMBEDDING_MODEL,
    INDEX_DIR,
    INDEX_MANIFEST_PATH,
    LEXICAL_INDEX_PATH,
)
from icd10_search.data.parse import load_corpus
from icd10_search.data.text import document_text, lexical_text
from icd10_search.indexing.embedder import Embedder
from icd10_search.indexing.quantize import quantize
from icd10_search.search.lexical import LexicalIndex

FLOAT32_VECTORS_PATH = INDEX_DIR / "dense_float32.npy"


def corpus_fingerprint(path: Path = CORPUS_PATH) -> str:
    """SHA-256 of the corpus file, truncated - identifies which data an index came from."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def build(
    *,
    model_name: str = EMBEDDING_MODEL,
    doc_text_variant: str = DOC_TEXT_VARIANT,
    index_dir: Path = INDEX_DIR,
    keep_float32: bool = False,
    limit: int | None = None,
) -> dict[str, object]:
    index_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    records = load_corpus()
    if limit:
        records = records[:limit]
    print(f"[build] {len(records):,} codes, text variant '{doc_text_variant}'")

    doc_texts = [document_text(record, doc_text_variant) for record in records]
    embedder = Embedder(model_name)
    embed_started = time.perf_counter()
    vectors = embedder.embed_documents(doc_texts)
    embed_seconds = time.perf_counter() - embed_started
    print(f"[build] embedded in {embed_seconds:,.1f}s -> {vectors.shape}")

    quantized, scales = quantize(vectors)
    np.save(index_dir / DENSE_VECTORS_PATH.name, quantized)
    np.save(index_dir / DENSE_SCALES_PATH.name, scales)
    if keep_float32:
        np.save(index_dir / FLOAT32_VECTORS_PATH.name, vectors)

    lexical_started = time.perf_counter()
    lexical = LexicalIndex.build([lexical_text(record) for record in records])
    lexical.save(index_dir / LEXICAL_INDEX_PATH.name)
    lexical_seconds = time.perf_counter() - lexical_started
    print(
        f"[build] BM25 index: {len(lexical.vocab):,} terms, "
        f"{lexical.postings_docs.shape[0]:,} postings in {lexical_seconds:,.1f}s"
    )

    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus_path": str(CORPUS_PATH.name),
        "corpus_sha256_prefix": corpus_fingerprint(),
        "num_documents": int(len(records)),
        "embedding_model": model_name,
        "embedding_dim": int(vectors.shape[1]),
        "doc_text_variant": doc_text_variant,
        "quantization": "int8-symmetric-per-row",
        "bm25": {
            "k1": lexical.k1,
            "b": lexical.b,
            "vocabulary_terms": int(len(lexical.vocab)),
            "postings": int(lexical.postings_docs.shape[0]),
            "avg_doc_length": round(lexical.avg_doc_length, 3),
        },
        "build_seconds": round(time.perf_counter() - started, 1),
        "artifact_bytes": {
            path.name: path.stat().st_size
            for path in (
                index_dir / DENSE_VECTORS_PATH.name,
                index_dir / DENSE_SCALES_PATH.name,
                index_dir / LEXICAL_INDEX_PATH.name,
            )
        },
    }
    (index_dir / INDEX_MANIFEST_PATH.name).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[build] done in {manifest['build_seconds']}s -> {index_dir}")
    return manifest


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Build the ICD-10 search index")
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--variant", default=DOC_TEXT_VARIANT)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--keep-float32", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(
        json.dumps(
            build(
                model_name=args.model,
                doc_text_variant=args.variant,
                index_dir=args.index_dir,
                keep_float32=args.keep_float32,
                limit=args.limit,
            ),
            indent=2,
        )
    )
