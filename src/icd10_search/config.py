"""Central configuration.

Every tunable lives here so that the data pipeline, the index builder, the API
and the evaluation harness cannot drift apart. Paths are resolved relative to
the repository root, which makes the project work the same way from a local
checkout and from inside the Docker image.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

DATA_DIR = Path(os.environ.get("ICD10_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
WEB_DIR = Path(os.environ.get("ICD10_WEB_DIR", REPO_ROOT / "web"))

CORPUS_PATH = PROCESSED_DIR / "corpus.jsonl.gz"
GOLD_INDEX_DEV_PATH = PROCESSED_DIR / "gold_index_dev.jsonl"
GOLD_INDEX_TEST_PATH = PROCESSED_DIR / "gold_index_test.jsonl"
GOLD_LAY_PATH = PROCESSED_DIR / "gold_lay_queries.jsonl"
EVAL_REPORT_PATH = REPO_ROOT / "docs" / "evaluation_report.json"
TUNING_REPORT_PATH = REPO_ROOT / "docs" / "tuning_report.json"

DENSE_VECTORS_PATH = INDEX_DIR / "dense_int8.npy"
DENSE_SCALES_PATH = INDEX_DIR / "dense_scales.npy"
LEXICAL_INDEX_PATH = INDEX_DIR / "lexical_bm25.npz"
INDEX_MANIFEST_PATH = INDEX_DIR / "manifest.json"


ICD10_RELEASE_YEAR = 2026

CMS_DOWNLOADS: dict[str, str] = {
    "tabular_order.zip": "https://www.cms.gov/files/zip/2026-code-descriptions-tabular-order.zip",
    "tables_index.zip": "https://www.cms.gov/files/zip/2026-code-tables-tabular-and-index.zip",
}

ORDER_FILE_MEMBER = "icd10cm_order_2026.txt"
TABULAR_XML_MEMBER = "icd10cm_tabular_2026.xml"
INDEX_XML_MEMBER = "icd10cm_index_2026.xml"


EMBEDDING_MODEL = os.environ.get("ICD10_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 256

QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def query_prefix(model_name: str = EMBEDDING_MODEL) -> str:
    """Return the instruction prefix a model expects on the query side."""
    return QUERY_PREFIXES.get(model_name, "")


DEFAULT_TOP_K = 10
MAX_TOP_K = 50
CANDIDATE_POOL = 100
RRF_K = int(os.environ.get("ICD10_RRF_K", "10"))

RRF_WEIGHT_DENSE = float(os.environ.get("ICD10_RRF_WEIGHT_DENSE", "1.0"))
RRF_WEIGHT_LEXICAL = float(os.environ.get("ICD10_RRF_WEIGHT_LEXICAL", "0.75"))

CODE_LOOKUP_LIMIT = 8

BM25_K1 = 1.5
BM25_B = 0.75

DOC_TEXT_VARIANT = os.environ.get("ICD10_DOC_TEXT_VARIANT", "desc_incl_parent_block")


MAX_QUERY_CHARS = 400
RATE_LIMIT_REQUESTS = int(os.environ.get("ICD10_RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("ICD10_RATE_LIMIT_WINDOW", "60"))

DISCLAIMER = (
    "Demonstration project only. Results are retrieval suggestions from the public "
    "CMS ICD-10-CM 2026 code set and must not be used for real clinical coding, "
    "billing, or medical decisions."
)
