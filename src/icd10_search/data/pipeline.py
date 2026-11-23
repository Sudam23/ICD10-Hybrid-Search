"""One command that takes the project from nothing to a parsed corpus.

Run order: download -> parse -> gold sets. Everything it writes is
deterministic, so two runs on the same CMS release produce identical files.
"""

from __future__ import annotations

import json

from icd10_search.config import (
    CORPUS_PATH,
    GOLD_INDEX_DEV_PATH,
    GOLD_INDEX_TEST_PATH,
    GOLD_LAY_PATH,
    INDEX_XML_MEMBER,
    ORDER_FILE_MEMBER,
    PROCESSED_DIR,
    TABULAR_XML_MEMBER,
)
from icd10_search.data.download import ensure_raw_data
from icd10_search.data.goldset import (
    build_index_gold_set,
    build_lay_gold_set,
    split_gold_set,
    write_gold_set,
)
from icd10_search.data.parse import build_corpus, corpus_stats, write_corpus


def run(
    *,
    force_download: bool = False,
    gold_dev_size: int = 1000,
    gold_test_size: int = 1000,
) -> dict[str, object]:
    raw = ensure_raw_data(force=force_download)

    print("[pipeline] parsing order file + tabular XML ...")
    records = build_corpus(raw[ORDER_FILE_MEMBER], raw[TABULAR_XML_MEMBER])
    write_corpus(records, CORPUS_PATH)
    stats = corpus_stats(records)
    print(f"[pipeline] corpus -> {CORPUS_PATH} ({stats['total_codes']:,} codes)")

    valid_codes = {record.code for record in records}

    print("[pipeline] building gold set from the CMS Alphabetic Index ...")
    pool = build_index_gold_set(raw[INDEX_XML_MEMBER], valid_codes, sample_size=0)
    dev, test = split_gold_set(pool, dev_size=gold_dev_size, test_size=gold_test_size)
    write_gold_set(dev, GOLD_INDEX_DEV_PATH)
    write_gold_set(test, GOLD_INDEX_TEST_PATH)
    print(
        f"[pipeline] gold pool: {len(pool):,} usable index entries -> "
        f"dev {len(dev):,} / test {len(test):,} (disjoint)"
    )

    lay_gold = build_lay_gold_set(valid_codes)
    write_gold_set(lay_gold, GOLD_LAY_PATH)
    print(f"[pipeline] gold (lay)   -> {GOLD_LAY_PATH} ({len(lay_gold):,} queries)")

    summary = {
        **stats,
        "gold_index_pool": len(pool),
        "gold_index_dev": len(dev),
        "gold_index_test": len(test),
        "gold_lay_queries": len(lay_gold),
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "corpus_stats.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2))
