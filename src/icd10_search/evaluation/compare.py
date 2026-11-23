"""Compare indexes built from different document-text variants.

The question this answers: does attaching the official synonyms and the parent
description to each code actually help retrieval, or is it just noise that
dilutes the embedding? Each variant needs its own full index, so this script
takes the index directories as arguments rather than building them itself.

Only the dense arm is compared, because the variants only change what is
embedded; the BM25 side is built from a separate text builder that does not
vary.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from icd10_search.config import (
    GOLD_INDEX_DEV_PATH,
    GOLD_INDEX_TEST_PATH,
    GOLD_LAY_PATH,
)
from icd10_search.data.goldset import load_gold_set
from icd10_search.evaluation.metrics import summarise
from icd10_search.evaluation.run import run_mode
from icd10_search.search.service import SearchService


def compare(index_dirs: list[Path], *, limit: int | None = None, split: str = "dev") -> dict:
    """Compare variants. Defaults to the dev split, because this comparison is a
    *selection*: whichever variant wins here becomes the shipped index, so
    running it on the test split would make the reported numbers self-selected.
    """
    if split == "dev":
        gold_sets = {"cms_index_dev": load_gold_set(GOLD_INDEX_DEV_PATH)}
    else:
        gold_sets = {
            "cms_index_test": load_gold_set(GOLD_INDEX_TEST_PATH),
            "hand_written_lay": load_gold_set(GOLD_LAY_PATH),
        }
    if limit:
        gold_sets = {name: queries[:limit] for name, queries in gold_sets.items()}

    rows: list[dict] = []
    for index_dir in index_dirs:
        service = SearchService.load(index_dir)
        variant = service.manifest.get("doc_text_variant", index_dir.name)
        for set_name, queries in gold_sets.items():
            outcomes = run_mode(service, queries, "dense")
            metrics = summarise(variant, outcomes).to_dict()
            rows.append(
                {
                    "variant": variant,
                    "index_dir": str(index_dir),
                    "gold_set": set_name,
                    "num_queries": len(queries),
                    "recall_at_1": metrics["recall_at"]["1"],
                    "recall_at_5": metrics["recall_at"]["5"],
                    "recall_at_10": metrics["recall_at"]["10"],
                    "mrr_at_10": metrics["mrr_at_10"],
                }
            )
            print(
                f"[compare] {variant:26} {set_name:18} "
                f"r@1={metrics['recall_at']['1']:.3f} r@10={metrics['recall_at']['10']:.3f} "
                f"mrr={metrics['mrr_at_10']:.3f}"
            )
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm": "dense",
        "split": split,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare document-text variants")
    parser.add_argument("index_dirs", type=Path, nargs="+")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--out", type=Path, default=Path("docs/variant_comparison.json"))
    args = parser.parse_args()

    report = compare(args.index_dirs, limit=args.limit, split=args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[compare] report written to {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
