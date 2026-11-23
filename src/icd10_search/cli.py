"""Command line entry point: ``icd10 <command>``.

Wraps the same functions the Makefile and the tests call, so there is one
implementation of each step rather than one per entry point.
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_data(args: argparse.Namespace) -> int:
    from icd10_search.data.pipeline import run

    print(json.dumps(run(force_download=args.force), indent=2))
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    from icd10_search.indexing.build import build

    build(doc_text_variant=args.variant, keep_float32=args.keep_float32)
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from icd10_search.search.service import SearchService

    service = SearchService.load()
    response = service.search(args.query, k=args.k, mode=args.mode)
    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
        return 0
    print(f"\n{response.query!r}  [{response.mode}]  {response.timings_ms['total']} ms\n")
    for result in response.results:
        flag = "billable" if result.billable else "not billable"
        print(f"{result.rank:>2}. {result.code_dotted:<9} {result.description}")
        print(f"    {flag} | ch {result.chapter} {result.chapter_title} | {result.match_reason}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("icd10_search.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from icd10_search.evaluation.run import main as eval_main

    sys.argv = ["eval"] + (["--limit", str(args.limit)] if args.limit else [])
    eval_main()
    return 0


def _cmd_tune(args: argparse.Namespace) -> int:
    from icd10_search.evaluation.tune import main as tune_main

    sys.argv = ["tune"] + (["--limit", str(args.limit)] if args.limit else [])
    tune_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="icd10", description="ICD-10-CM semantic code finder")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="download the CMS release and build the corpus")
    data.add_argument("--force", action="store_true", help="re-download even if cached")
    data.set_defaults(func=_cmd_data)

    index = sub.add_parser("index", help="build the dense and lexical indexes")
    index.add_argument("--variant", default=None)
    index.add_argument("--keep-float32", action="store_true")
    index.set_defaults(func=_cmd_index)

    search = sub.add_parser("search", help="run one query from the terminal")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=10)
    search.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "lexical"])
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=_cmd_search)

    serve = sub.add_parser("serve", help="run the web application")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_cmd_serve)

    evaluate = sub.add_parser("eval", help="measure retrieval quality on the held-out sets")
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.set_defaults(func=_cmd_eval)

    tune = sub.add_parser("tune", help="sweep the fusion parameters on the dev split")
    tune.add_argument("--limit", type=int, default=None)
    tune.set_defaults(func=_cmd_tune)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "variant", "unset") is None:
        from icd10_search.config import DOC_TEXT_VARIANT

        args.variant = DOC_TEXT_VARIANT
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
