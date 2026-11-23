
PY := .venv/bin/python
export PYTHONPATH := src

.PHONY: help venv data index serve dev test lint eval tune clean docker-build docker-run

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

venv: ## create the virtualenv and install the project
	uv venv --python 3.12
	uv pip install -e ".[dev]"

data: ## download the CMS release and build the corpus + gold sets
	$(PY) -m icd10_search.data.pipeline

index: ## build the dense (int8) and BM25 indexes
	$(PY) -m icd10_search.indexing.build

serve: ## run the web app on http://127.0.0.1:8000
	$(PY) -m uvicorn icd10_search.api.main:app --host 127.0.0.1 --port 8000

dev: ## run the web app with autoreload
	$(PY) -m uvicorn icd10_search.api.main:app --reload --port 8000

test: ## run the test suite
	$(PY) -m pytest

lint: ## check formatting and lint rules
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

eval: ## measure retrieval quality and write the numbers into the docs
	$(PY) -m icd10_search.evaluation.run
	$(PY) scripts/render_results.py

tune: ## sweep fusion parameters on the dev split
	$(PY) -m icd10_search.evaluation.tune

docker-build: ## build the container image
	docker build -t icd10-hybrid-search .

docker-run: ## run the container on http://127.0.0.1:8000
	docker run --rm -p 8000:8000 icd10-hybrid-search

clean: ## remove caches (keeps data and indexes)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
