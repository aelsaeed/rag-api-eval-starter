.PHONY: setup run lint typecheck test fmt demo smoke eval eval-ci clean

PYTHON ?= python
PIP ?= pip

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e ".[dev]"

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

lint:
	ruff check app eval tests

fmt:
	ruff format app eval tests
	ruff check --fix app eval tests

typecheck:
	mypy app eval

test:
	pytest

smoke:
	bash scripts/smoke_test.sh

demo:
	bash scripts/demo.sh

eval:
	$(PYTHON) -m eval.run --dataset data/eval.jsonl --out reports/latest.md --k 5 \
		--min-hit-rate 0.50 --min-rubric-score 0.40 --max-p95-ms 250.0

eval-ci:
	$(PYTHON) -m eval.run --dataset data/eval.jsonl --out reports/latest.md --k 5 \
		--min-hit-rate 0.50 --min-rubric-score 0.40 --max-p95-ms 250.0

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache reports
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
