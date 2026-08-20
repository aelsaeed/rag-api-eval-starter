.PHONY: setup setup-ml run lint typecheck test fmt demo smoke eval eval-ablation eval-ci build clean

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
HOST ?= 127.0.0.1
PORT ?= 8000

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e ".[dev]"

setup-ml:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e ".[dev,ml]"

run:
	uvicorn app.main:app --reload --host $(HOST) --port $(PORT)

lint:
	ruff check app eval tests
	ruff format --check app eval tests

fmt:
	ruff format app eval tests
	ruff check --fix app eval tests

typecheck:
	mypy app eval

test:
	pytest --cov=app --cov=eval --cov-report=term-missing --cov-report=xml

smoke:
	bash scripts/smoke_test.sh

demo:
	bash scripts/demo.sh

eval:
	$(PYTHON) -m eval.run --adapter application --strategy hybrid \
		--dataset data/eval.jsonl --corpus data/sample_docs \
		--out reports/latest.md --json-out reports/latest.json --k 5

eval-ablation:
	$(PYTHON) -m eval.compare --dataset data/eval.jsonl --corpus data/sample_docs \
		--out reports/ablation.md --json-out reports/ablation.json

eval-ci:
	$(MAKE) eval
	$(MAKE) eval-ablation

build:
	$(PYTHON) -m build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache reports
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
