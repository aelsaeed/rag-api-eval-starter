.PHONY: run test lint eval

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check app eval tests
	ruff format --check app eval tests
	mypy app eval

eval:
	python -m eval.run --dataset data/eval.jsonl --out reports/report.md
