#!/usr/bin/env bash
set -euo pipefail

echo "[smoke] lint + format"
ruff check app eval tests
ruff format --check app eval tests

echo "[smoke] mypy"
mypy app eval

echo "[smoke] pytest + coverage"
pytest --cov=app --cov=eval --cov-report=term-missing --cov-report=xml

echo "[smoke] application quality gate"
python -m eval.run --adapter application --strategy hybrid \
  --dataset data/eval.jsonl --corpus data/sample_docs \
  --out reports/latest.md --json-out reports/latest.json --k 5

echo "[smoke] strategy ablation"
python -m eval.compare --dataset data/eval.jsonl --corpus data/sample_docs \
  --out reports/ablation.md --json-out reports/ablation.json

echo "[smoke] ✅ all checks passed"
