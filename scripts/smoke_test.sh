#!/usr/bin/env bash
set -euo pipefail

echo "[smoke] ruff"
ruff check app eval tests

echo "[smoke] mypy"
mypy app eval

echo "[smoke] pytest"
pytest -q

echo "[smoke] eval-ci"
python -m eval.run --dataset data/eval.jsonl --out reports/latest.md --k 5 \
  --min-hit-rate 0.50 --min-rubric-score 0.40 --max-p95-ms 250.0

echo "[smoke] ✅ all checks passed"
