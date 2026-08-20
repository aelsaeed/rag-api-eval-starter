# Contributing

Thanks for helping improve the retrieval service or the evidence used to evaluate it. A change is
complete when its behavior, tests, benchmark impact, and documentation agree.

## Local setup

Python 3.11 or newer is required. The base development install uses deterministic hashed
embeddings and does not download a model.

```bash
python -m venv .venv
source .venv/bin/activate
make setup
pre-commit install
```

Use `make setup-ml` instead when work requires the optional sentence-transformer dependency. The
default tests and CI intentionally do not require it.

## Useful commands

| Command | Purpose |
|---|---|
| `make run` | Start the API on port 8000 |
| `make demo` | Run the deterministic end-to-end demo |
| `make lint` | Check Ruff lint and formatting |
| `make typecheck` | Run mypy over `app/` and `eval/` |
| `make test` | Run tests with branch coverage |
| `make eval` | Evaluate the real hybrid application pipeline |
| `make eval-ablation` | Compare dense, lexical, and hybrid retrieval |
| `make eval-ci` | Run both application evaluation commands |
| `make smoke` | Run the full local quality sequence |
| `make build` | Build source and wheel distributions |

## Development workflow

1. Create a focused branch, following the repository's existing naming convention.
2. Add or update tests alongside functional changes.
3. Run fast checks while iterating:

   ```bash
   make lint typecheck test
   ```

4. Before opening a pull request, run the application quality gates:

   ```bash
   make eval-ci
   ```

5. Run `make smoke` for the complete local quality suite. CI additionally installs the built wheel
   and runs a real Compose ingest/query check. Update the README, detailed docs, and changelog when
   public behavior changes.
6. Open a pull request using the repository template. Describe the user-visible change and include
   the relevant test and evaluation results.

## Changing the golden dataset

`data/eval.jsonl` is a versioned, human-reviewed fixture. Each JSONL record has:

- a unique stable `id` and non-empty `question`;
- an `answerable` flag;
- exact `gold_contexts` anchors with source and graded relevance for answerable cases;
- `required_fact_groups`, where each inner list contains accepted aliases for one fact; and
- tags that explain the scenario, such as `paraphrase`, `exact-term`, or `hard-negative`.

Answerable cases require at least one context and fact group. Unanswerable cases must have neither.
When changing the corpus or dataset:

1. Keep anchors exact and confirm they occur in the named source.
2. Prefer examples that represent a real retrieval risk, not wording tailored to the current
   implementation.
3. Add metric unit tests if scoring semantics change.
4. Run `make eval-ci` and explain every material strategy delta in the pull request.

The generated `reports/` directory is intentionally ignored. CI publishes Markdown/JSON reports
and `coverage.xml` as per-commit artifacts; do not commit local report output unless a maintainer
explicitly requests a frozen baseline.

## Code quality expectations

- Keep PRs small and reviewable.
- Add or update tests for functional changes.
- Preserve the public API contract unless the change is explicitly planned and documented.
- Prefer explicit typing and clear docstrings for non-trivial logic.
- Pass settings explicitly in tests so ambient `RAG_*` variables cannot change outcomes.
- Keep the deterministic evaluation profile offline and reproducible.
- Treat a failed quality gate as a behavior change to investigate, not a threshold to weaken without
  evidence.

## Pull request evidence

Include:

- the problem and intended behavior;
- commands run and their outcomes;
- test coverage for new failure modes;
- dense/lexical/hybrid metric deltas when retrieval, chunking, embeddings, scoring, abstention, or
  the corpus changes; and
- operational or security tradeoffs introduced by the change.

## Commit style

Use clear, imperative commit messages, for example:

- `Add hard-negative abstention cases`
- `Calibrate hybrid retrieval confidence`
- `Document pgvector readiness checks`
