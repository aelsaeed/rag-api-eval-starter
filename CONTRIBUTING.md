# Contributing

Thanks for your interest in improving this project.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
make setup
pre-commit install
```

## Development workflow

1. Create a feature branch.
2. Make focused changes with tests.
3. Run checks locally:
   ```bash
   make lint typecheck test smoke
   ```
4. Update docs/changelog if behavior changes.
5. Open a PR using the provided template.

## Code quality expectations

- Keep PRs small and reviewable.
- Add or update tests for functional changes.
- Preserve API compatibility unless explicitly planned.
- Prefer explicit typing and clear docstrings for non-trivial logic.

## Commit style

Use clear, imperative commit messages, for example:

- `Add smoke test script for CI parity`
- `Document offline demo workflow in README`
