import json
from pathlib import Path

import pytest

from eval.models import QueryResult
from eval.run import main, run_benchmark, run_eval

ROOT = Path(__file__).resolve().parents[2]


class AbstainingAdapter:
    name = "injected-adapter"

    def __init__(self) -> None:
        self.depths: list[int] = []

    def query(self, question: str, *, k: int) -> QueryResult:
        del question
        self.depths.append(k)
        return QueryResult(answer="", retrieval_hits=(), citations=(), abstained=True)


def test_run_benchmark_accepts_injected_adapter_and_requests_depth_five() -> None:
    adapter = AbstainingAdapter()

    report = run_benchmark(ROOT / "data/eval.jsonl", adapter)

    assert report.adapter == "injected-adapter"
    assert adapter.depths == [5] * 12


def test_offline_runner_ignores_external_store_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "pgvector")
    monkeypatch.setenv("RAG_POSTGRES_URL", "postgresql://must-not-be-used")
    monkeypatch.setenv("RAG_QDRANT_URL", "https://must-not-be-used.invalid")

    report = run_eval(ROOT / "data/eval.jsonl", corpus_path=ROOT / "data/sample_docs")

    assert report.adapter == "offline-lexical-baseline"
    assert report.summary.samples == 12
    assert report.dataset_sha256


def test_cli_writes_markdown_and_json_reports(tmp_path: Path) -> None:
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    main(
        [
            "--dataset",
            str(ROOT / "data/eval.jsonl"),
            "--corpus",
            str(ROOT / "data/sample_docs"),
            "--out",
            str(markdown_path),
            "--json-out",
            str(json_path),
        ]
    )

    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# Deterministic RAG Evaluation Report"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["adapter"] == "application-hybrid"
    assert payload["summary"]["samples"] == 12


@pytest.mark.parametrize("depth", [3, 6])
def test_run_eval_rejects_nonstandard_depth(depth: int) -> None:
    with pytest.raises(ValueError, match="exactly 5"):
        run_eval(ROOT / "data/eval.jsonl", k=depth, adapter=AbstainingAdapter())
