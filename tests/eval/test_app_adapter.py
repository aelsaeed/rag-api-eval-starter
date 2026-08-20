from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.answering import answer_from_retrieval
from app.services.retrieval import retrieve
from eval.app_adapter import ApplicationQueryAdapter
from eval.run import run_benchmark

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/sample_docs"
DATASET = ROOT / "data/eval.jsonl"


@pytest.mark.parametrize("strategy", ["dense", "lexical", "hybrid"])
def test_adapter_runs_real_retrieval_and_answer_pipeline(strategy: str) -> None:
    with ApplicationQueryAdapter(CORPUS, strategy=strategy) as adapter:
        with (
            patch("eval.app_adapter.retrieve", wraps=retrieve) as retrieve_spy,
            patch(
                "eval.app_adapter.answer_from_retrieval",
                wraps=answer_from_retrieval,
            ) as answer_spy,
        ):
            result = adapter.query("Which components does Docker Compose bring up?", k=5)

        assert result.abstained is False
        assert result.retrieval_hits
        assert result.citations
        assert result.retrieval_hits[0].source == "operations.md"
        assert result.citations[0].source == "operations.md"
        assert "Docker Compose" in result.answer
        assert adapter.name == f"application-{strategy}"
        assert retrieve_spy.call_args.kwargs["strategy"] == strategy
        assert retrieve_spy.call_args.kwargs["settings"] is adapter.settings
        assert answer_spy.call_args.kwargs["settings"] is adapter.settings

    assert adapter.closed is True


def test_adapter_ignores_hostile_backend_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "pgvector")
    monkeypatch.setenv("RAG_POSTGRES_URL", "postgresql://must-not-be-used")
    monkeypatch.setenv("RAG_QDRANT_URL", "https://must-not-be-used.invalid")
    monkeypatch.setenv("RAG_QDRANT_API_KEY", "must-not-be-used")
    monkeypatch.setenv("RAG_FAKE_EMBEDDINGS", "not-a-boolean")
    monkeypatch.setenv("RAG_RRF_LEXICAL_WEIGHT", "0.01")
    monkeypatch.setenv("RAG_MAX_FILENAME_CHARS", "32")
    monkeypatch.setenv("RAG_MAX_DOCUMENT_CHUNKS", "1")

    with ApplicationQueryAdapter(CORPUS, strategy="hybrid") as adapter:
        assert adapter.settings.vector_backend == "qdrant"
        assert adapter.settings.qdrant_url is None
        assert adapter.settings.qdrant_api_key is None
        assert adapter.settings.postgres_url is None
        assert adapter.settings.fake_embeddings is True
        assert adapter.settings.rrf_lexical_weight == 0.65
        assert adapter.settings.max_filename_chars == 255
        assert adapter.settings.max_document_chunks == 2_000

        result = adapter.query("Which components does Docker Compose bring up?", k=3)

    assert result.abstained is False
    assert result.retrieval_hits[0].source == "operations.md"
    assert result.citations[0].source == "operations.md"


def test_adapter_converts_application_abstention_to_evaluator_contract() -> None:
    with ApplicationQueryAdapter(CORPUS, strategy="lexical") as adapter:
        result = adapter.query("Which post-quantum encryption algorithm is configured?", k=5)

    assert result.answer == ""
    assert result.retrieval_hits
    assert result.citations == ()
    assert result.abstained is True


def test_adapter_can_evaluate_golden_cases_with_real_application_code() -> None:
    with ApplicationQueryAdapter(CORPUS, strategy="hybrid") as adapter:
        report = run_benchmark(DATASET, adapter)

    local_stack = next(case for case in report.cases if case.case_id == "local-stack-001")
    assert report.adapter == "application-hybrid"
    assert report.summary.samples == 12
    assert local_stack.context_recall_at[1] == 1.0
    assert local_stack.answer_fact_coverage == 1.0


def test_close_is_idempotent_and_prevents_further_queries() -> None:
    adapter = ApplicationQueryAdapter(CORPUS)

    with patch.object(adapter._store, "close", wraps=adapter._store.close) as close_spy:
        adapter.close()
        adapter.close()

    close_spy.assert_called_once_with()
    assert adapter.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        adapter.query("question", k=1)
