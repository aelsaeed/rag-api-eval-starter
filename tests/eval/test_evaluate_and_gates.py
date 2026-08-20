import math

import pytest

from eval.metrics import QualityGateError, enforce_thresholds, evaluate
from eval.models import EvalCase, GoldContext, QueryResult, RetrievalHit, Thresholds


class RecordingAdapter:
    name = "recording-fixture"

    def __init__(self, results: dict[str, QueryResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def query(self, question: str, *, k: int) -> QueryResult:
        self.calls.append((question, k))
        return self.results[question]


def _cases() -> tuple[EvalCase, EvalCase]:
    answerable = EvalCase(
        id="answerable-001",
        question="answerable",
        answerable=True,
        gold_contexts=(GoldContext("gold.md", "gold context", 2),),
        required_fact_groups=(("required fact",),),
    )
    negative = EvalCase(
        id="negative-001",
        question="negative",
        answerable=False,
        gold_contexts=(),
        required_fact_groups=(),
    )
    return answerable, negative


def test_evaluate_uses_k_five_and_aggregates_answerable_and_negative_cases() -> None:
    adapter = RecordingAdapter(
        {
            "answerable": QueryResult(
                answer="required fact",
                retrieval_hits=(RetrievalHit("gold.md", "gold-1", "gold context", 1.0),),
                citations=(RetrievalHit("gold.md", "gold-1", "gold context", 1.0),),
            ),
            "negative": QueryResult(answer="", retrieval_hits=(), citations=(), abstained=True),
        }
    )

    report = evaluate(_cases(), adapter, dataset_hash="abc123")

    assert adapter.calls == [("answerable", 5), ("negative", 5)]
    assert report.adapter == "recording-fixture"
    assert report.summary.context_recall_at == {1: 1.0, 3: 1.0, 5: 1.0}
    assert report.summary.mrr_at_5 == 1.0
    assert report.summary.ndcg_at_5 == 1.0
    assert report.summary.answer_fact_coverage == 1.0
    assert report.summary.citation_precision == 1.0
    assert report.summary.abstention_accuracy == 1.0


def test_quality_gate_reports_every_regression() -> None:
    adapter = RecordingAdapter(
        {
            "answerable": QueryResult(
                answer="missing",
                retrieval_hits=(RetrievalHit("wrong.md", "wrong-1", "distractor", 1.0),),
                citations=(RetrievalHit("wrong.md", "wrong-1", "distractor", 1.0),),
            ),
            "negative": QueryResult(
                answer="unsupported answer",
                retrieval_hits=(RetrievalHit("wrong.md", "wrong-2", "distractor", 1.0),),
                citations=(RetrievalHit("wrong.md", "wrong-2", "distractor", 1.0),),
            ),
        }
    )
    report = evaluate(_cases(), adapter, dataset_hash="abc123")
    thresholds = Thresholds(
        min_context_recall_at_1=0.5,
        min_context_recall_at_3=0.5,
        min_context_recall_at_5=0.5,
        min_mrr_at_5=0.5,
        min_ndcg_at_5=0.5,
        min_answer_fact_coverage=0.5,
        min_citation_precision=0.5,
        min_abstention_accuracy=1.0,
    )

    with pytest.raises(QualityGateError) as exc_info:
        enforce_thresholds(report, thresholds)

    assert len(exc_info.value.failures) == 8
    assert any("context_recall_at_1" in failure for failure in exc_info.value.failures)
    assert any("abstention_accuracy" in failure for failure in exc_info.value.failures)


def test_quality_gate_rejects_nan_threshold() -> None:
    adapter = RecordingAdapter(
        {
            "answerable": QueryResult(
                answer="required fact",
                retrieval_hits=(RetrievalHit("gold.md", "gold", "gold context", 1.0),),
                citations=(RetrievalHit("gold.md", "gold", "gold context", 1.0),),
            ),
            "negative": QueryResult(answer="", retrieval_hits=(), citations=(), abstained=True),
        }
    )
    report = evaluate(_cases(), adapter, dataset_hash="abc123")

    with pytest.raises(ValueError, match="finite"):
        enforce_thresholds(report, Thresholds(min_mrr_at_5=math.nan))
