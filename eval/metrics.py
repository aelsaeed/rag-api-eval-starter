from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from statistics import fmean

from eval.models import (
    CaseMetrics,
    EvalCase,
    EvaluationReport,
    GoldContext,
    QueryAdapter,
    QueryResult,
    RetrievalHit,
    SummaryMetrics,
    Thresholds,
)
from eval.text import normalize_text

DEFAULT_KS = (1, 3, 5)


class AdapterContractError(ValueError):
    """Raised when an injected adapter returns an invalid query result."""


class QualityGateError(RuntimeError):
    """Raised when one or more deterministic quality thresholds fail."""

    def __init__(self, failures: Sequence[str]) -> None:
        self.failures = tuple(failures)
        super().__init__("Quality gate failure: " + "; ".join(self.failures))


def _mean(values: Iterable[float]) -> float:
    values_tuple = tuple(values)
    return fmean(values_tuple) if values_tuple else 0.0


def _contains_normalized(text: str, fragment: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_fragment = normalize_text(fragment)
    if not normalized_fragment:
        return False
    return f" {normalized_fragment} " in f" {normalized_text} "


def _matching_context_indexes(hit: RetrievalHit, contexts: Sequence[GoldContext]) -> set[int]:
    return {
        index
        for index, context in enumerate(contexts)
        if hit.source == context.source and _contains_normalized(hit.text, context.anchor)
    }


def _dcg(grades: Sequence[int]) -> float:
    return float(
        sum(
            ((2**grade) - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(grades, start=1)
            if grade > 0
        )
    )


def _ndcg_at_5(case: EvalCase, retrieval_hits: Sequence[RetrievalHit]) -> float:
    seen_contexts: set[int] = set()
    actual_grades: list[int] = []
    for hit in retrieval_hits[:5]:
        matches = _matching_context_indexes(hit, case.gold_contexts) - seen_contexts
        if not matches:
            actual_grades.append(0)
            continue
        selected = max(matches, key=lambda index: (case.gold_contexts[index].relevance, -index))
        seen_contexts.add(selected)
        actual_grades.append(case.gold_contexts[selected].relevance)
    ideal_grades = sorted((context.relevance for context in case.gold_contexts), reverse=True)[:5]
    ideal = _dcg(ideal_grades)
    return _dcg(actual_grades) / ideal if ideal else 0.0


def _validate_result(case: EvalCase, result: QueryResult, max_k: int) -> None:
    if result.abstained and (result.answer.strip() or result.citations):
        raise AdapterContractError(
            f"case {case.id!r}: an abstained result must have no answer or citations"
        )
    if not result.abstained and not result.citations:
        raise AdapterContractError(
            f"case {case.id!r}: a non-abstained result must contain citations"
        )
    if len(result.retrieval_hits) > max_k:
        raise AdapterContractError(
            f"case {case.id!r}: adapter returned {len(result.retrieval_hits)} retrieval hits "
            f"for k={max_k}"
        )

    def validate_hits(hits: Sequence[RetrievalHit], label: str) -> set[str]:
        chunk_ids: set[str] = set()
        for hit in hits:
            if not hit.source.strip() or not hit.chunk_id.strip() or not hit.text.strip():
                raise AdapterContractError(
                    f"case {case.id!r}: {label} require source, chunk_id, and text"
                )
            if not math.isfinite(hit.score):
                raise AdapterContractError(f"case {case.id!r}: {label} scores must be finite")
            if hit.chunk_id in chunk_ids:
                raise AdapterContractError(
                    f"case {case.id!r}: duplicate {label[:-1]} chunk_id {hit.chunk_id!r}"
                )
            chunk_ids.add(hit.chunk_id)
        return chunk_ids

    retrieved_chunk_ids = validate_hits(result.retrieval_hits, "retrieval hits")
    citation_chunk_ids = validate_hits(result.citations, "citations")
    missing_citations = sorted(citation_chunk_ids - retrieved_chunk_ids)
    if missing_citations:
        raise AdapterContractError(
            f"case {case.id!r}: citations are not present in retrieval hits: {missing_citations}"
        )
    retrieved_by_id = {hit.chunk_id: hit for hit in result.retrieval_hits}
    mismatched_citations = [
        citation.chunk_id
        for citation in result.citations
        if (
            citation.source != retrieved_by_id[citation.chunk_id].source
            or citation.text != retrieved_by_id[citation.chunk_id].text
        )
    ]
    if mismatched_citations:
        raise AdapterContractError(
            f"case {case.id!r}: citations differ from retrieved evidence: {mismatched_citations}"
        )


def score_case(
    case: EvalCase,
    result: QueryResult,
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> CaseMetrics:
    ks_tuple = tuple(ks)
    if not ks_tuple or any(k <= 0 for k in ks_tuple) or tuple(sorted(set(ks_tuple))) != ks_tuple:
        raise ValueError("ks must be unique positive integers in ascending order")
    _validate_result(case, result, max(ks_tuple))

    abstention_correct = result.abstained == (not case.answerable)
    if not case.answerable:
        return CaseMetrics(
            case_id=case.id,
            question=case.question,
            answerable=False,
            tags=case.tags,
            context_recall_at={k: 0.0 for k in ks_tuple},
            reciprocal_rank_at_5=0.0,
            ndcg_at_5=0.0,
            answer_fact_coverage=None,
            citation_precision=None,
            abstention_correct=abstention_correct,
            retrieved_sources=tuple(hit.source for hit in result.retrieval_hits),
            missing_fact_groups=(),
        )

    matched_by_rank = [
        _matching_context_indexes(hit, case.gold_contexts) for hit in result.retrieval_hits
    ]
    recall_at: dict[int, float] = {}
    for k in ks_tuple:
        matched = set().union(*matched_by_rank[:k]) if matched_by_rank[:k] else set()
        recall_at[k] = len(matched) / len(case.gold_contexts)

    first_relevant_rank = next(
        (rank for rank, matched in enumerate(matched_by_rank[:5], start=1) if matched),
        None,
    )
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    missing_groups = tuple(
        group
        for group in case.required_fact_groups
        if not any(_contains_normalized(result.answer, alias) for alias in group)
    )
    fact_coverage = 1.0 - (len(missing_groups) / len(case.required_fact_groups))
    relevant_citations = sum(
        bool(_matching_context_indexes(citation, case.gold_contexts))
        for citation in result.citations
    )
    citation_precision = relevant_citations / len(result.citations) if result.citations else 0.0

    return CaseMetrics(
        case_id=case.id,
        question=case.question,
        answerable=True,
        tags=case.tags,
        context_recall_at=recall_at,
        reciprocal_rank_at_5=reciprocal_rank,
        ndcg_at_5=_ndcg_at_5(case, result.retrieval_hits),
        answer_fact_coverage=fact_coverage,
        citation_precision=citation_precision,
        abstention_correct=abstention_correct,
        retrieved_sources=tuple(hit.source for hit in result.retrieval_hits),
        missing_fact_groups=missing_groups,
    )


def evaluate(
    cases: Sequence[EvalCase],
    adapter: QueryAdapter,
    *,
    dataset_hash: str,
    ks: Sequence[int] = DEFAULT_KS,
) -> EvaluationReport:
    if not cases:
        raise ValueError("cases must not be empty")
    ks_tuple = tuple(ks)
    if ks_tuple != DEFAULT_KS:
        raise ValueError(f"this benchmark requires ks={DEFAULT_KS}")

    case_metrics = tuple(
        score_case(case, adapter.query(case.question, k=max(ks_tuple)), ks=ks_tuple)
        for case in cases
    )
    answerable = tuple(item for item in case_metrics if item.answerable)
    unanswerable_count = len(case_metrics) - len(answerable)
    summary = SummaryMetrics(
        samples=len(case_metrics),
        answerable_samples=len(answerable),
        unanswerable_samples=unanswerable_count,
        context_recall_at={
            k: _mean(item.context_recall_at[k] for item in answerable) for k in ks_tuple
        },
        mrr_at_5=_mean(item.reciprocal_rank_at_5 for item in answerable),
        ndcg_at_5=_mean(item.ndcg_at_5 for item in answerable),
        answer_fact_coverage=_mean(
            item.answer_fact_coverage
            for item in answerable
            if item.answer_fact_coverage is not None
        ),
        citation_precision=_mean(
            item.citation_precision for item in answerable if item.citation_precision is not None
        ),
        abstention_accuracy=_mean(float(item.abstention_correct) for item in case_metrics),
    )
    return EvaluationReport(
        schema_version=1,
        adapter=adapter.name,
        dataset_sha256=dataset_hash,
        summary=summary,
        cases=case_metrics,
    )


def enforce_thresholds(report: EvaluationReport, thresholds: Thresholds) -> None:
    configured = {
        "context_recall_at_1": thresholds.min_context_recall_at_1,
        "context_recall_at_3": thresholds.min_context_recall_at_3,
        "context_recall_at_5": thresholds.min_context_recall_at_5,
        "mrr_at_5": thresholds.min_mrr_at_5,
        "ndcg_at_5": thresholds.min_ndcg_at_5,
        "answer_fact_coverage": thresholds.min_answer_fact_coverage,
        "citation_precision": thresholds.min_citation_precision,
        "abstention_accuracy": thresholds.min_abstention_accuracy,
    }
    for name, threshold in configured.items():
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{name} threshold must be finite and between 0 and 1")

    summary = report.summary
    actual = {
        "context_recall_at_1": summary.context_recall_at[1],
        "context_recall_at_3": summary.context_recall_at[3],
        "context_recall_at_5": summary.context_recall_at[5],
        "mrr_at_5": summary.mrr_at_5,
        "ndcg_at_5": summary.ndcg_at_5,
        "answer_fact_coverage": summary.answer_fact_coverage,
        "citation_precision": summary.citation_precision,
        "abstention_accuracy": summary.abstention_accuracy,
    }
    failures = [
        f"{name} {actual[name]:.3f} < {threshold:.3f}"
        for name, threshold in configured.items()
        if actual[name] < threshold
    ]
    if failures:
        raise QualityGateError(failures)
