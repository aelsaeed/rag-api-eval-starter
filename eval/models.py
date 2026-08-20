from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GoldContext:
    """A source and text anchor that identify one relevant context."""

    source: str
    anchor: str
    relevance: int = 1


@dataclass(frozen=True)
class EvalCase:
    """One versioned, human-reviewed golden evaluation example."""

    id: str
    question: str
    answerable: bool
    gold_contexts: tuple[GoldContext, ...]
    required_fact_groups: tuple[tuple[str, ...], ...]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalHit:
    """A ranked context returned by a query adapter."""

    source: str
    chunk_id: str
    text: str
    score: float = 0.0


@dataclass(frozen=True)
class QueryResult:
    """Evaluator contract separating ranked retrieval from cited answer evidence."""

    answer: str
    retrieval_hits: tuple[RetrievalHit, ...]
    citations: tuple[RetrievalHit, ...]
    abstained: bool = False


class QueryAdapter(Protocol):
    """Minimal seam for evaluating an API, service, or deterministic fixture."""

    @property
    def name(self) -> str: ...

    def query(self, question: str, *, k: int) -> QueryResult: ...


@dataclass(frozen=True)
class CaseMetrics:
    case_id: str
    question: str
    answerable: bool
    tags: tuple[str, ...]
    context_recall_at: dict[int, float]
    reciprocal_rank_at_5: float
    ndcg_at_5: float
    answer_fact_coverage: float | None
    citation_precision: float | None
    abstention_correct: bool
    retrieved_sources: tuple[str, ...]
    missing_fact_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SummaryMetrics:
    samples: int
    answerable_samples: int
    unanswerable_samples: int
    context_recall_at: dict[int, float]
    mrr_at_5: float
    ndcg_at_5: float
    answer_fact_coverage: float
    citation_precision: float
    abstention_accuracy: float


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: int
    adapter: str
    dataset_sha256: str
    summary: SummaryMetrics
    cases: tuple[CaseMetrics, ...]


@dataclass(frozen=True)
class Thresholds:
    min_context_recall_at_1: float = 0.7
    min_context_recall_at_3: float = 0.9
    min_context_recall_at_5: float = 0.95
    min_mrr_at_5: float = 0.75
    min_ndcg_at_5: float = 0.8
    min_answer_fact_coverage: float = 0.8
    min_citation_precision: float = 0.5
    min_abstention_accuracy: float = 1.0


def as_tuple(values: Sequence[str]) -> tuple[str, ...]:
    """Keep tuple conversion in one place for adapters built outside this package."""

    return tuple(values)
