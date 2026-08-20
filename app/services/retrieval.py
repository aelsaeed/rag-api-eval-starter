import math
from dataclasses import dataclass
from typing import Any, Literal, cast

from app.core.config import Settings, get_settings
from app.services.embeddings import embed_query
from app.services.storage import BaseStore
from app.services.text import lexical_overlap_score, tokenize

RetrievalStrategy = Literal["dense", "lexical", "hybrid"]
SUPPORTED_STRATEGIES = {"dense", "lexical", "hybrid"}


@dataclass(frozen=True)
class RetrievalOutcome:
    hits: list[dict[str, Any]]
    strategy: RetrievalStrategy
    dense_candidates: int
    lexical_candidates: int

    @property
    def confidence(self) -> float:
        finite_scores = (
            score
            for hit in self.hits
            if math.isfinite(score := float(hit.get("evidence_score", 0.0)))
        )
        return max(finite_scores, default=0.0)


def _dense_confidence(score: float) -> float:
    """Treat non-positive cosine similarity as no supporting evidence."""

    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


def _dense_rrf_signal(score: float) -> float:
    """Map cosine similarity from [-1, 1] into a bounded ranking signal."""

    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _result_key(payload: dict[str, Any]) -> str:
    chunk_id = payload.get("chunk_id")
    if not chunk_id:
        raise ValueError("Stored retrieval payload is missing chunk_id")
    return str(chunk_id)


def retrieve(
    query: str,
    store: BaseStore,
    *,
    strategy: RetrievalStrategy | None = None,
    top_k: int | None = None,
    settings: Settings | None = None,
) -> RetrievalOutcome:
    """Retrieve independently ranked dense/lexical candidates and optionally fuse them.

    Hybrid mode uses reciprocal-rank fusion (RRF), which avoids treating cosine and
    lexical-overlap scores as though they share a calibrated numeric scale.
    """

    settings = settings or get_settings()
    selected_strategy = strategy or settings.retrieval_strategy
    if selected_strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported retrieval strategy: {selected_strategy}")
    selected_strategy = cast(RetrievalStrategy, selected_strategy)

    limit = top_k if top_k is not None else settings.top_k
    if limit < 1 or limit > 50:
        raise ValueError("top_k must be between 1 and 50")
    candidate_limit = min(200, max(limit, limit * settings.candidate_multiplier))
    query_tokens = tokenize(query)
    lexical_weight = settings.rrf_lexical_weight
    dense_weight = 1.0 - lexical_weight

    dense_hits: list[dict[str, Any]] = []
    lexical_hits: list[dict[str, Any]] = []
    if selected_strategy == "dense" or (selected_strategy == "hybrid" and dense_weight > 0):
        dense_hits = store.dense_search(
            embed_query(query, settings=settings), limit=candidate_limit
        )
    if selected_strategy == "lexical" or (selected_strategy == "hybrid" and lexical_weight > 0):
        lexical_hits = store.keyword_search(query_tokens, limit=candidate_limit)

    candidates: dict[str, dict[str, Any]] = {}
    dense_ranks: dict[str, int] = {}
    lexical_ranks: dict[str, int] = {}
    dense_scores: dict[str, float] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        payload = hit.get("payload", {})
        key = _result_key(payload)
        candidates[key] = payload
        dense_ranks[key] = rank
        raw_dense_score = float(hit.get("score", 0.0))
        dense_scores[key] = raw_dense_score if math.isfinite(raw_dense_score) else 0.0

    for rank, hit in enumerate(lexical_hits, start=1):
        payload = hit.get("payload", {})
        key = _result_key(payload)
        candidates[key] = payload
        lexical_ranks[key] = rank

    results: list[dict[str, Any]] = []
    max_rrf_score = 1.0 / (settings.rrf_k + 1)
    for key, payload in candidates.items():
        dense_rank = dense_ranks.get(key)
        lexical_rank = lexical_ranks.get(key)
        dense_score = dense_scores.get(key)
        keyword_score = lexical_overlap_score(payload.get("tokens", []), query_tokens)
        dense_confidence = _dense_confidence(dense_score or 0.0)

        if selected_strategy == "hybrid":
            raw_rrf = 0.0
            if dense_rank is not None:
                raw_rrf += (
                    dense_weight
                    * _dense_rrf_signal(dense_score or 0.0)
                    / (settings.rrf_k + dense_rank)
                )
            if lexical_rank is not None:
                raw_rrf += lexical_weight * keyword_score / (settings.rrf_k + lexical_rank)
            score = raw_rrf / max_rrf_score
            evidence_signals = [
                signal
                for signal, is_present in (
                    (dense_confidence, dense_rank is not None),
                    (keyword_score, lexical_rank is not None),
                )
                if is_present
            ]
            evidence_score = max(evidence_signals, default=0.0)
        elif selected_strategy == "dense":
            score = evidence_score = dense_confidence
        else:
            score = evidence_score = keyword_score

        results.append(
            {
                "doc_id": payload.get("doc_id"),
                "chunk_id": payload.get("chunk_id"),
                "snippet": payload.get("text"),
                "score": max(0.0, min(1.0, score)),
                "evidence_score": max(0.0, min(1.0, evidence_score)),
                "dense_score": dense_score,
                "keyword_score": keyword_score,
                "dense_rank": dense_rank,
                "keyword_rank": lexical_rank,
                "source": payload.get("source"),
            }
        )

    results.sort(
        key=lambda item: (
            -item["score"],
            str(item.get("source") or ""),
            str(item.get("chunk_id") or ""),
        )
    )
    return RetrievalOutcome(
        hits=results[:limit],
        strategy=selected_strategy,
        dense_candidates=len(dense_hits),
        lexical_candidates=len(lexical_hits),
    )


def hybrid_search(
    query: str,
    store: BaseStore,
    *,
    top_k: int | None = None,
    settings: Settings | None = None,
) -> list[dict]:
    """Compatibility wrapper for callers that only need hybrid hits."""

    return retrieve(
        query,
        store,
        strategy="hybrid",
        top_k=top_k,
        settings=settings,
    ).hits
