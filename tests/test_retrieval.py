from typing import Any

import pytest

from app.core.config import Settings
from app.services.answering import answer_from_retrieval
from app.services.retrieval import RetrievalOutcome, retrieve
from app.services.text import lexical_overlap_score, tokenize


def _hit(chunk_id: str, text: str, score: float, source: str = "doc.md") -> dict[str, Any]:
    return {
        "score": score,
        "payload": {
            "doc_id": source,
            "chunk_id": chunk_id,
            "text": text,
            "tokens": tokenize(text),
            "source": source,
        },
    }


class FakeStore:
    def __init__(self, dense: list[dict], lexical: list[dict]) -> None:
        self.dense = dense
        self.lexical = lexical

    def dense_search(self, vector: list[float], limit: int) -> list[dict]:
        return self.dense[:limit]

    def keyword_search(self, tokens: list[str], limit: int) -> list[dict]:
        return self.lexical[:limit]

    def ensure_collection(self) -> None:
        return None

    def delete_document(self, doc_id: str) -> None:
        return None

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        return None

    def is_ready(self) -> bool:
        return True

    def close(self) -> None:
        return None


def test_hybrid_recovers_candidate_missing_from_dense_pool(monkeypatch) -> None:
    monkeypatch.setattr("app.services.retrieval.embed_query", lambda _text, **_kwargs: [1.0, 0.0])
    dense = [_hit("dense-1", "general deployment guidance", 0.95)]
    lexical = [_hit("lexical-1", "rotate the emergency signing key", 1.0, "security.md")]

    outcome = retrieve(
        "emergency signing key",
        FakeStore(dense, lexical),
        strategy="hybrid",
        top_k=2,
    )

    assert {hit["chunk_id"] for hit in outcome.hits} == {"dense-1", "lexical-1"}
    lexical_hit = next(hit for hit in outcome.hits if hit["chunk_id"] == "lexical-1")
    assert lexical_hit["dense_rank"] is None
    assert lexical_hit["keyword_rank"] == 1


def test_hybrid_ties_have_stable_source_order(monkeypatch) -> None:
    monkeypatch.setattr("app.services.retrieval.embed_query", lambda _text, **_kwargs: [1.0, 0.0])
    alpha = _hit("alpha", "shared token", 0.8, "a.md")
    beta = _hit("beta", "shared token", 0.8, "b.md")
    store = FakeStore([beta, alpha], [alpha, beta])

    outcome = retrieve("shared token", store, strategy="hybrid", top_k=2)

    assert [hit["source"] for hit in outcome.hits] == ["a.md", "b.md"]


def test_weak_single_token_overlap_triggers_abstention(monkeypatch) -> None:
    monkeypatch.setattr("app.services.retrieval.embed_query", lambda _text, **_kwargs: [1.0, 0.0])
    weak = _hit("weak", "a configured confidence threshold", 0.2)
    store = FakeStore([weak], [weak])

    outcome = retrieve(
        "post quantum encryption algorithm configured",
        store,
        strategy="hybrid",
        top_k=1,
    )
    answer = answer_from_retrieval(outcome)

    assert outcome.confidence < 0.4
    assert answer.abstained is True
    assert answer.citations == []


def test_dense_only_hybrid_hit_uses_evidence_confidence_not_weighted_rank_score(
    monkeypatch,
) -> None:
    monkeypatch.setattr("app.services.retrieval.embed_query", lambda _text, **_kwargs: [1.0])
    semantic = _hit("semantic", "rollback procedure for a failed rollout", 0.9)

    outcome = retrieve(
        "how do we reverse a broken deployment",
        FakeStore([semantic], []),
        strategy="hybrid",
        top_k=1,
    )
    answer = answer_from_retrieval(outcome)

    assert outcome.hits[0]["score"] < 0.4
    assert outcome.hits[0]["evidence_score"] == pytest.approx(0.9)
    assert outcome.confidence == pytest.approx(0.9)
    assert answer.abstained is False


def test_orthogonal_dense_hit_does_not_clear_answer_threshold(monkeypatch) -> None:
    monkeypatch.setattr("app.services.retrieval.embed_query", lambda _text, **_kwargs: [1.0])
    unrelated = _hit("orthogonal", "unrelated evidence", 0.0)

    outcome = retrieve(
        "question",
        FakeStore([unrelated], []),
        strategy="dense",
        top_k=1,
    )
    answer = answer_from_retrieval(outcome)

    assert outcome.hits[0]["score"] == 0.0
    assert outcome.confidence == 0.0
    assert answer.abstained is True


def test_non_finite_dense_similarity_is_treated_as_no_evidence(monkeypatch) -> None:
    monkeypatch.setattr("app.services.retrieval.embed_query", lambda _text, **_kwargs: [0.0])
    invalid = _hit("invalid", "irrelevant evidence", float("nan"))

    outcome = retrieve(
        "???",
        FakeStore([invalid], []),
        strategy="hybrid",
        top_k=1,
    )
    answer = answer_from_retrieval(outcome)

    assert outcome.hits[0]["dense_score"] == 0.0
    assert outcome.confidence == 0.0
    assert answer.abstained is True


def test_zero_dense_weight_excludes_dense_only_hybrid_evidence(monkeypatch) -> None:
    monkeypatch.setattr("app.services.retrieval.embed_query", lambda _text, **_kwargs: [1.0])
    semantic = _hit("semantic", "strong semantic evidence", 0.9)

    outcome = retrieve(
        "a paraphrased question",
        FakeStore([semantic], []),
        strategy="hybrid",
        top_k=1,
        settings=Settings(rrf_lexical_weight=1.0),
    )

    assert outcome.hits == []
    assert outcome.dense_candidates == 0
    assert answer_from_retrieval(outcome).abstained is True


def test_zero_lexical_weight_excludes_lexical_only_hybrid_evidence(monkeypatch) -> None:
    monkeypatch.setattr("app.services.retrieval.embed_query", lambda _text, **_kwargs: [1.0])
    exact = _hit("exact", "emergency signing key", 1.0)

    outcome = retrieve(
        "emergency signing key",
        FakeStore([], [exact]),
        strategy="hybrid",
        top_k=1,
        settings=Settings(rrf_lexical_weight=0.0),
    )

    assert outcome.hits == []
    assert outcome.lexical_candidates == 0
    assert answer_from_retrieval(outcome).abstained is True


def test_answer_cites_only_hits_that_clear_the_evidence_threshold() -> None:
    strong = {
        "chunk_id": "strong",
        "snippet": "Grounded evidence.",
        "source": "gold.md",
        "score": 0.7,
        "evidence_score": 0.8,
    }
    weak = {
        "chunk_id": "weak",
        "snippet": "A low-confidence distractor.",
        "source": "wrong.md",
        "score": 0.6,
        "evidence_score": 0.2,
    }
    outcome = RetrievalOutcome(
        hits=[strong, weak],
        strategy="hybrid",
        dense_candidates=2,
        lexical_candidates=1,
    )

    answer = answer_from_retrieval(outcome, min_score=0.4)

    assert answer.abstained is False
    assert answer.citations == [strong]
    assert "Grounded evidence" in answer.answer
    assert "distractor" not in answer.answer


def test_answer_uses_strong_evidence_below_a_weak_fused_top_hit() -> None:
    weak = {
        "chunk_id": "weak",
        "snippet": "A dual-channel but weakly supported result.",
        "source": "weak.md",
        "score": 0.46,
        "evidence_score": 0.39,
    }
    strong = {
        "chunk_id": "strong",
        "snippet": "Strong semantic evidence.",
        "source": "strong.md",
        "score": 0.33,
        "evidence_score": 0.90,
    }
    outcome = RetrievalOutcome(
        hits=[weak, strong],
        strategy="hybrid",
        dense_candidates=2,
        lexical_candidates=1,
    )

    answer = answer_from_retrieval(outcome, min_score=0.4)

    assert outcome.confidence == pytest.approx(0.9)
    assert answer.abstained is False
    assert answer.citations == [strong]
    assert "Strong semantic evidence" in answer.answer
    assert "weakly supported" not in answer.answer


def test_tokenization_normalizes_punctuation_and_basic_inflection() -> None:
    query_tokens = tokenize("Which files are being ingested?")
    document_tokens = tokenize("The API can ingest a file.")

    assert "file" in query_tokens
    assert "ingest" in query_tokens
    assert lexical_overlap_score(document_tokens, query_tokens) == pytest.approx(2 / 3)


@pytest.mark.parametrize("top_k", [0, 51])
def test_top_k_is_bounded(monkeypatch, top_k: int) -> None:
    monkeypatch.setattr("app.services.retrieval.embed_query", lambda _text, **_kwargs: [1.0])
    with pytest.raises(ValueError, match="between 1 and 50"):
        retrieve("question", FakeStore([], []), strategy="dense", top_k=top_k)
