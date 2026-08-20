from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.services.retrieval import RetrievalOutcome

ABSTENTION_MESSAGE = "I don't have enough evidence in the indexed documents to answer that."


@dataclass(frozen=True)
class AnswerOutcome:
    answer: str
    citations: list[dict[str, Any]]
    confidence: float
    abstained: bool


def answer_from_retrieval(
    retrieval: RetrievalOutcome,
    *,
    min_score: float | None = None,
    settings: Settings | None = None,
) -> AnswerOutcome:
    """Compose a grounded extractive response or explicitly abstain."""

    settings = settings or get_settings()
    threshold = settings.min_relevance_score if min_score is None else min_score
    confidence = retrieval.confidence
    evidence = [
        item
        for item in retrieval.hits
        if item.get("snippet") and float(item.get("evidence_score", 0.0)) >= threshold
    ][:2]
    abstained = not evidence
    if abstained:
        answer = ABSTENTION_MESSAGE
    else:
        answer = "\n".join(
            [
                "Answer (extractive evidence):",
                *[
                    f"- {item['snippet']} [{item.get('source') or 'unknown source'}]"
                    for item in evidence
                ],
            ]
        )

    return AnswerOutcome(
        answer=answer,
        citations=[] if abstained else evidence,
        confidence=confidence,
        abstained=abstained,
    )
