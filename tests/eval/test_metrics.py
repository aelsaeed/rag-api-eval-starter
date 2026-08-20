import math

import pytest

from eval.metrics import AdapterContractError, score_case
from eval.models import EvalCase, GoldContext, QueryResult, RetrievalHit


def _case() -> EvalCase:
    return EvalCase(
        id="ranking-001",
        question="Where is the golden fact?",
        answerable=True,
        gold_contexts=(
            GoldContext(source="gold.md", anchor="The golden fact is documented.", relevance=2),
        ),
        required_fact_groups=(("golden fact",), ("documented", "written down")),
        tags=("ranking",),
    )


def _hit(source: str, chunk_id: str, text: str) -> RetrievalHit:
    return RetrievalHit(source=source, chunk_id=chunk_id, text=text, score=1.0)


def test_ranked_metrics_have_exact_values() -> None:
    hits = (
        _hit("wrong.md", "wrong-1", "A distractor."),
        _hit("wrong.md", "wrong-2", "Another distractor."),
        _hit("gold.md", "gold-1", "The golden fact is documented."),
    )
    result = QueryResult(
        answer="The GOLDEN fact is written down!",
        retrieval_hits=hits,
        citations=hits,
    )

    metrics = score_case(_case(), result)

    assert metrics.context_recall_at == {1: 0.0, 3: 1.0, 5: 1.0}
    assert metrics.reciprocal_rank_at_5 == pytest.approx(1 / 3)
    assert metrics.ndcg_at_5 == pytest.approx(0.5)
    assert metrics.answer_fact_coverage == 1.0
    assert metrics.citation_precision == pytest.approx(1 / 3)
    assert metrics.missing_fact_groups == ()


def test_context_requires_both_source_and_anchor() -> None:
    hit = _hit("wrong.md", "wrong-source", "The golden fact is documented.")
    result = QueryResult(
        answer="The golden fact is documented.",
        retrieval_hits=(hit,),
        citations=(hit,),
    )

    metrics = score_case(_case(), result)

    assert metrics.context_recall_at[1] == 0.0
    assert metrics.citation_precision == 0.0


def test_unanswerable_case_scores_correct_abstention() -> None:
    case = EvalCase(
        id="negative-001",
        question="What is the uptime guarantee?",
        answerable=False,
        gold_contexts=(),
        required_fact_groups=(),
    )

    metrics = score_case(
        case,
        QueryResult(
            answer="",
            retrieval_hits=(_hit("wrong.md", "wrong-1", "A distractor."),),
            citations=(),
            abstained=True,
        ),
    )

    assert metrics.abstention_correct is True
    assert metrics.answer_fact_coverage is None
    assert metrics.citation_precision is None


def test_adapter_contract_rejects_duplicate_chunks_and_non_finite_scores() -> None:
    duplicate = _hit("gold.md", "same", "The golden fact is documented.")
    with pytest.raises(AdapterContractError, match="duplicate retrieval hit"):
        score_case(
            _case(),
            QueryResult(
                answer="answer",
                retrieval_hits=(duplicate, duplicate),
                citations=(duplicate,),
            ),
        )

    invalid = RetrievalHit("gold.md", "bad-score", "text", score=math.nan)
    with pytest.raises(AdapterContractError, match="finite"):
        score_case(
            _case(),
            QueryResult(answer="answer", retrieval_hits=(invalid,), citations=(invalid,)),
        )


def test_ranking_and_citation_metrics_use_distinct_result_sets() -> None:
    wrong = _hit("wrong.md", "wrong-1", "A distractor.")
    gold = _hit("gold.md", "gold-1", "The golden fact is documented.")
    result = QueryResult(
        answer="The golden fact is documented.",
        retrieval_hits=(wrong, gold),
        citations=(gold,),
    )

    metrics = score_case(_case(), result)

    assert metrics.context_recall_at[1] == 0.0
    assert metrics.context_recall_at[3] == 1.0
    assert metrics.reciprocal_rank_at_5 == 0.5
    assert metrics.citation_precision == 1.0


def test_citations_must_come_from_ranked_retrieval_hits() -> None:
    retrieved = _hit("wrong.md", "wrong-1", "A distractor.")
    citation = _hit("gold.md", "gold-1", "The golden fact is documented.")

    with pytest.raises(AdapterContractError, match="not present in retrieval hits"):
        score_case(
            _case(),
            QueryResult(
                answer="The golden fact is documented.",
                retrieval_hits=(retrieved,),
                citations=(citation,),
            ),
        )
