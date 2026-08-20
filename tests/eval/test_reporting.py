import json

from eval.metrics import evaluate
from eval.models import EvalCase, GoldContext, QueryResult, RetrievalHit
from eval.reporting import render_json, render_markdown


class PipeAdapter:
    name = "adapter|fixture"

    def query(self, question: str, *, k: int) -> QueryResult:
        del question, k
        hit = RetrievalHit("gold|source.md", "chunk-1", "gold context", 1.0)
        return QueryResult(
            answer="required fact",
            retrieval_hits=(hit,),
            citations=(hit,),
        )


def test_json_and_markdown_reports_are_deterministic_and_machine_readable() -> None:
    case = EvalCase(
        id="report-001",
        question="question",
        answerable=True,
        gold_contexts=(GoldContext("gold|source.md", "gold context", 2),),
        required_fact_groups=(("required fact",),),
    )
    report = evaluate((case,), PipeAdapter(), dataset_hash="deadbeef")

    json_report = render_json(report)
    markdown_report = render_markdown(report)

    assert render_json(report) == json_report
    assert render_markdown(report) == markdown_report
    payload = json.loads(json_report)
    assert payload["schema_version"] == 1
    assert payload["summary"]["context_recall_at_1"] == 1.0
    assert "gold\\|source.md" in markdown_report
    assert "Context recall@1" in markdown_report
