from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.models import CaseMetrics, EvaluationReport, SummaryMetrics


def _summary_dict(summary: SummaryMetrics) -> dict[str, Any]:
    return {
        "samples": summary.samples,
        "answerable_samples": summary.answerable_samples,
        "unanswerable_samples": summary.unanswerable_samples,
        "context_recall_at_1": summary.context_recall_at[1],
        "context_recall_at_3": summary.context_recall_at[3],
        "context_recall_at_5": summary.context_recall_at[5],
        "mrr_at_5": summary.mrr_at_5,
        "ndcg_at_5": summary.ndcg_at_5,
        "answer_fact_coverage": summary.answer_fact_coverage,
        "citation_precision": summary.citation_precision,
        "abstention_accuracy": summary.abstention_accuracy,
    }


def _case_dict(case: CaseMetrics) -> dict[str, Any]:
    return {
        "id": case.case_id,
        "question": case.question,
        "answerable": case.answerable,
        "tags": list(case.tags),
        "context_recall_at_1": case.context_recall_at[1],
        "context_recall_at_3": case.context_recall_at[3],
        "context_recall_at_5": case.context_recall_at[5],
        "reciprocal_rank_at_5": case.reciprocal_rank_at_5,
        "ndcg_at_5": case.ndcg_at_5,
        "answer_fact_coverage": case.answer_fact_coverage,
        "citation_precision": case.citation_precision,
        "abstention_correct": case.abstention_correct,
        "retrieved_sources": list(case.retrieved_sources),
        "missing_fact_groups": [list(group) for group in case.missing_fact_groups],
    }


def report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "adapter": report.adapter,
        "dataset_sha256": report.dataset_sha256,
        "summary": _summary_dict(report.summary),
        "cases": [_case_dict(case) for case in report.cases],
    }


def render_json(report: EvaluationReport) -> str:
    return json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n"


def _escape_table(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_markdown(report: EvaluationReport) -> str:
    summary = report.summary
    lines = [
        "# Deterministic RAG Evaluation Report",
        "",
        f"- Adapter: `{report.adapter}`",
        f"- Dataset SHA-256: `{report.dataset_sha256}`",
        f"- Samples: **{summary.samples}** "
        f"({summary.answerable_samples} answerable, "
        f"{summary.unanswerable_samples} unanswerable)",
        "",
        "## Summary",
        "",
        "| Metric | Score |",
        "|---|---:|",
        f"| Context recall@1 | {summary.context_recall_at[1]:.3f} |",
        f"| Context recall@3 | {summary.context_recall_at[3]:.3f} |",
        f"| Context recall@5 | {summary.context_recall_at[5]:.3f} |",
        f"| MRR@5 | {summary.mrr_at_5:.3f} |",
        f"| nDCG@5 | {summary.ndcg_at_5:.3f} |",
        f"| Answer fact coverage | {summary.answer_fact_coverage:.3f} |",
        f"| Citation precision | {summary.citation_precision:.3f} |",
        f"| Abstention accuracy | {summary.abstention_accuracy:.3f} |",
        "",
        "## Per-case results",
        "",
        "| ID | R@1 | R@3 | R@5 | RR@5 | nDCG@5 | Facts | Citations | Abstain | Sources |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for case in report.cases:
        sources = ", ".join(case.retrieved_sources) or "-"
        lines.append(
            f"| {_escape_table(case.case_id)} "
            f"| {case.context_recall_at[1]:.3f} "
            f"| {case.context_recall_at[3]:.3f} "
            f"| {case.context_recall_at[5]:.3f} "
            f"| {case.reciprocal_rank_at_5:.3f} "
            f"| {case.ndcg_at_5:.3f} "
            f"| {_format_optional(case.answer_fact_coverage)} "
            f"| {_format_optional(case.citation_precision)} "
            f"| {'pass' if case.abstention_correct else 'FAIL'} "
            f"| {_escape_table(sources)} |"
        )

    failures = [case for case in report.cases if case.missing_fact_groups]
    if failures:
        lines.extend(["", "## Missing required facts", ""])
        for case in failures:
            groups = [" / ".join(group) for group in case.missing_fact_groups]
            lines.append(f"- `{case.case_id}`: {_escape_table('; '.join(groups))}")
    return "\n".join(lines) + "\n"


def write_reports(
    report: EvaluationReport,
    *,
    markdown_path: str | Path,
    json_path: str | Path,
) -> None:
    markdown_output = Path(markdown_path)
    json_output = Path(json_path)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    json_output.write_text(render_json(report), encoding="utf-8")
