from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.services.retrieval import RetrievalStrategy
from eval.app_adapter import ApplicationQueryAdapter
from eval.dataset import dataset_sha256, load_cases
from eval.metrics import enforce_thresholds, evaluate
from eval.models import EvaluationReport, Thresholds
from eval.reporting import report_to_dict

STRATEGIES: tuple[RetrievalStrategy, ...] = ("dense", "lexical", "hybrid")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_ablation(
    dataset_path: str | Path,
    corpus_path: str | Path,
) -> dict[str, EvaluationReport]:
    cases = load_cases(dataset_path)
    dataset_hash = dataset_sha256(dataset_path)
    reports: dict[str, EvaluationReport] = {}
    for strategy in STRATEGIES:
        with ApplicationQueryAdapter(corpus_path, strategy=strategy) as adapter:
            reports[strategy] = evaluate(cases, adapter, dataset_hash=dataset_hash)
    return reports


def render_ablation_markdown(reports: Mapping[str, EvaluationReport]) -> str:
    dataset_hashes = {report.dataset_sha256 for report in reports.values()}
    if set(reports) != set(STRATEGIES) or len(dataset_hashes) != 1:
        raise ValueError("ablation requires dense, lexical, and hybrid reports for one dataset")

    lines = [
        "# Retrieval Strategy Ablation",
        "",
        f"Dataset SHA-256: `{next(iter(dataset_hashes))}`",
        "",
        "| Strategy | Recall@1 | Recall@3 | Recall@5 | MRR@5 | nDCG@5 "
        "| Facts | Citations | Abstain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        summary = reports[strategy].summary
        lines.append(
            f"| {strategy} "
            f"| {summary.context_recall_at[1]:.3f} "
            f"| {summary.context_recall_at[3]:.3f} "
            f"| {summary.context_recall_at[5]:.3f} "
            f"| {summary.mrr_at_5:.3f} "
            f"| {summary.ndcg_at_5:.3f} "
            f"| {summary.answer_fact_coverage:.3f} "
            f"| {summary.citation_precision:.3f} "
            f"| {summary.abstention_accuracy:.3f} |"
        )
    lines.extend(
        [
            "",
            "The CI quality gate applies to `hybrid`; dense and lexical runs are "
            "diagnostic baselines.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_ablation_json(reports: Mapping[str, EvaluationReport]) -> str:
    payload = {
        "schema_version": 1,
        "strategies": {strategy: report_to_dict(reports[strategy]) for strategy in STRATEGIES},
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dense, lexical, and hybrid retrieval on the golden dataset"
    )
    parser.add_argument("--dataset", default=str(_PROJECT_ROOT / "data/eval.jsonl"))
    parser.add_argument("--corpus", default=str(_PROJECT_ROOT / "data/sample_docs"))
    parser.add_argument("--out", default=str(Path.cwd() / "reports/ablation.md"))
    parser.add_argument("--json-out", default=str(Path.cwd() / "reports/ablation.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    reports = run_ablation(args.dataset, args.corpus)
    enforce_thresholds(reports["hybrid"], Thresholds())

    markdown_path = Path(args.out)
    json_path = Path(args.json_out)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_ablation_markdown(reports), encoding="utf-8")
    json_path.write_text(render_ablation_json(reports), encoding="utf-8")


if __name__ == "__main__":
    main()
