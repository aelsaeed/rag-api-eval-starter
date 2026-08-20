from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from eval.adapters import OfflineCorpusAdapter, load_corpus
from eval.app_adapter import ApplicationQueryAdapter
from eval.dataset import dataset_sha256, load_cases, validate_gold_contexts
from eval.metrics import QualityGateError, enforce_thresholds, evaluate
from eval.models import EvaluationReport, QueryAdapter, Thresholds
from eval.reporting import write_reports

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = Thresholds()
    parser = argparse.ArgumentParser(
        description="Run the deterministic, offline RAG golden benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=str(_PROJECT_ROOT / "data/eval.jsonl"))
    parser.add_argument("--corpus", default=str(_PROJECT_ROOT / "data/sample_docs"))
    parser.add_argument("--out", default=str(Path.cwd() / "reports/latest.md"))
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--adapter",
        choices=("application", "offline"),
        default="application",
        help="evaluate the real application pipeline or the standalone lexical reference",
    )
    parser.add_argument(
        "--strategy",
        choices=("dense", "lexical", "hybrid"),
        default="hybrid",
        help="application retrieval strategy",
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=(5,),
        default=5,
        help="fixed retrieval depth for metrics reported at 1, 3, and 5",
    )
    parser.add_argument(
        "--min-context-recall-at-1", type=float, default=defaults.min_context_recall_at_1
    )
    parser.add_argument(
        "--min-context-recall-at-3", type=float, default=defaults.min_context_recall_at_3
    )
    parser.add_argument(
        "--min-context-recall-at-5", type=float, default=defaults.min_context_recall_at_5
    )
    parser.add_argument("--min-mrr-at-5", type=float, default=defaults.min_mrr_at_5)
    parser.add_argument("--min-ndcg-at-5", type=float, default=defaults.min_ndcg_at_5)
    parser.add_argument(
        "--min-answer-fact-coverage",
        type=float,
        default=defaults.min_answer_fact_coverage,
    )
    parser.add_argument(
        "--min-citation-precision", type=float, default=defaults.min_citation_precision
    )
    parser.add_argument(
        "--min-abstention-accuracy", type=float, default=defaults.min_abstention_accuracy
    )

    return parser.parse_args(argv)


def run_benchmark(
    dataset_path: str | Path,
    adapter: QueryAdapter,
) -> EvaluationReport:
    """Evaluate an injected query implementation without importing application state."""

    cases = load_cases(dataset_path)
    return evaluate(cases, adapter, dataset_hash=dataset_sha256(dataset_path))


def run_eval(
    dataset_path: str | Path,
    k: int = 5,
    *,
    adapter: QueryAdapter | None = None,
    corpus_path: str | Path = _PROJECT_ROOT / "data/sample_docs",
) -> EvaluationReport:
    """Compatibility entry point using the deterministic offline adapter by default."""

    if k != 5:
        raise ValueError("k must be exactly 5 for the fixed recall/MRR/nDCG benchmark")
    cases = load_cases(dataset_path)
    selected_adapter = adapter
    if selected_adapter is None:
        corpus = load_corpus(corpus_path)
        validate_gold_contexts(cases, corpus)
        selected_adapter = OfflineCorpusAdapter(corpus)
    return evaluate(cases, selected_adapter, dataset_hash=dataset_sha256(dataset_path))


def _thresholds_from_args(args: argparse.Namespace) -> Thresholds:
    return Thresholds(
        min_context_recall_at_1=args.min_context_recall_at_1,
        min_context_recall_at_3=args.min_context_recall_at_3,
        min_context_recall_at_5=args.min_context_recall_at_5,
        min_mrr_at_5=args.min_mrr_at_5,
        min_ndcg_at_5=args.min_ndcg_at_5,
        min_answer_fact_coverage=args.min_answer_fact_coverage,
        min_citation_precision=args.min_citation_precision,
        min_abstention_accuracy=args.min_abstention_accuracy,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.adapter == "application":
        with ApplicationQueryAdapter(args.corpus, strategy=args.strategy) as adapter:
            report = run_eval(args.dataset, args.k, adapter=adapter)
    else:
        report = run_eval(args.dataset, args.k, corpus_path=args.corpus)
    markdown_path = Path(args.out)
    json_path = Path(args.json_out) if args.json_out else markdown_path.with_suffix(".json")
    write_reports(report, markdown_path=markdown_path, json_path=json_path)

    try:
        enforce_thresholds(report, _thresholds_from_args(args))
    except QualityGateError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
