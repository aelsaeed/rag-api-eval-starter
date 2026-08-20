from dataclasses import dataclass, field
from threading import Lock

LATENCY_BUCKETS_SECONDS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


@dataclass
class Metrics:
    ingest_requests: int = 0
    query_requests: int = 0
    errors: int = 0
    query_abstentions: int = 0
    query_duration_sum_seconds: float = 0.0
    query_duration_count: int = 0
    query_duration_buckets: dict[float, int] = field(
        default_factory=lambda: {bucket: 0 for bucket in LATENCY_BUCKETS_SECONDS}
    )
    retrieval_strategies: dict[str, int] = field(default_factory=dict)
    returned_citations: int = 0


_metrics = Metrics()
_lock = Lock()


def increment(field: str) -> None:
    with _lock:
        setattr(_metrics, field, getattr(_metrics, field) + 1)


def observe_query(*, strategy: str, latency_ms: float, abstained: bool, citations: int) -> None:
    latency_seconds = latency_ms / 1_000.0
    with _lock:
        _metrics.query_requests += 1
        _metrics.query_abstentions += int(abstained)
        _metrics.query_duration_sum_seconds += latency_seconds
        _metrics.query_duration_count += 1
        _metrics.returned_citations += citations
        _metrics.retrieval_strategies[strategy] = _metrics.retrieval_strategies.get(strategy, 0) + 1
        for bucket in LATENCY_BUCKETS_SECONDS:
            if latency_seconds <= bucket:
                _metrics.query_duration_buckets[bucket] += 1


def render_prometheus() -> str:
    with _lock:
        lines = [
            "# HELP rag_ingest_requests_total Total ingest requests\n"
            "# TYPE rag_ingest_requests_total counter\n"
            f"rag_ingest_requests_total {_metrics.ingest_requests}\n",
            "# HELP rag_query_requests_total Total query requests\n"
            "# TYPE rag_query_requests_total counter\n"
            f"rag_query_requests_total {_metrics.query_requests}\n",
            "# HELP rag_errors_total Total errors\n"
            "# TYPE rag_errors_total counter\n"
            f"rag_errors_total {_metrics.errors}\n",
            "# HELP rag_query_abstentions_total Queries rejected for insufficient evidence\n"
            "# TYPE rag_query_abstentions_total counter\n"
            f"rag_query_abstentions_total {_metrics.query_abstentions}\n",
            "# HELP rag_query_duration_seconds Query retrieval and answer latency\n"
            "# TYPE rag_query_duration_seconds histogram\n",
        ]
        for bucket in LATENCY_BUCKETS_SECONDS:
            lines.append(
                f'rag_query_duration_seconds_bucket{{le="{bucket:g}"}} '
                f"{_metrics.query_duration_buckets[bucket]}\n"
            )
        lines.extend(
            [
                f'rag_query_duration_seconds_bucket{{le="+Inf"}} {_metrics.query_duration_count}\n',
                f"rag_query_duration_seconds_sum {_metrics.query_duration_sum_seconds:.9f}\n",
                f"rag_query_duration_seconds_count {_metrics.query_duration_count}\n",
                "# HELP rag_retrieval_strategy_total Queries by retrieval strategy\n"
                "# TYPE rag_retrieval_strategy_total counter\n",
            ]
        )
        for strategy in sorted(_metrics.retrieval_strategies):
            lines.append(
                f'rag_retrieval_strategy_total{{strategy="{strategy}"}} '
                f"{_metrics.retrieval_strategies[strategy]}\n"
            )
        lines.extend(
            [
                "# HELP rag_returned_citations_total Citations returned to callers\n"
                "# TYPE rag_returned_citations_total counter\n",
                f"rag_returned_citations_total {_metrics.returned_citations}\n",
            ]
        )
        return "".join(lines)
