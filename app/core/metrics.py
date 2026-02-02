from dataclasses import dataclass
from threading import Lock


@dataclass
class Metrics:
    ingest_requests: int = 0
    query_requests: int = 0
    errors: int = 0


_metrics = Metrics()
_lock = Lock()


def increment(field: str) -> None:
    with _lock:
        setattr(_metrics, field, getattr(_metrics, field) + 1)


def render_prometheus() -> str:
    with _lock:
        return (
            "# HELP rag_ingest_requests_total Total ingest requests\n"
            "# TYPE rag_ingest_requests_total counter\n"
            f"rag_ingest_requests_total {_metrics.ingest_requests}\n"
            "# HELP rag_query_requests_total Total query requests\n"
            "# TYPE rag_query_requests_total counter\n"
            f"rag_query_requests_total {_metrics.query_requests}\n"
            "# HELP rag_errors_total Total errors\n"
            "# TYPE rag_errors_total counter\n"
            f"rag_errors_total {_metrics.errors}\n"
        )
