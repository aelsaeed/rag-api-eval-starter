import argparse
import json
import os
import statistics
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/eval.jsonl")
    parser.add_argument("--out", default="reports/latest.md")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-hit-rate", type=float, default=0.5)
    parser.add_argument("--min-rubric-score", type=float, default=0.4)
    parser.add_argument("--max-p95-ms", type=float, default=250.0)
    return parser.parse_args()


def load_eval_records(path: str) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError("Evaluation dataset is empty")
    return records


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[rank]


def run_eval(dataset_path: str, k: int) -> dict:
    os.environ.setdefault("RAG_FAKE_EMBEDDINGS", "1")

    from app.core.config import get_settings
    from app.services.ingest import ingest_document
    from app.services.retrieval import hybrid_search
    from app.services.storage import get_store

    get_settings.cache_clear()
    store = get_store()
    store.ensure_collection()

    for path in sorted(Path("data/sample_docs").glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            ingest_document(path.name, path.read_bytes(), store)

    records = load_eval_records(dataset_path)
    hits = 0
    rubric_hits = 0
    latencies_ms: list[float] = []

    details: list[dict] = []
    for record in records:
        question = record["question"]
        expected = record["answer"]

        started = time.perf_counter()
        results = hybrid_search(question, store)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(latency_ms)

        top_k = results[:k]
        snippets = [item.get("snippet", "") or "" for item in top_k]
        expected_l = expected.lower()
        hit = any(expected_l in snippet.lower() for snippet in snippets)
        top1_hit = bool(snippets) and expected_l in snippets[0].lower()

        hits += int(hit)
        rubric_hits += int(top1_hit)

        details.append(
            {
                "question": question,
                "hit": hit,
                "top1_rubric": top1_hit,
                "latency_ms": latency_ms,
            }
        )

    total = len(records)
    metrics = {
        "samples": total,
        "hit_rate": hits / total,
        "recall_at_k": hits / total,
        "rubric_score": rubric_hits / total,
        "latency_p50_ms": statistics.median(latencies_ms),
        "latency_p95_ms": percentile(latencies_ms, 95),
        "details": details,
    }
    return metrics


def render_report(metrics: dict, k: int) -> str:
    lines = [
        "# Evaluation Report",
        "",
        "## Summary Metrics",
        f"- Samples: **{metrics['samples']}**",
        f"- Hit rate: **{metrics['hit_rate']:.3f}**",
        f"- Recall@{k}: **{metrics['recall_at_k']:.3f}**",
        f"- Rubric score (top-1 contains expected answer): **{metrics['rubric_score']:.3f}**",
        f"- Latency p50: **{metrics['latency_p50_ms']:.2f} ms**",
        f"- Latency p95: **{metrics['latency_p95_ms']:.2f} ms**",
        "",
        "## Per-question Results",
        "| # | Hit | Top1 Rubric | Latency (ms) | Question |",
        "|---|-----|-------------|--------------|----------|",
    ]

    for idx, item in enumerate(metrics["details"], start=1):
        lines.append(
            "| "
            f"{idx} | {item['hit']} | {item['top1_rubric']} | {item['latency_ms']:.2f} | "
            f"{item['question']} |"
        )

    return "\n".join(lines) + "\n"


def enforce_thresholds(metrics: dict, args: argparse.Namespace) -> None:
    failures = []
    if metrics["hit_rate"] < args.min_hit_rate:
        failures.append(f"hit_rate {metrics['hit_rate']:.3f} < {args.min_hit_rate:.3f}")
    if metrics["rubric_score"] < args.min_rubric_score:
        failures.append(
            f"rubric_score {metrics['rubric_score']:.3f} < {args.min_rubric_score:.3f}"
        )
    if metrics["latency_p95_ms"] > args.max_p95_ms:
        failures.append(f"latency_p95_ms {metrics['latency_p95_ms']:.2f} > {args.max_p95_ms:.2f}")

    if failures:
        raise SystemExit("Quality gate failure: " + "; ".join(failures))


def main() -> None:
    args = parse_args()
    metrics = run_eval(args.dataset, args.k)

    report = render_report(metrics, args.k)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    enforce_thresholds(metrics, args)


if __name__ == "__main__":
    main()
