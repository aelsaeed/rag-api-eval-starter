# Portfolio notes

This page translates the implementation into defensible interview material. Keep claims tied to the
deterministic benchmark and distinguish implemented safeguards from production infrastructure.

## CV bullet

> Built a FastAPI retrieval service with independently ranked dense/lexical search, weighted RRF,
> evidence citations, and confidence-based abstention; on a 12-case deterministic local benchmark,
> hybrid improved nDCG@5 from 0.702 (dense) to 0.963, reached Recall@3 and Recall@5 of 1.000, and
> abstained on both hard negatives, with the same metrics enforced as CI gates.

Add benchmark numbers to a CV only when copying them from a fresh run for the exact commit. This
repository's default benchmark uses hashed embeddings and a small fixture, so describe the result as
“deterministic local evaluation,” not as a general production-quality score.

## 60-second project story

“Retrieval changes are easy to ship and hard to notice: the API can stay healthy while answer evidence
gets worse. I built a small retrieval-focused RAG backend where those regressions become testable. The
service creates independent dense and lexical candidate lists, combines them with weighted RRF, and
returns the component ranks and scores that explain the result. A golden dataset exercises the real
ingestion, storage, retrieval, answering, citation, and abstention path. CI fails when ranking or
grounding metrics fall below explicit thresholds and preserves Markdown/JSON reports as evidence. I
kept hashed embeddings as the default so reviewers can reproduce the gate offline, while a
sentence-transformer extra demonstrates how the same service can use a real model.”

## Talking points

### The problem is silent quality regression

An HTTP 200 and a low latency do not show that the right context was retrieved. The primary design
decision was to make retrieval quality a build contract, alongside unit tests and container health.
The quality job runs the application adapter and returns a non-zero status when any hybrid threshold
fails.

### Hybrid means two candidate lists

Dense and lexical search run independently. Weighted reciprocal-rank fusion preserves rank evidence
while calibrating each contribution with a bounded component score; it does not add raw cosine and
token-overlap values as though they shared a scale. Per-query strategy selection makes the two
components useful diagnostic baselines, not hidden implementation details.

Ranking and answer confidence are separate: RRF orders candidates, while the evidence score uses the
strongest available non-negative cosine or lexical-coverage signal for each hit. The answer selects
the first two threshold-clearing hits in fused-rank order and reports the strongest returned evidence
as confidence. That avoids
rejecting a strong dense-only paraphrase merely because the lexical RRF weight is absent, avoids
diluting an exact match with a weak dense score, and treats an orthogonal dense result as irrelevant.

Useful code to discuss:

- [retrieval and weighted RRF](../app/services/retrieval.py)
- [Qdrant and pgvector adapters](../app/services/storage.py)
- [extractive answer and abstention](../app/services/answering.py)

### The evaluator crosses the application boundary

The application adapter creates an isolated in-memory store, ingests the same sample corpus as the
service, runs the selected retrieval strategy, and scores the actual answer/citation contract. Fully
explicit settings prevent a developer's ambient `RAG_*` environment from silently switching the
backend or embedding mode.

Useful code to discuss:

- [application evaluation adapter](../eval/app_adapter.py)
- [metric definitions and gates](../eval/metrics.py)
- [versioned golden cases](../data/eval.jsonl)
- [CI evidence workflow](../.github/workflows/ci.yml)

### Evaluation covers more than “did a chunk match?”

- Recall@1/3/5 measures whether gold evidence is recovered in the full ranked list by each depth.
- MRR@5 rewards putting the first relevant result early.
- nDCG@5 includes graded relevance and ranking order.
- Fact coverage checks whether the extractive answer contains every required fact group.
- Citation precision checks whether citations actually attached to the answer match human-reviewed
  source/anchor pairs.
- Abstention accuracy includes answerable cases and two unanswerable hard negatives.

The ranked top five and the answer's first two citations are scored separately. That distinction
prevents a rank-five hit from disappearing from Recall@5 and prevents uncited retrieval candidates
from diluting citation precision.

The benchmark is deliberately small and deterministic. Its value is fast regression detection; it
does not estimate quality on a broad user distribution. [Evaluation methodology](evaluation.md)
documents the boundary.

The current lexical baseline scores 1.000 on the ranking metrics and beats hybrid on this literal
fixture. That is a useful interview point: the ablation is there to challenge an architecture choice,
not decorate it. A larger, less literal held-out set is needed before arguing that fusion wins across
domains.

### Operational choices are visible, not implied

The project separates liveness from storage-backed readiness, offloads blocking storage/embedding
work from async handlers, exposes Prometheus metrics, produces request-correlated JSON logs, bounds
document processing, and runs the API container without root privileges. It also names the remaining
gaps: a process-local limiter, shared-key ingestion auth, no TLS/tenant model, and a bounded lexical
scan. See [operations and security](operations.md).

## Evidence to show in an interview

1. Run `make demo` to show ingestion, hybrid retrieval metadata, extractive evidence, citations, and
   cleanup without Docker or model downloads.
2. Open a local `reports/ablation.md` to compare dense, lexical, and hybrid rankings on the same
   hashed dataset.
3. Change a relevant paragraph or retrieval setting on a branch and run `make eval-ci` to demonstrate
   the regression gate's failure message.
4. Walk from one golden JSONL case through the application adapter, metric calculation, Markdown/JSON
   report, and CI artifact upload.

## Claims to avoid

- Do not call the current answer path generative: it returns extractive evidence.
- Do not present hashed-vector results as sentence-transformer model quality.
- Do not describe the 12-case fixture as a statistically representative benchmark.
- Do not call the in-memory limiter distributed or the optional ingestion key full authentication.
- Do not claim a remote CI run is green without linking to the successful run for the same commit.
- Do not claim the bounded token-overlap scan is BM25 or full-text search.

## Strong next-step answer

If asked what you would build next, start with evaluation depth: split calibration and held-out data,
add more adversarial cases, calibrate abstention per strategy, and run a recorded real-embedding job.
Then replace token overlap with indexed BM25/full-text retrieval and evaluate a reranker under a
latency budget. This sequence improves the measurement before increasing system complexity.
