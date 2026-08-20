# Roadmap

This roadmap favors measurable quality and operational evidence over feature count. Items are not
commitments or claims about current behavior.

## Shipped in v0.2

- [x] Reproducible local demo with deterministic hashed embeddings.
- [x] Independent dense and lexical retrieval with weighted RRF.
- [x] Application-level golden evaluation and dense/lexical/hybrid ablation.
- [x] CI thresholds for ranking, answer evidence, citations, and abstention.
- [x] Liveness, storage-backed readiness, Prometheus metrics, and structured logs.
- [x] Optional ingestion API key and bounded document processing.
- [x] Non-root API image and no host-published Qdrant port in Compose.

## Next: strengthen the evidence

- Expand the 12-case fixture into separate calibration and held-out sets with more documents,
  ambiguous questions, paraphrases, exact-term queries, and adversarial hard negatives.
- Expand citation judgments and calibrate citation filtering on a held-out set.
- Add confidence calibration by retrieval strategy and select thresholds on calibration data.
- Run a scheduled real-embedding benchmark and compare it with the deterministic CI profile;
  record model revision, hardware, warm-up, and latency separately.
- Add regression deltas against a checked baseline so reviewers can see direction, not only a
  pass/fail snapshot.

## Next: improve retrieval

- Replace bounded token-overlap scanning with indexed BM25 or database-native full-text search.
- Evaluate structure-aware and semantic chunking against the current sentence-aware heuristic.
- Measure a cross-encoder reranker behind a latency budget.
- Add multi-document and duplicate-context judgments to test citation diversity.

## Later: deployment hardening

- Move rate limiting and metrics aggregation to shared infrastructure for multi-worker deployments.
- Add production authentication/authorization, tenant isolation, TLS termination guidance, and a
  secret-manager integration.
- Add OpenTelemetry traces, dashboards, alert examples, load tests, and explicit SLOs.
- Test backup/restore and schema migration procedures for both supported vector backends.
- Pin deployable dependencies and publish an SBOM and vulnerability scan in CI.

## Later: optional generation

- Add a pluggable answer generator only with citation verification and faithfulness evaluation.
- Keep the extractive path as a deterministic baseline and fallback.
