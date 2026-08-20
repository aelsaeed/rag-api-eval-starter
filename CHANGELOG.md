# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation

- Reframed the project around measurable retrieval regressions, reproducible evidence, and
  explicit operational limits.

## [0.2.0] - 2026-08-20

### Added

- Independent dense and lexical candidate retrieval with weighted reciprocal-rank fusion (RRF).
- Per-request retrieval strategy selection and auditable response metadata, including component
  scores, ranks, candidate counts, confidence, latency, and abstention state.
- Threshold-based abstention and grounded, extractive answers with source citations.
- A versioned 12-case golden dataset with answerable questions, hard negatives, graded contexts,
  required fact aliases, and dataset hashing.
- Application-level evaluation for context recall@1/3/5, MRR@5, nDCG@5, answer fact coverage,
  citation precision, and abstention accuracy.
- An evaluator contract that keeps the full ranked retrieval list separate from cited answer
  evidence, so ranking metrics and citation precision measure distinct behaviors.
- Dense/lexical/hybrid ablation reports in Markdown and JSON, plus CI quality gates on hybrid.
- Tests for configuration, retrieval, ingestion, API behavior, evaluation contracts, metric math,
  reporting, and application adapter isolation.
- Storage-backed `/ready`, process liveness at `/health`, Prometheus metrics, structured JSON logs,
  and request IDs.
- Optional `X-API-Key` protection for ingestion, bounded upload/PDF/text inputs, and per-client
  in-memory request limiting.
- Optional API-key authentication for direct connections to secured Qdrant services.
- Content-derived document IDs and replacement semantics for repeatable ingestion.
- Gap-free Qdrant replacement and transactional pgvector replacement for re-ingestion.
- Optional real sentence-transformer embeddings through the `ml` dependency extra.
- Non-root container execution and a Compose profile with an internal Qdrant service.

### Changed

- Deterministic hashed embeddings and in-memory Qdrant are now the default local profile; model
  weights are no longer required for the base install.
- The demo now uses local deterministic mode by default, honors `API_URL`, waits on readiness, and
  uses Docker only when explicitly requested.
- Chunking now prefers sentence/paragraph boundaries, preserves bounded overlap between chunks, and
  applies fixed windows to a single text unit that exceeds the configured chunk size.
- Settings and API inputs now enforce bounds and cross-field validation.
- The development runner binds to loopback by default and accepts explicit `HOST`/`PORT` overrides.
- Qdrant and pgvector storage expose separate dense and lexical search paths.
- PostgreSQL identifiers are composed safely, and existing Qdrant vector dimensions are checked
  against the configured embedder.
- CI now builds the package, runs lint/format/type/test/coverage checks, executes the application
  benchmark and strategy ablation, uploads evidence artifacts, and smoke-tests the container.

### Security

- The API container runs as an unprivileged user with Linux capabilities dropped in Compose.
- Qdrant is no longer published to the host by the default Compose configuration.
- Ingestion rejects unsupported extensions, empty documents, oversized bodies, excessive PDF
  pages, excessive extracted text, oversized filenames, and excessive chunk counts before embedding.
- Dependency floors exclude the known multipart and PDF text-extraction denial-of-service ranges
  reachable through the ingestion endpoint.

## [0.1.0] - 2026-02-02

### Added

- Initial FastAPI ingestion/query service with Qdrant and pgvector adapters.
- Fixed-size overlapping chunks, dense embeddings, sample documents, a basic evaluator, Docker,
  and local development commands.
