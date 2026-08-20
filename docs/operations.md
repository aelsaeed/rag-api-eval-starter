# Operations and security

This service has a safe, reproducible development profile and several deployment-oriented controls.
It is not a turnkey production platform: authentication, distributed rate limiting, TLS, tenant
isolation, and external observability infrastructure remain deployment responsibilities.

## Runtime profiles

### Deterministic local process

The default settings use 64-dimensional hashed embeddings and Qdrant's in-memory client. This is
the fastest offline path for development and evaluation:

```bash
python -m venv .venv
source .venv/bin/activate
make setup
make run
```

Data disappears when the process stops. No model is downloaded. `make run` listens only on
`127.0.0.1` by default; use an explicit override such as `HOST=0.0.0.0 make run` only when network
exposure is intended and the surrounding controls are in place.

### Docker Compose with persistent Qdrant

```bash
docker compose up --build
```

Compose starts the API and Qdrant, binds the API to `127.0.0.1:8000`, does not publish Qdrant to the
host, and stores vectors in the `qdrant_data` volume. `docker compose down` preserves the volume;
`docker compose down --volumes` removes it.

The API image runs as UID/GID 10001, uses a readiness health check, and drops Linux capabilities in
Compose. The Qdrant image and the surrounding deployment still require their own hardening review.

### Real sentence-transformer embeddings

Real embeddings are optional so the base install and CI remain offline and deterministic:

```bash
make setup-ml
export RAG_FAKE_EMBEDDINGS=0
make run
```

The default model is `sentence-transformers/all-MiniLM-L6-v2`. Its weights are downloaded on first
use unless already cached. To build the optional ML dependencies into the Compose image:

```bash
RAG_INSTALL_ML=true RAG_FAKE_EMBEDDINGS=0 docker compose up --build
```

Changing embedding dimensions requires a new collection or a compatible stored collection. Startup
rejects a Qdrant collection or pgvector table whose vector size differs from the configured embedder.

### pgvector

The pgvector adapter requires an existing PostgreSQL service where the application can create the
`vector` extension and its table:

```bash
export RAG_VECTOR_BACKEND=pgvector
export RAG_POSTGRES_URL='postgresql://user:password@host:5432/database'
make run
```

The default Compose stack does not provision PostgreSQL. Table identifiers are validated and safely
composed, but database roles, network policy, TLS, backups, and extension permissions remain operator
concerns.

## Configuration

`Settings` reads `RAG_*` environment variables at process startup. `.env.example` is a reference;
the Python application does not load it automatically. Export variables in the shell or inject them
with the process supervisor. Restart after a change.

| Variable | Default | Purpose |
|---|---:|---|
| `RAG_FAKE_EMBEDDINGS` | `1` | Use deterministic hashed vectors instead of the optional ML model |
| `RAG_FAKE_EMBEDDING_DIM` | `64` | Hashed vector dimension |
| `RAG_VECTOR_BACKEND` | `qdrant` | Select `qdrant` or `pgvector` |
| `RAG_QDRANT_URL` | unset | Use remote Qdrant; unset selects the in-memory client |
| `RAG_QDRANT_API_KEY` | unset | Optional ASCII secret for an authenticated remote Qdrant service |
| `RAG_POSTGRES_URL` | unset | Required connection string for pgvector |
| `RAG_RETRIEVAL_STRATEGY` | `hybrid` | Default `dense`, `lexical`, or `hybrid` strategy |
| `RAG_CANDIDATE_MULTIPLIER` | `4` | Candidate depth relative to requested `top_k` |
| `RAG_RRF_K` | `60` | RRF rank constant |
| `RAG_RRF_LEXICAL_WEIGHT` | `0.65` | Lexical share of weighted RRF; dense uses the remainder |
| `RAG_MIN_RELEVANCE_SCORE` | `0.40` | Minimum retrieved-hit evidence confidence before answering |
| `RAG_LEXICAL_SCAN_LIMIT` | `1000` | Maximum Qdrant records scored in one lexical search |
| `RAG_INGEST_API_KEY` | unset | Optional ASCII ingestion secret; when set it must contain at least 16 characters |
| `RAG_RATE_LIMIT_PER_MINUTE` | `60` | Per-client, per-process request limit |
| `RAG_REQUEST_SIZE_LIMIT_MB` | `5` | Request and uploaded-document byte limit |
| `RAG_MAX_FILENAME_CHARS` | `255` | Maximum stored basename length before chunk fan-out |
| `RAG_MAX_PDF_PAGES` | `50` | Maximum PDF pages parsed |
| `RAG_MAX_DOCUMENT_CHARS` | `2000000` | Maximum extracted characters |
| `RAG_MAX_DOCUMENT_CHUNKS` | `2000` | Maximum chunks created before embedding |

See [.env.example](../.env.example) for the full configuration surface, including chunking, model,
collection, table, log-level, and `top_k` settings. Bounds and related-field constraints are checked
at startup.

## Health and observability

| Endpoint | Meaning |
|---|---|
| `GET /health` | Process liveness; does not prove storage is reachable |
| `GET /ready` | Calls the configured store and returns 503 when it is unavailable |
| `GET /metrics` | Prometheus text exposition for process-local counters and query latency |
| `GET /docs` | OpenAPI/Swagger UI |

Logs are JSON and carry the `X-Request-ID` supplied by the caller or generated by the service. Query
logs record strategy, result count, confidence, abstention, and latency; ingestion logs record a
content-derived document ID and chunk count. Raw document contents are not intentionally logged.

Prometheus metrics include ingest/query/error/abstention counters, a query-latency histogram,
strategy counts, and returned-citation counts. They are in process memory, so aggregate externally in
a multi-worker or replicated deployment.

## Ingestion controls

- `/ingest` accepts `.md`, `.txt`, and `.pdf` filenames.
- The endpoint reads at most the configured byte limit plus one byte, even when `Content-Length` is
  absent; middleware also rejects oversized declared requests early.
- Empty documents, PDFs over the page cap, extracted text over the character cap, and documents over
  the chunk-count cap are rejected before embedding.
- Stored source names are reduced to a basename and bounded before being copied into chunk payloads.
- The document ID is derived from SHA-256 content. Re-ingesting identical bytes replaces the same
  document's chunks rather than accumulating duplicates.
- Setting a non-blank, ASCII, 16+ character `RAG_INGEST_API_KEY` enables a constant-time key
  comparison for ingestion only; invalid secrets fail startup instead of disabling protection.

These are guardrails, not a complete trust boundary. Extension checks do not validate file signatures,
the API key is a single shared secret, query/metrics/docs endpoints are unauthenticated, and the app
does not terminate TLS.

## Known scaling limits

- Qdrant lexical retrieval scrolls and scores at most `RAG_LEXICAL_SCAN_LIMIT` matching records. It
  is deterministic and bounded, but it is not BM25 and may miss candidates in a large collection.
- The rate limiter is held in one process and keyed by the directly observed client address. It is
  neither shared across workers nor configured to interpret proxy headers.
- Metrics are process-local and reset on restart.
- Sentence-aware chunks can still split a single text unit that exceeds the configured size.
- PDF ingestion extracts text only; scanned documents need OCR before upload.
- In-memory Qdrant is intentionally ephemeral.

For an internet-facing deployment, put the service behind an authenticated TLS gateway, use shared
rate limiting, restrict `/metrics` and `/docs`, isolate each tenant's data, manage secrets outside the
repository, and test backup/restore for the chosen store.

## Troubleshooting

- **`/ready` returns 503:** verify the configured Qdrant/PostgreSQL address, credentials, collection
  initialization permissions, and network route. `/health` can still return 200 in this state.
- **Real embeddings fail to import:** run `make setup-ml` or restore `RAG_FAKE_EMBEDDINGS=1`.
- **Qdrant reports a dimension mismatch:** use a collection created for the current embedding model
  or select a new `RAG_QDRANT_COLLECTION`.
- **Docker demo rejects the environment:** `DEMO_RUNTIME=docker` requires Docker Compose v2 and an
  available daemon. Plain `make demo` does not require Docker.
- **The demo says the address is already serving traffic:** choose an unused address, for example
  `API_URL=http://127.0.0.1:8100 make demo`. The script refuses to ingest into an unrelated service.
- **Python fails to import `typing.Self`:** the shell is using Python older than 3.11. Activate the
  project virtual environment or run a Python-based target with `PYTHON=.venv/bin/python`.
