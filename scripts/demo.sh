#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://127.0.0.1:8000}
DEMO_RUNTIME=${DEMO_RUNTIME:-local}
DEMO_STARTUP_TIMEOUT=${DEMO_STARTUP_TIMEOUT:-60}
COMPOSE_PROJECT_NAME=${DEMO_COMPOSE_PROJECT:-rag-api-eval-demo-${UID:-0}-$$}

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN=$PYTHON
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN=python
fi

API_PID=""
API_LOG=""
COMPOSE_STARTED=0

fail() {
  echo "[demo] ERROR: $*" >&2
  exit 1
}

cleanup() {
  exit_code=$?

  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" >/dev/null 2>&1; then
    kill "$API_PID" >/dev/null 2>&1 || true
    wait "$API_PID" >/dev/null 2>&1 || true
  fi

  if [[ "$COMPOSE_STARTED" == "1" ]]; then
    docker compose --project-name "$COMPOSE_PROJECT_NAME" down \
      --volumes --remove-orphans >/dev/null 2>&1 || true
  fi

  if [[ -n "$API_LOG" && -f "$API_LOG" ]]; then
    if [[ "$exit_code" != "0" ]]; then
      echo "[demo] Local API log:" >&2
      tail -n 80 "$API_LOG" >&2 || true
    fi
    rm -f -- "$API_LOG"
  fi

  return "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python executable not found: $PYTHON_BIN"
[[ "$DEMO_RUNTIME" == "local" || "$DEMO_RUNTIME" == "docker" ]] || \
  fail "DEMO_RUNTIME must be 'local' or 'docker'"
[[ "$DEMO_STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || \
  fail "DEMO_STARTUP_TIMEOUT must be a positive integer"

API_ADDRESS=$(
  "$PYTHON_BIN" - "$API_URL" <<'PY'
import sys
from urllib.parse import urlsplit

url = urlsplit(sys.argv[1])
if url.scheme != "http":
    raise SystemExit("[demo] API_URL must use http:// for the local demo")
if not url.hostname or url.username or url.password:
    raise SystemExit("[demo] API_URL must contain a host and must not contain credentials")
if url.path.rstrip("/") or url.query or url.fragment:
    raise SystemExit("[demo] API_URL must not contain a path, query, or fragment")

print(f"{url.hostname}\t{url.port or 80}")
PY
)
IFS=$'\t' read -r API_BIND_HOST API_BIND_PORT <<<"$API_ADDRESS"

DOCKER_BIND_HOST=$API_BIND_HOST
if [[ "$DOCKER_BIND_HOST" == "localhost" ]]; then
  DOCKER_BIND_HOST=127.0.0.1
fi
[[ "$DEMO_RUNTIME" != "docker" || "$DOCKER_BIND_HOST" != *:* ]] || \
  fail "Docker demo mode currently requires an IPv4 API_URL host"

export API_URL
export DEMO_API_BIND_HOST="$DOCKER_BIND_HOST"
export DEMO_API_PORT="$API_BIND_PORT"

api_is_responding() {
  "$PYTHON_BIN" - "$API_URL" <<'PY'
import sys
import urllib.error
import urllib.request

url = sys.argv[1].rstrip("/") + "/health"
try:
    urllib.request.urlopen(url, timeout=1).close()
except urllib.error.HTTPError:
    raise SystemExit(0)
except (OSError, urllib.error.URLError):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

start_local_api() {
  echo "[demo] Starting deterministic local API at $API_URL"
  API_LOG=$(mktemp "${TMPDIR:-/tmp}/rag-api-demo.XXXXXX.log")

  local_env=(
    env -i
    "PATH=$PATH"
    RAG_ENVIRONMENT=dev
    RAG_LOG_LEVEL=INFO
    RAG_FAKE_EMBEDDINGS=1
    RAG_FAKE_EMBEDDING_DIM=64
    RAG_VECTOR_BACKEND=qdrant
    RAG_QDRANT_COLLECTION=rag_demo_documents
    RAG_CHUNK_SIZE=800
    RAG_CHUNK_OVERLAP=120
    RAG_TOP_K=5
    RAG_RETRIEVAL_STRATEGY=hybrid
    RAG_CANDIDATE_MULTIPLIER=4
    RAG_RRF_K=60
    RAG_RRF_LEXICAL_WEIGHT=0.65
    RAG_MIN_RELEVANCE_SCORE=0.40
    RAG_LEXICAL_SCAN_LIMIT=1000
    RAG_REQUEST_SIZE_LIMIT_MB=5
    RAG_MAX_FILENAME_CHARS=255
    RAG_MAX_PDF_PAGES=50
    RAG_MAX_DOCUMENT_CHARS=2000000
    RAG_MAX_DOCUMENT_CHUNKS=2000
    RAG_RATE_LIMIT_PER_MINUTE=600
  )
  if [[ -n "${RAG_INGEST_API_KEY:-}" ]]; then
    local_env+=("RAG_INGEST_API_KEY=$RAG_INGEST_API_KEY")
  fi
  "${local_env[@]}" "$PYTHON_BIN" -m uvicorn app.main:app \
    --host "$API_BIND_HOST" --port "$API_BIND_PORT" >"$API_LOG" 2>&1 &
  API_PID=$!
}

start_docker_api() {
  [[ -f docker-compose.yml ]] || fail "docker-compose.yml was not found"
  command -v docker >/dev/null 2>&1 || fail "Docker is not installed"
  docker compose version >/dev/null 2>&1 || \
    fail "DEMO_RUNTIME=docker requires the Docker Compose v2 plugin"
  docker info >/dev/null 2>&1 || fail "The Docker daemon is not available"

  echo "[demo] Starting isolated Docker project $COMPOSE_PROJECT_NAME at $API_URL"
  COMPOSE_STARTED=1
  docker compose --project-name "$COMPOSE_PROJECT_NAME" up --detach --build
}

wait_for_api() {
  echo "[demo] Waiting up to ${DEMO_STARTUP_TIMEOUT}s for retrieval readiness"
  if ! "$PYTHON_BIN" - "$API_URL" "$DEMO_STARTUP_TIMEOUT" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

url = sys.argv[1].rstrip("/") + "/ready"
deadline = time.monotonic() + int(sys.argv[2])
last_error = "no response"

while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "ready":
                print(f"[demo] API ready (backend={payload.get('backend', 'unknown')})")
                raise SystemExit(0)
            last_error = f"unexpected payload: {payload!r}"
    except (OSError, ValueError, urllib.error.URLError) as exc:
        last_error = str(exc)
    time.sleep(0.5)

raise SystemExit(f"[demo] API did not become ready: {last_error}")
PY
  then
    if [[ "$COMPOSE_STARTED" == "1" ]]; then
      docker compose --project-name "$COMPOSE_PROJECT_NAME" logs \
        --no-color --tail=80 api qdrant >&2 || true
    fi
    return 1
  fi
}

run_demo_flow() {
  echo "[demo] Ingesting sample corpus"
  "$PYTHON_BIN" - <<'PY'
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

api_url = os.environ["API_URL"].rstrip("/")
api_key = os.environ.get("RAG_INGEST_API_KEY", "")
docs = sorted(
    path
    for path in Path("data/sample_docs").iterdir()
    if path.is_file() and path.suffix.lower() in {".md", ".txt"}
)
queries = [
    "How does hybrid retrieval combine dense and keyword methods?",
    "How should operators reduce cold start and embedding overhead?",
    "Which metrics are exposed in Prometheus format?",
]


def send(request: urllib.request.Request) -> dict:
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{request.full_url} returned HTTP {exc.code}: {body}") from exc


def ingest_file(path: Path) -> None:
    boundary = "----ragdemoboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: text/plain\r\n\r\n"
    ).encode("utf-8") + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("utf-8")
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if api_key:
        headers["X-API-Key"] = api_key

    payload = send(
        urllib.request.Request(
            f"{api_url}/ingest",
            data=body,
            method="POST",
            headers=headers,
        )
    )
    print(f"[demo] ingested {path.name}: chunks={payload['chunks']}")


if not docs:
    raise SystemExit("[demo] No .md or .txt documents found in data/sample_docs")
for document in docs:
    ingest_file(document)

print("\n[demo] Running hybrid retrieval queries")
for index, question in enumerate(queries, start=1):
    started = time.perf_counter()
    body = send(
        urllib.request.Request(
            f"{api_url}/query",
            data=json.dumps(
                {"question": question, "strategy": "hybrid", "top_k": 3}
            ).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    retrieval = body.get("retrieval")
    citations = body.get("citations")
    if not isinstance(retrieval, dict) or not isinstance(citations, list):
        raise RuntimeError("Query response is missing retrieval metadata or citations")

    sources = [item.get("source") for item in citations if item.get("source")]
    print(f"\nQuery {index}: {question}")
    print(
        "Retrieval: "
        f"strategy={retrieval.get('strategy')} "
        f"confidence={retrieval.get('confidence', 0.0):.3f} "
        f"abstained={retrieval.get('abstained')} "
        f"dense_candidates={retrieval.get('dense_candidates')} "
        f"lexical_candidates={retrieval.get('lexical_candidates')}"
    )
    print(f"Round-trip latency: {elapsed_ms:.2f} ms")
    print(body.get("answer", ""))
    print(f"Sources: {sources}")
PY
}

if api_is_responding; then
  fail "$API_URL is already serving traffic; choose another API_URL to avoid modifying it"
fi

if [[ "$DEMO_RUNTIME" == "docker" ]]; then
  start_docker_api
else
  start_local_api
fi

wait_for_api
run_demo_flow

echo "[demo] Success"
