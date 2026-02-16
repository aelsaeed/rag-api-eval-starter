#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://127.0.0.1:8000}
DEMO_MODE=""
API_PID=""

cleanup() {
  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" >/dev/null 2>&1; then
    kill "$API_PID" >/dev/null 2>&1 || true
    wait "$API_PID" >/dev/null 2>&1 || true
  fi
  if [[ "$DEMO_MODE" == "docker" ]]; then
    docker compose down >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

start_api() {
  if [[ -f "docker-compose.yml" ]] && command -v docker >/dev/null 2>&1; then
    echo "[demo] Starting API with docker compose..."
    docker compose up -d --build
    DEMO_MODE="docker"
  else
    echo "[demo] Starting API with uvicorn (RAG_FAKE_EMBEDDINGS=1)..."
    RAG_FAKE_EMBEDDINGS=1 uvicorn app.main:app --host 0.0.0.0 --port 8000 >/tmp/rag-demo-api.log 2>&1 &
    API_PID=$!
    DEMO_MODE="uvicorn"
  fi
}

wait_for_api() {
  echo "[demo] Waiting for API readiness..."
  python - <<'PY'
import json
import time
import urllib.request

url = "http://127.0.0.1:8000/health"
for _ in range(60):
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") == "ok":
                print("[demo] API is healthy")
                raise SystemExit(0)
    except Exception:
        time.sleep(1)

raise SystemExit("[demo] API did not become healthy in time")
PY
}

run_demo_flow() {
  echo "[demo] Ingesting tiny sample corpus..."
  python - <<'PY'
import json
import os
import time
import urllib.request
from pathlib import Path

api_url = os.environ.get("API_URL", "http://127.0.0.1:8000")

docs = [
    Path("data/sample_docs/platform_overview.md"),
    Path("data/sample_docs/operations.md"),
    Path("data/sample_docs/observability.md"),
]
queries = [
    "How does hybrid retrieval combine dense and keyword methods?",
    "How should operators reduce cold start and embedding overhead?",
    "What observability data does the service expose?",
]

def ingest_file(path: Path) -> None:
    boundary = "----ragdemoboundary"
    payload = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
        f"Content-Type: text/plain\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{boundary}--\r\n".encode("utf-8")

    request = urllib.request.Request(
        f"{api_url}/ingest",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        parsed = json.loads(response.read().decode("utf-8"))
        print(f"[demo] ingested {path.name}: doc_id={parsed['doc_id']} chunks={parsed['chunks']}")

for doc in docs:
    ingest_file(doc)

print("\n[demo] Running 3 example queries...")
for index, query in enumerate(queries, start=1):
    started = time.perf_counter()
    payload = json.dumps({"question": query}).encode("utf-8")
    request = urllib.request.Request(
        f"{api_url}/query",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        body = json.loads(response.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - started) * 1000.0

    citations = body.get("citations", [])
    chunk_ids = [item.get("chunk_id") for item in citations if item.get("chunk_id")]

    print(f"\nQuery {index}: {query}")
    print(f"Latency: {latency_ms:.2f} ms")
    print("Answer:")
    print(body.get("answer", ""))
    print("Cited chunk ids:")
    print(chunk_ids if chunk_ids else "[]")
PY
}

start_api
wait_for_api
run_demo_flow

echo "[demo] ✅ Success"
