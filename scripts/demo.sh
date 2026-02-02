#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000}

curl -F "file=@data/sample_docs/platform_overview.md" "$API_URL/ingest"
curl -F "file=@data/sample_docs/security_notes.txt" "$API_URL/ingest"

curl -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the service do to protect against abuse?"}'
