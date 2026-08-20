from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.core.config import Settings, get_settings
from app.main import app, require_ingest_access


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("RAG_FAKE_EMBEDDINGS", "1")
    monkeypatch.delenv("RAG_QDRANT_URL", raising=False)
    monkeypatch.delenv("RAG_INGEST_API_KEY", raising=False)
    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_liveness_and_readiness_include_request_ids(client: TestClient) -> None:
    health = client.get("/health", headers={"x-request-id": "portfolio-check"})
    ready = client.get("/ready")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "environment": "dev"}
    assert health.headers["x-request-id"] == "portfolio-check"
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "backend": "qdrant"}


def test_ingest_is_idempotent_and_query_is_explainable(client: TestClient) -> None:
    file_content = b"Qdrant stores vectors. Hybrid retrieval also matches exact keywords."
    first = client.post(
        "/ingest",
        files={"file": ("sample.txt", file_content, "text/plain")},
    )
    second = client.post(
        "/ingest",
        files={"file": ("renamed.txt", file_content, "text/plain")},
    )

    assert first.status_code == 200
    assert first.json()["doc_id"] == second.json()["doc_id"]

    response = client.post(
        "/query",
        json={"question": "How does hybrid retrieval match exact keywords?", "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval"]["strategy"] == "hybrid"
    assert payload["retrieval"]["abstained"] is False
    assert 0.0 <= payload["retrieval"]["confidence"] <= 1.0
    assert payload["citations"][0]["keyword_rank"] == 1
    assert payload["citations"][0]["dense_rank"] == 1
    assert "sample.txt" not in payload["answer"]
    assert "renamed.txt" in payload["answer"]


def test_query_rejects_blank_question(client: TestClient) -> None:
    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 422
    assert response.headers.get("x-request-id")


def test_ingest_rejects_oversized_filename(client: TestClient) -> None:
    response = client.post(
        "/ingest",
        files={"file": (f"{'x' * 256}.txt", b"small body", "text/plain")},
    )

    assert response.status_code == 400
    assert "filename exceeds" in response.json()["message"]


def test_ingest_uses_the_normalized_filename_for_parsing(client: TestClient) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("nested\\document.txt  ", b"canonical source name", "text/plain")},
    )

    assert response.status_code == 200


def test_ingest_api_key_can_protect_mutations(monkeypatch) -> None:
    monkeypatch.setenv("RAG_FAKE_EMBEDDINGS", "1")
    monkeypatch.setenv("RAG_INGEST_API_KEY", "portfolio-secret")
    monkeypatch.delenv("RAG_QDRANT_URL", raising=False)
    get_settings.cache_clear()

    with TestClient(app) as protected_client:
        denied = protected_client.post(
            "/ingest",
            files={"file": ("sample.txt", b"protected corpus", "text/plain")},
        )
        allowed = protected_client.post(
            "/ingest",
            headers={"x-api-key": "portfolio-secret"},
            files={"file": ("sample.txt", b"protected corpus", "text/plain")},
        )

    get_settings.cache_clear()
    assert denied.status_code == 401
    assert denied.json()["code"] == "http_error"
    assert allowed.status_code == 200


def test_chunked_request_body_is_bounded_without_content_length(monkeypatch) -> None:
    monkeypatch.setenv("RAG_FAKE_EMBEDDINGS", "1")
    monkeypatch.setenv("RAG_REQUEST_SIZE_LIMIT_MB", "1")
    monkeypatch.delenv("RAG_QDRANT_URL", raising=False)
    get_settings.cache_clear()
    chunks = iter([b'{"question":"', b"x" * (1024 * 1024), b'"}'])

    with TestClient(app) as bounded_client:
        response = bounded_client.post(
            "/query",
            content=chunks,
            headers={"content-type": "application/json"},
        )

    get_settings.cache_clear()
    assert response.status_code == 413
    assert response.json()["code"] == "request_too_large"
    assert response.headers.get("x-request-id")


def test_non_ascii_ingest_key_is_rejected_without_server_error() -> None:
    request = Request(
        {
            "type": "http",
            "headers": [(b"x-api-key", "påss".encode("latin-1"))],
            "app": SimpleNamespace(
                state=SimpleNamespace(settings=Settings(ingest_api_key="expected-secret-1"))
            ),
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        require_ingest_access(request)

    assert exc_info.value.status_code == 401
