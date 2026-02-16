from fastapi.testclient import TestClient


def test_health(monkeypatch) -> None:
    monkeypatch.setattr("app.services.storage.embed_query", lambda _: [0.0, 0.0])
    from app.main import app

    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_and_query(monkeypatch) -> None:
    monkeypatch.setattr("app.services.storage.embed_query", lambda _: [0.0, 0.0])
    monkeypatch.setattr(
        "app.services.ingest.embed_texts",
        lambda texts: [[0.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr("app.services.embeddings.embed_query", lambda _: [0.0, 0.0])
    monkeypatch.setattr("app.services.retrieval.embed_query", lambda _: [0.0, 0.0])

    from app.main import app

    file_content = b"This is a test document about Qdrant and hybrid retrieval."
    with TestClient(app) as client:
        response = client.post(
            "/ingest",
            files={"file": ("sample.txt", file_content, "text/plain")},
        )
        assert response.status_code == 200

        query_response = client.post("/query", json={"question": "What is Qdrant?"})
        assert query_response.status_code == 200
        payload = query_response.json()
        assert "answer" in payload
        assert "citations" in payload
