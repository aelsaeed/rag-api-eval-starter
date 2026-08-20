from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qdrant_client.http import models as rest

from app.core.config import Settings
from app.services.storage import PgvectorStore, QdrantStore


def _connection(cursor: MagicMock) -> MagicMock:
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    return connection


def test_pgvector_bootstraps_extension_before_registering_vector_type() -> None:
    bootstrap_cursor = MagicMock()
    application_cursor = MagicMock()
    application_cursor.fetchall.return_value = [
        ("id", "text"),
        ("embedding", "vector(64)"),
        ("payload", "jsonb"),
    ]
    application_cursor.fetchone.return_value = (True,)
    bootstrap_connection = _connection(bootstrap_cursor)
    application_connection = _connection(application_cursor)
    events: list[str] = []
    bootstrap_cursor.execute.side_effect = lambda *_args: events.append("extension")

    settings = Settings(
        vector_backend="pgvector",
        postgres_url="postgresql://rag:rag@localhost/rag",
    )
    store = PgvectorStore(settings)

    with (
        patch(
            "app.services.storage.psycopg.connect",
            side_effect=[bootstrap_connection, application_connection],
        ) as connect,
        patch("app.services.storage.register_vector") as register,
        patch("app.services.storage.embed_query", return_value=[0.0] * 64),
    ):
        register.side_effect = lambda *_args: events.append("register")
        store.ensure_collection()

    assert events == ["extension", "register"]
    bootstrap_cursor.execute.assert_called_once_with("CREATE EXTENSION IF NOT EXISTS vector")
    bootstrap_connection.commit.assert_called_once_with()
    register.assert_called_once_with(application_connection)
    assert application_cursor.execute.call_count == 4
    assert connect.call_count == 2


def test_pgvector_converts_python_vectors_to_registered_numpy_values() -> None:
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    connection = _connection(cursor)
    settings = Settings(
        vector_backend="pgvector",
        postgres_url="postgresql://rag:rag@localhost/rag",
    )
    store = PgvectorStore(settings)

    with patch.object(store, "_connect", return_value=connection):
        store.upsert(
            ["chunk-1"],
            [[0.1, 0.2]],
            [{"doc_id": "doc-1", "chunk_id": "chunk-1", "tokens": ["term"]}],
        )
        store.dense_search([0.1, 0.2], limit=1)

    upsert_rows = cursor.executemany.call_args.args[1]
    search_params = cursor.execute.call_args.args[1]
    assert isinstance(upsert_rows[0][1], np.ndarray)
    assert isinstance(search_params[0], np.ndarray)
    assert search_params[0].dtype == np.float32


def test_pgvector_rejects_existing_table_with_wrong_dimension() -> None:
    bootstrap_cursor = MagicMock()
    application_cursor = MagicMock()
    application_cursor.fetchall.return_value = [
        ("id", "text"),
        ("embedding", "vector(32)"),
        ("payload", "jsonb"),
    ]
    settings = Settings(
        vector_backend="pgvector",
        postgres_url="postgresql://rag:rag@localhost/rag",
    )
    store = PgvectorStore(settings)

    with (
        patch(
            "app.services.storage.psycopg.connect",
            side_effect=[
                _connection(bootstrap_cursor),
                _connection(application_cursor),
            ],
        ),
        patch("app.services.storage.register_vector"),
        patch("app.services.storage.embed_query", return_value=[0.0] * 64),
        pytest.raises(ValueError, match=r"vector\(32\).*vector\(64\)"),
    ):
        store.ensure_collection()


def test_pgvector_rejects_table_without_unique_id_index() -> None:
    bootstrap_cursor = MagicMock()
    application_cursor = MagicMock()
    application_cursor.fetchall.return_value = [
        ("id", "text"),
        ("embedding", "vector(64)"),
        ("payload", "jsonb"),
    ]
    application_cursor.fetchone.return_value = (False,)
    settings = Settings(
        vector_backend="pgvector",
        postgres_url="postgresql://rag:rag@localhost/rag",
    )
    store = PgvectorStore(settings)

    with (
        patch(
            "app.services.storage.psycopg.connect",
            side_effect=[
                _connection(bootstrap_cursor),
                _connection(application_cursor),
            ],
        ),
        patch("app.services.storage.register_vector"),
        patch("app.services.storage.embed_query", return_value=[0.0] * 64),
        pytest.raises(ValueError, match="unique or primary-key index on id"),
    ):
        store.ensure_collection()


def test_external_qdrant_receives_configured_api_key() -> None:
    settings = Settings(
        qdrant_url="https://qdrant.example.test",
        qdrant_api_key="qdrant-secret",
    )

    with patch("app.services.storage.QdrantClient") as client:
        store = QdrantStore(settings)

    client.assert_called_once_with(
        url="https://qdrant.example.test",
        api_key="qdrant-secret",
    )
    assert store.collection == settings.qdrant_collection


def test_qdrant_rejects_named_vector_collection() -> None:
    settings = Settings(qdrant_collection="named_vectors")
    store = QdrantStore(settings)
    store.client.create_collection(
        collection_name=store.collection,
        vectors_config={
            "named": rest.VectorParams(
                size=settings.fake_embedding_dim,
                distance=rest.Distance.COSINE,
            )
        },
    )
    try:
        with pytest.raises(ValueError, match="named vectors"):
            store.ensure_collection()
    finally:
        store.close()


def test_qdrant_rejects_incompatible_distance_metric() -> None:
    settings = Settings(qdrant_collection="dot_vectors")
    store = QdrantStore(settings)
    store.client.create_collection(
        collection_name=store.collection,
        vectors_config=rest.VectorParams(
            size=settings.fake_embedding_dim,
            distance=rest.Distance.DOT,
        ),
    )
    try:
        with pytest.raises(ValueError, match="requires Cosine"):
            store.ensure_collection()
    finally:
        store.close()


def test_qdrant_failed_replacement_preserves_existing_points() -> None:
    store = QdrantStore(Settings(qdrant_collection="replace_failure"))
    try:
        with (
            patch.object(store.client, "upsert", side_effect=RuntimeError("write failed")),
            patch.object(store.client, "delete") as delete,
            pytest.raises(RuntimeError, match="write failed"),
        ):
            store.replace_document(
                "doc-1",
                ["doc-1-0"],
                [[0.0] * store.settings.fake_embedding_dim],
                [{"doc_id": "doc-1", "chunk_id": "doc-1-0", "text": "text"}],
            )

        delete.assert_not_called()
    finally:
        store.close()
