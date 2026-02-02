import json
from collections.abc import Iterable
from typing import Protocol

import psycopg
from pgvector.psycopg import register_vector
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.core.config import get_settings
from app.services.embeddings import embed_query


class BaseStore(Protocol):
    def ensure_collection(self) -> None:
        ...

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        ...

    def search(self, vector: list[float], limit: int) -> list[dict]:
        ...


class QdrantStore:
    def __init__(self) -> None:
        settings = get_settings()
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url)
        else:
            self.client = QdrantClient(":memory:")
        self.collection = settings.qdrant_collection

    def ensure_collection(self) -> None:
        if self.collection in [c.name for c in self.client.get_collections().collections]:
            return
        vector_size = len(embed_query("dimension check"))
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            ),
        )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        points = [
            rest.PointStruct(id=point_id, vector=vector, payload=payload)
            for point_id, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], limit: int) -> list[dict]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
        return [{"payload": hit.payload or {}, "score": float(hit.score)} for hit in hits]


class PgvectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.postgres_url:
            raise ValueError("RAG_POSTGRES_URL is required for pgvector backend")
        self.dsn = settings.postgres_url
        self.table = settings.pgvector_table

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.dsn)
        register_vector(conn)
        return conn

    def ensure_collection(self) -> None:
        vector_size = len(embed_query("dimension check"))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    embedding VECTOR({vector_size}),
                    payload JSONB
                )
                """
            )
            conn.commit()

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        rows: Iterable[tuple[str, list[float], str]] = [
            (point_id, vector, json.dumps(payload))
            for point_id, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.table} (id, embedding, payload)
                VALUES (%s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload
                """,
                rows,
            )
            conn.commit()

    def search(self, vector: list[float], limit: int) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT payload, 1 - (embedding <=> %s) AS score
                FROM {self.table}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (vector, vector, limit),
            )
            rows = cur.fetchall()
        return [{"payload": payload or {}, "score": float(score)} for payload, score in rows]


def get_store() -> BaseStore:
    settings = get_settings()
    if settings.vector_backend.lower() == "pgvector":
        return PgvectorStore()
    return QdrantStore()
