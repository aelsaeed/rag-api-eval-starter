import json
import uuid
from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.core.config import Settings, get_settings
from app.services.embeddings import embed_query
from app.services.text import lexical_overlap_score


class BaseStore(Protocol):
    def ensure_collection(self) -> None: ...

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None: ...

    def replace_document(
        self,
        doc_id: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None: ...

    def dense_search(self, vector: list[float], limit: int) -> list[dict]: ...

    def keyword_search(self, tokens: list[str], limit: int) -> list[dict]: ...

    def delete_document(self, doc_id: str) -> None: ...

    def is_ready(self) -> bool: ...

    def close(self) -> None: ...


class QdrantStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if self.settings.qdrant_url:
            api_key = self.settings.qdrant_api_key
            self.client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=api_key.get_secret_value() if api_key is not None else None,
            )
        else:
            self.client = QdrantClient(":memory:")
        self.collection = self.settings.qdrant_collection

    def ensure_collection(self) -> None:
        vector_size = len(embed_query("dimension check", settings=self.settings))
        if self.client.collection_exists(self.collection):
            collection = self.client.get_collection(self.collection)
            configured_vectors = collection.config.params.vectors
            if isinstance(configured_vectors, dict):
                raise ValueError(
                    f"Collection {self.collection!r} uses named vectors; "
                    "the application requires one unnamed vector"
                )
            existing_size = getattr(configured_vectors, "size", None)
            existing_distance = getattr(configured_vectors, "distance", None)
            if existing_size is None or existing_distance is None:
                raise ValueError(
                    f"Collection {self.collection!r} has an unsupported vector configuration"
                )
            if getattr(configured_vectors, "multivector_config", None) is not None:
                raise ValueError(
                    f"Collection {self.collection!r} uses multivectors; "
                    "the application requires one vector per chunk"
                )
            if existing_size != vector_size:
                raise ValueError(
                    f"Collection {self.collection!r} uses {existing_size}-dimensional vectors; "
                    f"the configured embedder produces {vector_size}"
                )
            if existing_distance != rest.Distance.COSINE:
                raise ValueError(
                    f"Collection {self.collection!r} uses {existing_distance} distance; "
                    "the application requires Cosine distance"
                )
            if self.settings.qdrant_url and "tokens" not in collection.payload_schema:
                self._create_keyword_index()
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            ),
        )
        if self.settings.qdrant_url:
            self._create_keyword_index()

    def _create_keyword_index(self) -> None:
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="tokens",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        points = self._points(ids, vectors, payloads)
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    @staticmethod
    def _points(
        ids: list[str], vectors: list[list[float]], payloads: list[dict]
    ) -> list[rest.PointStruct]:
        points = []
        for point_id, vector, payload in zip(ids, vectors, payloads, strict=True):
            point_payload = {**payload, "chunk_id": point_id}
            try:
                qdrant_id: str | int = str(uuid.UUID(point_id))
            except ValueError:
                qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_id))
            points.append(rest.PointStruct(id=qdrant_id, vector=vector, payload=point_payload))
        return points

    def replace_document(
        self,
        doc_id: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        # Upsert first so failed writes preserve the previous document. Once the
        # new points are durable, remove only stale chunks from an older layout.
        self.client.upsert(
            collection_name=self.collection,
            points=self._points(ids, vectors, payloads),
            wait=True,
        )
        self.client.delete(
            collection_name=self.collection,
            points_selector=rest.FilterSelector(
                filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="doc_id",
                            match=rest.MatchValue(value=doc_id),
                        )
                    ],
                    must_not=[
                        rest.FieldCondition(
                            key="chunk_id",
                            match=rest.MatchAny(any=ids),
                        )
                    ],
                )
            ),
            wait=True,
        )

    def dense_search(self, vector: list[float], limit: int) -> list[dict]:
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=limit,
                with_payload=True,
            )
            hits = result.points
        else:
            hits = self.client.search(  # type: ignore[attr-defined]
                collection_name=self.collection,
                query_vector=vector,
                limit=limit,
                with_payload=True,
            )
        return [{"payload": hit.payload or {}, "score": float(hit.score)} for hit in hits]

    def keyword_search(self, tokens: list[str], limit: int) -> list[dict]:
        if not tokens:
            return []
        records, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="tokens",
                        match=rest.MatchAny(any=sorted(set(tokens))),
                    )
                ]
            ),
            limit=self.settings.lexical_scan_limit,
            with_payload=True,
            with_vectors=False,
        )
        scored: list[dict[str, Any]] = []
        for record in records:
            payload = record.payload or {}
            score = lexical_overlap_score(payload.get("tokens", []), tokens)
            if score > 0:
                scored.append({"payload": payload, "score": score})
        scored.sort(
            key=lambda item: (
                -item["score"],
                str(item["payload"].get("source", "")),
                str(item["payload"].get("chunk_id", "")),
            )
        )
        return scored[:limit]

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=rest.FilterSelector(
                filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="doc_id",
                            match=rest.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def is_ready(self) -> bool:
        try:
            self.client.get_collection(self.collection)
        except Exception:
            return False
        return True

    def close(self) -> None:
        self.client.close()


class PgvectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.postgres_url:
            raise ValueError("RAG_POSTGRES_URL is required for pgvector backend")
        self.dsn = self.settings.postgres_url
        self.table = self.settings.pgvector_table

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.dsn)
        register_vector(conn)
        return conn

    def ensure_collection(self) -> None:
        vector_size = len(embed_query("dimension check", settings=self.settings))
        # pgvector's Python type registration requires the extension to exist.
        # Bootstrap it with an unregistered connection before opening the normal
        # registered connection used by vector reads and writes.
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                CREATE TABLE IF NOT EXISTS {} (
                    id TEXT PRIMARY KEY,
                    embedding VECTOR({}),
                    payload JSONB
                )
                """
                ).format(sql.Identifier(self.table), sql.SQL(str(vector_size)))
            )
            cur.execute(
                """
                SELECT attribute.attname,
                       format_type(attribute.atttypid, attribute.atttypmod)
                FROM pg_attribute AS attribute
                WHERE attribute.attrelid = to_regclass(quote_ident(%s))
                  AND attribute.attname = ANY(%s::text[])
                  AND NOT attribute.attisdropped
                """,
                (self.table, ["id", "embedding", "payload"]),
            )
            actual_schema: dict[str, str] = dict(cur.fetchall())
            expected_schema = {
                "id": "text",
                "embedding": f"vector({vector_size})",
                "payload": "jsonb",
            }
            if actual_schema != expected_schema:
                raise ValueError(
                    f"Table {self.table!r} has incompatible columns {actual_schema}; "
                    f"expected {expected_schema}"
                )
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_index AS index_definition
                    JOIN pg_attribute AS id_column
                      ON id_column.attrelid = index_definition.indrelid
                     AND id_column.attname = 'id'
                     AND NOT id_column.attisdropped
                    WHERE index_definition.indrelid = to_regclass(quote_ident(%s))
                      AND index_definition.indisunique
                      AND index_definition.indisvalid
                      AND index_definition.indimmediate
                      AND index_definition.indpred IS NULL
                      AND index_definition.indexprs IS NULL
                      AND index_definition.indnkeyatts = 1
                      AND index_definition.indkey[0] = id_column.attnum
                )
                """,
                (self.table,),
            )
            unique_id_row = cur.fetchone()
            if not unique_id_row or not unique_id_row[0]:
                raise ValueError(
                    f"Table {self.table!r} must have a valid unique or primary-key index on id"
                )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} USING GIN ((payload -> 'tokens'))"
                ).format(
                    sql.Identifier(f"{self.table}_tokens_gin"),
                    sql.Identifier(self.table),
                )
            )
            conn.commit()

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        rows: Iterable[tuple[str, Any, str]] = [
            (point_id, np.asarray(vector, dtype=np.float32), json.dumps(payload))
            for point_id, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    """
                INSERT INTO {} (id, embedding, payload)
                VALUES (%s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload
                """
                ).format(sql.Identifier(self.table)),
                rows,
            )
            conn.commit()

    def replace_document(
        self,
        doc_id: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        rows: Iterable[tuple[str, Any, str]] = [
            (point_id, np.asarray(vector, dtype=np.float32), json.dumps(payload))
            for point_id, vector, payload in zip(ids, vectors, payloads, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {} WHERE payload ->> 'doc_id' = %s").format(
                    sql.Identifier(self.table)
                ),
                (doc_id,),
            )
            cur.executemany(
                sql.SQL(
                    """
                INSERT INTO {} (id, embedding, payload)
                VALUES (%s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload
                """
                ).format(sql.Identifier(self.table)),
                rows,
            )
            conn.commit()

    def dense_search(self, vector: list[float], limit: int) -> list[dict]:
        query_vector = np.asarray(vector, dtype=np.float32)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                SELECT payload, 1 - (embedding <=> %s) AS score
                FROM {}
                ORDER BY embedding <=> %s, payload ->> 'source', payload ->> 'chunk_id'
                LIMIT %s
                """
                ).format(sql.Identifier(self.table)),
                (query_vector, query_vector, limit),
            )
            rows = cur.fetchall()
        return [{"payload": payload or {}, "score": float(score)} for payload, score in rows]

    def keyword_search(self, tokens: list[str], limit: int) -> list[dict]:
        unique_tokens = sorted(set(tokens))
        if not unique_tokens:
            return []
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                WITH scored AS (
                    SELECT payload,
                           (
                               SELECT COUNT(DISTINCT token)::float
                               FROM jsonb_array_elements_text(payload -> 'tokens') AS terms(token)
                               WHERE token = ANY(%s::text[])
                           ) / %s AS score
                    FROM {}
                    WHERE payload -> 'tokens' ?| %s::text[]
                )
                SELECT payload, score
                FROM scored
                WHERE score > 0
                ORDER BY score DESC, payload ->> 'source', payload ->> 'chunk_id'
                LIMIT %s
                """
                ).format(sql.Identifier(self.table)),
                (unique_tokens, len(unique_tokens), unique_tokens, limit),
            )
            rows = cur.fetchall()
        return [{"payload": payload or {}, "score": float(score)} for payload, score in rows]

    def delete_document(self, doc_id: str) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {} WHERE payload ->> 'doc_id' = %s").format(
                    sql.Identifier(self.table)
                ),
                (doc_id,),
            )
            conn.commit()

    def is_ready(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT 1 FROM {} LIMIT 1").format(sql.Identifier(self.table)))
                cur.fetchone()
                return True
        except Exception:
            return False

    def close(self) -> None:
        return None


def get_store(settings: Settings | None = None) -> BaseStore:
    settings = settings or get_settings()
    if settings.vector_backend == "pgvector":
        return PgvectorStore(settings)
    return QdrantStore(settings)
