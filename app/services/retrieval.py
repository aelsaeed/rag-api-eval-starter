from collections import Counter

from app.core.config import get_settings
from app.services.embeddings import embed_query
from app.services.storage import BaseStore


def _keyword_score(chunk_tokens: list[str], query_tokens: list[str]) -> float:
    if not chunk_tokens or not query_tokens:
        return 0.0
    chunk_counts = Counter(chunk_tokens)
    score = 0.0
    for token in query_tokens:
        score += 1.0 if token in chunk_counts else 0.0
    return score / len(query_tokens)


def hybrid_search(query: str, store: BaseStore) -> list[dict]:
    settings = get_settings()
    query_vector = embed_query(query)
    query_tokens = [token.lower() for token in query.split()]
    dense_hits = store.search(query_vector, limit=settings.top_k * 4)

    results = []
    for hit in dense_hits:
        payload = hit.get("payload", {})
        tokens = payload.get("tokens", [])
        keyword_score = _keyword_score(tokens, query_tokens)
        blended = settings.hybrid_alpha * float(hit.get("score", 0.0)) + (
            1 - settings.hybrid_alpha
        ) * keyword_score
        results.append(
            {
                "doc_id": payload.get("doc_id"),
                "chunk_id": payload.get("chunk_id"),
                "snippet": payload.get("text"),
                "score": blended,
                "dense_score": float(hit.get("score", 0.0)),
                "keyword_score": keyword_score,
                "source": payload.get("source"),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[: settings.top_k]
