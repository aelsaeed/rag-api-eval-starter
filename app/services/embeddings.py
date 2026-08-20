import hashlib
from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from app.core.config import Settings, get_settings
from app.services.text import tokenize

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _model(model_name: str) -> "SentenceTransformer":
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Real embeddings require the optional ML dependencies; "
            "install with `pip install -e '.[ml]'` or set RAG_FAKE_EMBEDDINGS=1"
        ) from exc
    return SentenceTransformer(model_name)


def _fake_embed(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    tokens = tokenize(text)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        vector[idx] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_texts(texts: Iterable[str], settings: Settings | None = None) -> list[list[float]]:
    settings = settings or get_settings()
    values = list(texts)
    if settings.fake_embeddings:
        return [_fake_embed(text, settings.fake_embedding_dim) for text in values]

    model = _model(settings.embedding_model_name)
    embeddings: Any = model.encode(values, normalize_embeddings=True)
    return [list(map(float, row)) for row in embeddings.tolist()]


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    return embed_texts([text], settings=settings)[0]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
