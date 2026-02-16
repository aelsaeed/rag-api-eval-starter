import hashlib
from collections.abc import Iterable
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model_name)


def _fake_embed(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    tokens = text.lower().split()
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        vector[idx] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    settings = get_settings()
    values = list(texts)
    if settings.fake_embeddings:
        return [_fake_embed(text, settings.fake_embedding_dim) for text in values]

    model = _model()
    embeddings = model.encode(values, normalize_embeddings=True)
    return [list(map(float, row)) for row in embeddings.tolist()]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
