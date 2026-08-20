import pytest
from pydantic import ValidationError

from app.core.config import Settings


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chunk_size": 0}, "greater than or equal to 64"),
        ({"chunk_size": 64, "chunk_overlap": 64}, "smaller than chunk_size"),
        ({"chunk_size": 64, "chunk_overlap": 33}, "must not exceed half"),
        ({"top_k": 0}, "greater than or equal to 1"),
        ({"rrf_k": 0}, "greater than or equal to 1"),
        ({"rrf_lexical_weight": 1.1}, "less than or equal to 1"),
        ({"min_relevance_score": 1.1}, "less than or equal to 1"),
        ({"vector_backend": "unknown"}, "'qdrant' or 'pgvector'"),
    ],
)
def test_invalid_settings_fail_fast(overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(**overrides)
    assert message in str(exc_info.value)


def test_pgvector_requires_dsn() -> None:
    with pytest.raises(ValidationError, match="postgres_url is required"):
        Settings(vector_backend="pgvector", postgres_url=None)


@pytest.mark.parametrize("secret", ["", "short-secret"])
def test_configured_ingest_key_must_be_non_blank_and_long_enough(secret: str) -> None:
    with pytest.raises(ValidationError, match="ingest_api_key"):
        Settings(ingest_api_key=secret)


def test_configured_ingest_key_must_be_ascii() -> None:
    with pytest.raises(ValidationError, match="only ASCII"):
        Settings(ingest_api_key="é" * 16)


def test_configured_qdrant_key_must_not_be_blank() -> None:
    with pytest.raises(ValidationError, match="qdrant_api_key"):
        Settings(qdrant_api_key="  ")


def test_configured_qdrant_key_must_be_ascii() -> None:
    with pytest.raises(ValidationError, match="only ASCII"):
        Settings(qdrant_api_key="clé-secrète")
