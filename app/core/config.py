from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "rag-api-eval-starter"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    fake_embeddings: bool = True
    fake_embedding_dim: int = Field(default=64, ge=8, le=4096)
    vector_backend: Literal["qdrant", "pgvector"] = "qdrant"
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = Field(default="rag_documents", pattern=r"^[A-Za-z0-9_-]+$")
    postgres_url: str | None = None
    pgvector_table: str = Field(default="rag_documents", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    chunk_size: int = Field(default=800, ge=64, le=32_000)
    chunk_overlap: int = Field(default=120, ge=0, le=8_000)
    top_k: int = Field(default=5, ge=1, le=50)
    retrieval_strategy: Literal["dense", "lexical", "hybrid"] = "hybrid"
    candidate_multiplier: int = Field(default=4, ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1, le=1_000)
    rrf_lexical_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    min_relevance_score: float = Field(default=0.40, ge=0.0, le=1.0)
    lexical_scan_limit: int = Field(default=1_000, ge=10, le=10_000)
    request_size_limit_mb: int = Field(default=5, ge=1, le=100)
    max_filename_chars: int = Field(default=255, ge=32, le=1_024)
    max_pdf_pages: int = Field(default=50, ge=1, le=1_000)
    max_document_chars: int = Field(default=2_000_000, ge=1_000, le=20_000_000)
    max_document_chunks: int = Field(default=2_000, ge=1, le=20_000)
    rate_limit_per_minute: int = Field(default=60, ge=1, le=100_000)
    ingest_api_key: SecretStr | None = None

    model_config = SettingsConfigDict(env_prefix="RAG_")

    @field_validator("qdrant_api_key")
    @classmethod
    def validate_qdrant_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        secret = value.get_secret_value()
        if not secret.strip():
            raise ValueError("qdrant_api_key must not be blank when configured")
        if not secret.isascii():
            raise ValueError("qdrant_api_key must contain only ASCII characters")
        return value

    @field_validator("ingest_api_key")
    @classmethod
    def validate_ingest_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        secret = value.get_secret_value()
        if not secret.strip():
            raise ValueError("ingest_api_key must not be blank when configured")
        if len(secret) < 16:
            raise ValueError("ingest_api_key must contain at least 16 characters")
        if not secret.isascii():
            raise ValueError("ingest_api_key must contain only ASCII characters")
        return value

    @model_validator(mode="after")
    def validate_related_fields(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.chunk_overlap > self.chunk_size // 2:
            raise ValueError("chunk_overlap must not exceed half of chunk_size")
        if self.vector_backend == "pgvector" and not self.postgres_url:
            raise ValueError("postgres_url is required when using pgvector backend")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


def validate_settings(settings: Settings) -> None:
    """Retained as an explicit startup hook; Pydantic validates all settings eagerly."""

    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
