from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "rag-api-eval-starter"
    environment: str = "dev"
    log_level: str = "INFO"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    fake_embeddings: bool = False
    fake_embedding_dim: int = 64
    vector_backend: str = "qdrant"
    qdrant_url: str | None = None
    qdrant_collection: str = "rag_documents"
    postgres_url: str | None = None
    pgvector_table: str = "rag_documents"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    hybrid_alpha: float = 0.7
    request_size_limit_mb: int = 5
    rate_limit_per_minute: int = 60

    model_config = SettingsConfigDict(env_prefix="RAG_")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def validate_settings(settings: Settings) -> None:
    supported_backends = {"qdrant", "pgvector"}
    if settings.vector_backend.lower() not in supported_backends:
        raise ValueError(f"Unsupported vector backend: {settings.vector_backend}")
    if settings.vector_backend.lower() == "pgvector" and not settings.postgres_url:
        raise ValueError("RAG_POSTGRES_URL is required when using pgvector backend")
