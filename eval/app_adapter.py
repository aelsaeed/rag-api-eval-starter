from __future__ import annotations

from pathlib import Path
from types import TracebackType

from app.core.config import Settings
from app.services.answering import answer_from_retrieval
from app.services.ingest import ingest_document
from app.services.retrieval import RetrievalStrategy, retrieve
from app.services.storage import BaseStore, get_store
from eval.models import QueryResult, RetrievalHit


class ApplicationQueryAdapter:
    """Run the real application pipeline against an isolated in-memory corpus.

    Every application dependency receives the same fully explicit ``Settings``
    object, so ambient ``RAG_*`` variables cannot influence the evaluator.
    """

    def __init__(
        self,
        corpus_path: str | Path,
        *,
        strategy: RetrievalStrategy = "hybrid",
        min_relevance_score: float = 0.40,
    ) -> None:
        # BaseSettings accepts this runtime-only constructor option; the Pydantic
        # mypy plugin does not expose it in the generated Settings signature.
        self._settings = Settings(  # type: ignore[call-arg]
            _env_prefix="RAG_EVAL_ISOLATED_",
            app_name="rag-api-eval-starter",
            environment="test",
            log_level="INFO",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            fake_embeddings=True,
            fake_embedding_dim=64,
            vector_backend="qdrant",
            qdrant_url=None,
            qdrant_api_key=None,
            qdrant_collection="rag_eval_documents",
            postgres_url=None,
            pgvector_table="rag_documents",
            chunk_size=800,
            chunk_overlap=120,
            top_k=5,
            retrieval_strategy=strategy,
            candidate_multiplier=4,
            rrf_k=60,
            rrf_lexical_weight=0.65,
            min_relevance_score=min_relevance_score,
            lexical_scan_limit=1_000,
            request_size_limit_mb=5,
            max_filename_chars=255,
            max_pdf_pages=50,
            max_document_chars=2_000_000,
            max_document_chunks=2_000,
            rate_limit_per_minute=60,
            ingest_api_key=None,
        )
        self._strategy = strategy
        self._store: BaseStore = get_store(self._settings)
        self._closed = False
        try:
            self._store.ensure_collection()
            self._ingest_corpus(Path(corpus_path))
        except BaseException:
            self.close()
            raise

    @property
    def name(self) -> str:
        return f"application-{self._strategy}"

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def closed(self) -> bool:
        return self._closed

    def _ingest_corpus(self, corpus_path: Path) -> None:
        if not corpus_path.is_dir():
            raise ValueError(f"Corpus directory does not exist: {corpus_path}")
        documents = [
            path
            for path in sorted(corpus_path.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.suffix.casefold() in {".md", ".pdf", ".txt"}
        ]
        if not documents:
            raise ValueError(f"No supported documents found in {corpus_path}")
        for document in documents:
            ingest_document(
                document.name,
                document.read_bytes(),
                self._store,
                self._settings,
            )

    def query(self, question: str, *, k: int) -> QueryResult:
        if self._closed:
            raise RuntimeError("ApplicationQueryAdapter is closed")
        retrieval = retrieve(
            question,
            self._store,
            strategy=self._strategy,
            top_k=k,
            settings=self._settings,
        )
        answer = answer_from_retrieval(retrieval, settings=self._settings)
        retrieval_hits = tuple(
            RetrievalHit(
                source=str(hit.get("source") or ""),
                chunk_id=str(hit.get("chunk_id") or ""),
                text=str(hit.get("snippet") or ""),
                score=float(hit.get("score", 0.0)),
            )
            for hit in retrieval.hits
        )
        if answer.abstained:
            return QueryResult(
                answer="",
                retrieval_hits=retrieval_hits,
                citations=(),
                abstained=True,
            )

        citations = tuple(
            RetrievalHit(
                source=str(citation.get("source") or ""),
                chunk_id=str(citation.get("chunk_id") or ""),
                text=str(citation.get("snippet") or ""),
                score=float(citation.get("score", 0.0)),
            )
            for citation in answer.citations
        )
        return QueryResult(
            answer=answer.answer,
            retrieval_hits=retrieval_hits,
            citations=citations,
            abstained=False,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._store.close()
        self._closed = True

    def __enter__(self) -> ApplicationQueryAdapter:
        if self._closed:
            raise RuntimeError("ApplicationQueryAdapter is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()
