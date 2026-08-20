from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from eval.models import QueryResult, RetrievalHit
from eval.text import tokens

_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "be",
    "can",
    "does",
    "for",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "provide",
    "should",
    "service",
    "the",
    "to",
    "what",
    "which",
    "why",
    "with",
}


def _lexical_tokens(text: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for token in tokens(text):
        if len(token) > 5 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 4 and token.endswith("ied"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        normalized.append(token)
    return tuple(normalized)


def load_corpus(path: str | Path) -> dict[str, str]:
    corpus_path = Path(path)
    documents: dict[str, str] = {}
    for document_path in sorted(corpus_path.iterdir(), key=lambda item: item.name):
        if not document_path.is_file() or document_path.suffix.casefold() not in {".md", ".txt"}:
            continue
        documents[document_path.name] = document_path.read_text(encoding="utf-8")
    if not documents:
        raise ValueError(f"No markdown or text documents found in {corpus_path}")
    return documents


def _split_text(text: str, *, chunk_size: int, overlap: int) -> tuple[str, ...]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return tuple(chunks)


class OfflineCorpusAdapter:
    """A deterministic lexical baseline used by the offline CLI and tests.

    This adapter intentionally imports no application or vector-store modules. An
    application adapter only needs to implement the same ``query(question, k)``
    contract to reuse the scorer and reports.
    """

    def __init__(
        self,
        documents: Mapping[str, str],
        *,
        chunk_size: int = 800,
        overlap: int = 120,
    ) -> None:
        if not documents:
            raise ValueError("documents must not be empty")
        chunks: list[RetrievalHit] = []
        for source, text in sorted(documents.items()):
            for index, chunk in enumerate(
                _split_text(text, chunk_size=chunk_size, overlap=overlap)
            ):
                digest = hashlib.sha256(f"{source}\0{index}\0{chunk}".encode()).hexdigest()[:16]
                chunks.append(
                    RetrievalHit(
                        source=source,
                        chunk_id=f"{source}:{index}:{digest}",
                        text=chunk,
                    )
                )
        if not chunks:
            raise ValueError("documents must contain text")
        self._chunks = tuple(chunks)
        self._chunk_tokens = tuple(Counter(_lexical_tokens(chunk.text)) for chunk in self._chunks)
        document_frequency: Counter[str] = Counter()
        for chunk_tokens in self._chunk_tokens:
            document_frequency.update(chunk_tokens.keys())
        total = len(self._chunks)
        self._idf = {
            token: math.log((total + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }

    @classmethod
    def from_path(cls, path: str | Path) -> OfflineCorpusAdapter:
        return cls(load_corpus(path))

    @property
    def name(self) -> str:
        return "offline-lexical-baseline"

    def query(self, question: str, *, k: int) -> QueryResult:
        if k <= 0:
            raise ValueError("k must be positive")
        query_tokens = tuple(
            token for token in _lexical_tokens(question) if token not in _STOPWORDS
        )
        if not query_tokens:
            return QueryResult(answer="", retrieval_hits=(), citations=(), abstained=True)

        query_counts = Counter(query_tokens)
        ranked: list[RetrievalHit] = []
        for index, chunk in enumerate(self._chunks):
            chunk_tokens = self._chunk_tokens[index]
            score = sum(
                self._idf.get(token, 0.0) * min(query_frequency, chunk_tokens.get(token, 0))
                for token, query_frequency in query_counts.items()
            )
            if score <= 0:
                continue
            ranked.append(
                RetrievalHit(
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=score,
                )
            )
        ranked.sort(key=lambda hit: (-hit.score, hit.source, hit.chunk_id))
        retrieval_hits = tuple(ranked[:k])
        if not retrieval_hits:
            return QueryResult(answer="", retrieval_hits=(), citations=(), abstained=True)

        citations = retrieval_hits[:2]

        answer = "Answer (extractive):\n" + "\n".join(
            f"- {citation.text}" for citation in citations[:2]
        )
        return QueryResult(
            answer=answer,
            retrieval_hits=retrieval_hits,
            citations=citations,
            abstained=False,
        )
