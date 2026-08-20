from unittest.mock import MagicMock

import pytest

from app.core.config import Settings
from app.services.ingest import _read_text_from_file, _split_text, ingest_document


def test_split_text_overlap() -> None:
    text = "a" * 2000
    chunks = _split_text(text)
    assert len(chunks) >= 2
    assert all(chunk for chunk in chunks)


def test_split_text_keeps_sentences_intact_when_they_fit() -> None:
    first = "Retrieval quality gates catch regressions before release."
    second = "Sentence-aware chunks keep evaluation anchors intact."
    settings = Settings(chunk_size=64, chunk_overlap=10)

    chunks = _split_text(f"{first} {second}", settings)

    assert chunks == [first, second]


def test_split_text_never_exceeds_configured_size_when_overlap_would_overflow() -> None:
    settings = Settings(chunk_size=64, chunk_overlap=12)
    text = f"{'A' * 10}. {'B' * 60}."

    chunks = _split_text(text, settings)

    assert len(chunks) == 2
    assert all(len(chunk) <= settings.chunk_size for chunk in chunks)


class RecordingStore:
    def __init__(self) -> None:
        self.replacements: list[tuple[str, list[str], list[dict]]] = []

    def replace_document(
        self,
        doc_id: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        del vectors
        self.replacements.append((doc_id, ids, payloads))


def test_ingest_is_content_addressed_and_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.ingest.embed_texts",
        lambda texts, **_: [[1.0, 0.0] for _ in texts],
    )
    store = RecordingStore()
    content = b"A stable document about retrieval quality gates."

    first = ingest_document("first.md", content, store)  # type: ignore[arg-type]
    second = ingest_document("renamed.md", content, store)  # type: ignore[arg-type]

    assert first["doc_id"] == second["doc_id"]
    assert [replacement[0] for replacement in store.replacements] == [
        first["doc_id"],
        first["doc_id"],
    ]
    assert store.replacements[0][1] == store.replacements[1][1]


def test_ingest_rejects_empty_documents() -> None:
    with pytest.raises(ValueError, match="empty"):
        ingest_document("empty.txt", b"", RecordingStore())  # type: ignore[arg-type]


def test_ingest_rejects_oversized_filename_before_embedding(monkeypatch) -> None:
    embed = MagicMock()
    monkeypatch.setattr("app.services.ingest.embed_texts", embed)
    settings = Settings(max_filename_chars=32)

    with pytest.raises(ValueError, match="filename exceeds the 32-character limit"):
        ingest_document(
            f"{'x' * 33}.txt",
            b"bounded source metadata",
            RecordingStore(),  # type: ignore[arg-type]
            settings,
        )

    embed.assert_not_called()


def test_pdf_text_limit_stops_extraction_before_later_pages(monkeypatch) -> None:
    first = MagicMock()
    second = MagicMock()
    unvisited = MagicMock()
    first.extract_text.return_value = "a" * 800
    second.extract_text.return_value = "b" * 300
    reader = MagicMock()
    reader.pages = [first, second, unvisited]
    monkeypatch.setattr("app.services.ingest.PdfReader", lambda _stream: reader)
    settings = Settings(max_document_chars=1_000)

    with pytest.raises(ValueError, match="character limit"):
        _read_text_from_file("large.pdf", b"pdf", settings)

    first.extract_text.assert_called_once_with()
    second.extract_text.assert_called_once_with()
    unvisited.extract_text.assert_not_called()


def test_malformed_pdf_is_reported_as_invalid_input() -> None:
    with pytest.raises(ValueError, match="could not be parsed"):
        _read_text_from_file("broken.pdf", b"not a PDF")


def test_chunk_count_limit_rejects_before_embedding(monkeypatch) -> None:
    embed = MagicMock()
    monkeypatch.setattr("app.services.ingest.embed_texts", embed)
    settings = Settings(
        chunk_size=64,
        chunk_overlap=0,
        max_document_chunks=2,
    )

    with pytest.raises(ValueError, match="exceeds the 2-chunk limit"):
        ingest_document(
            "many-chunks.txt",
            b"a" * 200,
            RecordingStore(),  # type: ignore[arg-type]
            settings,
        )

    embed.assert_not_called()


def test_chunk_count_limit_stops_a_pathological_long_unit_early(monkeypatch) -> None:
    embed = MagicMock()
    monkeypatch.setattr("app.services.ingest.embed_texts", embed)
    settings = Settings(
        chunk_size=64,
        chunk_overlap=32,
        max_document_chars=6_000_000,
        max_document_chunks=2,
    )

    with pytest.raises(ValueError, match="exceeds the 2-chunk limit"):
        ingest_document(
            "long-unit.txt",
            b"x" * (5 * 1024 * 1024 - 1),
            RecordingStore(),  # type: ignore[arg-type]
            settings,
        )

    embed.assert_not_called()
