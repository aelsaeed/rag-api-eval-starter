import hashlib
import io
import re
from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfReader

from app.core.config import Settings, get_settings
from app.services.embeddings import embed_texts
from app.services.storage import BaseStore
from app.services.text import tokenize

_TEXT_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def normalize_source_name(filename: str, settings: Settings | None = None) -> str:
    """Return a bounded basename that is safe to repeat in stored chunk payloads."""

    settings = settings or get_settings()
    source_name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not source_name:
        raise ValueError("Document filename is empty")
    if len(source_name) > settings.max_filename_chars:
        raise ValueError(
            f"Document filename exceeds the {settings.max_filename_chars}-character limit"
        )
    if any(ord(character) < 32 for character in source_name):
        raise ValueError("Document filename contains control characters")
    return source_name


def _iter_text_units(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    """Yield sentence/paragraph units and window oversized units without a full split list."""

    normalized = text.strip()

    def window(unit_text: str) -> Iterator[str]:
        unit = unit_text.strip()
        if not unit:
            return
        if len(unit) <= chunk_size:
            yield unit
            return
        start = 0
        while start < len(unit):
            end = min(len(unit), start + chunk_size)
            chunk = unit[start:end].strip()
            if chunk:
                yield chunk
            if end == len(unit):
                break
            start = end - overlap

    segment_start = 0
    for boundary in _TEXT_BOUNDARY.finditer(normalized):
        yield from window(normalized[segment_start : boundary.start()])
        segment_start = boundary.end()
    yield from window(normalized[segment_start:])


def _split_text(text: str, settings: Settings | None = None) -> list[str]:
    settings = settings or get_settings()
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap

    chunks: list[str] = []
    current: list[str] = []

    def append_chunk(chunk: str) -> None:
        if len(chunks) >= settings.max_document_chunks:
            raise ValueError(f"Document exceeds the {settings.max_document_chunks}-chunk limit")
        chunks.append(chunk)

    for unit in _iter_text_units(text, chunk_size, overlap):
        candidate = "\n\n".join([*current, unit])
        if current and len(candidate) > chunk_size:
            append_chunk("\n\n".join(current))
            carry: list[str] = []
            for previous in reversed(current):
                possible = [previous, *carry]
                if len("\n\n".join(possible)) > overlap:
                    break
                carry = possible
            if carry and len("\n\n".join([*carry, unit])) > chunk_size:
                carry = []
            current = carry
        current.append(unit)

    if current:
        append_chunk("\n\n".join(current))
    return chunks


def _read_text_from_file(filename: str, data: bytes, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md"}:
        text = data.decode("utf-8", errors="ignore")
    elif ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            if len(reader.pages) > settings.max_pdf_pages:
                raise ValueError(f"PDF exceeds the {settings.max_pdf_pages}-page limit")
            page_texts: list[str] = []
            extracted_chars = 0
            for page in reader.pages:
                page_text = page.extract_text() or ""
                extracted_chars += len(page_text) + int(bool(page_texts))
                if extracted_chars > settings.max_document_chars:
                    raise ValueError(
                        f"Extracted text exceeds the {settings.max_document_chars}-character limit"
                    )
                page_texts.append(page_text)
            text = "\n".join(page_texts)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("PDF could not be parsed or decrypted") from exc
    else:
        raise ValueError("Unsupported file type")

    text = text.strip()
    if not text:
        raise ValueError("Document contains no extractable text")
    if len(text) > settings.max_document_chars:
        raise ValueError(
            f"Extracted text exceeds the {settings.max_document_chars}-character limit"
        )
    return text


def ingest_document(
    filename: str,
    data: bytes,
    store: BaseStore,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    source_name = normalize_source_name(filename, settings)
    if not data:
        raise ValueError("Document is empty")
    if len(data) > settings.request_size_limit_mb * 1024 * 1024:
        raise ValueError(f"Document exceeds the {settings.request_size_limit_mb} MB limit")

    text = _read_text_from_file(filename, data, settings)
    doc_id = hashlib.sha256(data).hexdigest()[:32]
    chunks = _split_text(text, settings)
    embeddings = embed_texts(chunks, settings=settings)

    ids = []
    payloads = []
    for index, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}-{index}"
        ids.append(chunk_id)
        payloads.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk,
                "tokens": tokenize(chunk),
                "source": source_name,
            }
        )

    store.replace_document(doc_id, ids, embeddings, payloads)
    return {"doc_id": doc_id, "chunks": len(chunks)}
