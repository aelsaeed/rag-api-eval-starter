import io
import os
import uuid
from pathlib import Path

from pypdf import PdfReader

from app.core.config import get_settings
from app.services.embeddings import embed_texts
from app.services.storage import BaseStore


def _split_text(text: str) -> list[str]:
    settings = get_settings()
    chunk_size = settings.chunk_size
    overlap = min(settings.chunk_overlap, max(0, chunk_size - 1))
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def _read_text_from_file(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md"}:
        return data.decode("utf-8", errors="ignore")
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError("Unsupported file type")


def ingest_document(filename: str, data: bytes, store: BaseStore) -> dict:
    text = _read_text_from_file(filename, data)
    doc_id = str(uuid.uuid4())
    chunks = _split_text(text)
    embeddings = embed_texts(chunks)

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
                "tokens": [token.lower() for token in chunk.split()],
                "source": os.path.basename(filename),
            }
        )

    store.upsert(ids, embeddings, payloads)
    return {"doc_id": doc_id, "chunks": len(chunks)}
