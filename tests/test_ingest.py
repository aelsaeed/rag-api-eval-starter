from app.services.ingest import _split_text


def test_split_text_overlap() -> None:
    text = "a" * 2000
    chunks = _split_text(text)
    assert len(chunks) >= 2
    assert all(chunk for chunk in chunks)
