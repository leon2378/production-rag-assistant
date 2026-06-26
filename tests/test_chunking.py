from app.rag.chunking import chunk_text


def test_chunk_text_returns_chunks():
    text = "This is a test document. " * 200
    chunks = chunk_text(text, chunk_size=200, overlap=50)


    assert len(chunks) > 1
    assert chunks[0].chunk_id == 0
    assert isinstance(chunks[0].text, str)