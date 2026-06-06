"""Basic RAG tests."""

from rag.chunking import chunk_text

def test_chunking():
    text = "a" * 1024
    chunks = chunk_text(text, chunk_size=512)
    assert len(chunks) == 2
