"""Retriever placeholder combining embeddings + vector store."""

from .embeddings import embed

def retrieve(query: str, store, k=5):
    qvec = embed([query])[0]
    return store.search(qvec, k=k)
