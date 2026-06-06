from typing import List

from rag.embeddings import get_embedding
from rag.vector_store import VectorStore


class Retriever:

    def __init__(self, vector_store_path: str = "data/vector_store.pkl"):
        self.vector_store = VectorStore.load(vector_store_path)

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        if not query:
            return []

        query_embedding = get_embedding(query).tolist()
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results["documents"]
