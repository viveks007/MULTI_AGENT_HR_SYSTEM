import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

Document = Dict[str, Any]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class VectorStore:

    def __init__(self) -> None:
        self.documents: List[Document] = []
        self.embedding_dim: int = 0

    def add_document(
        self,
        chunk_text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.embedding_dim == 0 and embedding is not None:
            self.embedding_dim = len(embedding)

        self.documents.append(
            {
                "chunk_text": chunk_text,
                "embedding": embedding,
                "metadata": metadata or {},
            }
        )

    def add_documents(self, documents: List[Document]) -> None:
        self.documents.extend(documents)
        if documents and self.embedding_dim == 0:
            first_embedding = documents[0].get("embedding")
            if first_embedding is not None:
                self.embedding_dim = len(first_embedding)

    def similarity_search(self, query_embedding: List[float], top_k: int = 3) -> Dict[str, Any]:
        scored = [
            {
                "score": cosine_similarity(query_embedding, document["embedding"]),
                "document": document,
            }
            for document in self.documents
        ]

        top_results = sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

        return {
            "documents": [item["document"] for item in top_results],
            "scores": [item["score"] for item in top_results],
        }

    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("wb") as handle:
            pickle.dump({"documents": self.documents, "embedding_dim": self.embedding_dim}, handle)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        store = cls()
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Vector store file not found: {path}")

        with path_obj.open("rb") as handle:
            data = pickle.load(handle)
            if isinstance(data, dict) and "documents" in data:
                store.documents = data["documents"]
                store.embedding_dim = data.get("embedding_dim", 0)
            else:
                store.documents = data
                if store.documents:
                    first_embedding = store.documents[0].get("embedding")
                    store.embedding_dim = len(first_embedding) if first_embedding is not None else 0

        return store

    def is_compatible(self, expected_dim: int) -> bool:
        if self.embedding_dim == 0:
            return True
        return self.embedding_dim == expected_dim
