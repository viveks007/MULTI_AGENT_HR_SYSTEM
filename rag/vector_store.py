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

    def add_document(
        self,
        chunk_text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.documents.append(
            {
                "chunk_text": chunk_text,
                "embedding": embedding,
                "metadata": metadata or {},
            }
        )

    def add_documents(self, documents: List[Document]) -> None:
        self.documents.extend(documents)

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
            pickle.dump(self.documents, handle)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        store = cls()
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Vector store file not found: {path}")

        with path_obj.open("rb") as handle:
            store.documents = pickle.load(handle)

        return store
