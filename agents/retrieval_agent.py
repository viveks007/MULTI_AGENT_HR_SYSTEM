"""Retrieval agent that loads the vector store and returns relevant chunks."""

from typing import List

from rag.retriever import Retriever
from prompts.retrieval_prompt import RETRIEVAL_PROMPT


class RetrievalAgent:

    def __init__(self, vector_store_path: str = "data/vector_store.pkl"):
        self.prompt = RETRIEVAL_PROMPT
        self.retriever = Retriever(vector_store_path)

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        if not query or not query.strip():
            return []

        return self.retriever.retrieve(query, top_k=top_k)
