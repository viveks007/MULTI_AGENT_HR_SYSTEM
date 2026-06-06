"""Response agent that generates a simple answer from retrieved context."""

from prompts.response_prompt import RESPONSE_PROMPT


class ResponseAgent:

    def __init__(self):
        self.prompt = RESPONSE_PROMPT

    def generate(self, query: str, retrieved_documents: list) -> str:
        if not query or not query.strip():
            return "Please provide a question or topic so I can help."

        if not retrieved_documents:
            return "I could not find any matching information in the knowledge base. Please refine your question."

        snippets = []
        for doc in retrieved_documents[:3]:
            text = doc.get("chunk_text") or str(doc)
            source = doc.get("metadata", {}).get("source", "unknown source")
            snippets.append(f"From {source}: {text[:300].strip()}")

        summary = "\n\n".join(snippets)
        return f"{self.prompt}\n\n{summary}"
