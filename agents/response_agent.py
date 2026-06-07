"""Response agent that generates an answer using an LLM and retrieved context."""

from prompts.response_prompt import RESPONSE_PROMPT
from llm.groq_client import generate as llm_generate


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
            snippets.append(f"Source: {source}\n{text.strip()}")

        context = "\n\n".join(snippets)
        prompt = (
            f"{self.prompt}\n\n"
            f"User query:\n{query}\n\n"
            f"Retrieved knowledge base content:\n{context}\n\n"
            "Use this information to answer the query in a concise and professional way. "
            "If the content is insufficient, explain that more information is needed."
        )

        try:
            llm_response = llm_generate(prompt)
            if llm_response and llm_response.strip():
                return llm_response.strip()
        except Exception:
            pass

        # Fallback if LLM call fails
        summary = "\n\n".join(snippets)
        return (
            "Based on the retrieved HR documentation:\n\n"
            f"{summary}\n\n"
            "For more detailed information, please contact the HR department."
        )
