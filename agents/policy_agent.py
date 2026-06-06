"""Policy agent that selects the next action based on context."""

from prompts.policy_prompt import POLICY_PROMPT


class PolicyAgent:

    def __init__(self):
        self.prompt = POLICY_PROMPT

    def decide(self, intent: str, retrieved_documents: list) -> dict:
        if intent == "unknown":
            return {"action": "ask_clarification", "reason": "User intent is unclear."}

        if not retrieved_documents:
            return {"action": "escalate", "reason": "No relevant knowledge base documents were found."}

        if intent in {"leave_request", "policy_question", "benefits_question", "payroll_question", "general_query"}:
            return {"action": "answer", "reason": "Relevant information is available to answer the query."}

        return {"action": "summary", "reason": "Provide a concise summary based on retrieved context."}
