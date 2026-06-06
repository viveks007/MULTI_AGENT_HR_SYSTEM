"""Intent agent that infers user intent from text."""

from prompts.intent_prompt import INTENT_PROMPT


class IntentAgent:

    def __init__(self):
        self.prompt = INTENT_PROMPT
        self.intent_keywords = {
            "leave_request": ["leave", "vacation", "pto", "time off"],
            "policy_question": ["policy", "procedure", "guideline", "rule"],
            "benefits_question": ["benefits", "health insurance", "401k", "paid time off"],
            "payroll_question": ["salary", "paycheck", "bonus", "payroll"],
            "complaint": ["issue", "problem", "complain", "concern"],
            "greeting": ["hello", "hi", "good morning", "good afternoon"],
        }

    def predict_intent(self, text: str) -> str:
        if not text or not text.strip():
            return "unknown"

        normalized = text.lower()
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in normalized for keyword in keywords):
                return intent

        if "help" in normalized or "assist" in normalized:
            return "help_request"

        return "general_query"
