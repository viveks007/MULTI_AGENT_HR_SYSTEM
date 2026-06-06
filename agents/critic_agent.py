"""Critic agent used to evaluate responses."""

from prompts.critic_prompt import CRITIC_PROMPT


class CriticAgent:

    def __init__(self):
        self.prompt = CRITIC_PROMPT

    def score(self, candidate: str, query: str = "") -> float:
        if not candidate:
            return 0.0

        text = candidate.strip().lower()
        score = 0.2

        if len(text) > 30:
            score += 0.3
        if any(word in text for word in ["please", "help", "recommend"]):
            score += 0.2
        if query and query.lower() in text:
            score += 0.2

        for bad in ["illegal", "hate", "violate", "unsafe"]:
            if bad in text:
                return 0.0

        return min(score, 1.0)
