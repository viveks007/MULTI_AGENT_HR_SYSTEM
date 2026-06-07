import os

try:
    from groq import Groq
except ModuleNotFoundError:
    Groq = None

client = None
if Groq is not None:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate(prompt):
    if client is None:
        raise RuntimeError(
            "The groq package is not installed or failed to load. "
            "Install it and set GROQ_API_KEY to use the LLM integration."
        )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content