import httpx
from app.core.config import settings


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)

    return f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the context below.
If the answer is not in the context, say:
"I could not find the answer in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
""".strip()

async def generate_answer(question: str, context_chunks: list[str]) -> str:
    prompt = build_prompt(question, context_chunks)

    url = f"{settings.ollama_base_url}/api/chat"

    payload = {
        "model": settings.ollama_model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You answer questions using only the provided document context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        return data["message"]["content"]