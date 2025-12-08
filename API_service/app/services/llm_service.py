
from openai import OpenAI
from app.config import settings

client_llm = OpenAI(base_url=f"{settings.TOOL_HOST_IP}:8000/v1", api_key="EMPTY")
client_embed = OpenAI(base_url=f"{settings.TOOL_HOST_IP}:8002/v1", api_key="EMPTY")

def embed_text(text: str):
    res = client_embed.embeddings.create(
        input=[text], model="Qwen/Qwen3-Embedding-0.6B"
    )
    return res.data[0].embedding

def llm_generate(query: str, chunks: list[str]):
    context = "\n\n".join(chunks)
    msg = [
        {"role": "system", "content": settings.SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\nQuestion:{query}"}
    ]
    out = client_llm.chat.completions.create(
        model="Qwen/Qwen3-8B-FP8",
        messages=msg,
        temperature=0.1,
        max_tokens=200
    )
    return out.choices[0].message.content
