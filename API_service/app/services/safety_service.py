
from openai import OpenAI
from app.config import settings

client_guard = OpenAI(base_url=f"{settings.TOOL_HOST_IP}:8001/v1", api_key="EMPTY")

def safety_check(text: str, output=False) -> bool:
    prompt = (
        "Check safety of output: " if output else "Check safety of input: "
    ) + text
    try:
        res = client_guard.chat.completions.create(
            model="Qwen/Qwen3Guard-Gen-0.6B",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=10
        )
        return "SAFE" in res.choices[0].message.content.upper()
    except:
        return True
