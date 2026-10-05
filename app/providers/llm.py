import json
from pathlib import Path

import httpx

from app.config import settings

PERSONA = json.loads(Path("config/persona.json").read_text())

class LLMProvider:
    def generate(self, system: str, user: str) -> str:
        if settings.mock_llm:
            return json.dumps({
                "title": "AI is getting promoted before you",
                "hook": "Your office has one coworker who already replaced half their job with AI.",
                "script": "POV: the quiet coworker discovers one AI tool.\n\nNOVA: I automated the report.\nCoworker: How long did that take?\nNOVA: Seven minutes.\nCoworker: That's it?\nNOVA: No. Finding the seven-minute tool took three hours.\n\nBeat. NOVA looks at camera.\n\nNOVA: Technology is amazing. Search is still terrible.",
                "caption": "The productivity paradox: automate the task, then spend your life finding the automation.",
                "originality_score": 0.93,
                "risk_score": 0.04
            })

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing and MOCK_LLM=false")

        response = httpx.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={
                "model": settings.openai_model,
                "input": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
            },
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        return data["output"][0]["content"][0]["text"]

provider = LLMProvider()
