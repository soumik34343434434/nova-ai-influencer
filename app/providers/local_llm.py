"""Optional local LLM adapter. Network/cloud AI is not required."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx


class LocalLLM:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.getenv("NOVA_LOCAL_MODEL", "qwen3:1.7b")
        self.timeout = timeout

    def health(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            return response.is_success
        except Exception:
            return False

    def models(self) -> list[str]:
        try:
            data = httpx.get(f"{self.base_url}/api/tags", timeout=5).json()
            return [str(row.get("name", "")) for row in data.get("models", []) if row.get("name")]
        except Exception:
            return []

    def chat(self, system: str, user: str) -> str | None:
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return str(response.json().get("message", {}).get("content", "")).strip() or None
        except Exception:
            return None

    @staticmethod
    def _json_object(text: str) -> dict[str, Any] | None:
        clean = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None

    def chat_json(self, system: str, user: str) -> dict[str, Any] | None:
        text = self.chat(system, user)
        return self._json_object(text) if text else None


local_llm = LocalLLM()
