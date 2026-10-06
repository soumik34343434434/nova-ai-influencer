"""Local-AI-assisted cartoon episode director with deterministic fallback."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from app.agents.cartoon_factory import PREMISES, action_asset
from app.providers.local_llm import LocalLLM


ALLOWED_CHARACTERS = {"PIP", "MOMO", "BUBU"}


def _seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def _base_candidates(topic: str) -> list[dict[str, Any]]:
    ordered = sorted(PREMISES, key=lambda p: _seed(topic + p["slug"]))
    # Prefer premises whose curated action pack is already complete. This
    # prevents the director from selecting an attractive idea we cannot render.
    ready = [
        p for p in ordered
        if all(action_asset(p["character"], beat[1]) for beat in p["beats"])
    ]
    ordered = ready + [p for p in ordered if p not in ready]
    candidates = []
    for premise in ordered[: max(1, min(5, len(ordered)))]:
        beats = [
            {"beat": name, "asset_action": action, "voice": voice, "sfx": sfx}
            for name, action, voice, sfx in premise["beats"]
        ]
        candidates.append({
            "premise_slug": premise["slug"],
            "character": premise["character"],
            "title": premise["title"],
            "hook": premise["hook"],
            "beats": beats,
            "caption": f"{premise['title']} — funny original cartoon short.",
            "generation_mode": "deterministic_template",
        })
    return candidates


def _normalise_candidate(data: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    result = dict(fallback)
    result.update({k: data[k] for k in ("title", "hook", "caption") if isinstance(data.get(k), str)})
    # The model may rewrite dialogue, but it cannot swap the character:
    # keeping the supplied character preserves the curated asset pack.
    if isinstance(data.get("beats"), list) and len(data["beats"]) == 4:
        cleaned = []
        for src, base in zip(data["beats"], fallback["beats"]):
            row = dict(base)
            if isinstance(src, dict) and isinstance(src.get("voice"), str):
                row["voice"] = " ".join(src["voice"].split())
            cleaned.append(row)
        result["beats"] = cleaned
    return result


class CartoonDirector:
    SYSTEM = (
        "You write wholesome original cartoon shorts. Return ONLY JSON. "
        "Use one supplied premise and keep exactly four beats. No real people, "
        "no copyrighted characters, no dangerous imitation, no hateful or sexual content. "
        "Make the hook immediate, visual and specific. Escalate the visible problem twice, then land a short reversal or deadpan punchline. Favor concrete props and reactions over explanatory dialogue."
    )

    def __init__(self, llm: LocalLLM | None = None):
        self.llm = llm or LocalLLM()

    def candidates(self, topic: str, count: int = 3, strategy: str = "") -> list[dict[str, Any]]:
        base = _base_candidates(topic or "today")
        # Never spend local-model inference on a concept whose current asset
        # pack cannot render it. As the library grows, more ready concepts
        # naturally enter this pool.
        ready = [c for c in base if all(action_asset(c["character"], b["asset_action"]) for b in c["beats"])]
        base = (ready or base)[: max(1, min(count, 3))]
        if not self.llm.health() or self.llm.model not in self.llm.models():
            return base
        result = []
        for fallback in base:
            prompt = json.dumps({
                "topic": topic,
                "strategy": strategy,
                "base": fallback,
                "task": "Rewrite the dialogue and hook to make this topical and funnier, while preserving premise_slug, character, beat count and asset_action values. Apply the supplied strategy without changing the safe original-cartoon constraints.",
                "required": {"title": "string", "hook": "string", "character": fallback["character"], "beats": "array of 4 objects with beat, asset_action, voice, sfx", "caption": "string"},
            }, ensure_ascii=False)
            generated = self.llm.chat_json(self.SYSTEM, prompt)
            result.append(_normalise_candidate(generated, fallback) if generated else fallback)
        return result
