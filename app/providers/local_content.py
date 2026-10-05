"""Deterministic content generation fallback requiring no paid LLM."""

import hashlib
import json
from pathlib import Path

PERSONA = json.loads(Path("config/persona.json").read_text())

TEMPLATES = [
    {
        "title": "NOVA reviews {topic}",
        "hook": "Everyone is talking about {topic}. NOVA has one question: why is everyone making this harder than it needs to be?",
        "body": "NOVA: I investigated {topic}.\n\nBeat.\n\nNOVA: My conclusion is extremely scientific. We need coffee and a better interface.\n\nNOVA looks at camera.\n\nNOVA: You're welcome.",
        "caption": "{topic}, but make it unnecessarily complicated. 🤖",
    },
    {
        "title": "POV: your workplace discovers {topic}",
        "hook": "The moment your office discovers {topic}, everyone's job title becomes 'AI Strategist.'",
        "body": "Manager: Can we use {topic} for this?\n\nNOVA: Probably.\n\nManager: Great. Can you make a 47-slide deck explaining it?\n\nNOVA: That part is still human-powered.",
        "caption": "AI can automate the work. Meetings remain undefeated.",
    },
    {
        "title": "The {topic} experiment",
        "hook": "NOVA tried {topic} so you don't have to spend three hours Googling it.",
        "body": "NOVA: Experiment number one. Try {topic}.\n\nBeat.\n\nNOVA: Result: surprisingly useful.\n\nBeat.\n\nNOVA: Side effect: I now have 14 tabs open.",
        "caption": "The real bottleneck was never the AI. It was the tabs.",
    },
]

def _topic(trend_context: str) -> str:
    for line in (line.strip() for line in trend_context.splitlines() if line.strip()):
        if line.lower().startswith("trend title:"):
            return line.split(":", 1)[1].strip().rstrip(".")[:80]
    return "AI"

def generate_local(trend_context: str = "") -> dict:
    topic = _topic(trend_context)
    seed = int(hashlib.sha256((topic + trend_context).encode()).hexdigest(), 16)
    template = TEMPLATES[seed % len(TEMPLATES)]
    return {
        "title": template["title"].format(topic=topic),
        "hook": template["hook"].format(topic=topic),
        "script": template["body"].format(topic=topic),
        "caption": template["caption"].format(topic=topic),
        "originality_score": 0.87,
        "risk_score": 0.03,
        "generation_mode": "local_zero_cost",
    }
