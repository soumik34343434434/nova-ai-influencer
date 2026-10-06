"""Deterministic three-angle content engine for NOVA."""

from __future__ import annotations

import hashlib


ANGLES = ("workplace", "pov", "experiment")


def _topic(title: str) -> str:
    return title.rstrip(". ")[:110]


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def generate_variants(trend_title: str) -> list[dict]:
    topic = _topic(trend_title)
    options = [
        {
            "angle": "workplace",
            "title": f"NOVA's office discovers {topic}",
            "hook": f"My office discovered {topic}. Five minutes later, someone scheduled a meeting about it.",
            "script": (
                f"NOVA: Apparently everyone is talking about {topic}.\n\n"
                "Manager: Can we use it at work?\n\n"
                "NOVA: Yes.\n\n"
                "Manager: Great. Can you make a deck explaining why we need it?\n\n"
                "NOVA stares at camera.\n\n"
                "NOVA: We have invented a new form of productivity."
            ),
            "caption": "AI can automate work. Meetings remain beautifully manual. 🤖",
        },
        {
            "angle": "pov",
            "title": f"POV: your company discovers {topic}",
            "hook": f"POV: your company discovers {topic} and suddenly everyone becomes an AI strategist.",
            "script": (
                f"NOVA: Today the company discovered {topic}.\n\n"
                "Beat.\n\n"
                "NOVA checks the calendar.\n\n"
                "NOVA: I have six new meetings.\n\n"
                "NOVA looks at camera.\n\n"
                "NOVA: The future arrived with a calendar invite."
            ),
            "caption": "The future is here. It sent a meeting invite.",
        },
        {
            "angle": "experiment",
            "title": f"NOVA tests {topic}",
            "hook": f"I tested {topic} so you don't have to open fourteen tabs.",
            "script": (
                f"NOVA: Experiment: {topic}.\n\n"
                "NOVA tries it.\n\n"
                "NOVA: Useful.\n\n"
                "Beat.\n\n"
                "NOVA looks at the laptop.\n\n"
                "NOVA: Unexpected result. I now have seventeen tabs.\n\n"
                "NOVA: Science has a cost."
            ),
            "caption": "The real productivity metric: tabs still open.",
        },
    ]
    # Deterministically rotate angle order per topic so the system does not
    # always privilege one structure before the evaluator scores them.
    start = _seed(topic) % len(options)
    return [options[(start + i) % len(options)] for i in range(len(options))]
