import json

from app.config import settings
from app.providers.llm import provider, PERSONA
from app.providers.local_content import generate_local

SYSTEM = """You are the content director for an original fictional AI influencer.
Never imitate a real creator or celebrity. Use trends only as high-level inspiration.
Create an original short-form concept with a distinct premise, a strong opening hook,
one clear useful or comedic idea, and a memorable final beat.
Return strict JSON with: title, hook, script, caption, originality_score, risk_score.
"""

def generate_content(trend_context: str = "") -> dict:
    user = (
        f"Persona: {json.dumps(PERSONA)}\n"
        f"Trend context: {trend_context or 'No live trend. Create an evergreen original idea.'}"
    )

    raw = provider.generate(SYSTEM, user) if not settings.mock_llm else json.dumps(generate_local(trend_context))
    item = json.loads(raw)

    if item["originality_score"] < 0.80:
        raise ValueError("Rejected: originality score below 0.80")
    if item["risk_score"] > 0.20:
        raise ValueError("Rejected: content risk score too high")
    return item
