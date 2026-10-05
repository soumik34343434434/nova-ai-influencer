"""Orchestrates the autonomous daily content discovery cycle."""

from sqlalchemy.orm import Session

from app.agents.trend_agent import discover_trends, trend_to_dict
from app.services.content_service import create_draft

def run_content_cycle(db: Session, max_candidates: int = 3) -> dict:
    trends = discover_trends()
    selected = trends[:max_candidates]
    created = []

    if not selected:
        item = create_draft(db, "No live trend feed available. Create an evergreen original AI/tech idea.")
        return {"trends": [], "created": [item.id]}

    for trend in selected:
        context = (
            f"Trend title: {trend.title}\n"
            f"Category: {trend.category}\n"
            f"Why now: {trend.why_now}\n"
            "Rule: make an original scene and script; do not reuse source footage or wording."
        )
        item = create_draft(db, context)
        created.append(item.id)

    return {"trends": [trend_to_dict(t) for t in selected], "created": created}
