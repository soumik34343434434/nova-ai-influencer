from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.agents.content_agent import generate_content
from app.config import settings
from app.db.models import ContentItem

def create_draft(db: Session, trend_context: str = "") -> ContentItem:
    data = generate_content(trend_context)
    item = ContentItem(
        status="pending_approval" if settings.approval_required else "approved",
        title=data["title"],
        hook=data["hook"],
        script=data["script"],
        content_json=json.dumps(data),
        platform_caption=data["caption"],
        originality_score=data["originality_score"],
        risk_score=data["risk_score"],
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

def create_draft_from_data(db: Session, data: dict, metadata: dict | None = None) -> ContentItem:
    payload = dict(data)
    if metadata:
        payload["intelligence"] = metadata
    item = ContentItem(
        status="pending_approval" if settings.approval_required else "approved",
        title=payload["title"],
        hook=payload["hook"],
        script=payload["script"],
        content_json=json.dumps(payload),
        platform_caption=payload["caption"],
        originality_score=payload.get("originality_score", 0.87),
        risk_score=payload.get("risk_score", 0.03),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def latest_draft(db: Session):
    row = db.query(ContentItem).order_by(ContentItem.created_at.desc()).first()
    if not row:
        return None
    return {
        "id": row.id,
        "title": row.title,
        "hook": row.hook,
        "script": row.script,
        "caption": row.platform_caption,
        "originality_score": row.originality_score,
        "risk_score": row.risk_score,
    }
