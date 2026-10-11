from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import require_token
from sqlalchemy.orm import Session

from app.db.models import ContentItem, Metric
from app.db.session import SessionLocal
from app.services.content_service import create_draft

router = APIRouter()

def db_dep():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/health")
def health():
    return {"ok": True, "service": "nova", "version": "0.2.0"}

@router.post("/content/generate", dependencies=[Depends(require_token)])
def generate(trend_context: str = "", db: Session = Depends(db_dep)):
    return create_draft(db, trend_context).__dict__

@router.get("/content/queue", dependencies=[Depends(require_token)])
def queue(db: Session = Depends(db_dep)):
    rows = db.query(ContentItem).order_by(ContentItem.id.desc()).all()
    return [r.__dict__ for r in rows]

@router.post("/content/{content_id}/approve", dependencies=[Depends(require_token)])
def approve(content_id: int, db: Session = Depends(db_dep)):
    row = db.get(ContentItem, content_id)
    if not row:
        raise HTTPException(404, "content not found")
    row.status = "approved"
    db.commit()
    return {"id": row.id, "status": row.status}

@router.post("/metrics", dependencies=[Depends(require_token)])
def add_metrics(payload: dict, db: Session = Depends(db_dep)):
    metric = Metric(**payload)
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric.__dict__
