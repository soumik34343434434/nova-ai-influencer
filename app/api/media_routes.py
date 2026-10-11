from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import require_token

from app.db.session import SessionLocal
from app.services.content_service import latest_draft
from app.services.media_service import build_local_video

router = APIRouter(prefix="/media", tags=["media"], dependencies=[Depends(require_token)])

@router.post("/build-latest")
def build_latest_video():
    db = SessionLocal()
    try:
        draft = latest_draft(db)
        if not draft:
            raise HTTPException(status_code=404, detail="No draft content exists")
        try:
            output = build_local_video(draft)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return {"status": "created", "video": str(output)}
    finally:
        db.close()
