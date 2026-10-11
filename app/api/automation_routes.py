from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.auth import require_token
from app.api.routes import db_dep
from app.services.automation_service import run_content_cycle

router = APIRouter(dependencies=[Depends(require_token)])

@router.post("/automation/run")
def run_automation(db: Session = Depends(db_dep)):
    return run_content_cycle(db)
