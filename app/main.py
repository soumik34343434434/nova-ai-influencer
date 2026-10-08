from fastapi import FastAPI

from app.db.init import init_db
from app.api.routes import router
from app.api.automation_routes import router as automation_router
from app.api.media_routes import router as media_router
from app.api.n8n_routes import router as n8n_router

app = FastAPI(title="NOVA AI Influencer", version="0.2.0")

@app.on_event("startup")
def startup():
    init_db()

app.include_router(router, prefix="/api")
app.include_router(automation_router, prefix="/api")
app.include_router(media_router, prefix="/api")
app.include_router(n8n_router, prefix="/api")
