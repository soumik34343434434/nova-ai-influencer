import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.init import init_db
from app.db.session import SessionLocal
from app.services.content_service import create_draft

init_db()
db = SessionLocal()
try:
    item = create_draft(db, "AI tools, productivity, and office life.")
    print(f"Created content #{item.id}: {item.title}")
    print(f"Status: {item.status}")
    print(f"Hook: {item.hook}")
finally:
    db.close()
