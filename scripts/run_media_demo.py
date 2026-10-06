from app.db.init import init_db
from app.db.session import SessionLocal
from app.services.content_service import latest_draft
from app.services.media_service import build_local_video

def main():
    init_db()
    db = SessionLocal()
    try:
        draft = latest_draft(db)
        if not draft:
            raise SystemExit("No draft found. Run: python -m app.workers.daily_cycle")
        output = build_local_video(draft)
        print(output)
    finally:
        db.close()

if __name__ == "__main__":
    main()
