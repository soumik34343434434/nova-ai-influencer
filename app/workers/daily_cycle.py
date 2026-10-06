from app.db.init import init_db
from app.db.session import SessionLocal
from app.services.automation_service import run_content_cycle

def main():
    init_db()
    db = SessionLocal()
    try:
        result = run_content_cycle(db)
        print(result)
    finally:
        db.close()

if __name__ == "__main__":
    main()
