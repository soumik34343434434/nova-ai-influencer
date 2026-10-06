from app.db.models import Base
from app.db.session import SessionLocal
from app.state_machine import DurableRun, Phase


def test_durable_run_transitions_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'state.db'}")
    # SessionLocal is configured at import time, so use the app's current DB and isolate via a fresh in-memory engine.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()

    run = DurableRun.start(db, {"trend": "AI agents"})
    run.transition(Phase.TREND_VALIDATED)
    run.transition(Phase.IDEA_CREATED)
    run.transition(Phase.SCRIPT_READY)
    run.fail("media provider unavailable")
    resumed = DurableRun.resume(db, run.run.id)
    assert resumed.snapshot().status == "failed"
    resumed.resume_after_failure(Phase.SCRIPT_READY)
    resumed.transition(Phase.MEDIA_READY)
    assert resumed.snapshot().phase is Phase.MEDIA_READY
