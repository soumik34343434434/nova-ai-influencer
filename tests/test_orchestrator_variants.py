from pathlib import Path


def test_process_drafts_selects_one_of_three_variants(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db.models import Base, ContentItem
    from app.orchestrator import Trend, Publisher, _process_drafts, rank_trends

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    publisher = Publisher()
    char = tmp_path / "nova.png"
    char.write_bytes(b"character")

    fake_video = tmp_path / "nova.mp4"
    monkeypatch.setattr(
        "app.orchestrator.build_video_from_draft",
        lambda *args, **kwargs: fake_video.write_bytes(b"video") or fake_video,
    )
    monkeypatch.setattr("app.orchestrator.Compatibility.check_video", lambda path: None)

    trend = Trend("AI workplace agents create meeting chaos", "test", category="ai")
    trend = rank_trends([trend])[0]
    drafts, media, blocked = _process_drafts(db, [trend], publisher, char)

    assert drafts == 3
    assert media == 1
    assert blocked == 0
    assert len(publisher.queue) == 1
    assert db.query(ContentItem).count() == 1
    stored = db.query(ContentItem).first()
    assert "variant_pool_size" in stored.content_json
    db.close()
