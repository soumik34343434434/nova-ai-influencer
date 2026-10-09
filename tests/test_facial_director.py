from app.agents.facial_director import build_facial_timeline


def test_facial_timeline_has_attention_then_expression_then_recovery():
    rows = build_facial_timeline("checks_phone")
    assert len(rows) == 13
    peak = max(range(len(rows)), key=lambda i: rows[i]["mouth_open"] + rows[i]["brow"])
    first_gaze = next(i for i, row in enumerate(rows) if row["gaze_lead"] > 0.2)
    assert first_gaze < peak
    blink = next(i for i, row in enumerate(rows) if row["blink"])
    assert blink > peak
    assert rows[-1]["emotion"] == "deadpan"


def test_facial_timeline_is_bounded_for_all_core_actions():
    for action in ("checks_phone", "sees_calendar", "reaches_for_coffee", "glances_at_many_tabs", "leans_toward_camera", "deadpan_camera_reaction"):
        for row in build_facial_timeline(action):
            assert 0.0 <= row["t"] <= 1.0
            assert 0.45 <= row["eye_open"] <= 1.35
            assert -0.4 <= row["brow"] <= 1.0
            assert 0.0 <= row["mouth_open"] <= 1.0
            assert 0.0 <= row["gaze_lead"] <= 1.0
            assert -1.0 <= row["gaze_x"] <= 1.0
            assert row["blink"] in (0.0, 1.0)
