from app.agents.storyboard import build_storyboard


def test_storyboard_creates_multiple_visual_beats():
    draft = {
        "hook": "Your office installed an AI agent and somehow created three new meetings.",
        "script": "Manager: Can the AI do this?\n\nNOVA checks the phone and sees a new calendar invite.\n\nNOVA: The work is finished.\n\nNOVA stares at camera.",
    }
    board = build_storyboard(draft)
    assert len(board.shots) >= 4
    assert 10 <= board.total_duration <= 18
    assert board.shots[0].camera == "hook_close_up"
    assert board.shots[-1].action == "deadpan_camera_reaction"
    assert abs(sum(s.duration for s in board.shots) - board.total_duration) < 0.1
