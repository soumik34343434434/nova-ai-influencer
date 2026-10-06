from pathlib import Path

from app.agents.cartoon_animatic import render_animatic
from app.agents.cartoon_animatic_judge import AnimaticJudge
from app.agents.cartoon_format_intelligence import best_format, infer_patterns
from app.agents.cartoon_jury import CartoonIdeaJury


ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ROOT / "assets" / "cartoon" / "actions"


def _pip_candidate():
    return {
        "title": "PIP vs the Smart Kitchen",
        "hook": "PIP asked for one sandwich. Big mistake!",
        "character": "PIP",
        "caption": "Original cartoon",
        "beats": [
            {"beat": "setup", "asset_action": "press_button", "voice": "One sandwich, please.", "sfx": "button"},
            {"beat": "escalation", "asset_action": "shock", "voice": "TWELVE?!", "sfx": "chaos"},
            {"beat": "reaction", "asset_action": "overwhelmed", "voice": "Why did I press that?", "sfx": "overwhelmed"},
            {"beat": "punchline", "asset_action": "deadpan", "voice": "The kitchen misunderstood one.", "sfx": "punchline"},
        ],
    }


def test_format_intelligence_detects_a_visual_gadget_format():
    patterns = infer_patterns("Smart robot gadget creates unexpected kitchen chaos", "ai_gadgets")
    assert patterns
    assert patterns[0].confidence >= 0.35
    assert "robot" in patterns[0].evidence


def test_best_format_has_hook_escalation_and_payoff():
    pattern = best_format("Funny robot gadget fails")
    assert pattern.hook_pattern
    assert pattern.escalation_pattern
    assert pattern.payoff_pattern


def test_jury_rejects_weak_generic_gag():
    draft = _pip_candidate()
    draft["hook"] = "Something happened today."
    draft["beats"] = [
        {**draft["beats"][0], "voice": "Okay."},
        {**draft["beats"][1], "voice": "Okay."},
        {**draft["beats"][2], "voice": "Okay."},
        {**draft["beats"][3], "voice": "Okay."},
    ]
    result = CartoonIdeaJury().judge(draft)
    assert not result.passed
    assert any(score.score < 0.70 for score in result.judges)


def test_jury_accepts_specific_original_cartoon_structure():
    result = CartoonIdeaJury().judge(_pip_candidate())
    assert result.passed
    assert result.score >= 0.78


def test_animatic_renders_and_passes_visual_sanity(tmp_path):
    images = [ACTIONS / name for name in (
        "pip_press_button.png",
        "pip_shock.png",
        "pip_overwhelmed.png",
        "pip_deadpan.png",
    )]
    output = tmp_path / "animatic.mp4"
    render_animatic(images, ["One.", "Twelve?!", "Why?!", "Oh."], output)
    result = AnimaticJudge().inspect(output)
    assert result["passed"], result
    assert result["sample_frames"] >= 4
