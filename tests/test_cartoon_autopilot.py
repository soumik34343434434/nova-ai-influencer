from app.agents.cartoon_director import CartoonDirector
from app.agents.cartoon_quality import CartoonQuality
from app.agents.cartoon_research import ResearchSignal, rank
from app.providers.local_llm import LocalLLM


def test_research_ranks_visual_fun_topics():
    rows = rank([
        ResearchSignal("Funny robot gadget causes kitchen chaos", "test", "home"),
        ResearchSignal("Long market report", "test", "finance"),
    ])
    assert rows[0].score >= rows[1].score


def test_local_llm_json_parser_strips_thinking():
    data = LocalLLM._json_object('<think>hidden</think> {"title":"ok"}')
    assert data == {"title": "ok"}


def test_director_fallback_produces_four_asset_backed_beats(monkeypatch):
    class Offline:
        def health(self): return False
        def models(self): return []
    candidates = CartoonDirector(Offline()).candidates("smart kitchen", 3)
    assert candidates
    assert all(len(x["beats"]) == 4 for x in candidates)
    assert all(x["character"] in {"PIP", "MOMO", "BUBU"} for x in candidates)


def test_cartoon_quality_rejects_unsafe_content():
    result = CartoonQuality().evaluate({
        "title": "Celebrity lookalike prank",
        "hook": "Watch this now",
        "character": "PIP",
        "beats": [{"asset_action":"press_button"}] * 4,
    })
    assert not result.passed
    assert result.safety == 0.0
