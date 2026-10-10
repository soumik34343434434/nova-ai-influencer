"""Every generated idea must be renderable and pass every pre-production gate.

Before this contract existed, the generator emitted legacy office actions with no
cartoon artwork, so every idea was dropped before the jury and no run could
reach rendering.
"""
import pytest

from app.agents.cartoon_factory import action_asset
from app.agents.cartoon_jury import CartoonIdeaJury
from app.agents.cartoon_quality import CartoonQuality
from app.agents.idea_generator_v2 import FORMAT_RECIPES, generate_ideas
from app.agents.script_engine import ScriptEngine
from app.agents.story_engine import StoryEngine

TREND_TITLES = (
    "Cats refusing to get up on Monday mornings",
    "Robot vacuum fails dance challenge",
    "new phone notification meme",
    "AI homework helper goes viral",
)


@pytest.fixture(scope="module")
def ideas():
    trends = [
        {
            "trend_id": f"trend-{index}",
            "title": title,
            # The generator uses at most eight formats per trend; rotate the
            # order so the pool as a whole exercises every format.
            "formats": tuple(list(FORMAT_RECIPES)[3 * index:] + list(FORMAT_RECIPES)[:3 * index]),
            "emotions": ("cute", "amusement", "surprise"),
            "opportunity_score": 0.5,
        }
        for index, title in enumerate(TREND_TITLES)
    ]
    return generate_ideas(trends, count=10_000)


def test_pool_covers_every_format_and_character(ideas):
    assert {idea["format_name"] for idea in ideas} == set(FORMAT_RECIPES)
    assert {idea["character"] for idea in ideas} == {"PIP", "MOMO", "BUBU"}


def test_every_beat_has_a_renderable_asset(ideas, monkeypatch):
    monkeypatch.delenv("NOVA_CARTOON_ANIMATION_BACKEND", raising=False)
    for idea in ideas:
        missing = [beat["asset_action"] for beat in idea["beats"] if action_asset(idea["character"], beat["asset_action"]) is None]
        assert not missing, (idea["idea_id"], missing)


def test_hooks_name_the_idea_character(ideas):
    for idea in ideas:
        others = {"PIP", "MOMO", "BUBU"} - {idea["character"]}
        assert not any(name in idea["hook"] for name in others), idea["hook"]


def test_every_idea_passes_story_jury_quality_and_script_gates(ideas):
    story, script, jury, quality = StoryEngine(), ScriptEngine(), CartoonIdeaJury(), CartoonQuality()
    for idea in ideas:
        candidate = dict(idea, caption=idea["title"])
        assert story.evaluate(candidate).passed, (idea["idea_id"], story.evaluate(candidate).blockers)
        assert jury.judge(candidate).passed, (idea["idea_id"], jury.judge(candidate).reasons)
        assert quality.evaluate(candidate, []).passed, (idea["idea_id"], quality.evaluate(candidate, []).reasons)
        script.build(story.build(candidate).to_dict())


def test_ideas_carry_their_own_hook_style(ideas):
    assert all(idea["hook_pattern"] for idea in ideas)
    assert len({idea["hook_pattern"] for idea in ideas}) >= 4
