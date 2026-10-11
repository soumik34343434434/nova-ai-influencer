"""Research-to-idea behaviour: family safety, de-duplication, degraded mode, variety and gates.

Everything runs offline. History and queue files live in tmp_path, and no test relies
on the operator's environment variables.
"""
import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

import app.agents.cartoon_research as research
import app.cartoon_factory_orchestrator as orch_module
import app.trend_history as history
from app.agents.cartoon_jury import CartoonIdeaJury
from app.agents.cartoon_quality import CartoonQuality
from app.agents.cartoon_safety import public_text_problems, sanitize_topic, topic_problems
from app.agents.idea_generator_v2 import FORMAT_RECIPES, PROPS, generate_ideas
from app.agents.script_engine import _sentence
from app.cartoon_factory_orchestrator import CartoonFactoryOrchestrator
from app.trend_history import attach_historical_signals


UNSAFE_HEADLINES = [
    "Teen killed after phone explodes while charging",
    "Gunman kills 12 in school shooting",
    "Floods kill 12 as heavy rain lashes Kerala",
    "Onlyfans star arrested for drugs",
    "Beer pong tournament gets wild",
    "Minister says election result will be protested",
    "Bomb threat email sent to 40 schools",
    "Shah Rukh Khan reacts to viral AI video",
]
SAFE_HEADLINES = [
    "MOMO says whatever",
    "PIP fixes a minor glitch",
    "BUBU visits the chateau",
    "Minority scholarship results announced",
    "Robot vacuum goes viral after chasing a cat",
]


@pytest.mark.parametrize("headline", UNSAFE_HEADLINES)
def test_unsafe_or_real_person_headlines_are_not_usable_topics(headline):
    assert topic_problems(headline), headline


@pytest.mark.parametrize("headline", SAFE_HEADLINES)
def test_harmless_words_are_not_blocked(headline):
    # 'hate' used to match 'whatever' and 'chateau'; 'minor' matched 'minority' and 'minor glitch'.
    assert public_text_problems(headline) == [], headline
    assert topic_problems(headline) == [], headline


def test_unsafe_topic_is_replaced_by_a_stable_evergreen_premise():
    title, status = sanitize_topic("Teen killed after phone explodes", research.EVERGREEN_TITLES, "2026-10-10|gdelt")
    assert status.startswith("replaced:")
    assert title in research.EVERGREEN_TITLES
    assert sanitize_topic("Teen killed after phone explodes", research.EVERGREEN_TITLES, "2026-10-10|gdelt") == (title, status)
    assert sanitize_topic("Robot vacuum goes viral", research.EVERGREEN_TITLES, "x") == ("Robot vacuum goes viral", "ok")


def test_discover_report_screens_unsafe_news_and_counts_replacements(monkeypatch):
    feed = """<?xml version="1.0"?><rss version="2.0"><channel>
    <item><title>Gunman kills 12 in school shooting - NDTV</title><link>https://a</link>
      <pubDate>Sat, 10 Oct 2026 05:02:00 GMT</pubDate><source url="https://ndtv.com">NDTV</source></item>
    <item><title>Robot vacuum goes viral after chasing a cat - Times of India</title><link>https://b</link>
      <pubDate>Sat, 10 Oct 2026 06:12:00 GMT</pubDate><source url="https://toi.com">Times of India</source></item>
    </channel></rss>"""

    class Response:
        status_code = 200
        headers = {"content-type": "application/xml"}
        text = feed

        def raise_for_status(self):
            return None

    monkeypatch.setattr(research.httpx, "get", lambda url, **kw: Response())
    signals, status = research.discover_report(limit=30, timeout=1)
    titles = {s.title for s in signals}
    assert "Robot vacuum goes viral after chasing a cat" in titles
    assert all(not topic_problems(t) for t in titles)
    assert "Gunman kills 12 in school shooting" not in titles
    # Five Google News queries each return the same two items; the unsafe item is replaced five times.
    assert status["safety"] == "replaced:5"


def test_evergreen_pool_rotates_by_day_and_avoids_recent_topics():
    first_days = {research.evergreen_topics(f"2026-10-{d:02d}", count=1)[0]["title"] for d in range(1, 31)}
    assert len(first_days) >= 8, "30 days should not keep reproducing the same premise"
    day = "2026-10-11"
    assert research.evergreen_topics(day, count=8) == research.evergreen_topics(day, count=8)
    recent = [research.evergreen_topics(day, count=1)[0]["title"]]
    assert research.evergreen_topics(day, recent, count=1)[0]["title"] not in recent
    assert len(research.evergreen_topics(day, recent, count=12)) == 12


def test_every_evergreen_premise_is_safe_and_renders_as_a_pool():
    for topic in research.EVERGREEN_TOPICS:
        assert public_text_problems(topic["title"]) == []
        assert topic_problems(topic["title"]) == []
        assert set(topic["formats"]) <= set(FORMAT_RECIPES)


def test_research_stage_degrades_to_evergreen_and_records_why(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(orch_module, "OUTBOX", tmp_path / "publish_queue.json")
    monkeypatch.setattr(history, "HISTORY_FILE", tmp_path / "trend_history.json")
    monkeypatch.setattr(orch_module, "snapshot_trends", lambda rows: rows)
    queued = {"title": "BUBU and the sandwich: fails in the cutest way",
              "experiment_metadata": {"trend_id": "trend-old"}}
    (tmp_path / "publish_queue.json").write_text(json.dumps([queued]))
    failed = {"google_news:ai_gadgets": "error:ConnectError:blocked", "gdelt:global_news": "error:ConnectError:blocked"}
    monkeypatch.setattr(orch_module, "discover_report", lambda **kw: ([], failed))
    monkeypatch.setattr(orch_module, "discover_social_trends", lambda timeout=8.0: ([], {"reddit": "public_rss:aww=http_403"}))

    store = SimpleNamespace(set_payload=lambda run, **values: run.payload.update(values))
    orchestrator = CartoonFactoryOrchestrator.__new__(CartoonFactoryOrchestrator)
    orchestrator.plane = SimpleNamespace(store=store)
    run = SimpleNamespace(payload={})

    result = orchestrator._research(run)

    assert result["mode"] == "degraded_evergreen" and result["degraded"] is True
    assert result["evergreen_records"] > 0
    assert run.payload["research_status"]["mode"] == "degraded_evergreen"
    assert run.payload["research_status"]["sources"] == failed
    assert {row["title"] for row in run.payload["research_signals"]} <= set(research.EVERGREEN_TITLES)
    assert run.payload["recent_titles"] == [queued["title"]]
    assert run.payload["recent_trend_ids"] == ["trend-old"]
    assert "degraded" in capsys.readouterr().err


def test_research_stage_live_mode_is_not_flagged_degraded(monkeypatch, tmp_path):
    monkeypatch.setattr(orch_module, "OUTBOX", tmp_path / "publish_queue.json")
    monkeypatch.setattr(history, "HISTORY_FILE", tmp_path / "trend_history.json")
    monkeypatch.setattr(orch_module, "snapshot_trends", lambda rows: rows)
    live = research.ResearchSignal(title="Cats refuse to get up on Monday", source="google_news", category="home_tech")
    monkeypatch.setattr(orch_module, "discover_report", lambda **kw: ([live], {"google_news:home_tech": "ok:1"}))
    monkeypatch.setattr(orch_module, "discover_social_trends", lambda timeout=8.0: ([], {}))
    store = SimpleNamespace(set_payload=lambda run, **values: run.payload.update(values))
    orchestrator = CartoonFactoryOrchestrator.__new__(CartoonFactoryOrchestrator)
    orchestrator.plane = SimpleNamespace(store=store)
    run = SimpleNamespace(payload={})
    result = orchestrator._research(run)
    assert result["mode"] == "live" and result["degraded"] is False
    assert result["sources_ok"] == ["google_news:home_tech"]


def test_trend_stage_skips_trends_that_already_made_an_episode(monkeypatch):
    monkeypatch.setattr(orch_module, "build_strategy_memory", lambda: {"observations": 0})
    monkeypatch.setattr(orch_module, "rank_format_candidates", lambda rows, memory, limit: [])
    rows = [{"trend_id": f"trend-{i}", "title": f"topic {i}", "opportunity_score": 1.0 - i / 100} for i in range(8)]
    store = SimpleNamespace(set_payload=lambda run, **values: run.payload.update(values))
    orchestrator = CartoonFactoryOrchestrator.__new__(CartoonFactoryOrchestrator)
    orchestrator.plane = SimpleNamespace(store=store)
    run = SimpleNamespace(payload={"trend_signals_v2": rows, "recent_trend_ids": ["trend-0", "trend-1"]})
    orchestrator._trend_intelligence(run)
    kept = [row["trend_id"] for row in run.payload["ranked_signals_v2"]]
    assert "trend-0" not in kept and "trend-1" not in kept
    assert len(kept) == 6


def test_publish_text_is_checked_before_anything_is_queued():
    with pytest.raises(RuntimeError, match="family-safety"):
        orch_module._guard_public_text({"title": "Gunman kills 12", "caption": "x"})
    orch_module._guard_public_text({"title": "BUBU and the sandwich", "caption": "A short."})


# --- history: de-duplication and repeat penalty -------------------------------------

def test_history_compares_with_the_previous_run_not_two_runs_ago():
    past = [
        {"trend_id": "t", "views": 1000, "captured_at": "2026-10-08T00:00:00+00:00"},
        {"trend_id": "t", "views": 2000, "captured_at": "2026-10-09T00:00:00+00:00"},
    ]
    row = {"trend_id": "t", "title": "x", "views": 3000, "saturation_score": 0.1, "velocity_score": 0.1}
    out = attach_historical_signals([row], past)[0]
    assert out["historical_growth_rate"] == pytest.approx((3000 - 2000) / 2000, abs=1e-4)


def test_repeats_raise_saturation_instead_of_being_masked_by_the_in_run_proxy():
    past = [{"trend_id": "t", "views": 10, "captured_at": f"2026-10-0{d}T00:00:00+00:00"} for d in range(1, 4)]
    row = {"trend_id": "t", "title": "x", "views": 0, "saturation_score": 0.3, "velocity_score": 0.1}
    out = attach_historical_signals([row], past)[0]
    assert out["saturation_score"] > 0.3
    assert out["saturation_score"] == pytest.approx(0.3 + 0.35 * 3 / 8, abs=1e-3)


def test_recent_rows_returns_the_newest_snapshots(monkeypatch, tmp_path):
    rows = [{"trend_id": f"t{i}", "title": f"topic {i}"} for i in range(5)]
    monkeypatch.setattr(history, "HISTORY_FILE", tmp_path / "trend_history.json")
    (tmp_path / "trend_history.json").write_text(json.dumps(rows))
    assert [r["trend_id"] for r in history.recent_rows(2)] == ["t3", "t4"]


def test_rank_prefers_fresh_stories_over_stale_ones():
    now = datetime.now(timezone.utc)
    fresh = research.ResearchSignal(title="Robot cat goes viral", source="google_news", category="x",
                                    published_at=format_datetime(now - timedelta(hours=2)))
    stale = research.ResearchSignal(title="Robot dog goes viral", source="google_news", category="x",
                                    published_at=format_datetime(now - timedelta(days=10)))
    ranked = research.rank([stale, fresh])
    assert ranked[0].title == fresh.title


def test_rank_penalises_topics_seen_in_recent_history():
    signal = research.ResearchSignal(title="Robot cat goes viral", source="google_news", category="x")
    free = research.rank([signal])[0].score
    seen = research.rank([signal], recent_topics=["Robot cat goes viral today"])[0].score
    assert seen < free


# --- ideas: viewer-facing text, variety, scene contract -----------------------------

TREND_TITLES = (
    "Cats refusing to get up on Monday mornings",
    "Robot vacuum fails dance challenge",
    "new phone notification meme",
    "AI homework helper goes viral",
    "Teen killed after phone explodes",
)


@pytest.fixture(scope="module")
def ideas():
    trends = [
        {"trend_id": f"trend-{i}", "title": title,
         "formats": tuple(list(FORMAT_RECIPES)[3 * i:] + list(FORMAT_RECIPES)[:3 * i]),
         "emotions": ("cute", "amusement", "surprise"), "opportunity_score": 0.5}
        for i, title in enumerate(TREND_TITLES)
    ]
    return generate_ideas(trends, count=10_000)


def test_each_format_has_at_least_six_premises():
    for fmt, recipe in FORMAT_RECIPES.items():
        premises = recipe["premises"]
        assert len(premises) >= 6, fmt
        assert len({p.hook for p in premises}) == len(premises), fmt
        assert len({p.goal for p in premises}) == len(premises), fmt


def test_generated_pool_uses_every_premise_of_every_format(ideas):
    for fmt, recipe in FORMAT_RECIPES.items():
        used = {idea["premise_id"] for idea in ideas if idea["format_name"] == fmt}
        assert len(used) == len(recipe["premises"]), fmt


def test_titles_and_captions_are_viewer_facing(ideas):
    for idea in ideas:
        title = idea["title"]
        assert "_" not in title and "idea-" not in title and "trend" not in title.lower(), title
        assert "evergreen" not in title.lower(), title
        assert not public_text_problems(title, idea["hook"]), title
        assert idea["hook"] and len(idea["hook"].split()) >= 4
        assert not any(word in title for word in ("explodes", "killed", "Teen", "homework helper")), title


def test_no_headline_words_reach_public_copy(ideas):
    # Words that occur only in the research headlines above, never in curated copy.
    headline_words = {"cats", "refusing", "mornings", "vacuum", "homework", "helper", "explodes", "teen", "shah"}
    for idea in ideas:
        text = (idea["title"] + " " + idea["hook"] + " " + " ".join(b["voice"] for b in idea["beats"])).lower()
        assert not headline_words & set(text.replace(":", " ").replace(",", " ").split()), idea["title"]


def test_scene_contract_names_only_the_prop_on_screen(ideas):
    settings = {"kitchen", "bedroom", "classroom", "park", "bakery", "living_room", "garden", "studio"}
    label_to_key = {label: key for key, (_, label, _) in PROPS.items()}
    for idea in ideas:
        scene = idea["scene"]
        assert scene["setting"] in settings
        assert scene["prop_label"] in label_to_key
        assert scene["prop"] == PROPS[label_to_key[scene["prop_label"]]][0]
        assert scene["prop_label"] in idea["hook"].lower(), idea["hook"]
        assert 0 in scene["prop_beats"], idea["idea_id"]
        for index, beat in enumerate(idea["beats"]):
            named = [label for label in label_to_key if label in beat["voice"].lower()]
            if index in scene["prop_beats"]:
                assert named == [scene["prop_label"]], (idea["idea_id"], beat["voice"])
            else:
                assert scene["prop_label"] not in beat["voice"].lower()
                assert not named, (idea["idea_id"], beat["voice"])


def test_generated_ideas_pass_every_pre_production_gate(ideas):
    from app.agents.story_engine import StoryEngine
    from app.agents.script_engine import ScriptEngine

    story, script, jury, quality = StoryEngine(), ScriptEngine(), CartoonIdeaJury(), CartoonQuality()
    for idea in ideas[:120]:
        candidate = dict(idea, caption=idea["title"])
        assert story.evaluate(candidate).passed, idea["title"]
        assert jury.judge(candidate).passed, idea["title"]
        assert quality.evaluate(candidate, []).passed, idea["title"]
        script.build(story.build(candidate).to_dict())


def test_quality_gate_flags_unsafe_viewer_text_and_repeated_titles(ideas):
    candidate = dict(ideas[0])
    assert CartoonQuality().evaluate(candidate, []).safety == 1.0
    unsafe = dict(candidate, title="Gunman kills 12 in school shooting")
    evaluation = CartoonQuality().evaluate(unsafe, [])
    assert evaluation.safety == 0.0
    assert "forbidden_or_unsafe_signal" in evaluation.reasons
    repeated = CartoonQuality().evaluate(candidate, [candidate["title"]])
    assert "too_similar_to_recent_title" in repeated.reasons
    assert CartoonQuality().evaluate(candidate, ["Completely unrelated cooking show"]).novelty == 1.0


def test_jury_flags_unsafe_viewer_text(ideas):
    unsafe = dict(ideas[0], hook="A bomb went off during the parade!")
    result = CartoonIdeaJury().judge(unsafe)
    assert not result.passed
    assert "unsafe_or_reuse_signal" in result.reasons


# --- script: reaction punctuation -----------------------------------------------------

def test_reaction_mark_comes_from_the_last_sentence_only():
    assert _sentence("Wait, why? Turns out it was the wrong one", "reaction") == "Wait, why? Turns out it was the wrong one."
    assert _sentence("Instead, it was a reminder to relax?", "reaction") == "Instead, it was a reminder to relax?"
    assert _sentence("Why is this happening? It was literally a sticker.", "reaction") == "Why is this happening? It was literally a sticker."
