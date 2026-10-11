"""Runway execution is opt-in, budget-capped, resumable, and produces one clip per shot."""
import io
import json
import shutil
import subprocess
import threading
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.providers import runway_video
from app.providers.runway_video import RunwayExecutionBlocked, RunwayVideoClient, execute_plan

FROZEN_NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
RUNWAY_ENV = (
    "NOVA_RUNWAY_EXECUTE", "RUNWAYML_API_SECRET", "NOVA_RUNWAY_MAX_CREDITS", "NOVA_RUNWAY_DAILY_CREDITS",
    "NOVA_RUNWAY_MODEL_MAP", "NOVA_RUNWAY_DOWNLOAD_HOSTS", "NOVA_RUNWAY_CLIPS_DIR", "NOVA_RUNWAY_PAID_REPAIRS",
    "NOVA_RUNWAY_LEDGER_RESTORED", "GITHUB_ACTIONS", "RUNWAY_API_BASE",
)


@pytest.fixture(autouse=True)
def _isolated_ledger(monkeypatch, tmp_path):
    """Every test gets its own ledger, a frozen clock, a reachable network and no operator settings."""
    monkeypatch.setattr(runway_video, "LEDGER_FILE", tmp_path / "ledger.json")
    monkeypatch.setattr(runway_video, "host_reachable", lambda host, timeout=8.0: True)
    monkeypatch.setattr(runway_video, "_utc_now", lambda: FROZEN_NOW)
    for name in RUNWAY_ENV:
        monkeypatch.delenv(name, raising=False)


def _plan(tmp_path, durations=(2.4, 3.1), prompt="shot {i} prompt"):
    image = tmp_path / "shot.png"
    if not image.exists():
        image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return {
        "provider": "runway",
        "shots": [
            {"shot_index": i, "model": "gen-4-turbo", "image": str(image), "duration": d,
             "aspect_ratio": "9:16", "prompt": prompt.format(i=i)}
            for i, d in enumerate(durations)
        ],
    }


class FakeRunway:
    """Runway stand-in. Tasks finish on their second poll unless listed in ``failing``."""

    def __init__(self, *, failing=(), post_errors=None):
        self.posts = []
        self.polls = {}
        self.failing = set(failing)  # shot numbers whose task reports FAILED
        self.post_errors = dict(post_errors or {})  # post number (1-based) -> exception to raise
        self.shot_of_task = {}

    def transport(self, method, url, payload):
        if method == "POST":
            self.posts.append(payload)
            if len(self.posts) in self.post_errors:
                raise self.post_errors.pop(len(self.posts))
            task_id = f"task-{len(self.posts)}"
            self.shot_of_task[task_id] = len(self.posts) - 1
            return {"id": task_id}
        task_id = url.rsplit("/", 1)[-1]
        self.polls[task_id] = self.polls.get(task_id, 0) + 1
        if self.shot_of_task.get(task_id) in self.failing:
            return {"status": "FAILED", "failure": "moderation"}
        if self.polls[task_id] < 2:
            return {"status": "RUNNING"}
        return {"status": "SUCCEEDED", "output": [f"https://cdn.example/{task_id}.mp4"]}

    def download(self, url, path):
        path.write_bytes(url.encode())


def _enable(monkeypatch, budget="100", daily=None):
    monkeypatch.setenv("NOVA_RUNWAY_EXECUTE", "true")
    monkeypatch.setenv("RUNWAYML_API_SECRET", "test-secret")
    monkeypatch.setenv("NOVA_RUNWAY_MAX_CREDITS", budget)
    if daily is not None:
        monkeypatch.setenv("NOVA_RUNWAY_DAILY_CREDITS", daily)


def _client(fake=None, downloader=None):
    fake = fake or FakeRunway()
    return fake, RunwayVideoClient("test-secret", transport=fake.transport, downloader=downloader or fake.download)


def test_execution_is_disabled_by_default(monkeypatch, tmp_path):
    with pytest.raises(RunwayExecutionBlocked, match="NOVA_RUNWAY_EXECUTE"):
        execute_plan(_plan(tmp_path), tmp_path)


def test_execution_needs_a_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("NOVA_RUNWAY_EXECUTE", "true")
    with pytest.raises(RunwayExecutionBlocked, match="RUNWAYML_API_SECRET"):
        execute_plan(_plan(tmp_path), tmp_path)


def test_budget_counts_billed_seconds_not_shot_seconds(monkeypatch, tmp_path):
    # Two short shots bill 5s each at 5 credits/s = 50 credits.
    _enable(monkeypatch, budget="49")
    with pytest.raises(RunwayExecutionBlocked, match="50 credits"):
        execute_plan(_plan(tmp_path), tmp_path)


def test_plan_executes_and_downloads_one_clip_per_shot(monkeypatch, tmp_path):
    _enable(monkeypatch, budget="50")
    fake, client = _client()
    result = execute_plan(_plan(tmp_path), tmp_path / "clips", client=client, poll_seconds=0)

    assert sorted(result["clips"]) == [0, 1]
    assert all(Path(p).exists() for p in result["clips"].values())
    first = fake.posts[0]
    assert first["model"] == "gen4_turbo"
    assert first["ratio"] == "720:1280"
    assert first["duration"] == 5
    assert first["promptImage"].startswith("data:image/png;base64,")
    assert result["estimated_credits"] == 50
    assert result["submitted"] == 2


def test_repair_plan_regenerates_only_named_shots(monkeypatch, tmp_path):
    _enable(monkeypatch)
    fake, client = _client()
    plan = _plan(tmp_path, durations=(2.0, 2.0, 2.0))
    repair = {"shots": [dict(plan["shots"][2], prompt="repair")]}
    result = execute_plan(repair, tmp_path / "repair", client=client, poll_seconds=0)
    assert list(result["clips"]) == [2]
    assert len(fake.posts) == 1


def test_unknown_models_need_an_explicit_mapping(monkeypatch, tmp_path):
    _enable(monkeypatch)
    plan = _plan(tmp_path)
    plan["shots"][0]["model"] = "kling-3-standard"
    plan["shots"] = plan["shots"][:1]
    fake, client = _client()
    with pytest.raises(RunwayExecutionBlocked, match="NOVA_RUNWAY_MODEL_MAP"):
        execute_plan(plan, tmp_path, client=client, poll_seconds=0)
    monkeypatch.setenv("NOVA_RUNWAY_MODEL_MAP", json.dumps({"kling-3-standard": "kling3_std"}))
    execute_plan(plan, tmp_path, client=client, poll_seconds=0)
    assert fake.posts[-1]["model"] == "kling3_std"


def test_failed_task_surfaces_the_failure(monkeypatch, tmp_path):
    client = RunwayVideoClient("s", transport=lambda m, u, p: {"id": "t"} if m == "POST" else {"status": "FAILED", "failure": "moderation"})
    with pytest.raises(RuntimeError, match="moderation"):
        client.wait("t", poll_seconds=0)
    assert runway_video.billed_seconds(5.0) == 5 and runway_video.billed_seconds(5.1) == 10


def test_long_director_prompts_are_compacted_not_cut_mid_sentence():
    """Runway rejects ~1000+ char prompts; the Director's full prompt is ~3.5k."""
    from app.agents.shot_performance import build_shot_performance_spec, performance_prompt
    from app.providers.ai_performance import build_runway_repair_request
    from app.providers.runway_video import PROMPT_LIMIT, compact_prompt

    spec = build_shot_performance_spec(action="shock", dialogue="Then twelve sandwiches appear!", duration=3.3, camera="wide")
    full = performance_prompt(spec)
    assert len(full) > PROMPT_LIMIT
    short = compact_prompt(full)
    assert len(short) <= PROMPT_LIMIT and short.endswith(".")
    for label in ("Performance arc:", "Action:", "Gaze:", "Avoid:", "Preserve:"):
        assert label in short
    repair = build_runway_repair_request({"prompt": full}, {"reasons": ["possible_pose_popping"], "repair_shots": [1]})
    assert "Targeted repair:" in compact_prompt(repair["prompt"])
    assert compact_prompt("Short prompt.") == "Short prompt."


def test_compact_prompt_keeps_the_timing_and_orientation_guidance():
    """The planned length is 2.8s but Runway renders 5s: the timing must survive compaction."""
    from app.agents.shot_performance import build_shot_performance_spec, performance_prompt
    from app.providers.runway_video import PROMPT_LIMIT, compact_prompt, shot_prompt

    spec = build_shot_performance_spec(action="shock", dialogue="Then twelve sandwiches appear!", duration=2.8, camera="wide")
    request = {"prompt": performance_prompt(spec), "duration": 2.8}
    text = compact_prompt(shot_prompt(request))
    assert len(text) <= PROMPT_LIMIT
    assert text.startswith("Animate the existing character. Timing: the action completes by 2.8s of this 5s clip, "
                           "then the character holds the final pose until the end.")
    assert "exactly" not in text
    assert "Orientation lock:" in text
    assert "Performance arc:" in text


def test_shot_payload_bills_the_clip_length_and_says_where_the_action_ends(tmp_path):
    request = _plan(tmp_path, durations=(2.8,), prompt="Animate the existing character for exactly 2.8s. Action: wave.")["shots"][0]
    payload = runway_video.shot_payload(request, {"gen-4-turbo": "gen4_turbo"})
    assert payload["duration"] == 5
    assert payload["promptText"].startswith("Animate the existing character. Timing: the action completes by 2.8s of this 5s clip")
    assert "Action: wave." in payload["promptText"]


def test_blank_caps_mean_unset(monkeypatch, tmp_path):
    """`.env.example` ships NOVA_RUNWAY_DAILY_CREDITS= blank; that must mean the episode cap, not 0."""
    monkeypatch.setenv("NOVA_RUNWAY_MAX_CREDITS", "100")
    monkeypatch.setenv("NOVA_RUNWAY_DAILY_CREDITS", "")
    assert runway_video.quote(_plan(tmp_path), reachable=lambda h: True)["daily_cap"] == 100.0
    _enable(monkeypatch, budget="100", daily="")
    fake, client = _client()
    execute_plan(_plan(tmp_path, durations=(2.0,)), tmp_path / "a", client=client, poll_seconds=0, run_id="ep")
    assert len(fake.posts) == 1


@pytest.mark.parametrize("value", ["nan", "inf", "-5", "lots"])
def test_non_finite_or_negative_caps_are_rejected(monkeypatch, tmp_path, value):
    _enable(monkeypatch, budget=value)
    fake, client = _client()
    with pytest.raises(RunwayExecutionBlocked, match="NOVA_RUNWAY_MAX_CREDITS"):
        execute_plan(_plan(tmp_path, durations=(2.0,)), tmp_path, client=client, poll_seconds=0)
    assert fake.posts == []


def test_episode_cap_counts_what_the_episode_already_spent(monkeypatch, tmp_path):
    _enable(monkeypatch, budget="60")
    fake, client = _client()
    plan = _plan(tmp_path, durations=(2.0,))  # 25 credits
    execute_plan(plan, tmp_path / "a", client=client, poll_seconds=0, run_id="ep-1")
    execute_plan(plan, tmp_path / "b", client=client, poll_seconds=0, run_id="ep-1")  # 50 total
    with pytest.raises(RunwayExecutionBlocked, match="already spent 50"):
        execute_plan(plan, tmp_path / "c", client=client, poll_seconds=0, run_id="ep-1", kind="repair")
    assert len(fake.posts) == 2
    assert runway_video.credits_spent(run_id="ep-1") == 50
    assert [row["kind"] for row in json.loads(runway_video.LEDGER_FILE.read_text())] == ["generate", "generate"]


def test_daily_cap_spans_episodes(monkeypatch, tmp_path):
    _enable(monkeypatch, budget="100", daily="30")
    fake, client = _client()
    plan = _plan(tmp_path, durations=(2.0,))
    execute_plan(plan, tmp_path / "a", client=client, poll_seconds=0, run_id="ep-1")
    with pytest.raises(RunwayExecutionBlocked, match="NOVA_RUNWAY_DAILY_CREDITS is 30"):
        execute_plan(plan, tmp_path / "b", client=client, poll_seconds=0, run_id="ep-2")
    assert len(fake.posts) == 1


def test_unreachable_download_host_blocks_before_any_spend(monkeypatch, tmp_path):
    """A clip that cannot be downloaded is wasted credits, so nothing is submitted."""
    _enable(monkeypatch)
    fake, client = _client()
    blocked = runway_video.DEFAULT_DOWNLOAD_HOSTS[0]
    with pytest.raises(RunwayExecutionBlocked, match=f"no credits spent.*{blocked}"):
        execute_plan(_plan(tmp_path), tmp_path, client=client, poll_seconds=0,
                     reachable=lambda host: host != blocked)
    assert fake.posts == []
    assert runway_video.credits_spent() == 0


def test_missing_start_image_blocks_before_any_spend(monkeypatch, tmp_path):
    _enable(monkeypatch)
    fake, client = _client()
    plan = _plan(tmp_path)
    plan["shots"][1]["image"] = str(tmp_path / "missing.png")
    with pytest.raises(RunwayExecutionBlocked, match="shot 1: start image missing"):
        execute_plan(plan, tmp_path, client=client, poll_seconds=0)
    assert fake.posts == []


def test_quote_reports_cost_and_problems_without_spending(monkeypatch, tmp_path):
    monkeypatch.setenv("NOVA_RUNWAY_MAX_CREDITS", "100")
    result = runway_video.quote(_plan(tmp_path), run_id="ep-9", reachable=lambda host: False)
    assert result["total_credits"] == 50
    assert [row["billed_seconds"] for row in result["shots"]] == [5, 5]
    assert any("cannot reach" in problem for problem in result["problems"])
    assert result["can_execute"] is False
    assert runway_video.credits_spent() == 0


def test_quote_says_when_execution_would_be_refused(monkeypatch, tmp_path):
    """The default environment cannot spend: the quote must say so, as the gate does."""
    plan = _plan(tmp_path)
    quoted = runway_video.quote(plan, reachable=lambda host: True)
    assert quoted["can_execute"] is False
    assert any("NOVA_RUNWAY_EXECUTE" in problem for problem in quoted["problems"])
    assert any("RUNWAYML_API_SECRET" in problem for problem in quoted["problems"])
    with pytest.raises(RunwayExecutionBlocked):
        runway_video.execution_gate(plan)
    _enable(monkeypatch, budget="100")
    assert runway_video.quote(plan, reachable=lambda host: True)["can_execute"] is True
    runway_video.execution_gate(plan)


def test_quote_prices_unknown_models_as_a_problem(monkeypatch, tmp_path):
    _enable(monkeypatch)
    plan = _plan(tmp_path)
    plan["shots"][0]["model"] = "veo-3"
    quoted = runway_video.quote(plan, reachable=lambda host: True)
    assert any("no credit rate for model veo-3" in problem for problem in quoted["problems"])
    assert quoted["can_execute"] is False


def test_corrupt_ledger_fails_closed(monkeypatch, tmp_path):
    """A torn or unreadable ledger used to read as zero spend; it must now block."""
    _enable(monkeypatch, budget="100")
    runway_video.record_spend(run_id="ep", shot=0, task_id="t0", credits=100, kind="generate", model="gen4_turbo")
    data = runway_video.LEDGER_FILE.read_text()
    runway_video.LEDGER_FILE.write_text(data[: len(data) // 2])
    fake, client = _client()
    with pytest.raises(RunwayExecutionBlocked, match="corrupt"):
        execute_plan(_plan(tmp_path, durations=(2.0,)), tmp_path, client=client, poll_seconds=0)
    assert fake.posts == []
    with pytest.raises(RunwayExecutionBlocked, match="corrupt"):
        runway_video.credits_spent()
    assert any("corrupt" in problem for problem in runway_video.quote(_plan(tmp_path), reachable=lambda h: True)["problems"])


def test_ledger_that_is_not_a_list_fails_closed(monkeypatch, tmp_path):
    runway_video.LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    runway_video.LEDGER_FILE.write_text('{"credits": 0}')
    with pytest.raises(RunwayExecutionBlocked, match="not a list"):
        runway_video.credits_spent()
    runway_video.LEDGER_FILE.write_text("")
    with pytest.raises(RunwayExecutionBlocked, match="corrupt"):
        runway_video.credits_spent()


def test_missing_ledger_is_an_empty_ledger(monkeypatch):
    assert not runway_video.LEDGER_FILE.exists()
    assert runway_video.credits_spent() == 0


def test_ci_refuses_to_spend_without_the_restored_ledger(monkeypatch, tmp_path):
    """A cold Actions cache must not look like a fresh budget."""
    _enable(monkeypatch, budget="100")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    plan = _plan(tmp_path, durations=(2.0,))
    with pytest.raises(RunwayExecutionBlocked, match="missing on GitHub Actions"):
        runway_video.execution_gate(plan)
    monkeypatch.setenv("NOVA_RUNWAY_LEDGER_RESTORED", "true")
    assert runway_video.execution_gate(plan)["estimated_credits"] == 25
    monkeypatch.delenv("NOVA_RUNWAY_LEDGER_RESTORED")
    runway_video.LEDGER_FILE.write_text("[]")
    runway_video.execution_gate(plan)


def test_ledger_writes_are_atomic(monkeypatch):
    """A failure while writing leaves the previous ledger intact and no temp file behind."""
    runway_video.record_spend(run_id="ep", shot=0, task_id="t0", credits=25, kind="generate", model="m")
    before = runway_video.LEDGER_FILE.read_text()

    def interrupted_replace(src, dst):
        raise OSError("disk went away")

    real_replace = runway_video.os.replace
    monkeypatch.setattr(runway_video.os, "replace", interrupted_replace)
    with pytest.raises(OSError):
        runway_video.record_spend(run_id="ep", shot=1, task_id="t1", credits=25, kind="generate", model="m")
    monkeypatch.setattr(runway_video.os, "replace", real_replace)
    assert runway_video.LEDGER_FILE.read_text() == before
    assert sorted(p.name for p in runway_video.LEDGER_FILE.parent.iterdir()) == ["ledger.json", "ledger.json.lock"]


def test_concurrent_writers_keep_every_row(monkeypatch):
    """Four threads recording 25 shots each used to keep 4 of the 100 rows."""
    def writer(worker):
        for shot in range(25):
            runway_video.record_spend(run_id="race", shot=shot, task_id=f"{worker}-{shot}", credits=1,
                                      kind="generate", model="m")

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rows = json.loads(runway_video.LEDGER_FILE.read_text())
    assert len(rows) == 100
    assert runway_video.credits_spent(run_id="race") == 100


def test_concurrent_reservations_cannot_overspend_the_cap(monkeypatch):
    """Two episodes passing the gate together used to both be allowed past the daily cap."""
    results = []

    def reserve(i):
        try:
            runway_video._reserve_spend(run_id=f"ep-{i}", shot=0, model="m", credits=25, kind="generate",
                                        episode_cap=1000, daily_cap=100)
            results.append("ok")
        except RunwayExecutionBlocked:
            results.append("blocked")

    threads = [threading.Thread(target=reserve, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count("ok") == 4
    assert runway_video.credits_spent() == 100


def test_resumed_plan_never_pays_twice(monkeypatch, tmp_path):
    """A download blip on shot 2 must not cost a second generation for shots 1 and 2 on retry."""
    _enable(monkeypatch, budget="100")
    fake = FakeRunway()
    lost = {"once": True}

    def flaky_download(url, path):
        if url.endswith("task-3.mp4") and lost["once"]:  # shot 2 (the third task)
            lost["once"] = False
            raise RuntimeError("connection reset while downloading")
        fake.download(url, path)

    _, client = _client(fake, downloader=flaky_download)
    plan = _plan(tmp_path, durations=(2.0, 2.0, 2.0, 2.0))  # 4 x 25 = 100 credits
    with pytest.raises(RuntimeError, match="connection reset"):
        execute_plan(plan, tmp_path / "gen", client=client, poll_seconds=0, run_id="ep-1")
    assert len(fake.posts) == 3

    result = execute_plan(plan, tmp_path / "gen", client=client, poll_seconds=0, run_id="ep-1")
    assert len(fake.posts) == 4
    assert sorted(result["clips"]) == [0, 1, 2, 3]
    assert all(Path(p).stat().st_size > 0 for p in result["clips"].values())
    assert runway_video.credits_spent(run_id="ep-1") == 100
    reused = {row["shot"]: row["reused"] for row in result["tasks"]}
    assert reused == {0: True, 1: True, 2: False, 3: False}

    again = execute_plan(plan, tmp_path / "gen", client=client, poll_seconds=0, run_id="ep-1")
    assert len(fake.posts) == 4 and again["submitted"] == 0


def test_failed_task_is_resubmitted_for_that_shot_only(monkeypatch, tmp_path):
    _enable(monkeypatch, budget="150")
    fake = FakeRunway(failing={1})
    _, client = _client(fake)
    plan = _plan(tmp_path, durations=(2.0, 2.0, 2.0))
    with pytest.raises(RuntimeError, match="moderation"):
        execute_plan(plan, tmp_path / "gen", client=client, poll_seconds=0, run_id="ep-1")
    fake.failing.clear()
    result = execute_plan(plan, tmp_path / "gen", client=client, poll_seconds=0, run_id="ep-1")
    assert sorted(result["clips"]) == [0, 1, 2]
    assert len(fake.posts) == 4  # shots 0 and 1 first, then shot 1 again and shot 2
    assert runway_video.credits_spent(run_id="ep-1") == 100  # the failed task still counts


def test_pending_task_is_polled_not_resubmitted(monkeypatch, tmp_path):
    """A submitted task whose wait timed out is polled on the next attempt."""
    _enable(monkeypatch, budget="50")
    fake = FakeRunway()
    timed_out = {"once": True}

    class SlowClient(RunwayVideoClient):
        def wait(self, task_id, **kwargs):
            if timed_out["once"]:
                timed_out["once"] = False
                raise TimeoutError("still rendering")
            return super().wait(task_id, **kwargs)

    client = SlowClient("test-secret", transport=fake.transport, downloader=fake.download)
    plan = _plan(tmp_path, durations=(2.0,))
    with pytest.raises(TimeoutError):
        execute_plan(plan, tmp_path / "gen", client=client, poll_seconds=0, run_id="ep-1")
    result = execute_plan(plan, tmp_path / "gen", client=client, poll_seconds=0, run_id="ep-1")
    assert len(fake.posts) == 1
    assert result["tasks"][0]["reused"] is False and result["clips"]


def test_submission_without_a_task_id_blocks_resubmission(monkeypatch, tmp_path):
    """If the request may have reached Runway, the retry must not guess and pay again."""
    _enable(monkeypatch, budget="100")
    fake = FakeRunway(post_errors={1: ConnectionError("timed out after the request was sent")})
    _, client = _client(fake)
    plan = _plan(tmp_path, durations=(2.0,))
    with pytest.raises(ConnectionError):
        execute_plan(plan, tmp_path / "gen", client=client, poll_seconds=0, run_id="ep-1")
    with pytest.raises(RunwayExecutionBlocked, match="no recorded task id"):
        execute_plan(plan, tmp_path / "gen", client=client, poll_seconds=0, run_id="ep-1")
    assert len(fake.posts) == 1
    assert runway_video.credits_spent(run_id="ep-1") == 25


def test_refused_request_releases_its_reservation(monkeypatch, tmp_path):
    """An HTTP error from Runway means no task was created, so the credits are not counted."""
    _enable(monkeypatch, budget="100")
    refusal = urllib.error.HTTPError("https://api", 400, "bad request", {}, io.BytesIO(b"{}"))
    fake = FakeRunway(post_errors={1: refusal})
    _, client = _client(fake)
    plan = _plan(tmp_path, durations=(2.0,))
    with pytest.raises(urllib.error.HTTPError):
        execute_plan(plan, tmp_path / "gen", client=client, poll_seconds=0, run_id="ep-1")
    assert runway_video.credits_spent(run_id="ep-1") == 0
    execute_plan(plan, tmp_path / "gen", client=client, poll_seconds=0, run_id="ep-1")
    assert len(fake.posts) == 2 and runway_video.credits_spent(run_id="ep-1") == 25


def test_a_changed_plan_never_reuses_old_clips(monkeypatch, tmp_path):
    _enable(monkeypatch, budget="100")
    fake, client = _client()
    out = tmp_path / "gen"
    execute_plan(_plan(tmp_path, durations=(2.0,)), out, client=client, poll_seconds=0, run_id="ep-1")
    changed = execute_plan(_plan(tmp_path, durations=(2.0,), prompt="new {i}"), out, client=client,
                           poll_seconds=0, run_id="ep-1")
    assert len(fake.posts) == 2
    assert changed["tasks"][0]["reused"] is False
    assert len({p.parent for p in out.rglob("shot_00_*.mp4")}) == 2


def test_corrupt_run_manifest_blocks_instead_of_guessing(monkeypatch, tmp_path):
    _enable(monkeypatch, budget="100")
    fake, client = _client()
    plan = _plan(tmp_path, durations=(2.0,))
    run_dir = runway_video._run_dir(tmp_path / "gen", runway_video.plan_sha256(plan))
    run_dir.mkdir(parents=True)
    (run_dir / runway_video.RUN_MANIFEST).write_text("{not json")
    with pytest.raises(RunwayExecutionBlocked, match="unreadable"):
        execute_plan(plan, tmp_path / "gen", client=client, poll_seconds=0, run_id="ep-1")
    assert fake.posts == []


def test_download_retries_then_succeeds(monkeypatch, tmp_path):
    attempts = []

    class Body(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(url, timeout=None):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise urllib.error.URLError("reset")
        return Body(b"mp4-bytes")

    monkeypatch.setattr(runway_video.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(runway_video.time, "sleep", lambda seconds: None)
    target = tmp_path / "clip.mp4"
    runway_video._download_file("https://cdn.example/clip.mp4", target)
    assert target.read_bytes() == b"mp4-bytes"
    assert attempts == [120.0, 120.0]
    assert not list(tmp_path.glob("*.part"))


def test_download_gives_up_loudly_after_retries(monkeypatch, tmp_path):
    def slow(url, timeout=None):
        raise TimeoutError("slow")

    monkeypatch.setattr(runway_video.urllib.request, "urlopen", slow)
    monkeypatch.setattr(runway_video.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        runway_video._download_file("https://cdn.example/clip.mp4", tmp_path / "clip.mp4")
    assert not (tmp_path / "clip.mp4").exists()


def test_empty_download_is_not_accepted(monkeypatch, tmp_path):
    class Body(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(runway_video.urllib.request, "urlopen", lambda url, timeout=None: Body(b""))
    monkeypatch.setattr(runway_video.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        runway_video._download_file("https://cdn.example/clip.mp4", tmp_path / "clip.mp4")


def test_ledger_rows_record_the_kind_and_task(monkeypatch, tmp_path):
    _enable(monkeypatch, budget="100")
    _, client = _client()
    execute_plan(_plan(tmp_path, durations=(2.0,)), tmp_path / "rep", client=client, poll_seconds=0,
                 run_id="ep-7", kind="repair")
    (row,) = json.loads(runway_video.LEDGER_FILE.read_text())
    assert row["kind"] == "repair" and row["task_id"] == "task-1" and row["status"] == "submitted"
    assert row["run_id"] == "ep-7" and row["credits"] == 25 and row["at"] == "2026-10-11T12:00:00Z"


def _handoff_and_clips(tmp_path, plan, names):
    clips = tmp_path / "web"
    clips.mkdir(exist_ok=True)
    runway_video.write_handoff(plan, clips)
    for name in names:
        (clips / name).write_bytes(b"a")
    return clips


def _good_probe(path):
    return {"width": 720.0, "height": 1280.0, "duration": 5.0}


def test_import_requires_a_handoff_for_this_plan(tmp_path):
    plan = _plan(tmp_path)
    clips = tmp_path / "web"
    clips.mkdir()
    with pytest.raises(RunwayExecutionBlocked, match="No Runway handoff"):
        runway_video.import_clips(plan, clips, probe=_good_probe)
    runway_video.write_handoff(plan, clips)
    other = _plan(tmp_path, prompt="a different episode {i}")
    with pytest.raises(RunwayExecutionBlocked, match="different plan"):
        runway_video.import_clips(other, clips, probe=_good_probe)


def test_clips_made_elsewhere_are_imported_by_shot_without_spending(monkeypatch, tmp_path):
    """Web-app clips use plan credits the API key cannot, so the factory just imports them."""
    plan = _plan(tmp_path)
    clips = _handoff_and_clips(tmp_path, plan, ["shot_00.mp4"])
    with pytest.raises(RunwayExecutionBlocked, match="shot 1: expected one shot_01"):
        runway_video.import_clips(plan, clips, probe=_good_probe)
    (clips / "shot_01_web.mp4").write_bytes(b"b")
    result = runway_video.import_clips(plan, clips, probe=_good_probe)
    assert result["clips"] == {0: str(clips / "shot_00.mp4"), 1: str(clips / "shot_01_web.mp4")}
    assert result["estimated_credits"] == 0 and runway_video.credits_spent() == 0
    assert result["plan_sha256"] == runway_video.plan_sha256(plan)


def test_import_matches_exact_shot_numbers(tmp_path):
    """shot_01*.mp4 used to also match shot_010.mp4 and shot_011_x.mp4."""
    plan = _plan(tmp_path, durations=(2.0, 2.0))
    clips = _handoff_and_clips(tmp_path, plan, ["shot_00.mp4", "shot_010.mp4", "shot_01.mp4"])
    result = runway_video.import_clips(plan, clips, probe=_good_probe)
    assert result["clips"][1] == str(clips / "shot_01.mp4")
    (clips / "shot_01_extra.mp4").write_bytes(b"b")
    with pytest.raises(RunwayExecutionBlocked, match="shot 1: expected one shot_01.*found 2"):
        runway_video.import_clips(plan, clips, probe=_good_probe)


def test_imported_clips_must_cover_the_planned_shot_and_be_portrait(tmp_path):
    plan = _plan(tmp_path, durations=(2.0, 3.0))
    clips = _handoff_and_clips(tmp_path, plan, ["shot_00.mp4", "shot_01.mp4"])
    infos = {"shot_00.mp4": {"width": 720.0, "height": 1280.0, "duration": 1.0},
             "shot_01.mp4": {"width": 1280.0, "height": 720.0, "duration": 5.0}}

    def probe(path):
        return infos[path.name]

    with pytest.raises(RunwayExecutionBlocked) as blocked:
        runway_video.import_clips(plan, clips, probe=probe)
    message = str(blocked.value)
    assert "shot 0: shot_00.mp4 is 1.0s, shorter than the planned 2.0s" in message
    assert "shot 1: shot_01.mp4 is 1280x720; the plan needs 9:16" in message


def test_unreadable_import_clip_is_a_problem_not_a_crash(tmp_path):
    plan = _plan(tmp_path, durations=(2.0,))
    clips = _handoff_and_clips(tmp_path, plan, ["shot_00.mp4"])

    def probe(path):
        raise subprocess.CalledProcessError(1, "ffprobe")

    with pytest.raises(RunwayExecutionBlocked, match="not a readable video"):
        runway_video.import_clips(plan, clips, probe=probe)


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")
def test_import_probes_real_clips(tmp_path):
    plan = _plan(tmp_path, durations=(2.0,))
    clips = _handoff_and_clips(tmp_path, plan, [])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=720x1280:rate=24:duration=5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clips / "shot_00.mp4")], check=True)
    assert runway_video.import_clips(plan, clips)["clips"] == {0: str(clips / "shot_00.mp4")}
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=24:duration=5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clips / "shot_00.mp4")], check=True)
    with pytest.raises(RunwayExecutionBlocked, match="1280x720"):
        runway_video.import_clips(plan, clips)


def test_handoff_cli_writes_names_prompts_and_the_plan_hash(tmp_path, capsys):
    plan = _plan(tmp_path, durations=(2.8,))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    clips = tmp_path / "web"
    assert runway_video.main(["handoff", str(plan_path), str(clips)]) == 0
    manifest = json.loads((clips / runway_video.HANDOFF_FILE).read_text())
    assert manifest["plan_sha256"] == runway_video.plan_sha256(plan)
    (shot,) = manifest["shots"]
    assert shot["file"] == "shot_00.mp4" and shot["billed_seconds"] == 5
    assert shot["prompt"].startswith("Animate the existing character. Timing: the action completes by 2.8s")
    assert "shot_00.mp4" in capsys.readouterr().out


def test_quote_cli_exits_non_zero_when_execution_would_be_refused(tmp_path, capsys, monkeypatch):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan(tmp_path)))
    assert runway_video.main(["quote", str(plan_path)]) == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["can_execute"] is False
    assert any("NOVA_RUNWAY_EXECUTE" in problem for problem in printed["problems"])
    assert printed["shots"][0]["compacted_prompt"].startswith("Animate the existing character. Timing:")

    _enable(monkeypatch, budget="100")
    assert runway_video.main(["quote", str(plan_path)]) == 0
    assert json.loads(capsys.readouterr().out)["can_execute"] is True


def test_download_hosts_come_from_the_environment_or_the_default(monkeypatch):
    assert runway_video.download_hosts() == runway_video.DEFAULT_DOWNLOAD_HOSTS
    monkeypatch.setenv("NOVA_RUNWAY_DOWNLOAD_HOSTS", "  a.example , ,b.example ")
    assert runway_video.download_hosts() == ("a.example", "b.example")
    monkeypatch.setenv("NOVA_RUNWAY_DOWNLOAD_HOSTS", "")
    assert runway_video.download_hosts() == runway_video.DEFAULT_DOWNLOAD_HOSTS


def test_quote_of_an_empty_ledger_path_is_zero_spend_and_frozen_day(tmp_path):
    result = runway_video.quote(_plan(tmp_path), reachable=lambda host: True)
    assert result["today_spent"] == 0 and result["episode_spent"] == 0
    runway_video.record_spend(run_id="ep", shot=0, task_id="t", credits=10, kind="generate", model="m")
    assert runway_video.credits_spent(day="2026-10-11") == 10
    assert runway_video.credits_spent(day="2026-10-10") == 0
