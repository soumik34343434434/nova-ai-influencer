"""Runway image-to-video execution for Director storyboard plans.

``RunwayPerformanceProvider.build_storyboard_plan`` decides *what* each shot
should be (model, prompt, duration) without spending credits. This module is
the opt-in worker that actually executes such a plan and downloads one clip per
shot for the Editor to assemble.

Execution is fail-closed and credit-careful. Before any request it requires:

* ``NOVA_RUNWAY_EXECUTE=true`` and ``RUNWAYML_API_SECRET``;
* the request to fit ``NOVA_RUNWAY_MAX_CREDITS`` *including everything this
  episode already spent* (repairs count), and ``NOVA_RUNWAY_DAILY_CREDITS``
  including everything spent today (defaults to the episode cap; blank means
  unset; caps must be finite numbers >= 0);
* a credit ledger that can be read (a corrupt ledger blocks spending);
* a no-cost preflight: every start image exists, every model maps to an API
  id, every prompt is present, and the API and download hosts are reachable.

Every submitted shot is reserved in, then recorded in, a credit ledger
(``artifacts/cartoon/runway_credit_ledger.json``), written under a file lock.
A stage retry is idempotent: each shot's task id and clip are kept in a
per-plan manifest, so clips already downloaded are reused, submitted tasks are
polled instead of posted again, and only shots that never reached Runway are
submitted.

Clips made in Runway's web app are imported only for the exact plan they were
prepared for: ``python -m app.providers.runway_video handoff <plan.json> <dir>``
writes the expected file names, prompts and plan hash into ``<dir>``, and
``import_clips`` refuses any other plan or clip that is not the right shape.
``python -m app.providers.runway_video quote <plan.json>`` prints the cost,
every blocker and the compacted prompts without spending anything.

Endpoint and model ids follow Runway's public API (``POST /v1/image_to_video``,
``GET /v1/tasks/{id}``). NOVA's routing names map to API model ids through
``DEFAULT_MODEL_IDS``; override or extend the map with ``NOVA_RUNWAY_MODEL_MAP``
(a JSON object) when Runway adds or renames models.
"""
from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from app.providers.ai_performance import RUNWAY_MODELS

ROOT = Path(__file__).resolve().parents[2]
LEDGER_FILE = ROOT / "artifacts" / "cartoon" / "runway_credit_ledger.json"
# Hosts Runway serves generated clips from; a clip we cannot download is wasted.
DEFAULT_DOWNLOAD_HOSTS = ("dnznrvs05pmza.cloudfront.net",)
API_BASE = "https://api.dev.runwayml.com"
API_VERSION = "2024-11-06"
# Vertical 9:16 output ratio accepted by Runway image-to-video models.
RATIOS = {"9:16": "720:1280", "16:9": "1280:720", "1:1": "960:960"}
# Billed clip lengths; a shorter shot is generated at 5s and trimmed by the Editor.
BILLED_DURATIONS = (5, 10)
DEFAULT_MODEL_IDS = {"gen-4-turbo": "gen4_turbo", "gen-4.5": "gen4.5"}
# Written by `handoff` into the folder of web-app clips; checked by `import_clips`.
HANDOFF_FILE = "runway_handoff.json"
# Per-plan record of submitted tasks and downloaded clips, inside the plan's folder.
RUN_MANIFEST = "manifest.json"
# shot_00.mp4, shot_01_web.mp4, ...; the number must match exactly (shot_01 is not shot_010).
CLIP_NAME = re.compile(r"shot_(\d+)(?:[_.-][^/]*)?\.mp4")
# Tolerance for clip duration (seconds) and aspect ratio (fraction) when importing.
DURATION_TOLERANCE = 0.05
ASPECT_TOLERANCE = 0.05
# The Director's opening sentence names the planned length, not the billed clip.
_DIRECTOR_INTRO = re.compile(r"^Animate the existing character for exactly [\d.]+s\.\s*")

Transport = Callable[[str, str, "dict[str, Any] | None"], dict[str, Any]]
Downloader = Callable[[str, Path], None]


class RunwayExecutionBlocked(RuntimeError):
    """Raised when execution is disabled, unconfigured, over budget or unsafe to resume."""


class RunwayTaskFailed(RuntimeError):
    """Runway reported a finished task as failed or cancelled; the shot produced no clip."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _today() -> str:
    return _utc_now().strftime("%Y-%m-%d")


def _rates() -> dict[str, float]:
    return {option.model: option.credits_per_second for option in RUNWAY_MODELS}


def billed_seconds(duration: float) -> int:
    for option in BILLED_DURATIONS:
        if duration <= option:
            return option
    return BILLED_DURATIONS[-1]


def model_ids() -> dict[str, str]:
    mapping = dict(DEFAULT_MODEL_IDS)
    raw = os.getenv("NOVA_RUNWAY_MODEL_MAP", "").strip()
    if raw:
        try:
            extra = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RunwayExecutionBlocked("NOVA_RUNWAY_MODEL_MAP must be a JSON object") from exc
        if not isinstance(extra, dict):
            raise RunwayExecutionBlocked("NOVA_RUNWAY_MODEL_MAP must be a JSON object")
        mapping.update({str(k): str(v) for k, v in extra.items()})
    return mapping


def billed_credits(plan: dict[str, Any], shot_indexes: list[int] | None = None) -> float:
    rates = _rates()
    total = 0.0
    for position, shot in enumerate(plan.get("shots", [])):
        if shot_indexes is not None and position not in shot_indexes:
            continue
        rate = rates.get(str(shot.get("model")))
        if rate is None:
            raise RunwayExecutionBlocked(f"No credit rate for model {shot.get('model')}")
        total += rate * billed_seconds(float(shot.get("duration", 5.0)))
    return round(total, 2)


def plan_sha256(plan: dict[str, Any]) -> str:
    """Identity of a plan: clips, manifests and handoffs are tied to this hash."""
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _positions(plan: dict[str, Any], shot_indexes: list[int] | None = None) -> list[tuple[int, int, dict[str, Any]]]:
    """(plan position, shot number, request) for the requested positions, in plan order."""
    return [
        (position, int(request.get("shot_index", position)), request)
        for position, request in enumerate(plan.get("shots", []))
        if shot_indexes is None or position in shot_indexes
    ]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# Credit ledger ------------------------------------------------------------

@contextlib.contextmanager
def _ledger_lock() -> Iterator[None]:
    """Serialise ledger read-modify-write across threads and processes."""
    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_FILE.with_name(LEDGER_FILE.name + ".lock"), "a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _ledger_rows() -> list[dict[str, Any]]:
    """All ledger rows. A missing ledger is empty; an unreadable one blocks spending."""
    try:
        raw = LEDGER_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise RunwayExecutionBlocked(f"Runway credit ledger {LEDGER_FILE} cannot be read; refusing to spend") from exc
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RunwayExecutionBlocked(
            f"Runway credit ledger {LEDGER_FILE} is corrupt; refusing to spend until it is restored"
        ) from exc
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RunwayExecutionBlocked(f"Runway credit ledger {LEDGER_FILE} is not a list of rows; refusing to spend")
    return rows


def _write_ledger(rows: list[dict[str, Any]]) -> None:
    _atomic_write_text(LEDGER_FILE, json.dumps(rows, indent=2) + "\n")


def _sum_credits(rows: list[dict[str, Any]], *, run_id: str | None = None, day: str | None = None) -> float:
    total = 0.0
    for row in rows:
        if run_id is not None and row.get("run_id") != run_id:
            continue
        if day is not None and not str(row.get("at", "")).startswith(day):
            continue
        total += float(row.get("credits", 0.0) or 0.0)
    return round(total, 2)


def credits_spent(*, run_id: str | None = None, day: str | None = None) -> float:
    """Credits recorded for one episode (run_id) and/or one UTC day (YYYY-MM-DD)."""
    return _sum_credits(_ledger_rows(), run_id=run_id, day=day)


def _new_row(**fields: Any) -> dict[str, Any]:
    return {"at": _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"), **fields}


def record_spend(*, run_id: str, shot: int, task_id: str, credits: float, kind: str, model: str) -> None:
    """Append one submitted shot to the ledger (counted as spent even if it later fails)."""
    with _ledger_lock():
        rows = _ledger_rows()
        rows.append(_new_row(run_id=run_id, shot=shot, task_id=task_id,
                             credits=round(float(credits), 2), kind=kind, model=model))
        _write_ledger(rows)


def _reserve_spend(*, run_id: str, shot: int, model: str, credits: float, kind: str,
                   episode_cap: float, daily_cap: float) -> str:
    """Check the caps and record a submission about to be paid for, as one locked step.

    Two processes cannot both pass the gate for the same credits. The row stays
    ``reserved`` until its task id is recorded, so a crash after the request is
    sent still counts as spent and blocks a blind resubmission.
    """
    row_id = uuid.uuid4().hex
    with _ledger_lock():
        rows = _ledger_rows()
        episode_spent = _sum_credits(rows, run_id=run_id) if run_id else 0.0
        today_spent = _sum_credits(rows, day=_today())
        if episode_spent + credits > episode_cap:
            raise RunwayExecutionBlocked(
                f"Runway shot {shot} needs {credits:.0f} credits; this episode already spent {episode_spent:.0f} "
                f"and NOVA_RUNWAY_MAX_CREDITS is {episode_cap:.0f}"
            )
        if today_spent + credits > daily_cap:
            raise RunwayExecutionBlocked(
                f"Runway shot {shot} needs {credits:.0f} credits; {today_spent:.0f} already spent today "
                f"and NOVA_RUNWAY_DAILY_CREDITS is {daily_cap:.0f}"
            )
        rows.append(_new_row(id=row_id, run_id=run_id, shot=shot, task_id="", credits=round(credits, 2),
                             kind=kind, model=model, status="reserved"))
        _write_ledger(rows)
    return row_id


def _settle_spend(row_id: str, task_id: str | None) -> None:
    """Record the task id of a reservation, or release it (task_id None) when Runway refused the request."""
    with _ledger_lock():
        rows = _ledger_rows()
        kept: list[dict[str, Any]] = []
        found = False
        for row in rows:
            if row.get("id") == row_id:
                found = True
                if task_id is None:
                    continue
                row = {**row, "task_id": task_id, "status": "submitted"}
            kept.append(row)
        if not found:
            raise RuntimeError(f"Runway ledger row {row_id} is missing; refusing to continue")
        _write_ledger(kept)


def _unconfirmed_submissions(run_id: str) -> list[dict[str, Any]]:
    return [row for row in _ledger_rows() if row.get("status") == "reserved" and row.get("run_id") == run_id]


def _cap(name: str, default: float = 0.0) -> float:
    """A credit cap. Unset or blank means the default; anything else must be a finite number >= 0."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise RunwayExecutionBlocked(f"{name} must be a number of credits, got {raw!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise RunwayExecutionBlocked(f"{name} must be a finite number of credits >= 0, got {raw!r}")
    return value


def _ledger_missing_on_ci() -> bool:
    """On CI a cold cache looks like an empty ledger; that must not reset the caps."""
    return (os.getenv("GITHUB_ACTIONS", "").strip().lower() == "true" and not LEDGER_FILE.exists()
            and os.getenv("NOVA_RUNWAY_LEDGER_RESTORED", "").strip().lower() != "true")


def _budget(plan: dict[str, Any], shot_indexes: list[int] | None, run_id: str) -> tuple[dict[str, Any], list[str]]:
    """Every reason a request cannot run, plus the numbers behind the budget checks."""
    state: dict[str, Any] = {"api_secret": "", "episode_cap": 0.0, "daily_cap": 0.0,
                             "estimated_credits": 0.0, "episode_spent": 0.0, "today_spent": 0.0}
    problems: list[str] = []
    if os.getenv("NOVA_RUNWAY_EXECUTE", "false").strip().lower() != "true":
        problems.append("Runway execution is disabled; set NOVA_RUNWAY_EXECUTE=true to allow spending credits")
    state["api_secret"] = os.getenv("RUNWAYML_API_SECRET", "").strip()
    if not state["api_secret"]:
        problems.append("RUNWAYML_API_SECRET is required for Runway execution")
    try:
        state["episode_cap"] = _cap("NOVA_RUNWAY_MAX_CREDITS")
        state["daily_cap"] = _cap("NOVA_RUNWAY_DAILY_CREDITS", state["episode_cap"])
    except RunwayExecutionBlocked as exc:
        problems.append(str(exc))
    try:
        state["estimated_credits"] = billed_credits(plan, shot_indexes)
    except RunwayExecutionBlocked as exc:
        problems.append(str(exc))
    unconfirmed: list[dict[str, Any]] = []
    try:
        if run_id:
            state["episode_spent"] = credits_spent(run_id=run_id)
        state["today_spent"] = credits_spent(day=_today())
        unconfirmed = _unconfirmed_submissions(run_id)
    except RunwayExecutionBlocked as exc:
        problems.append(str(exc))
    if unconfirmed:
        ids = ", ".join(str(row.get("shot")) for row in unconfirmed)
        problems.append(
            f"{len(unconfirmed)} Runway submission(s) for shot(s) {ids} have no recorded task id; "
            f"check the Runway dashboard, then remove those rows from {LEDGER_FILE} before spending again"
        )
    estimate = float(state["estimated_credits"])
    if estimate > 0 and _ledger_missing_on_ci():
        problems.append(
            f"the Runway credit ledger {LEDGER_FILE} is missing on GitHub Actions, so earlier spend is unknown; "
            "restore it from the factory-state branch, or set NOVA_RUNWAY_LEDGER_RESTORED=true once you have "
            "confirmed there was no earlier spend"
        )
    if estimate > 0:
        if estimate + state["episode_spent"] > state["episode_cap"]:
            problems.append(
                f"Runway request needs about {estimate:.0f} credits; this episode already spent "
                f"{state['episode_spent']:.0f} and NOVA_RUNWAY_MAX_CREDITS is {state['episode_cap']:.0f}"
            )
        if estimate + state["today_spent"] > state["daily_cap"]:
            problems.append(
                f"Runway request needs about {estimate:.0f} credits; {state['today_spent']:.0f} already spent today "
                f"and NOVA_RUNWAY_DAILY_CREDITS is {state['daily_cap']:.0f}"
            )
    return state, problems


def execution_gate(plan: dict[str, Any], shot_indexes: list[int] | None = None, *, run_id: str = "") -> dict[str, Any]:
    """Return the execution config or raise RunwayExecutionBlocked listing every reason it is refused."""
    state, problems = _budget(plan, shot_indexes, run_id)
    if problems:
        raise RunwayExecutionBlocked("; ".join(problems))
    return {"api_secret": state["api_secret"], "budget": state["episode_cap"],
            "estimated_credits": state["estimated_credits"], "episode_spent": state["episode_spent"],
            "today_spent": state["today_spent"], "daily_cap": state["daily_cap"]}


def host_reachable(host: str, timeout: float = 8.0) -> bool:
    """True when an HTTPS request reaches the host at all (any HTTP status counts)."""
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://{host}/", method="HEAD"), timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def download_hosts() -> tuple[str, ...]:
    raw = os.getenv("NOVA_RUNWAY_DOWNLOAD_HOSTS", "").strip()
    return tuple(h.strip() for h in raw.split(",") if h.strip()) or DEFAULT_DOWNLOAD_HOSTS


def preflight(plan: dict[str, Any], shot_indexes: list[int] | None = None, *,
              reachable: Callable[[str], bool] | None = None) -> list[str]:
    """No-cost checks that must pass before any credit is spent; returns problems."""
    problems: list[str] = []
    mapping = model_ids()
    for position, shot, request in _positions(plan, shot_indexes):
        image = Path(str(request.get("image", "")))
        if not image.is_file() or image.stat().st_size == 0:
            problems.append(f"shot {position}: start image missing ({image})")
        if str(request.get("model", "")) not in mapping:
            problems.append(f"shot {position}: no Runway API id for model {request.get('model')}; add it via NOVA_RUNWAY_MODEL_MAP")
        if not str(request.get("prompt", "")).strip():
            problems.append(f"shot {position}: empty prompt")
    check = reachable or host_reachable
    api_host = urllib.parse.urlparse(os.getenv("RUNWAY_API_BASE", API_BASE)).hostname or ""
    for host in (api_host, *download_hosts()):
        if host and not check(host):
            problems.append(f"network: cannot reach {host}; generated clips could not be used")
    return problems


def quote(plan: dict[str, Any], shot_indexes: list[int] | None = None, *, run_id: str = "",
          reachable: Callable[[str], bool] | None = None) -> dict[str, Any]:
    """What a run would cost, every reason it would be refused, and the prompts it would send.

    Nothing is spent. ``can_execute`` is True only when ``execution_gate`` and the
    preflight would both pass, so the quote and the real run cannot disagree.
    """
    rates = _rates()
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    positions = _positions(plan, shot_indexes)
    for _, shot, request in positions:
        model = str(request.get("model"))
        seconds = billed_seconds(float(request.get("duration", 5.0)))
        rate = rates.get(model)
        if rate is None:
            problems.append(f"shot {shot}: no credit rate for model {model}; it cannot be priced")
        prompt = compact_prompt(shot_prompt(request))
        rows.append({
            "shot": shot, "model": request.get("model"), "billed_seconds": seconds,
            "credits": round((rate or 0.0) * seconds, 2), "image": str(request.get("image", "")),
            "prompt_chars": len(prompt), "compacted_prompt": prompt,
        })
    indexes = [position for position, _, _ in positions]
    budget, budget_problems = _budget(plan, indexes, run_id)
    problems = list(dict.fromkeys(
        preflight(plan, indexes, reachable=reachable) + problems + budget_problems
    ))
    return {
        "shots": rows,
        "total_credits": round(sum(r["credits"] for r in rows), 2),
        "episode_spent": budget["episode_spent"],
        "today_spent": budget["today_spent"],
        "episode_cap": budget["episode_cap"],
        "daily_cap": budget["daily_cap"],
        "problems": problems,
        "can_execute": not problems,
    }


# Prompts ------------------------------------------------------------------

def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _find_output_url(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith("https://"):
        return value
    if isinstance(value, list):
        for child in value:
            found = _find_output_url(child)
            if found:
                return found
    if isinstance(value, dict):
        for child in value.values():
            found = _find_output_url(child)
            if found:
                return found
    return None


# Runway rejects promptText longer than this; the Director's full prompt is ~3.5k.
PROMPT_LIMIT = 1000
_PROMPT_LABELS = (
    "Targeted repair", "Performance arc", "Action chain", "Action", "Gaze", "Head", "Body", "Hands",
    "Face", "Secondary motion", "Root lock", "Overshoot mode", "Facial performance", "Pose accents",
    "Orientation lock", "Motion priority", "Timeline", "Camera", "Preserve", "Avoid",
)
# Most important first: what to fix, what happens, then what must not change.
_PROMPT_PRIORITY = (
    "Targeted repair", "Performance arc", "Action", "Gaze", "Head", "Orientation lock", "Hands", "Avoid",
    "Preserve", "Root lock", "Body", "Camera", "Face",
)


def shot_prompt(request: dict[str, Any]) -> str:
    """The Runway prompt for one shot, before compaction.

    Runway renders a 5s or 10s clip, not the planned length. The Director's
    timeline is therefore reduced to one timing sentence: where the action ends
    in the billed clip, and that the pose holds afterwards. The Editor trims
    from the start, so the kept footage is the whole action, then a held pose.
    """
    planned = float(request.get("duration", 5.0))
    billed = billed_seconds(planned)
    timing = (f"Timing: the action completes by {min(planned, billed):.1f}s of this {billed}s clip, "
              "then the character holds the final pose until the end.")
    text = _DIRECTOR_INTRO.sub("", " ".join(str(request.get("prompt", "")).split()), count=1)
    return f"Animate the existing character. {timing} {text}".strip()


def compact_prompt(prompt: str, limit: int = PROMPT_LIMIT) -> str:
    """Fit a Director prompt into Runway's limit without cutting a sentence in half.

    The full prompt is a run of "Label: content." sections. Keep the opening
    instruction, then whole sections in priority order; long Preserve/Avoid
    lists are added clause by clause.
    """
    text = " ".join(str(prompt).split())
    if len(text) <= limit:
        return text
    labels = "|".join(re.escape(label) for label in _PROMPT_LABELS)
    parts = re.split(rf"(?:^|(?<=\.) )({labels})(?: v[\d.]+)?: ", text)
    intro = parts[0].strip()
    sections: dict[str, str] = {}
    for label, content in zip(parts[1::2], parts[2::2]):
        sections.setdefault(label, content.strip().rstrip("."))
    out = intro if len(intro) <= limit else intro[: limit - 1].rsplit(" ", 1)[0] + "."
    for label in _PROMPT_PRIORITY:
        content = sections.get(label)
        if not content:
            continue
        piece = f" {label}: {content}."
        if len(out) + len(piece) <= limit:
            out += piece
            continue
        clauses = [c.strip() for c in content.split(";") if c.strip()]
        kept: list[str] = []
        for clause in clauses:
            candidate = f" {label}: {'; '.join(kept + [clause])}."
            if len(out) + len(candidate) > limit:
                break
            kept.append(clause)
        if kept:
            out += f" {label}: {'; '.join(kept)}."
    return out.strip()


def shot_payload(request: dict[str, Any], mapping: dict[str, str] | None = None) -> dict[str, Any]:
    mapping = mapping or model_ids()
    model = str(request.get("model", ""))
    api_model = mapping.get(model)
    if not api_model:
        raise RunwayExecutionBlocked(
            f"No Runway API model id for {model}; add it via NOVA_RUNWAY_MODEL_MAP"
        )
    ratio = RATIOS.get(str(request.get("aspect_ratio", "9:16")), RATIOS["9:16"])
    return {
        "model": api_model,
        "promptImage": _data_uri(Path(request["image"])),
        "promptText": compact_prompt(shot_prompt(request)),
        "ratio": ratio,
        "duration": billed_seconds(float(request.get("duration", 5.0))),
    }


# Client -------------------------------------------------------------------

def _download_file(url: str, path: Path, *, attempts: int = 3, timeout: float = 120.0) -> None:
    """Fetch a generated clip atomically. A paid clip is worth a few retries."""
    partial = path.with_name(path.name + ".part")
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=timeout) as response, open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle)
            if partial.stat().st_size == 0:
                raise OSError("empty response")
            os.replace(partial, path)
            return
        except OSError as exc:
            last = exc
            partial.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Could not download {url} after {attempts} attempts: {last}") from last


class RunwayVideoClient:
    """Minimal Runway REST client; transport and downloader are injectable for tests."""

    def __init__(self, api_secret: str, *, base_url: str | None = None, api_version: str = API_VERSION,
                 transport: Transport | None = None, downloader: Downloader | None = None) -> None:
        self.api_secret = api_secret
        self.base_url = (base_url or os.getenv("RUNWAY_API_BASE", API_BASE)).rstrip("/")
        self.api_version = api_version
        self._transport = transport or self._http
        self._download = downloader or _download_file

    def _http(self, method: str, url: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode() if payload is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_secret}",
                "Content-Type": "application/json",
                "X-Runway-Version": self.api_version,
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode())
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected Runway response")
        return data

    def image_to_video(self, payload: dict[str, Any]) -> str:
        task = self._transport("POST", f"{self.base_url}/v1/image_to_video", payload)
        task_id = str(task.get("id", ""))
        if not task_id:
            raise RuntimeError("Runway did not return a task id")
        return task_id

    def wait(self, task_id: str, *, poll_seconds: float = 5.0, timeout_seconds: float = 900.0) -> str:
        started = time.monotonic()
        while time.monotonic() - started < timeout_seconds:
            status = self._transport("GET", f"{self.base_url}/v1/tasks/{task_id}", None)
            state = str(status.get("status", "")).upper()
            if state in {"SUCCEEDED", "SUCCESS", "COMPLETED"}:
                url = _find_output_url(status.get("output"))
                if not url:
                    raise RuntimeError(f"Runway task {task_id} finished without an output URL")
                return url
            if state in {"FAILED", "CANCELLED", "CANCELED"}:
                raise RunwayTaskFailed(f"Runway task {task_id} failed: {status.get('failure', state)}")
            time.sleep(poll_seconds)
        raise TimeoutError(f"Runway task {task_id} did not finish within {timeout_seconds:.0f}s")

    def download(self, url: str, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._download(url, path)
        return path


# Execution ----------------------------------------------------------------

def _run_dir(output_dir: Path, plan_hash: str) -> Path:
    """Each plan gets its own folder, so a changed storyboard never reuses old clips."""
    return Path(output_dir) / f"plan_{plan_hash[:12]}"


def _load_manifest(run_dir: Path, plan_hash: str) -> dict[str, Any]:
    path = run_dir / RUN_MANIFEST
    if not path.exists():
        return {"version": 1, "plan_sha256": plan_hash, "shots": {}}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunwayExecutionBlocked(
            f"Runway run manifest {path} is unreadable; refusing to guess what was already paid for"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("plan_sha256") != plan_hash or not isinstance(manifest.get("shots"), dict):
        raise RunwayExecutionBlocked(f"Runway run manifest {path} does not belong to this plan")
    return manifest


def _save_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    _atomic_write_text(run_dir / RUN_MANIFEST, json.dumps(manifest, indent=2) + "\n")


def _clip_ready(value: Any) -> bool:
    if not value:
        return False
    path = Path(str(value))
    return path.is_file() and path.stat().st_size > 0


def _needs_submission(entry: dict[str, Any] | None) -> bool:
    """A shot needs a new paid request unless Runway already has its task (a failed task is retried)."""
    return entry is None or not entry.get("task_id") or entry.get("status") == "failed"


def execute_plan(
    plan: dict[str, Any],
    output_dir: Path,
    *,
    shot_indexes: list[int] | None = None,
    client: RunwayVideoClient | None = None,
    poll_seconds: float = 5.0,
    run_id: str = "",
    kind: str = "generate",
    reachable: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Execute (part of) a storyboard plan and download one clip per shot. Safe to retry.

    ``shot_indexes`` limits execution to those plan positions, which is how a
    targeted repair regenerates only the failing shots. Each shot's task id and
    clip are written to the plan's manifest as they happen, so a failed attempt
    resumes where it stopped: downloaded clips are reused, submitted tasks are
    polled, and only shots never submitted (or whose task failed) cost credits.
    Nothing is submitted unless the budget gate and the preflight both pass.
    """
    plan_hash = plan_sha256(plan)
    run_dir = _run_dir(output_dir, plan_hash)
    manifest = _load_manifest(run_dir, plan_hash)
    recorded: dict[str, Any] = manifest["shots"]
    positions = _positions(plan, shot_indexes)
    to_submit = [position for position, shot, _ in positions if _needs_submission(recorded.get(str(shot)))]
    gate = execution_gate(plan, to_submit, run_id=run_id)
    problems = preflight(plan, to_submit, reachable=reachable)
    if problems:
        raise RunwayExecutionBlocked("Runway preflight failed, no credits spent: " + "; ".join(problems))
    client = client or RunwayVideoClient(gate["api_secret"])
    mapping = model_ids()
    rates = _rates()
    clips: dict[int, str] = {}
    tasks: list[dict[str, Any]] = []
    submitted = 0
    for _, shot, request in positions:
        entry = recorded.get(str(shot))
        if entry is not None and _clip_ready(entry.get("clip")):
            clips[shot] = str(entry["clip"])
            tasks.append(_task_row(shot, entry, reused=True))
            continue
        if _needs_submission(entry):
            payload = shot_payload(request, mapping)
            credits = rates[str(request.get("model"))] * payload["duration"]
            row_id = _reserve_spend(run_id=run_id, shot=shot, model=payload["model"], credits=credits, kind=kind,
                                    episode_cap=gate["budget"], daily_cap=gate["daily_cap"])
            try:
                task_id = client.image_to_video(payload)
            except urllib.error.HTTPError:
                _settle_spend(row_id, None)  # Runway refused the request, so no task exists
                raise
            _settle_spend(row_id, task_id)
            entry = {"task_id": task_id, "model": payload["model"], "duration": payload["duration"],
                     "credits": round(credits, 2), "status": "submitted", "clip": None}
            recorded[str(shot)] = entry
            _save_manifest(run_dir, manifest)
            submitted += 1
        task_id = str(entry["task_id"])
        try:
            url = client.wait(task_id, poll_seconds=poll_seconds)
        except RunwayTaskFailed:
            entry["status"] = "failed"
            _save_manifest(run_dir, manifest)
            raise
        clip = client.download(url, run_dir / f"shot_{shot:02d}_{task_id[:8]}.mp4")
        if not _clip_ready(clip):
            raise RuntimeError(f"Runway clip for shot {shot} downloaded empty: {clip}")
        entry.update(status="downloaded", clip=str(clip))
        _save_manifest(run_dir, manifest)
        clips[shot] = str(clip)
        tasks.append(_task_row(shot, entry, reused=False))
    return {
        "provider": "runway",
        "clips": clips,
        "tasks": tasks,
        "submitted": submitted,
        "estimated_credits": billed_credits(plan, [position for position, _, _ in positions]),
        "budget": gate["budget"],
        "plan_sha256": plan_hash,
    }


def _task_row(shot: int, entry: dict[str, Any], *, reused: bool) -> dict[str, Any]:
    return {"shot": shot, "task_id": entry["task_id"], "model": entry.get("model"),
            "duration": entry.get("duration"), "reused": reused}


# Web-app clips ------------------------------------------------------------

def write_handoff(plan: dict[str, Any], clips_dir: Path, *, plan_path: str = "") -> Path:
    """Write the clip names, prompts and plan hash that web-app clips must match."""
    shots = []
    for _, shot, request in _positions(plan):
        duration = float(request.get("duration", 5.0))
        shots.append({
            "shot": shot, "file": f"shot_{shot:02d}.mp4", "model": request.get("model"),
            "aspect_ratio": request.get("aspect_ratio", "9:16"), "image": str(request.get("image", "")),
            "planned_seconds": duration, "billed_seconds": billed_seconds(duration),
            "prompt": compact_prompt(shot_prompt(request)),
        })
    path = Path(clips_dir) / HANDOFF_FILE
    manifest = {"version": 1, "plan": plan_path, "plan_sha256": plan_sha256(plan), "shots": shots}
    _atomic_write_text(path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return path


def _read_handoff(clips_dir: Path) -> dict[str, Any]:
    path = Path(clips_dir) / HANDOFF_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunwayExecutionBlocked(
            f"No Runway handoff in {clips_dir}; run `python -m app.providers.runway_video handoff <plan.json> "
            f"{clips_dir}` for this plan and make its clips there"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RunwayExecutionBlocked(f"Runway handoff {path} is unreadable") from exc
    if not isinstance(data, dict):
        raise RunwayExecutionBlocked(f"Runway handoff {path} is not a JSON object")
    return data


def _probe_clip(path: Path) -> dict[str, float]:
    """Width, height and duration of a clip's first video stream, read with ffprobe."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
             "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=60,
        ).stdout
    except FileNotFoundError as exc:
        raise RunwayExecutionBlocked("ffprobe is required to verify clips imported from Runway's web app") from exc
    data = json.loads(out)
    stream = data["streams"][0]
    return {"width": float(stream["width"]), "height": float(stream["height"]),
            "duration": float(data["format"]["duration"])}


def _clip_problem(shot: int, path: Path, request: dict[str, Any], probe: Callable[[Path], dict[str, float]]) -> str | None:
    try:
        info = probe(path)
    except (subprocess.SubprocessError, KeyError, IndexError, TypeError, ValueError) as exc:
        return f"shot {shot}: {path.name} is not a readable video ({exc})"
    planned = float(request.get("duration", 5.0))
    if info["duration"] + DURATION_TOLERANCE < planned:
        return f"shot {shot}: {path.name} is {info['duration']:.1f}s, shorter than the planned {planned:.1f}s"
    width, height = (float(part) for part in RATIOS.get(str(request.get("aspect_ratio", "9:16")), RATIOS["9:16"]).split(":"))
    expected = height / width
    if not info["width"] or abs(info["height"] / info["width"] - expected) / expected > ASPECT_TOLERANCE:
        return (f"shot {shot}: {path.name} is {info['width']:.0f}x{info['height']:.0f}; "
                f"the plan needs {request.get('aspect_ratio', '9:16')}")
    return None


def import_clips(plan: dict[str, Any], clips_dir: Path, *,
                 probe: Callable[[Path], dict[str, float]] | None = None) -> dict[str, Any]:
    """Use clips made outside the factory for this exact plan, spending nothing.

    Runway's web app (and Claude's Runway connector) bill the workspace's plan
    credits, which the API key cannot use. The folder must hold a handoff written
    for this plan's hash (see ``write_handoff``), then one ``shot_NN*.mp4`` per
    shot whose duration covers the planned shot and whose aspect ratio matches.
    """
    clips_dir = Path(clips_dir)
    plan_hash = plan_sha256(plan)
    handoff = _read_handoff(clips_dir)
    if handoff.get("plan_sha256") != plan_hash:
        raise RunwayExecutionBlocked(
            f"Clips in {clips_dir} were prepared for a different plan; run `python -m app.providers.runway_video "
            f"handoff <plan.json> {clips_dir}` for this plan and make its clips"
        )
    files: dict[int, list[Path]] = {}
    for path in sorted(clips_dir.iterdir()):
        match = CLIP_NAME.fullmatch(path.name)
        if match and path.is_file():
            files.setdefault(int(match.group(1)), []).append(path)
    check = probe or _probe_clip
    clips: dict[int, str] = {}
    problems: list[str] = []
    for _, shot, request in _positions(plan):
        found = files.get(shot, [])
        if len(found) != 1:
            problems.append(f"shot {shot}: expected one shot_{shot:02d}*.mp4, found {len(found)}")
            continue
        problem = _clip_problem(shot, found[0], request, check)
        if problem:
            problems.append(problem)
        else:
            clips[shot] = str(found[0])
    if problems:
        raise RunwayExecutionBlocked(f"Cannot import clips from {clips_dir}: " + "; ".join(problems))
    return {"provider": "runway", "clips": clips, "tasks": [], "estimated_credits": 0.0, "budget": 0.0,
            "source": "imported", "plan_sha256": plan_hash}


def required_credits_note(plan: dict[str, Any]) -> str:
    return f"~{math.ceil(billed_credits(plan))} credits for {len(plan.get('shots', []))} shots"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m app.providers.runway_video",
        description="Quote and prepare Runway storyboard plans. Nothing here spends credits.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    quote_cmd = commands.add_parser("quote", help="print cost, blockers and prompts; exit 2 if execution would be refused")
    quote_cmd.add_argument("plan", type=Path)
    quote_cmd.add_argument("--run-id", default="")
    handoff_cmd = commands.add_parser("handoff", help="write the clip names, prompts and plan hash for web-app clips")
    handoff_cmd.add_argument("plan", type=Path)
    handoff_cmd.add_argument("clips_dir", type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.command == "quote":
        result = quote(plan, run_id=args.run_id)
        print(json.dumps(result, indent=2))
        return 0 if result["can_execute"] else 2
    path = write_handoff(plan, args.clips_dir, plan_path=str(args.plan))
    print(f"Wrote {path}. Make one clip per shot in the Runway web app, save it as shot_NN.mp4 in that folder:")
    for row in json.loads(path.read_text(encoding="utf-8"))["shots"]:
        print(f"\nshot {row['shot']} -> {row['file']} (plan {row['planned_seconds']}s, bill {row['billed_seconds']}s, "
              f"{row['model']}, {row['aspect_ratio']})")
        print(f"  start image: {row['image']}" + ("" if Path(row["image"]).is_file() else "  [MISSING]"))
        print(f"  prompt: {row['prompt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
