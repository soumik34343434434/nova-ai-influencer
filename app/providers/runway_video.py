"""Runway image-to-video execution for Director storyboard plans.

``RunwayPerformanceProvider.build_storyboard_plan`` decides *what* each shot
should be (model, prompt, duration) without spending credits. This module is
the opt-in worker that actually executes such a plan and downloads one clip per
shot for the Editor to assemble.

Execution is fail-closed and credit-careful. Before any request it requires:

* ``NOVA_RUNWAY_EXECUTE=true`` and ``RUNWAYML_API_SECRET``;
* the request to fit ``NOVA_RUNWAY_MAX_CREDITS`` *including everything this
  episode already spent* (repairs count), and ``NOVA_RUNWAY_DAILY_CREDITS``
  including everything spent today (defaults to the episode cap);
* a no-cost preflight: every start image exists, every model maps to an API
  id, every prompt fits, and the API and download hosts are reachable, so a
  paid clip is never generated that cannot be downloaded and used.

Every submitted shot is written to a credit ledger
(``artifacts/cartoon/runway_credit_ledger.json``). ``python -m
app.providers.runway_video quote <plan.json>`` prints the cost and preflight
result without spending anything.

Endpoint and model ids follow Runway's public API (``POST /v1/image_to_video``,
``GET /v1/tasks/{id}``). NOVA's routing names map to API model ids through
``DEFAULT_MODEL_IDS``; override or extend the map with ``NOVA_RUNWAY_MODEL_MAP``
(a JSON object) when Runway adds or renames models.
"""
from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

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

Transport = Callable[[str, str, "dict[str, Any] | None"], dict[str, Any]]
Downloader = Callable[[str, Path], None]


class RunwayExecutionBlocked(RuntimeError):
    """Raised when execution is disabled, unconfigured or over budget."""


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
    rates = {option.model: option.credits_per_second for option in RUNWAY_MODELS}
    total = 0.0
    for position, shot in enumerate(plan.get("shots", [])):
        if shot_indexes is not None and position not in shot_indexes:
            continue
        rate = rates.get(str(shot.get("model")))
        if rate is None:
            raise RunwayExecutionBlocked(f"No credit rate for model {shot.get('model')}")
        total += rate * billed_seconds(float(shot.get("duration", 5.0)))
    return round(total, 2)


def _ledger_rows() -> list[dict[str, Any]]:
    try:
        rows = json.loads(LEDGER_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return rows if isinstance(rows, list) else []


def credits_spent(*, run_id: str | None = None, day: str | None = None) -> float:
    """Credits recorded for one episode (run_id) and/or one UTC day (YYYY-MM-DD)."""
    total = 0.0
    for row in _ledger_rows():
        if run_id is not None and row.get("run_id") != run_id:
            continue
        if day is not None and not str(row.get("at", "")).startswith(day):
            continue
        total += float(row.get("credits", 0.0) or 0.0)
    return round(total, 2)


def record_spend(*, run_id: str, shot: int, task_id: str, credits: float, kind: str, model: str) -> None:
    """Append one submitted shot to the ledger (counted as spent even if it later fails)."""
    rows = _ledger_rows()
    rows.append({
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id, "shot": shot, "task_id": task_id,
        "credits": round(float(credits), 2), "kind": kind, "model": model,
    })
    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_FILE.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _cap(name: str, default: str = "0") -> float:
    try:
        return float(os.getenv(name, default) or 0)
    except ValueError as exc:
        raise RunwayExecutionBlocked(f"{name} must be a number") from exc


def execution_gate(plan: dict[str, Any], shot_indexes: list[int] | None = None, *, run_id: str = "") -> dict[str, Any]:
    """Return the execution config or raise RunwayExecutionBlocked explaining why."""
    if os.getenv("NOVA_RUNWAY_EXECUTE", "false").strip().lower() != "true":
        raise RunwayExecutionBlocked("Runway execution is disabled; set NOVA_RUNWAY_EXECUTE=true to allow spending credits")
    secret = os.getenv("RUNWAYML_API_SECRET", "").strip()
    if not secret:
        raise RunwayExecutionBlocked("RUNWAYML_API_SECRET is required for Runway execution")
    episode_cap = _cap("NOVA_RUNWAY_MAX_CREDITS")
    daily_cap = _cap("NOVA_RUNWAY_DAILY_CREDITS", str(episode_cap))
    estimate = billed_credits(plan, shot_indexes)
    episode_spent = credits_spent(run_id=run_id) if run_id else 0.0
    today_spent = credits_spent(day=time.strftime("%Y-%m-%d", time.gmtime()))
    if estimate + episode_spent > episode_cap:
        raise RunwayExecutionBlocked(
            f"Runway request needs about {estimate:.0f} credits; this episode already spent {episode_spent:.0f} "
            f"and NOVA_RUNWAY_MAX_CREDITS is {episode_cap:.0f}"
        )
    if estimate + today_spent > daily_cap:
        raise RunwayExecutionBlocked(
            f"Runway request needs about {estimate:.0f} credits; {today_spent:.0f} already spent today "
            f"and NOVA_RUNWAY_DAILY_CREDITS is {daily_cap:.0f}"
        )
    return {"api_secret": secret, "budget": episode_cap, "estimated_credits": estimate,
            "episode_spent": episode_spent, "today_spent": today_spent, "daily_cap": daily_cap}


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
    for position, request in enumerate(plan.get("shots", [])):
        if shot_indexes is not None and position not in shot_indexes:
            continue
        image = Path(str(request.get("image", "")))
        if not image.is_file() or image.stat().st_size == 0:
            problems.append(f"shot {position}: start image missing ({image})")
        if str(request.get("model", "")) not in mapping:
            problems.append(f"shot {position}: no Runway API id for model {request.get('model')}; add it via NOVA_RUNWAY_MODEL_MAP")
        if not compact_prompt(str(request.get("prompt", ""))):
            problems.append(f"shot {position}: empty prompt")
    check = reachable or host_reachable
    api_host = urllib.parse.urlparse(os.getenv("RUNWAY_API_BASE", API_BASE)).hostname or ""
    for host in (api_host, *download_hosts()):
        if host and not check(host):
            problems.append(f"network: cannot reach {host}; generated clips could not be used")
    return problems


def quote(plan: dict[str, Any], shot_indexes: list[int] | None = None, *, run_id: str = "",
          reachable: Callable[[str], bool] | None = None) -> dict[str, Any]:
    """What a run would cost and whether it could succeed, without spending anything."""
    rates = {option.model: option.credits_per_second for option in RUNWAY_MODELS}
    rows = []
    for position, request in enumerate(plan.get("shots", [])):
        if shot_indexes is not None and position not in shot_indexes:
            continue
        seconds = billed_seconds(float(request.get("duration", 5.0)))
        rows.append({
            "shot": int(request.get("shot_index", position)), "model": request.get("model"),
            "billed_seconds": seconds,
            "credits": round(rates.get(str(request.get("model")), 0.0) * seconds, 2),
            "prompt_chars": len(compact_prompt(str(request.get("prompt", "")))),
        })
    episode_cap = _cap("NOVA_RUNWAY_MAX_CREDITS")
    return {
        "shots": rows,
        "total_credits": round(sum(r["credits"] for r in rows), 2),
        "episode_spent": credits_spent(run_id=run_id) if run_id else 0.0,
        "today_spent": credits_spent(day=time.strftime("%Y-%m-%d", time.gmtime())),
        "episode_cap": episode_cap,
        "daily_cap": _cap("NOVA_RUNWAY_DAILY_CREDITS", str(episode_cap)),
        "problems": preflight(plan, shot_indexes, reachable=reachable),
    }


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


class RunwayVideoClient:
    """Minimal Runway REST client; transport and downloader are injectable for tests."""

    def __init__(self, api_secret: str, *, base_url: str | None = None, api_version: str = API_VERSION,
                 transport: Transport | None = None, downloader: Downloader | None = None) -> None:
        self.api_secret = api_secret
        self.base_url = (base_url or os.getenv("RUNWAY_API_BASE", API_BASE)).rstrip("/")
        self.api_version = api_version
        self._transport = transport or self._http
        self._download = downloader or (lambda url, path: urllib.request.urlretrieve(url, path))

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
                raise RuntimeError(f"Runway task {task_id} failed: {status.get('failure', state)}")
            time.sleep(poll_seconds)
        raise TimeoutError(f"Runway task {task_id} did not finish within {timeout_seconds:.0f}s")

    def download(self, url: str, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._download(url, path)
        return path


# Runway rejects promptText longer than this; the Director's full prompt is ~3.5k.
PROMPT_LIMIT = 1000
_PROMPT_LABELS = (
    "Targeted repair", "Performance arc", "Action chain", "Action", "Gaze", "Head", "Body", "Hands",
    "Face", "Secondary motion", "Root lock", "Overshoot mode", "Facial performance", "Pose accents",
    "Orientation lock", "Motion priority", "Timeline", "Camera", "Preserve", "Avoid",
)
# Most important first: what to fix, what happens, then what must not change.
_PROMPT_PRIORITY = (
    "Targeted repair", "Performance arc", "Action", "Gaze", "Head", "Hands", "Avoid", "Preserve",
    "Root lock", "Orientation lock", "Body", "Camera", "Face",
)


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
        "promptText": compact_prompt(str(request.get("prompt", ""))),
        "ratio": ratio,
        "duration": billed_seconds(float(request.get("duration", 5.0))),
    }


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
    """Execute (part of) a storyboard plan and download one clip per shot.

    ``shot_indexes`` limits execution to those plan positions, which is how a
    targeted repair regenerates only the failing shots. Nothing is submitted
    unless the budget gate and the no-cost preflight both pass.
    """
    gate = execution_gate(plan, shot_indexes, run_id=run_id)
    problems = preflight(plan, shot_indexes, reachable=reachable)
    if problems:
        raise RunwayExecutionBlocked("Runway preflight failed, no credits spent: " + "; ".join(problems))
    client = client or RunwayVideoClient(gate["api_secret"])
    mapping = model_ids()
    rates = {option.model: option.credits_per_second for option in RUNWAY_MODELS}
    clips: dict[int, str] = {}
    tasks = []
    for position, request in enumerate(plan.get("shots", [])):
        if shot_indexes is not None and position not in shot_indexes:
            continue
        payload = shot_payload(request, mapping)
        task_id = client.image_to_video(payload)
        shot = int(request.get("shot_index", position))
        record_spend(run_id=run_id, shot=shot, task_id=task_id, kind=kind, model=payload["model"],
                     credits=rates.get(str(request.get("model")), 0.0) * payload["duration"])
        url = client.wait(task_id, poll_seconds=poll_seconds)
        clip = client.download(url, Path(output_dir) / f"shot_{shot:02d}_{task_id[:8]}.mp4")
        clips[shot] = str(clip)
        tasks.append({"shot": shot, "task_id": task_id, "model": payload["model"], "duration": payload["duration"]})
    return {
        "provider": "runway",
        "clips": clips,
        "tasks": tasks,
        "estimated_credits": gate["estimated_credits"],
        "budget": gate["budget"],
    }


def required_credits_note(plan: dict[str, Any]) -> str:
    return f"~{math.ceil(billed_credits(plan))} credits for {len(plan.get('shots', []))} shots"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Quote a Runway storyboard plan without spending credits.")
    parser.add_argument("command", choices=["quote"])
    parser.add_argument("plan", type=Path)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    print(json.dumps(quote(json.loads(args.plan.read_text(encoding="utf-8")), run_id=args.run_id), indent=2))


if __name__ == "__main__":
    main()
