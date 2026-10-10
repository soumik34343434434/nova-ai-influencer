"""Runway image-to-video execution for Director storyboard plans.

``RunwayPerformanceProvider.build_storyboard_plan`` decides *what* each shot
should be (model, prompt, duration) without spending credits. This module is
the opt-in worker that actually executes such a plan and downloads one clip per
shot for the Editor to assemble.

Execution is fail-closed. It requires all of:

* ``NOVA_RUNWAY_EXECUTE=true``
* ``RUNWAYML_API_SECRET``
* ``NOVA_RUNWAY_MAX_CREDITS`` at least the plan's billed estimate

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
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from app.providers.ai_performance import RUNWAY_MODELS

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


def execution_gate(plan: dict[str, Any], shot_indexes: list[int] | None = None) -> dict[str, Any]:
    """Return the execution config or raise RunwayExecutionBlocked explaining why."""
    if os.getenv("NOVA_RUNWAY_EXECUTE", "false").strip().lower() != "true":
        raise RunwayExecutionBlocked("Runway execution is disabled; set NOVA_RUNWAY_EXECUTE=true to allow spending credits")
    secret = os.getenv("RUNWAYML_API_SECRET", "").strip()
    if not secret:
        raise RunwayExecutionBlocked("RUNWAYML_API_SECRET is required for Runway execution")
    try:
        budget = float(os.getenv("NOVA_RUNWAY_MAX_CREDITS", "0") or 0)
    except ValueError as exc:
        raise RunwayExecutionBlocked("NOVA_RUNWAY_MAX_CREDITS must be a number") from exc
    estimate = billed_credits(plan, shot_indexes)
    if estimate > budget:
        raise RunwayExecutionBlocked(
            f"Runway plan needs about {estimate:.0f} credits but NOVA_RUNWAY_MAX_CREDITS is {budget:.0f}"
        )
    return {"api_secret": secret, "budget": budget, "estimated_credits": estimate}


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
        "promptText": str(request.get("prompt", ""))[:1000],
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
) -> dict[str, Any]:
    """Execute (part of) a storyboard plan and download one clip per shot.

    ``shot_indexes`` limits execution to those plan positions, which is how a
    targeted repair regenerates only the failing shots.
    """
    gate = execution_gate(plan, shot_indexes)
    client = client or RunwayVideoClient(gate["api_secret"])
    mapping = model_ids()
    clips: dict[int, str] = {}
    tasks = []
    for position, request in enumerate(plan.get("shots", [])):
        if shot_indexes is not None and position not in shot_indexes:
            continue
        payload = shot_payload(request, mapping)
        task_id = client.image_to_video(payload)
        url = client.wait(task_id, poll_seconds=poll_seconds)
        shot = int(request.get("shot_index", position))
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
