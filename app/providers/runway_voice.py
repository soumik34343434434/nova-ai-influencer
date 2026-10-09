"""Runway speech backend with expressive script shaping and task polling.

Uses Runway's model-router audio endpoint so NOVA can keep provider choice
behind configuration while retaining deterministic voice timing metadata.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any


API_URL = "https://api.dev.runwayml.com/v1/generate/audio"
TASK_URL = "https://api.dev.runwayml.com/v1/tasks/{task_id}"
API_VERSION = "2024-11-06"


def render_runway_script(text: str, beat: str) -> str:
    clean = " ".join(str(text).split())
    if not clean:
        return clean
    if beat == "setup":
        return clean.rstrip(".!?") + ". [pause]"
    if beat == "escalation":
        return clean.rstrip(".!?") + " [pause]"
    if beat == "reaction":
        return "[pause] " + clean.rstrip(".!?") + "?"
    return "[pause] " + clean.rstrip(".!?") + ". [pause]"


def build_runway_speech_request(
    text: str,
    config_id: str,
    voice: Any | None = None,
    duration: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not str(config_id).strip():
        raise ValueError("config_id is required")
    if len(text) > 2500:
        raise ValueError("Runway expressive Eleven v4 scripts must be <= 2500 characters")
    payload: dict[str, Any] = {
        "configId": config_id,
        "input": {"type": "speech", "promptText": text},
    }
    if voice is not None:
        payload["input"]["voice"] = voice
    if duration is not None:
        if not 0.1 <= float(duration) <= 120.0:
            raise ValueError("duration must be between 0.1 and 120 seconds")
        payload["input"]["duration"] = float(duration)
    if dry_run:
        payload["dryRun"] = True
    return payload


def _find_output_url(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith("https://"):
        return value
    if isinstance(value, dict):
        for key in ("url", "audioUrl", "audio_url", "output"):
            found = _find_output_url(value.get(key))
            if found:
                return found
        for child in value.values():
            found = _find_output_url(child)
            if found:
                return found
    if isinstance(value, list):
        for child in value:
            found = _find_output_url(child)
            if found:
                return found
    return None


class RunwayVoiceProvider:
    name = "runway"

    def __init__(self, api_secret: str | None = None, config_id: str | None = None, api_version: str = API_VERSION) -> None:
        self.api_secret = api_secret or os.getenv("RUNWAYML_API_SECRET", "")
        self.config_id = config_id or os.getenv("RUNWAY_AUDIO_CONFIG_ID", "")
        self.api_version = api_version

    def request_payload(self, text: str, beat: str, voice: Any | None = None, duration: float | None = None, dry_run: bool = False) -> dict[str, Any]:
        return build_runway_speech_request(render_runway_script(text, beat), self.config_id, voice, duration, dry_run)

    def _request(self, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_secret:
            raise RuntimeError("RUNWAYML_API_SECRET is required for Runway voice generation")
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            url,
            data=body,
            method="POST" if payload is not None else "GET",
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

    def synthesize(
        self,
        text: str,
        beat: str,
        output: Path,
        voice: Any | None = None,
        duration: float | None = None,
        poll_seconds: float = 2.0,
        timeout_seconds: float = 300.0,
    ) -> dict[str, Any]:
        payload = self.request_payload(text, beat, voice, duration)
        started = time.monotonic()
        task = self._request(API_URL, payload)
        task_id = str(task.get("id", ""))
        if not task_id:
            raise RuntimeError("Runway did not return a task id")
        while time.monotonic() - started < timeout_seconds:
            status = self._request(TASK_URL.format(task_id=task_id))
            state = str(status.get("status", "")).upper()
            if state in {"SUCCEEDED", "SUCCESS", "COMPLETED"}:
                url = _find_output_url(status.get("output", status))
                if not url:
                    raise RuntimeError("Runway task completed without an audio output URL")
                output.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, output)
                return {"task_id": task_id, "output": str(output), "status": state, "provider": self.name}
            if state in {"FAILED", "CANCELLED", "CANCELED"}:
                raise RuntimeError(f"Runway voice task failed: {status.get('failure', status)}")
            time.sleep(poll_seconds)
        raise TimeoutError(f"Runway voice task {task_id} did not finish within timeout")
