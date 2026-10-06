"""Provider-neutral character motion layer for NOVA.

The router keeps the pipeline working locally at zero software cost while
allowing a future hosted AI motion service to be plugged in through a command
adapter without changing storyboard, QA, or publishing code.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.providers.local_puppet import make_puppet_video


@dataclass(frozen=True)
class MotionResult:
    output: Path
    provider: str
    mode: str


class MotionProvider(Protocol):
    name: str

    def render(self, image: Path, audio: Path, output: Path, duration: float, shot_index: int) -> MotionResult:
        ...


class LocalMotionProvider:
    name = "local"

    def render(self, image: Path, audio: Path, output: Path, duration: float, shot_index: int) -> MotionResult:
        result = make_puppet_video(image, audio, output, duration, shot_index)
        return MotionResult(result, self.name, "procedural_zero_cost")


class CommandMotionProvider:
    """Adapter for an externally hosted AI motion service.

    Configure NOVA_MOTION_COMMAND with a shell-free argv template represented
    as a JSON-like shell string, using these placeholders:
    {image}, {audio}, {output}, {duration}, {shot_index}
    """

    name = "command"

    def __init__(self, template: str):
        if not template.strip():
            raise ValueError("NOVA_MOTION_COMMAND cannot be empty")
        self.template = template

    def render(self, image: Path, audio: Path, output: Path, duration: float, shot_index: int) -> MotionResult:
        args = self.template.format(
            image=shlex.quote(str(image)),
            audio=shlex.quote(str(audio)),
            output=shlex.quote(str(output)),
            duration=f"{duration:.3f}",
            shot_index=shot_index,
        )
        completed = subprocess.run(["/bin/zsh", "-lc", args], capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr[-4000:] or "external motion provider failed")
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("external motion provider did not produce the expected output")
        return MotionResult(output, self.name, "external_command")


class MotionRouter:
    """Selects the best configured motion provider, with safe local fallback."""

    def __init__(self, provider: MotionProvider | None = None):
        if provider is not None:
            self.provider = provider
            return
        mode = os.getenv("NOVA_MOTION_PROVIDER", "local").strip().lower()
        if mode == "command":
            command = os.getenv("NOVA_MOTION_COMMAND", "")
            self.provider = CommandMotionProvider(command) if command else LocalMotionProvider()
        else:
            self.provider = LocalMotionProvider()

    @property
    def provider_name(self) -> str:
        return self.provider.name

    def render(self, image: Path, audio: Path, output: Path, duration: float, shot_index: int) -> MotionResult:
        return self.provider.render(image, audio, output, duration, shot_index)
