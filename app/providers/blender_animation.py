from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class BlenderAnimationProvider:
    """Optional Blender-backed 2D character performance renderer.

    Blender is an external free/open-source tool, so the core factory remains
    provider-independent. This adapter makes the rig deterministic and testable.
    """

    def __init__(self, blender_bin: str | None = None) -> None:
        self.blender_bin = blender_bin or shutil.which("blender") or "/Applications/Blender.app/Contents/MacOS/Blender"

    def available(self) -> bool:
        return Path(self.blender_bin).exists() or shutil.which(self.blender_bin) is not None

    def build_rig(self, repo_root: Path, output_blend: Path | None = None, resolution_percent: int = 35) -> Path:
        if not self.available():
            raise RuntimeError("Blender is not installed")
        output_blend = output_blend or repo_root / "artifacts" / "cartoon" / "blender" / "pip_cutout_rig_v2.blend"
        script = repo_root / "tools" / "blender" / "pip_cutout_rig_v2.py"
        env = {"NOVA_REPO": str(repo_root), "NOVA_BLENDER_RESOLUTION": str(resolution_percent)}
        cmd = [self.blender_bin, "--background", "--python", str(script), "--", "--build", "--blend-output", str(output_blend)]
        subprocess.run(cmd, cwd=str(repo_root), env={**__import__("os").environ, **env}, check=True, capture_output=True, text=True)
        return output_blend

    def render_preview(self, repo_root: Path, output_blend: Path | None = None, resolution_percent: int = 35) -> dict:
        blend = self.build_rig(repo_root, output_blend, resolution_percent)
        script = repo_root / "tools" / "blender" / "pip_cutout_rig_v2.py"
        env = {"NOVA_REPO": str(repo_root), "NOVA_BLENDER_RESOLUTION": str(resolution_percent)}
        cmd = [self.blender_bin, "--background", "--python", str(script), "--", "--build", "--render", "--blend-output", str(blend)]
        run = subprocess.run(cmd, cwd=str(repo_root), env={**__import__("os").environ, **env}, check=True, capture_output=True, text=True)
        frames = sorted((repo_root / "artifacts" / "cartoon" / "blender").glob("frame_*.png"))
        return {"blend": str(blend), "frame_count": len(frames), "stdout_tail": run.stdout[-1200:]}
