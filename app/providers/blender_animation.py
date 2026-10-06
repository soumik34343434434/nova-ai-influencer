from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from app.providers.cartoon_media import _audio_for_shots, _caption_lines, _font
from app.providers.cartoon_voice import prepare_voice_performance
from PIL import Image, ImageDraw


class BlenderAnimationProvider:
    """Optional Blender-backed character-performance renderer for PIP."""

    def __init__(self, blender_bin: str | None = None) -> None:
        self.blender_bin = blender_bin or shutil.which("blender") or "/Applications/Blender.app/Contents/MacOS/Blender"

    def available(self) -> bool:
        return Path(self.blender_bin).exists() or shutil.which(self.blender_bin) is not None

    def _script(self, repo_root: Path) -> Path:
        return repo_root / "tools" / "blender" / "pip_cutout_rig_v2.py"

    def ensure_character_assets(self, repo_root: Path, character: str) -> Path:
        """Compile non-PIP characters on demand so a clean clone is render-ready."""
        if character.upper() == "PIP":
            return repo_root / "artifacts" / "cartoon" / "blender" / "pip_cutout_layers"
        out = repo_root / "artifacts" / "cartoon" / "blender" / f"{character.lower()}_cutout_layers"
        required = [out / f"{character.lower()}_torso.png", out / f"{character.lower()}_head.png", out / f"{character.lower()}_left_arm.png", out / f"{character.lower()}_right_arm.png", out / (f"{character.lower()}_scarf.png" if character.upper()=="MOMO" else f"{character.lower()}_backpack.png"), out / "manifest.json"]
        if all(path.exists() for path in required):
            return out
        compiler = repo_root / "tools" / "blender" / "compile_cartoon_character_asset.py"
        if not compiler.exists():
            raise FileNotFoundError(f"Character compiler missing: {compiler}")
        subprocess.run([__import__("sys").executable, str(compiler), character], cwd=str(repo_root), check=True, capture_output=True, text=True)
        if not all(path.exists() for path in required):
            raise RuntimeError(f"Character compilation did not produce a complete pack for {character}")
        return out

    def build_rig(
        self, repo_root: Path, output_blend: Path | None = None,
        resolution_percent: int = 35, voice_json: Path | None = None, frames: int = 190,
        character: str = "PIP", background_path: Path | None = None,
    ) -> Path:
        if not self.available():
            raise RuntimeError("Blender is not installed")
        self.ensure_character_assets(repo_root, character)
        output_blend = output_blend or repo_root / "artifacts" / "cartoon" / "blender" / "pip_cutout_rig_v2.blend"
        output_blend.parent.mkdir(parents=True, exist_ok=True)
        env = {
            "NOVA_REPO": str(repo_root),
            "NOVA_BLENDER_RESOLUTION": str(resolution_percent),
        }
        cmd = [self.blender_bin, "--background", "--python", str(self._script(repo_root)), "--", "--build", "--frames", str(frames), "--character", character, "--blend-output", str(output_blend)]
        if voice_json:
            cmd.extend(["--voice-json", str(voice_json)])
        if background_path:
            cmd.extend(["--background", str(background_path)])
        subprocess.run(cmd, cwd=str(repo_root), env={**__import__("os").environ, **env}, check=True, capture_output=True, text=True)
        return output_blend

    def render_preview(
        self, repo_root: Path, output_blend: Path | None = None, resolution_percent: int = 35,
        voice_json: Path | None = None, frames: int = 190, key_frames: list[int] | None = None,
        character: str = "PIP", background_path: Path | None = None,
    ) -> dict:
        blend = self.build_rig(repo_root, output_blend, resolution_percent, voice_json=voice_json, frames=frames, character=character, background_path=background_path)
        env = {
            "NOVA_REPO": str(repo_root),
            "NOVA_BLENDER_RESOLUTION": str(resolution_percent),
        }
        if key_frames:
            env["NOVA_KEY_FRAMES"] = ",".join(str(x) for x in key_frames)
        cmd = [self.blender_bin, "--background", "--python", str(self._script(repo_root)), "--", "--build", "--render", "--frames", str(frames), "--character", character, "--blend-output", str(blend)]
        if voice_json:
            cmd.extend(["--voice-json", str(voice_json)])
        if background_path:
            cmd.extend(["--background", str(background_path)])
        run = subprocess.run(cmd, cwd=str(repo_root), env={**__import__("os").environ, **env}, check=True, capture_output=True, text=True)
        out_dir = repo_root / "artifacts" / "cartoon" / "blender" / "pip_cutout_rig"
        frames_found = sorted(out_dir.glob("frame_*.png"))
        keys_found = sorted(out_dir.glob("key_*.png"))
        return {"blend": str(blend), "frame_count": len(frames_found), "key_frame_count": len(keys_found), "output_dir": str(out_dir), "stdout_tail": run.stdout[-1600:]}



    def render_episode(
        self, repo_root: Path, output_mp4: Path, voice_lines: list[str],
        beats: list[str], durations: list[float], sfx_kinds: list[str],
        resolution_percent: int = 100, action_assets: list[Path] | None = None,
        background_path: Path | None = None, character: str = "PIP", actions: list[str] | None = None,
    ) -> dict:
        """Render a complete PIP performance clip and return its media/metadata paths."""
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_character_assets(repo_root, character)
        work = output_mp4.parent / (output_mp4.stem + "_blender_work")
        work.mkdir(parents=True, exist_ok=True)
        voice_performance = prepare_voice_performance(work / "voice", voice_lines, beats, durations)
        reaction_tails = (0.48,0.58,0.60,0.82)
        # Keep the same effective clock for Blender, captions and audio: each
        # voice beat owns a small reaction tail after the spoken line.
        effective_shot_durations = [
            round(max(1.0, float(row.get("duration", 0.0) or 0.0)) + tail, 3)
            for row, tail in zip(voice_performance[:4], reaction_tails)
        ]
        total_seconds = sum(effective_shot_durations)
        frames = max(24, round(total_seconds * 24))
        voice_json = work / "voice_performance.json"
        rows=[]
        for i,row in enumerate(voice_performance):
            item=dict(row)
            if actions and i < len(actions):
                item["action"] = actions[i]
            rows.append(item)
        voice_json.write_text(json.dumps({"audio_sync": {"voice_performance": rows}}, indent=2), encoding="utf-8")
        voice_performance=rows
        if action_assets:
            missing=[str(path) for path in action_assets if not Path(path).exists()]
            if missing:
                raise FileNotFoundError("Missing Blender action assets: " + ", ".join(missing))
        blend = self.build_rig(repo_root, voice_json=voice_json, frames=frames, resolution_percent=resolution_percent, character=character, background_path=background_path)
        env = {"NOVA_REPO": str(repo_root), "NOVA_BLENDER_RESOLUTION": str(resolution_percent)}
        render_cmd = [self.blender_bin, "--background", "--python", str(self._script(repo_root)), "--", "--build", "--render", "--frames", str(frames), "--character", character, "--voice-json", str(voice_json), "--blend-output", str(blend)]
        if background_path:
            render_cmd.extend(["--background", str(background_path)])
        subprocess.run(render_cmd, cwd=str(repo_root), env={**os.environ, **env}, check=True, capture_output=True, text=True)

        frame_dir = repo_root / "artifacts" / "cartoon" / "blender" / "pip_cutout_rig"
        frames_found = sorted(frame_dir.glob("frame_*.png"))
        if len(frames_found) < frames:
            raise RuntimeError(f"Blender rendered {len(frames_found)} frames; expected {frames}")

        # Reuse the factory's caption semantics so Blender clips enter the same
        # caption/publish path as the existing renderer.
        captioned_dir = work / "captioned"
        captioned_dir.mkdir(exist_ok=True)
        cumulative = []
        cursor = 0
        total = sum(durations) or 1.0
        for duration in durations:
            count = round(frames * float(duration) / total)
            cumulative.append((cursor, min(frames, cursor + max(1, count))))
            cursor += count
        if cumulative:
            cumulative[-1] = (cumulative[-1][0], frames)
        for idx, frame in enumerate(frames_found[:frames]):
            shot = next((i for i,(a,b) in enumerate(cumulative) if a <= idx < b), len(cumulative)-1)
            caption = voice_lines[shot] if voice_lines else ""
            img = Image.open(frame).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")
            font = _font(66)
            lines = _caption_lines(caption)
            for row,line in enumerate(lines):
                box=draw.textbbox((0,0),line,font=font,stroke_width=2); tw=box[2]-box[0]
                draw.text(((img.width-tw)/2,img.height-390+row*74),line,font=font,fill=(255,255,255,255),stroke_width=7,stroke_fill=(0,0,0,235))
            df=_font(30); disclosure="AI-generated • fictional cartoon"; db=draw.textbbox((0,0),disclosure,font=df)
            draw.text((img.width-(db[2]-db[0])-42,img.height-56),disclosure,font=df,fill=(245,245,245,230))
            img.save(captioned_dir/f"frame_{idx+1:05d}.png")

        video_only = work / "video_only.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required")
        subprocess.run([ffmpeg,"-y","-framerate","24","-i",str(captioned_dir/'frame_%05d.png'),"-c:v","libx264","-pix_fmt","yuv420p","-crf","18",str(video_only)],check=True,capture_output=True,text=True)
        audio = _audio_for_shots(work / "audio", voice_lines, effective_shot_durations, sfx_kinds, ffmpeg, [row.get("performance_peak_fraction",0.4) for row in voice_performance], [0.0]*len(effective_shot_durations), voice_performance=voice_performance)
        if not audio:
            raise RuntimeError("Blender episode audio generation failed")
        subprocess.run([ffmpeg,"-y","-i",str(video_only),"-i",str(audio),"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac","-b:a","192k","-shortest",str(output_mp4)],check=True,capture_output=True,text=True)
        metadata={
            "engine":"blender-cutout-performance-rig-v4",
            "character":character,
            "frames":frames,
            "resolution_percent":resolution_percent,
            "action_assets":[str(p) for p in (action_assets or [])],
            "background_path":str(background_path) if background_path else None,
            "voice_performance":voice_performance,
            "captions":True,
            "audio_sync":{
                "voice_performance":voice_performance,
                "effective_shot_durations":effective_shot_durations,
                "sfx_peak_fractions":[row.get("performance_peak_fraction",0.4) for row in voice_performance],
            },
            "camera":"keyframed-damped-root-follow",
            "deformable_limbs":True,
            "facial_controls":["blink","gaze","mouth_shape","speech_micro_motion"],
        }
        meta_path=output_mp4.with_suffix('.motion.json'); meta_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        return {"video":str(output_mp4),"metadata":str(meta_path),"frames":frames,"resolution_percent":resolution_percent}
