"""Zero-cost photo-mix renderer for original cartoon comedy shorts."""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.providers.cartoon_audio import loudness_normalize, make_music_bed, make_sfx, make_voice
from app.providers.cartoon_voice import prepare_voice_performance


W, H = 1080, 1920
# SFX bus level relative to the dialog (dialog is unity); SFX sit well under speech.
SFX_BUS_GAIN = 0.45


def _font(size: int):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _fit_cover(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    scale = max(W / img.width, H / img.height)
    size = (int(img.width * scale), int(img.height * scale))
    img = img.resize(size, Image.Resampling.LANCZOS)
    x = max(0, (img.width - W) // 2)
    y = max(0, (img.height - H) // 2)
    return img.crop((x, y, x + W, y + H))


def _caption_lines(text: str, max_width: int = 920) -> list[str]:
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    font = _font(66)
    lines: list[str] = []
    current = ""
    for word in " ".join(text.split()).split():
        candidate = (current + " " + word).strip()
        if draw.textbbox((0, 0), candidate, font=font, stroke_width=1)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def render_frame(image: Path, caption: str, progress: float, zoom: float, shot: int = 0) -> Image.Image:
    base = _fit_cover(Image.open(image))
    impact = 0.0
    if shot == 1:
        impact = 8.0 * math.sin(progress * math.pi * 6.0) * (1.0 - progress)
    crop_w = max(1, int(base.width / zoom))
    crop_h = max(1, int(base.height / zoom))
    phase = progress * math.tau
    x = int((base.width - crop_w) * (0.5 + 0.14 * math.sin(phase)) + impact)
    y = int((base.height - crop_h) * (0.5 + 0.06 * math.cos(phase)))
    x = max(0, min(x, base.width - crop_w))
    y = max(0, min(y, base.height - crop_h))
    frame = base.crop((x, y, x + crop_w, y + crop_h)).resize((W, H), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(frame, "RGBA")
    lines = _caption_lines(caption)
    font = _font(66)
    line_h = 74
    y0 = H - 390
    for i, line in enumerate(lines):
        box = draw.textbbox((0, 0), line, font=font, stroke_width=2)
        tw = box[2] - box[0]
        draw.text(
            ((W - tw) / 2, y0 + i * line_h),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=7,
            stroke_fill=(0, 0, 0, 235),
        )

    disclosure = "AI-generated • fictional cartoon"
    df = _font(30)
    db = draw.textbbox((0, 0), disclosure, font=df)
    draw.text((W - (db[2] - db[0]) - 42, H - 56), disclosure, font=df, fill=(245, 245, 245, 230))
    return frame


def _audio_for_shots(
    work: Path,
    voice_lines: list[str],
    durations: list[float],
    sfx_kinds: list[str],
    ffmpeg: str,
    sfx_peak_fractions: list[float] | None = None,
    voice_start_fractions: list[float] | None = None,
    voice_performance: list[dict] | None = None,
) -> Path | None:
    """Build the episode mix on the same per-shot clock as the animation engine.

    Dialog and SFX are summed without amix's 1/N scaling (which made the mix about
    24 dB too quiet), a soft procedural music bed is ducked under the dialog, and
    the result is loudness-normalised to -14 LUFS with a -1.5 dBTP true-peak ceiling.
    """
    work.mkdir(parents=True, exist_ok=True)
    total = sum(durations)
    peak_fractions = sfx_peak_fractions or [0.0] * len(durations)
    voice_starts = voice_start_fractions or [0.04] * len(durations)

    inputs: list[str] = []
    filters: list[str] = []
    voice_labels: list[str] = []
    sfx_labels: list[str] = []
    input_index = 0
    elapsed = 0.0

    for i, (line, duration) in enumerate(zip(voice_lines, durations)):
        if line.strip():
            path = Path(voice_performance[i]["path"]) if voice_performance and i < len(voice_performance) else work / f"voice_{i:02d}.wav"
            made = path if path.exists() else make_voice(line.strip(), path)
            if made and path.exists():
                start = 0.0 if voice_performance else max(0.0, min(0.35, float(voice_starts[i]) if i < len(voice_starts) else 0.04))
                available = max(0.20, duration - start - 0.04)
                import wave
                with wave.open(str(path), "rb") as wav:
                    source_duration = wav.getnframes() / max(1, wav.getframerate())
                # Natural-rate rendering happens in cartoon_voice.py. Only use a
                # tiny safety correction here for pathological overrun; never
                # stretch a normal line to fit a shot.
                tempo = 1.0
                if source_duration > available * 1.08:
                    tempo = min(1.06, source_duration / available)
                delay = round((elapsed + start) * 1000)
                filters.append(
                    f"[{input_index}:a]atempo={tempo:.5f},apad,atrim=0:{available:.3f},adelay={delay}|{delay},volume=0.95[v{i}]"
                )
                inputs.extend(["-i", str(path)])
                voice_labels.append(f"[v{i}]")
                input_index += 1
        elapsed += duration

    elapsed = 0.0
    for i, (duration, sfx) in enumerate(zip(durations, sfx_kinds)):
        if sfx and sfx != "none":
            sfx_duration = min(0.36, max(0.10, duration * 0.24))
            path = work / f"sfx_{i:02d}.wav"
            make_sfx(sfx, sfx_duration, path)
            fraction = max(0.0, min(0.90, float(peak_fractions[i]) if i < len(peak_fractions) else 0.0))
            delay_seconds = elapsed + max(0.0, duration * fraction - sfx_duration * 0.35)
            delay = round(delay_seconds * 1000)
            inputs.extend(["-i", str(path)])
            filters.append(f"[{input_index}:a]adelay={delay}|{delay},volume={SFX_BUS_GAIN:.3f}[s{i}]")
            sfx_labels.append(f"[s{i}]")
            input_index += 1
        elapsed += duration

    if not voice_labels and not sfx_labels:
        return None

    mix_inputs: list[str] = []
    if voice_labels:
        bed = work / "music_bed.wav"
        make_music_bed(total, bed)
        bed_index = input_index
        inputs.extend(["-i", str(bed)])
        filters.append(f"{''.join(voice_labels)}amix=inputs={len(voice_labels)}:duration=longest:normalize=0[vmix]")
        filters.append("[vmix]asplit=2[vkey][vdry]")
        # The bed ducks under the dialog itself, so it stays soft but never fights the words.
        filters.append(f"[{bed_index}:a]atrim=0:{total:.3f}[bed]")
        filters.append("[bed][vkey]sidechaincompress=threshold=0.02:ratio=4:attack=15:release=300[duck]")
        mix_inputs += ["[vdry]", "[duck]"]
    if sfx_labels:
        filters.append(f"{''.join(sfx_labels)}amix=inputs={len(sfx_labels)}:duration=longest:normalize=0[fmix]")
        mix_inputs.append("[fmix]")
    filters.append(
        f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=longest:normalize=0,atrim=0:{total:.3f}[mix]"
    )
    raw_mix = work / "episode_mix_raw.wav"
    subprocess.run(
        [ffmpeg, "-y", *inputs,
         "-filter_complex", ";".join(filters),
         "-map", "[mix]", "-ar", "44100", "-ac", "1", str(raw_mix)],
        check=True, capture_output=True, text=True,
    )
    final_audio = work / "episode_audio.wav"
    loudness_normalize(raw_mix, final_audio)
    raw_mix.unlink(missing_ok=True)
    return final_audio


def render_photo_mix(
    images: list[Path],
    captions: list[str],
    durations: list[float],
    output: Path,
    voice_lines: list[str] | None = None,
    sfx_kinds: list[str] | None = None,
) -> Path:
    if not images or len(images) != len(captions) or len(images) != len(durations):
        raise ValueError("images, captions and durations must have equal non-zero lengths")
    voice_lines = voice_lines or captions
    sfx_kinds = sfx_kinds or ["none"] * len(images)
    if len(voice_lines) != len(images) or len(sfx_kinds) != len(images):
        raise ValueError("voice_lines and sfx_kinds must match image count")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")

    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / (output.stem + "_frames")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    fps = 8
    frame_count = 0
    for shot, (img, caption, duration) in enumerate(zip(images, captions, durations)):
        n = max(1, round(duration * fps))
        for i in range(n):
            progress = i / max(1, n - 1)
            zoom = 1.06 + 0.10 * min(progress, 1.0) + 0.03 * (shot % 2)
            if shot == 3:
                zoom += 0.07
            frame = render_frame(img, caption, progress, zoom, shot)
            path = work / f"frame_{frame_count:05d}.jpg"
            frame.save(path, format="JPEG", quality=88, optimize=True)
            frame_count += 1

    video_raw = output.with_suffix(".raw.mp4")
    subprocess.run(
        [
            ffmpeg, "-y", "-framerate", str(fps), "-i", str(work / "frame_%05d.jpg"),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-vf", "scale=1080:1920,format=yuv420p", str(video_raw),
        ],
        check=True, capture_output=True, text=True,
    )

    # The Free-friendly trick: generate convincing camera motion locally and
    # use FFmpeg motion interpolation to turn 8fps keyframes into smoother
    # 24fps animation. No paid video API is required.
    video_noaudio = output.with_suffix(".silent.mp4")
    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(video_raw),
            "-vf", "minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(video_noaudio),
        ],
        check=True, capture_output=True, text=True,
    )
    video_raw.unlink(missing_ok=True)

    audio = _audio_for_shots(work, voice_lines, durations, sfx_kinds, ffmpeg)
    if audio:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(video_noaudio), "-i", str(audio), "-shortest",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", str(output)],
            check=True, capture_output=True, text=True,
        )
    else:
        video_noaudio.replace(output)

    video_noaudio.unlink(missing_ok=True)
    return output
