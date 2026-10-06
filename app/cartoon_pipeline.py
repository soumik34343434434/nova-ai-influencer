"""Production pipeline for original, family-friendly cartoon comedy shorts."""
from __future__ import annotations

import json
from pathlib import Path

from app.agents.cartoon_content import pick_episode
from app.providers.cartoon_media import render_photo_mix


ROOT = Path(__file__).resolve().parents[1]
CARTOON_ASSETS = ROOT / "assets" / "cartoon"
ACTION_DIR = CARTOON_ASSETS / "actions"


def build_pip_episode(output_dir: Path | None = None) -> dict:
    episode = pick_episode("smart kitchen")
    frames = [
        ACTION_DIR / "pip_press_button.png",
        ACTION_DIR / "pip_shock.png",
        ACTION_DIR / "pip_overwhelmed.png",
        ACTION_DIR / "pip_deadpan.png",
    ]
    missing = [str(p) for p in frames if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing production frames: " + ", ".join(missing))

    output_dir = output_dir or ROOT / "artifacts" / "cartoon"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "pip_vs_smart_kitchen.mp4"
    captions = [
        "PIP asked the smart kitchen to make ONE sandwich.",
        "The kitchen made TWELVE.",
        "PIP immediately regretted everything.",
        "Apparently, the kitchen misunderstood 'one'.",
    ]
    durations = [2.6, 2.8, 2.4, 3.0]
    render_photo_mix(frames, captions, durations, output)

    metadata = {
        "title": episode.title,
        "character": episode.character,
        "hook": episode.hook,
        "caption": episode.caption,
        "video": str(output),
        "family_friendly": True,
        "original_characters": True,
        "ai_generated": True,
        "audience_classification_review_required": True,
    }
    (output_dir / "pip_vs_smart_kitchen.json").write_text(
        json.dumps(metadata, indent=2)
    )
    return metadata


if __name__ == "__main__":
    print(json.dumps(build_pip_episode(), indent=2))
