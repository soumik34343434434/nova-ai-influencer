"""Production pipeline for original, family-friendly cartoon comedy shorts."""
from __future__ import annotations

import json
from pathlib import Path

from app.agents.cartoon_content import pick_episode
from app.agents.cartoon_factory import action_asset, asset_path, list_episode_specs, quality_gate
from app.providers.cartoon_media import render_photo_mix


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "cartoon"


def build_episode(slug: str = "smart_kitchen", output_dir: Path | None = None) -> dict:
    spec = next((s for s in list_episode_specs(10) if s.slug == slug), None)
    if spec is None:
        raise ValueError(f"Unknown cartoon episode slug: {slug}")
    allowed, reasons = quality_gate(spec)
    if not allowed:
        raise RuntimeError("Episode blocked by quality gate: " + ", ".join(reasons))

    frames = []
    for beat in spec.beats:
        frame = action_asset(spec.character, beat["asset_action"])
        if frame is None:
            raise FileNotFoundError(
                f"Missing action asset for {spec.character}/{beat['asset_action']}"
            )
        frames.append(frame)

    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{spec.slug}.mp4"
    captions = [b["voice"] for b in spec.beats]
    durations = [b["duration"] for b in spec.beats]
    voice_lines = captions
    sfx_kinds = [b["sfx"] for b in spec.beats]

    render_photo_mix(
        frames,
        captions,
        durations,
        output,
        voice_lines=voice_lines,
        sfx_kinds=sfx_kinds,
    )
    metadata = {
        "title": spec.title,
        "character": spec.character,
        "hook": spec.hook,
        "caption": spec.caption,
        "video": str(output),
        "duration_seconds": round(sum(durations), 2),
        "family_friendly": True,
        "original_characters": True,
        "ai_generated": True,
        "audience_classification_review_required": True,
        "quality_gate": {"allowed": True, "reasons": []},
    }
    (output_dir / f"{spec.slug}.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def plan_batch(limit: int = 5) -> list[dict]:
    plans = []
    for spec in list_episode_specs(limit):
        allowed, reasons = quality_gate(spec)
        plans.append(
            {
                "slug": spec.slug,
                "title": spec.title,
                "character": spec.character,
                "hook": spec.hook,
                "action_asset_coverage": sum(
                    1 for b in spec.beats
                    if action_asset(spec.character, b["asset_action"])
                ) / max(1, len(spec.beats)),
                "publish_ready": allowed,
                "blocked_reasons": reasons,
            }
        )
    return plans


def build_pip_episode(output_dir: Path | None = None) -> dict:
    # Backward-compatible entry point used by earlier tests/scripts.
    episode = pick_episode("smart kitchen")
    return build_episode("smart_kitchen", output_dir)


if __name__ == "__main__":
    print(json.dumps({"plans": plan_batch(5), "built": build_episode()}, indent=2))
