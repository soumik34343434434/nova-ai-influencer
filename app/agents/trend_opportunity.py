
from __future__ import annotations
from typing import Any

CHARACTER_FIT_KEYWORDS = {
    "MOMO": ("cat", "cute", "office", "work", "awkward", "sleepy", "relatable", "meme", "reaction", "frustrat"),
    "PIP": ("robot", "tech", "ai", "future", "challenge", "gaming", "science", "glitch", "absurd"),
    "BUBU": ("bird", "cute", "dance", "music", "travel", "nature", "wholesome", "reaction"),
}
FORMAT_CAPABILITY = {"story_comedy":1.0,"reaction":.96,"pov":.94,"cute_animal":.98,"absurdist":.92,"listicle":.88,"trend_audio":.82,"challenge":.82,"transformation":.78,"dance":.72,"tutorial":.68}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _get(signal: Any, key: str, default: Any = "") -> Any:
    if isinstance(signal, dict):
        return signal.get(key, default)
    return getattr(signal, key, default)


def score_content_opportunity(signal: Any) -> dict[str, Any]:
    text = " ".join(str(_get(signal, k, "")) for k in ("title", "description", "category", "hashtags", "music")).lower()
    formats = tuple(str(x) for x in (_get(signal, "formats", ()) or ()))
    emotions = tuple(str(x) for x in (_get(signal, "emotions", ()) or ()))
    hook_patterns = tuple(_get(signal, "hook_patterns", ()) or ())
    best_format = max(formats, key=lambda f: FORMAT_CAPABILITY.get(f, 0.55), default="")
    production_feasibility = FORMAT_CAPABILITY.get(best_format, 0.55)
    hook_strength = _clamp(.35 + .22 * min(2, len(hook_patterns)) + .12 * bool(_get(signal, "music", "")))
    emotion_fit = _clamp(.35 + .18 * min(3, len(emotions)))
    originality = _clamp(1.0 - .55 * float(_get(signal, "saturation_score", 0.0)))
    trend_strength = _clamp(.35 * float(_get(signal, "velocity_score", 0)) + .25 * float(_get(signal, "freshness_score", 0)) + .20 * float(_get(signal, "cross_platform_score", 0)) + .20 * float(_get(signal, "novelty_score", 0)))
    character_fit = {}
    for character, keywords in CHARACTER_FIT_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in text)
        character_fit[character] = _clamp(.30 + .10 * min(5, hits) + .12 * emotion_fit)
    best_character = max(character_fit, key=character_fit.get)
    content_opportunity = _clamp(.30 * trend_strength + .20 * production_feasibility + .18 * hook_strength + .14 * emotion_fit + .10 * originality + .08 * character_fit[best_character])
    return {"content_opportunity_score": round(content_opportunity, 4), "trend_strength_score": round(trend_strength, 4), "production_feasibility_score": round(production_feasibility, 4), "hook_strength_score": round(hook_strength, 4), "emotion_fit_score": round(emotion_fit, 4), "originality_score": round(originality, 4), "character_fit": {k: round(v, 4) for k, v in character_fit.items()}, "best_character": best_character, "best_format": best_format}
