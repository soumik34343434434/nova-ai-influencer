
from __future__ import annotations
import re
from typing import Any

CHARACTER_FIT_KEYWORDS = {
    "MOMO": ("cat", "cute", "office", "work", "awkward", "sleepy", "relatable", "meme", "reaction", "frustrat"),
    "PIP": ("robot", "tech", "ai", "future", "challenge", "gaming", "science", "glitch", "absurd"),
    "BUBU": ("bird", "cute", "dance", "music", "travel", "nature", "wholesome", "reaction"),
}
FORMAT_CAPABILITY = {"story_comedy":1.0,"reaction":.96,"pov":.94,"cute_animal":.98,"absurdist":.92,"listicle":.88,"trend_audio":.82,"challenge":.82,"transformation":.78,"dance":.72,"tutorial":.68}

CHARACTER_THEMES = {
    "MOMO": {"work", "office", "awkward", "sleepy", "relatable", "cat", "frustration", "amusement"},
    "PIP": {"tech", "ai", "robot", "gaming", "science", "glitch", "absurd", "curiosity"},
    "BUBU": {"bird", "dance", "music", "travel", "nature", "cute", "wholesome", "surprise"},
}
FORMAT_EMOTIONS = {
    "story_comedy": {"amusement", "cringe", "frustration"}, "reaction": {"surprise", "amusement", "cringe"},
    "pov": {"amusement", "anxiety", "frustration"}, "cute_animal": {"cute", "amusement"},
    "absurdist": {"amusement", "surprise", "curiosity"}, "dance": {"amusement", "awe"},
    "trend_audio": {"amusement", "nostalgia", "surprise"}, "challenge": {"anxiety", "amusement", "satisfaction"},
    "transformation": {"surprise", "satisfaction", "awe"}, "tutorial": {"curiosity", "satisfaction"},
    "listicle": {"curiosity", "amusement"}, "micro_short": {"surprise", "amusement"},
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _get(signal: Any, key: str, default: Any = "") -> Any:
    if isinstance(signal, dict):
        return signal.get(key, default)
    return getattr(signal, key, default)


def analyze_trend_transfer(signal: Any) -> dict[str, Any]:
    """Explain deterministic trend-to-NOVA fit; scores are inferences, inputs stay observations."""
    get = lambda key, default="": _get(signal, key, default)
    text = " ".join(str(get(k, "") or "") for k in ("title", "description", "category", "hashtags", "music")).lower()
    formats = tuple(str(x).lower() for x in (get("formats", ()) or ()))
    emotions = tuple(str(x).lower() for x in (get("emotions", ()) or ()))
    hooks = tuple(str(x).lower() for x in (get("hook_patterns", ()) or ()))
    words = set(re.findall(r"[a-z0-9]+", text))
    known_formats = [f for f in formats if f in FORMAT_CAPABILITY]
    best_format = max(known_formats, key=lambda f: FORMAT_CAPABILITY[f], default="")
    fit_format = min(1.0, .35 + .65 * min(1.0, len(known_formats) / 2))
    char_scores = {c: min(1.0, .3 + .14 * len(words & themes)) for c, themes in CHARACTER_THEMES.items()}
    best_character = max(char_scores, key=char_scores.get)
    emo_set = set(emotions)
    emo_fit = max((len(emo_set & FORMAT_EMOTIONS.get(f, set())) / max(1, len(emo_set)) for f in known_formats), default=.35)
    hook_fit = min(1.0, .25 + .25 * min(3, len(hooks)))
    feasibility = FORMAT_CAPABILITY.get(best_format, .55)
    saturation = float(get("saturation_score", 0) or 0)
    # Saturation is only a proxy for distinctiveness; owned-content comparison is unavailable here.
    originality = max(.1, min(.9, .65 - .45 * saturation))
    source_count = int(get("source_count", 0) or 0)
    evidence_confidence = min(.95, .2 + .16 * min(source_count, 3) + .12 * bool(hooks) + .10 * bool(formats) + .08 * bool(emotions))
    observed = []
    for label, key in (("source_count", "source_count"), ("mention_count", "mention_count"), ("views", "views"), ("likes", "likes"), ("comments", "comments"), ("shares", "shares"), ("music", "music"), ("duration_seconds", "duration_seconds")):
        value = get(key, 0 if key not in {"music"} else "")
        if value:
            observed.append({"field": label, "value": value})
    inferred = {"trend_to_format": round(fit_format, 4), "trend_to_character": round(char_scores[best_character], 4),
                "trend_to_emotion": round(emo_fit, 4), "trend_to_hook": round(hook_fit, 4),
                "originality_distance": round(originality, 4), "production_feasibility": round(feasibility, 4),
                "evidence_confidence": round(evidence_confidence, 4)}
    warnings = []
    if not known_formats: warnings.append("No format signal; format fit uses a neutral prior.")
    if not emotions: warnings.append("No emotion signal detected.")
    if not hooks: warnings.append("No hook pattern detected.")
    if not source_count: warnings.append("Source coverage is unknown.")
    warnings.append("Originality distance uses saturation as a proxy; no NOVA-owned content comparison is available.")
    if not any(x.get("field") == "duration_seconds" for x in observed): warnings.append("Duration and production requirements are unobserved.")
    if any(term in text for term in ("no dance", "not a dance", "without music")) and any(f in {"dance", "trend_audio"} for f in known_formats):
        warnings.append("Contradiction: format label suggests movement/audio but text explicitly negates it.")
        fit_format *= .45
        feasibility *= .8
        inferred["trend_to_format"] = round(fit_format, 4)
        inferred["production_feasibility"] = round(feasibility, 4)
    reasons = []
    if best_format: reasons.append(f"{best_format} is a supported NOVA format (capability prior {feasibility:.2f}).")
    if char_scores[best_character] >= .58: reasons.append(f"{best_character} matches the trend's observed themes.")
    if not reasons: reasons.append("Available trend signals provide little specific evidence of NOVA fit.")
    return {"score": round(sum(inferred.values()) / len(inferred), 4), "dimensions": inferred,
            "best_character": best_character, "best_format": best_format,
            "observed_evidence": observed, "inferred_fit": inferred,
            "reasons": reasons, "missing_evidence_warnings": warnings}


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
    return {"content_opportunity_score": round(content_opportunity, 4), "trend_strength_score": round(trend_strength, 4), "production_feasibility_score": round(production_feasibility, 4), "hook_strength_score": round(hook_strength, 4), "emotion_fit_score": round(emotion_fit, 4), "originality_score": round(originality, 4), "character_fit": {k: round(v, 4) for k, v in character_fit.items()}, "best_character": best_character, "best_format": best_format, "trend_transfer_analysis": analyze_trend_transfer(signal)}
