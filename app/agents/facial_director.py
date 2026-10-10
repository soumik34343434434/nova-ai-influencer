"""Temporal facial acting plans shared by local and hosted performers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FacialCue:
    t: float
    eye_open: float
    brow: float
    mouth_open: float
    gaze_lead: float
    gaze_x: float
    blink: float
    asymmetry: float
    emotion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ACTION_EMOTION = {
    "checks_phone": ("neutral", "surprised", "deadpan"),
    "sees_calendar": ("neutral", "annoyed", "deadpan"),
    "reaches_for_coffee": ("tired", "alert", "tired"),
    "glances_at_many_tabs": ("neutral", "overloaded", "deadpan"),
    "leans_toward_camera": ("neutral", "curious", "deadpan"),
    "deadpan_camera_reaction": ("reactive", "confused", "deadpan"),
    "finish_work": ("tired", "relieved", "deadpan"),
    "press_button": ("curious", "confident", "surprised"),
    "shock": ("neutral", "shocked", "stunned"),
    "overwhelmed": ("worried", "overloaded", "deflated"),
    "deadpan": ("neutral", "unimpressed", "deadpan"),
    "walk_in": ("sleepy", "content", "neutral"),
    "wave_hello": ("eager", "happy", "hopeful"),
    "yawn_stretch": ("tired", "exasperated", "sleepy"),
    "panic_run": ("worried", "alarmed", "panicked"),
    "curl_up": ("tired", "overwhelmed", "hiding"),
    "worried_close": ("uneasy", "worried", "worried"),
}


def build_facial_timeline(action: str, points: int = 13) -> list[dict[str, Any]]:
    """Create a small acting timeline with lead/peak/recovery separation."""
    count = max(5, int(points))
    start, peak, end = _ACTION_EMOTION.get(action, ("neutral", "reactive", "deadpan"))
    rows: list[dict[str, Any]] = []
    for i in range(count):
        t = i / max(1, count - 1)
        anticipation = max(0.0, min(1.0, (0.40 - t) / 0.28)) if t < 0.40 else 0.0
        impact = max(0.0, 1.0 - abs(t - 0.58) / 0.22)
        recovery = max(0.0, (t - 0.64) / 0.36)
        blink_window = 1.0 if 0.67 <= t <= 0.80 else 0.0
        gaze = min(1.0, anticipation * 0.72 + impact * 0.55)
        mouth = max(0.0, min(1.0, impact * (0.55 if peak in {"surprised", "alert", "reactive"} else 0.34)))
        brow = impact * (0.82 if peak in {"surprised", "alert", "curious"} else 0.48) - recovery * 0.25
        eye_open = 1.0 + impact * (0.22 if peak in {"surprised", "alert", "overloaded"} else -0.10 if peak == "annoyed" else 0.06)
        rows.append(
            FacialCue(
                t=round(t, 4),
                eye_open=round(max(0.45, min(1.35, eye_open)), 4),
                brow=round(max(-0.4, min(1.0, brow)), 4),
                mouth_open=round(mouth, 4),
                gaze_lead=round(gaze, 4),
                gaze_x=round((-0.55 if action in {"checks_phone", "glances_at_many_tabs"} else 0.35) * gaze, 4),
                blink=blink_window,
                asymmetry=round((0.16 if impact > 0.4 else 0.05) * (1.0 if action != "deadpan_camera_reaction" else -1.0), 4),
                emotion=start if t < 0.35 else peak if t < 0.72 else end,
            ).to_dict()
        )
    return rows
