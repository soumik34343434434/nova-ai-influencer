"""Temporal cartoon acting specs for NOVA Director AI."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
from app.agents.performance import plan_performance
from app.agents.facial_director import build_facial_timeline
from app.agents.reference_preflight import required_reference_props

@dataclass(frozen=True)
class PerformancePhase:
    name: str
    start: float
    end: float
    intensity: float
    purpose: str

@dataclass(frozen=True)
class ShotPerformanceSpec:
    duration: float
    emotion_start: str
    emotion_peak: str
    emotion_end: str
    action: str
    gaze: str
    head_motion: str
    body_motion: str
    hands: str
    facial_animation: tuple[str, ...]
    phases: tuple[PerformancePhase, ...]
    camera: str
    continuity: tuple[str, ...]
    negative_constraints: tuple[str, ...]
    micro_beats: tuple[str, ...] = ()
    secondary_motion: str = ""
    root_lock: str = "anchor the character's feet/base in screen space"
    overshoot_mode: str = "pose-space overshoot; no world-space displacement"
    facial_performance: tuple[str, ...] = ()
    pose_accents: tuple[str, ...] = ()
    orientation_lock: str = "keep the chest and face mostly front-facing; no large yaw or turn-away"
    motion_priority: str = "prioritize readable upper-body action over facial-only motion"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["phases"] = [asdict(p) for p in self.phases]
        for k in (
            "facial_animation",
            "continuity",
            "negative_constraints",
            "micro_beats",
        ):
            d[k] = list(d[k])
        return d

ACTIONS = {
    "checks_phone": ("relaxed","surprised","deadpan","glance to phone","scene -> phone -> camera","small check then recoil","anticipate -> action -> recoil -> settle","one hand holds phone; other stays stable"),
    "finish_work": ("tired","relieved","deadpan","finish the last task and relax","desk -> camera","small exhale then settle","anticipate -> release -> settle","hands lower naturally and remain coherent"),
    "sees_calendar": ("neutral","annoyed","deadpan","notice calendar then point","scene -> calendar -> camera","tiny turn then pull back","anticipate -> action -> recoil -> settle","controlled point then return to rest"),
    "reaches_for_coffee": ("tired","alert","tired","reach for coffee","camera -> coffee","small follow-through","weight shift -> action -> settle","one hand reaches with coherent elbow/wrist"),
    "glances_at_many_tabs": ("neutral","overloaded","deadpan","scan tabs then freeze","left -> right -> camera","small side-to-side follow","anticipate -> scan -> recoil -> settle","hands remain stable"),
    "leans_toward_camera": ("neutral","curious","deadpan","lean toward camera","camera","controlled forward tilt","anticipate -> lean -> settle","hands remain stable"),
    "deadpan_camera_reaction": ("reactive","confused","deadpan","small recoil then freeze","camera","tiny recoil then stillness","anticipate -> impact -> recoil -> settle","hands stay planted"),
    "press_button": ("curious","surprised","deadpan","press one button","button -> result -> camera","press then recoil from the result","anticipate -> press -> recoil -> settle","one hand presses; other stays stable"),
    "shock": ("neutral","shocked","stunned","jump back from the problem","problem -> camera","big recoil with arms up","anticipate -> impact -> recoil -> settle","arms rise together and return coherently"),
    "overwhelmed": ("worried","overloaded","deflated","sink under the problem","problem -> camera","slow sag then small shake","anticipate -> sag -> settle","hands stay connected to the body"),
    "deadpan": ("neutral","unimpressed","deadpan","hold a flat stare","camera","tiny head tilt then stillness","anticipate -> hold -> settle","hands stay planted"),
    "walk_in": ("sleepy","content","neutral","walk slowly into frame","path -> camera","small step bounce","anticipate -> step -> settle","paws alternate; scarf trails"),
    "wave_hello": ("eager","happy","hopeful","wave one wing","camera","bouncy wave with follow-through","anticipate -> wave -> settle","one wing waves; other stays close"),
    "yawn_stretch": ("tired","exasperated","sleepy","huge yawn with paws up","camera -> sky","stretch up then droop","anticipate -> stretch -> recoil -> settle","both paws rise together then drop"),
    "panic_run": ("worried","alarmed","panicked","sprint across frame","threat -> escape route","fast bounce with flapping wings","anticipate -> run -> recoil -> settle","wings flap symmetrically; feet alternate"),
    "curl_up": ("tired","overwhelmed","hiding","curl into a ball","floor","slow squash into a ball","anticipate -> squash -> settle","paws tuck under the body"),
    "worried_close": ("uneasy","worried","worried","close-up worried look","camera","tiny wobble then hold","anticipate -> react -> settle","face only; keep features stable"),
}
DEFAULT = ("neutral","reactive","deadpan","small reaction","scene -> camera","small reaction then settle","anticipate -> action -> recoil -> settle","keep hands stable and coherent")

ACTION_CHAINS: dict[str, tuple[str, ...]] = {
    "finish_work": (
        "anticipation: tiny final-task tension before finishing",
        "eye_lead: eyes confirm the last item is complete",
        "head_follow: head lifts slightly as the work ends",
        "primary_action: small finishing gesture and release",
        "impact: brief relief expression",
        "overshoot: tiny pose-space release while the root stays planted",
        "recoil: shoulders settle back toward neutral",
        "secondary_motion: scarf/fur settle after the release",
        "settle: relaxed deadpan resting pose",
    ),
    "checks_phone": (
        "anticipation: tiny body preparation while the eyes first notice the phone",
        "eye_lead: eyes move toward the phone before the head moves",
        "head_follow: head and upper torso follow the eye direction with a small delayed turn",
        "primary_action: brief readable look-down toward the phone",
        "impact: a compact facial change on recognition of the phone content",
        "overshoot: add a small pose-space lean past the neutral reaction point while the feet/root stay planted and chest remains mostly front-facing",
        "recoil: connected head, shoulders and torso recover toward the starting screen position without turning away",
        "secondary_motion: scarf and fur lag behind the torso, then catch up",
        "settle: return to a controlled deadpan pose without freezing abruptly; finish front-facing and fully in frame",
    ),
    "sees_calendar": (
        "anticipation: eyes notice the calendar before the head turns",
        "eye_lead: gaze shifts to the calendar",
        "head_follow: head and torso follow the gaze",
        "primary_action: small readable point toward the calendar",
        "impact: annoyed recognition",
        "overshoot: point and head add a small pose-space extension while the root stays planted",
        "recoil: arm and torso return together",
        "secondary_motion: scarf/fur follow the recoil",
        "settle: controlled deadpan recovery",
    ),
    "reaches_for_coffee": (
        "anticipation: weight shifts before the reach",
        "eye_lead: gaze settles on the coffee",
        "head_follow: head follows the target",
        "primary_action: arm reaches with coherent elbow and wrist",
        "impact: brief alert expression on contact",
        "overshoot: hand reaches a small pose-space extension before stopping while the body base stays anchored",
        "recoil: arm and shoulder recover naturally",
        "secondary_motion: torso and fur follow the arm movement",
        "settle: return to the tired resting pose",
    ),
    "glances_at_many_tabs": (
        "anticipation: prepare the head before scanning",
        "eye_lead: eyes scan left to right first",
        "head_follow: head follows the gaze with small delayed turns",
        "primary_action: readable multi-tab scan",
        "impact: overloaded recognition",
        "overshoot: final gaze and head add a small pose-space extension while the root stays planted",
        "recoil: head returns with a small controlled correction",
        "secondary_motion: scarf/fur follow the side-to-side movement",
        "settle: freeze only after a visible settling motion",
    ),
    "leans_toward_camera": (
        "anticipation: subtle weight shift backward",
        "eye_lead: eyes lock to camera",
        "head_follow: head begins the forward tilt",
        "primary_action: upper body leans toward camera",
        "impact: curious facial peak",
        "overshoot: tiny extra forward pose extension while the root stays planted",
        "recoil: controlled return from the overshoot",
        "secondary_motion: scarf/fur trail the torso",
        "settle: return to neutral without a hard stop",
    ),
}

DEFAULT_CHAIN = (
    "anticipation: prepare the body before the action",
    "eye_lead: establish the attention target",
    "head_follow: let the head follow the gaze",
    "primary_action: perform one clear readable action",
    "impact: show the emotional peak",
    "overshoot: travel slightly beyond the neutral endpoint",
    "recoil: recover through connected body motion",
    "secondary_motion: add restrained follow-through",
    "settle: ease into the final pose",
)


FACIAL_PERFORMANCE = {
    "checks_phone": (
        "eye lead is clearly visible: pupils/gaze shift toward the phone before the head moves",
        "eyelids narrow slightly during the look-down, then open wider at recognition",
        "brows lift briefly on the impact beat, then relax",
        "mouth makes a tiny surprised opening on impact, then returns toward deadpan",
        "one restrained blink happens after the reaction, not during the primary action",
    ),
    "sees_calendar": (
        "eyes lead toward the calendar before the head turns",
        "eyelids tighten subtly as annoyance appears",
        "brows compress on recognition, then release",
        "mouth gives a tiny irritated reaction before settling",
    ),
    "reaches_for_coffee": (
        "eyes lock onto the coffee before the hand reaches",
        "eyelids lift slightly at contact",
        "brows rise with the alert beat",
        "mouth makes a tiny relieved reaction, then returns to tired neutral",
    ),
    "glances_at_many_tabs": (
        "eyes scan left-to-right before the head follows",
        "eyelids widen during the overloaded peak",
        "brows lift unevenly as the number of tabs registers",
        "mouth holds a tiny overwhelmed shape before deadpan recovery",
    ),
    "leans_toward_camera": (
        "eyes stay locked to camera while the upper body leans",
        "eyelids open slightly at the curious peak",
        "brows lift subtly, then settle",
        "mouth makes a tiny curious change without lip-sync jitter",
    ),
}

POSE_ACCENTS = {
    "checks_phone": (
        "tiny upper-body squash on the reaction peak followed by a restrained return to neutral",
        "head and shoulders overshoot by pose only; feet/root remain fixed",
        "scarf/fur follow the torso with a short delayed lag and catch-up",
    ),
    "sees_calendar": (
        "small upper-body compression on annoyed recognition, then release",
        "pointing arm adds a restrained pose-space extension, never locomotion",
        "scarf/fur follow the return with a short lag",
    ),
    "reaches_for_coffee": (
        "small reach extension on contact followed by a soft torso release",
        "arm and shoulder carry one connected overshoot/recoil",
        "fur follows the arm/torso with subtle delayed motion",
    ),
    "glances_at_many_tabs": (
        "small side-to-side head compression during the scan, then release",
        "final gaze/head overshoot stays pose-space only",
        "scarf/fur lag the final head correction and settle afterward",
    ),
    "leans_toward_camera": (
        "subtle squash at the forward lean peak, then gentle recoil",
        "neck/head lead the pose while the feet/root remain anchored",
        "scarf/fur trail the lean and settle a fraction later",
    ),
}


def _facial_performance_for(action: str) -> tuple[str, ...]:
    return FACIAL_PERFORMANCE.get(
        action,
        (
            "eyes lead the attention target before the head",
            "use restrained eyelid/brow changes for the emotional peak",
            "use one tiny mouth change at the peak, then return to neutral",
        ),
    )


def _pose_accents_for(action: str) -> tuple[str, ...]:
    return POSE_ACCENTS.get(
        action,
        (
            "add one subtle squash/stretch accent at the emotional peak",
            "keep overshoot in pose-space while the character base stays anchored",
            "add restrained clothing/fur follow-through after the main body motion",
        ),
    )


SECONDARY_MOTION = {
    "finish_work": "teal scarf and fur settle after the small finishing release",
    "checks_phone": "teal scarf lags the torso by a fraction, then catches up; fur follows the recoil subtly",
    "sees_calendar": "teal scarf and forearm follow the point and return with a small delayed settle",
    "reaches_for_coffee": "torso and fur follow the reaching arm; the body settles a beat after the hand",
    "glances_at_many_tabs": "teal scarf and head fur lag slightly during the scan, then settle after the gaze stops",
    "leans_toward_camera": "teal scarf trails the forward lean and settles after the torso",
    "deadpan_camera_reaction": "teal scarf and fur absorb a tiny recoil before returning to rest",
}


def _secondary_motion_for(action: str) -> str:
    return SECONDARY_MOTION.get(
        action,
        "restrained fur, clothing or prop follow-through that lags the main body motion slightly",
    )


def _phases(duration: float, names: tuple[str,...]) -> tuple[PerformancePhase,...]:
    weights = {
        "anticipation": .10,
        "eye_lead": .08,
        "head_follow": .09,
        "action": .18,
        "impact": .08,
        "overshoot": .08,
        "recoil": .14,
        "secondary_motion": .08,
        "settle": .17,
    }
    purpose = {
        "anticipation": "prepare the body before the main reaction",
        "eye_lead": "let the eyes establish the target before the head follows",
        "head_follow": "carry the gaze into a connected head and upper-body follow",
        "action": "perform the primary readable action",
        "impact": "hit the strongest emotional change",
        "overshoot": "travel slightly beyond the intended endpoint for cartoon elasticity",
        "recoil": "recover through connected head, torso and limb motion",
        "secondary_motion": "add restrained scarf, fur or prop follow-through",
        "settle": "ease into the final resting pose instead of snapping to stillness",
    }
    intensity = {
        "anticipation": .38,
        "eye_lead": .52,
        "head_follow": .60,
        "action": .82,
        "impact": 1.0,
        "overshoot": .74,
        "recoil": .78,
        "secondary_motion": .42,
        "settle": .25,
    }
    total = sum(weights.get(n, 0.05) for n in names)
    cur = 0.0
    out = []
    for i, name in enumerate(names):
        weight = weights.get(name, 0.05)
        end = duration if i == len(names) - 1 else cur + duration * weight / total
        out.append(
            PerformancePhase(
                name,
                round(cur, 3),
                round(end, 3),
                intensity.get(name, .5),
                purpose.get(name, "perform a connected motion beat"),
            )
        )
        cur = end
    return tuple(out)

def build_shot_performance_spec(action: str, dialogue: str, duration: float=5.0, camera: str="stable camera with subtle push-in") -> dict[str,Any]:
    s = ACTIONS.get(action, DEFAULT)
    reference_requirements = required_reference_props(action)
    chain = ACTION_CHAINS.get(action, DEFAULT_CHAIN)

    # The nine-beat chain is intentional: it forces the model to distinguish
    # attention, action, recovery and follow-through instead of collapsing the
    # shot into one continuous pose change.
    names = (
        "anticipation",
        "eye_lead",
        "head_follow",
        "action",
        "impact",
        "overshoot",
        "recoil",
        "secondary_motion",
        "settle",
    )
    duration = round(float(duration), 3)
    p = ShotPerformanceSpec(
        duration,
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        (
            "eyes lead the reaction before the head moves",
            "eyelids and brow change through the emotional beat",
            "small head/face follow-through after the eye lead",
            "brief blink or micro-settle at the end",
        ),
        _phases(duration, names),
        camera,
        (
            "preserve character identity and proportions",
            "preserve face, eyes, ears, hands and costume",
            "preserve environment and object placement",
            "do not invent extra limbs, props or characters",
            "keep motion connected across the entire action chain",
            "keep the character root/feet anchored in screen space",
            "treat overshoot as a small pose change, not locomotion",
            "return the root to the original framing before the settle beat",
            "keep chest and face mostly front-facing throughout the shot",
            "do not rotate the character into profile during recoil or settle",
            "prioritize visible upper-body action before adding facial micro-expression",
        ),
        (
            "no morphing",
            "no rubber limbs",
            "no teleporting hands",
            "no expression popping",
            "no independent sprite-like part motion",
            "no unrelated camera drift",
            "no walking or world-space travel unless the action explicitly requires locomotion",
            "do not let the character leave the frame",
            "no large yaw, profile turn, spin or turn-away",
            "do not skip the anticipation, overshoot or settle beats",
        ),
        tuple(chain),
        _secondary_motion_for(action),
        "anchor the character's feet/base in screen space",
        "pose-space overshoot only; return to original framing before settle",
        _facial_performance_for(action),
        _pose_accents_for(action),
    ).to_dict()
    p["speaking_style"] = plan_performance(action, dialogue).get("speaking_style", "measured_deadpan")
    p["body_motion"] = (
        "anticipation -> eye lead -> head follow -> primary action -> impact -> "
        "overshoot -> recoil -> secondary motion -> settle"
    )
    p["source_action"] = action
    p["dialogue"] = dialogue
    p["facial_timeline"] = build_facial_timeline(action)
    p["facial_timing_contract"] = {
        "eye_lead_before_head": True,
        "expression_peak_after_attention": True,
        "blink_after_peak": True,
        "recovery_to_deadpan": True,
    }
    p["reference_requirements"] = list(reference_requirements)
    p["action_chain_version"] = 2.3
    p["root_lock"] = "anchor feet/root in screen space; no world-space travel"
    p["overshoot_mode"] = "pose-space overshoot only; return to original framing before settle"
    p["orientation_lock"] = "keep chest and face mostly front-facing; no large yaw or turn-away"
    p["motion_priority"] = "prioritize readable upper-body action over facial-only motion"
    return p

def performance_prompt(spec: dict[str,Any]) -> str:
    phases = "; ".join(
        f'{p["start"]:.1f}-{p["end"]:.1f}s {p["name"]}: {p["purpose"]}'
        for p in spec["phases"]
    )
    micro_beats = "; ".join(spec.get("micro_beats", ()))
    secondary_motion = spec.get("secondary_motion", "")
    return (
        f'Animate the existing character for exactly {spec["duration"]:.1f}s. '
        f'Performance arc: {spec["emotion_start"]} -> {spec["emotion_peak"]} -> {spec["emotion_end"]}. '
        f'Action: {spec["action"]}. Gaze: {spec["gaze"]}. Head: {spec["head_motion"]}. '
        f'Body: {spec["body_motion"]}. Hands: {spec["hands"]}. '
        f'Face: {", ".join(spec["facial_animation"])}. '
        f'Action chain v{spec.get("action_chain_version", 1)}: {micro_beats}. '
        f'Secondary motion: {secondary_motion}. '
        f'Root lock: {spec.get("root_lock", "anchor the character base in screen space")}. '
        f'Overshoot mode: {spec.get("overshoot_mode", "pose-space only")}. '
        f'Facial performance: {"; ".join(spec.get("facial_performance", ())) }. '
        f'Pose accents: {"; ".join(spec.get("pose_accents", ())) }. '
        f'Orientation lock: {spec.get("orientation_lock", "mostly front-facing")}. '
        f'Motion priority: {spec.get("motion_priority", "readable body action")}. '
        f'Timeline: {phases}. Camera: {spec["camera"]}. '
        f'Preserve: {"; ".join(spec["continuity"])}. '
        f'Avoid: {"; ".join(spec["negative_constraints"])}.'
    )
