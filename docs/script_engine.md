# Script Engine

The Script Engine sits after the approved Story Contract and before Storyboard/Director consumption.

## Contract

It converts the narrative beats into a production-facing script without changing the approved story structure. Each beat carries timed dialogue and voice text, delivery direction, caption and caption style, SFX cue, visual action, emotion and gaze intent, and timing start/end/duration.

The contract is deterministic and provider-neutral. No Runway generation is required.

## Quality Gate

The Script Engine evaluates ten dimensions and enforces an 8.5/10 floor:

1. hook delivery
2. story fidelity
3. dialogue economy
4. character voice
5. timing
6. visual sync
7. reaction/payoff
8. caption readiness
9. sound cues
10. performance direction

A script is accepted only when the overall score and every dimension are at least 8.5.

## Timing Policy

Speech is planned against the story duration target using a conservative speaking rate. Pauses are short and intentional so the audio stage does not create artificial gaps between separately rendered lines.

The final payoff receives protected timing and deadpan delivery direction.

## Downstream Usage

Storyboard and performance stages can consume the script contract's explicit action, performance, timing, caption and SFX fields rather than inferring production intent only from prose.
