# NOVA Story Engine

The Story Engine is a hard narrative-quality gate between Idea Jury and Script Engine.
It turns the selected idea into a production-facing story contract so downstream
stages inherit an explicit narrative intent rather than only raw dialogue.

## Quality gate

Every story is scored independently on ten dimensions:

Hook, Setup, Character Goal, Conflict, Escalation, Visual Storytelling, Surprise/Twist, Character Reaction, Payoff, Ending/Loopability.

The release floor is **8.5/10 on every dimension** and **8.5/10 overall**.
A high average cannot compensate for a weak dimension. Failed stories emit
blockers and deterministic repair guidance and are not promoted to Script Engine.

## Story contract

A passing story contains:

- premise and logline
- explicit character desire
- concrete conflict
- escalation statement
- twist/reversal
- payoff and ending
- normalized beats with story role, duration, and visual intent
- quality evidence and the exact floor used for the decision

The contract is stored on the durable run payload as `story` and `story_quality`.
Script Engine reads the contract and locks its dialogue to the approved beat structure.

## Zero-credit validation

The implementation is deterministic and locally testable. Story scoring and gating
do not call Runway or consume generation credits.

## Backward compatibility

The durable control plane migrates old 14-stage checkpoints in memory because Story
Engine insertion changes the downstream stage numbers from 5-14 to 6-15.
