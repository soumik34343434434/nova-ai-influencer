# NOVA Phase Validation

Last validated on 2026-10-10 in a Linux container matching CI (Python 3.13,
FFmpeg, eSpeak NG). The earlier 2026-10-06 validation on the 8 GB M2 Mac covered
the original 10-phase NOVA persona pipeline.

## Cartoon factory (16 stages)

| Stage | Status | Evidence |
|---|---|---|
| 1 Research | ✅ public sources; optional keyed connectors | offline fixture in end-to-end test |
| 2 Trend + format intelligence | ✅ | learned format boost applies (`test_closed_loop_simulation.py`) |
| 3 Idea generator | ✅ every idea is renderable and passes all gates | `test_idea_generator_gates.py`, `test_action_packs.py` |
| 4 Idea jury + cheap animatic | ✅ | end-to-end test |
| 5–6 Story + script | ✅ script lines are what is rendered | end-to-end test |
| 7 Storyboard / Director AI | ✅ one shot per beat; plans reach the renderer | end-to-end test, `test_character_world_bible.py` |
| 8 Assets + bible identity | ✅ palette check on every action image | `test_character_world_bible.py` |
| 9 Animation | ✅ local; Runway execution opt-in | end-to-end test, `test_runway_video.py`, `test_reel_editor.py` |
| 10–11 Audio + captions | ✅ per-character voices | `test_cartoon_voice.py` |
| 12 Editor | ✅ assembles hosted clips; conforms local reels | `test_reel_editor.py`, end-to-end test |
| 13 Video QA + repair loop | ✅ bounded loop, technical + motion | `test_cartoon_repair.py` |
| 14 Publish queue | ✅ approval-first; live adapters opt-in | `test_platform_publishers.py` |
| 15 Analytics | ✅ binds only published content | `test_closed_loop_simulation.py` |
| 16 Learning | ✅ next run loads the policy | `test_closed_loop_simulation.py`, end-to-end test |

## End-to-end run

`tests/test_factory_end_to_end.py` runs all 16 stages offline in order. In the
validation run it produced a ~10 s 1080×1920 episode with audio that passed
motion QA, the Editor conformance check and the final QA publish gate, queued
it for review, and the next run loaded the learned policy. It takes about
3 minutes.

## Not yet validated against live services

The Runway video worker, the live YouTube/Instagram/TikTok publishers and the
Instagram hashtag trend connector are tested only against simulated HTTP
responses, because no credentials were available. Their request shapes follow
the public API documentation; confirm a first live run with a private post and
a small `NOVA_RUNWAY_MAX_CREDITS` budget.

## Important zero-cost boundary

The core loop needs no paid API: public research sources, local rendering with
Pillow + FFmpeg, local speech (`say` or eSpeak NG), and an approval-first outbox.
Runway, live publishing and keyed trend APIs are optional and fail closed.

See `docs/factory_architecture.md` for the stage-to-code map.
