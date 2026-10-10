# NOVA AI Factory — architecture to code

This maps the 19-stage NOVA AI Factory design to the code that implements it.
The durable control plane (`app/cartoon_control.py`) runs 16 checkpointed stages
in order; several design stages live inside one control-plane stage. Every run
is saved under `artifacts/cartoon/runs/<run_id>/run.json`, and a failed stage can
be resumed or repaired without redoing earlier stages.

`tests/test_factory_end_to_end.py` runs all 16 stages offline in order and checks
that a reviewable 1080×1920 episode with audio comes out.

## Stage map

| # | Design stage | Control-plane stage | Main code |
|---|---|---|---|
| 1 | Internet / data | `1_research` | `agents/cartoon_research.py` (Google News, GDELT), `agents/trend_intelligence_v2.py` (Reddit, Google Trends, YouTube, TikTok, Instagram hashtag search) |
| 2 | Trend engine | `1_research`, `2_trend_intelligence` | `agents/trend_intelligence_v2.py`, `agents/trend_opportunity.py`, `trend_history.py` |
| 3 | Format engine | `2_trend_intelligence` | `agents/format_engine_v2.py` (10 formats + `micro_short`) |
| 4 | Idea generator | `3_idea_generator` | `agents/idea_generator_v2.py` (~100 ideas), `learning_v2.select_portfolio` (~60) |
| 5 | Idea jury | `4_idea_evaluator` | `agents/cartoon_quality.py`, `agents/cartoon_jury.py`, `agents/story_engine.py` gate, `agents/idea_prediction.py`, cheap animatic + `agents/cartoon_animatic_judge.py` |
| 6 | Story engine | `5_story_engine` | `agents/story_engine.py` |
| 7 | Script engine | `6_script_engine` | `agents/script_engine.py` |
| 8 | Director AI | `7_storyboard_engine` | `agents/storyboard.py`, `agents/director_decision.py`, `agents/shot_performance.py`, `agents/camera_director.py`, `agents/facial_director.py` |
| 9 | Character / world bible | `7_storyboard_engine`, `8_asset_engine` | `config/character_world_bible.json`, `agents/character_world_bible.py`, `config/cartoon_action_packs.json` |
| 10 | Visual pipeline | `8_asset_engine`, `9_animation_engine` | `agents/action_packs.py`, `providers/cartoon_animation.py` (local), `providers/blender_animation.py` |
| 11 | Performance AI / Runway | `9_animation_engine` | `providers/cartoon_performance.py`, `providers/character_acting.py`, `providers/ai_performance.py` (plans), `providers/runway_video.py` (execution) |
| 12 | Visual QA | `13_video_qa` | `agents/cartoon_quality.py`, `agents/cartoon_motion_qa.py`, `agents/cartoon_runtime_qa.py` |
| 13 | Repair AI | `13_video_qa` (loop) | `CartoonMotionQA.plan_repair`, `rerender_cartoon_shots`, hosted repair via `runway_video` |
| 14 | Audio engine | `9_animation_engine`, `10_audio_engine` | `providers/cartoon_voice.py`, `providers/cartoon_audio.py`, `providers/runway_voice.py`, `config/character_voices.json` |
| 15 | Editor AI | `11_caption_engine`, `12_editor_engine` | `agents/edit_decision.py`, `providers/reel_editor.py` |
| 16 | Final QA | `13_video_qa`, `14_publish` | `agents/final_qa.py` |
| 17 | Publishing | `14_publish` + `python -m app.publish_approved` | `providers/publisher.py`, `providers/platform_publishers.py` |
| 18 | Analytics | `15_analytics` | `providers/platform_analytics.py`, `experiment_system.py` |
| 19 | Learning AI | `16_learning` | `learning_v2.py`, `agent_memory.py` |

## How the stages connect

- **Ideas are always renderable.** Each character's action pack lists the
  poses that exist as artwork and which story role (hook, escalation, reaction,
  punchline) each serves. The idea generator only casts characters with a
  complete pack and only uses their registered actions.
- **The script is what you hear.** The Script Engine's lines (the approved hook
  replaces the first beat) are rendered, captioned and put in the edit plan.
- **Director AI drives the animation.** The storyboard makes one shot per
  scripted beat using that beat's action. Each shot's acting plan
  (anticipation, eye lead, head follow, impact, overshoot, recoil, secondary
  motion, settle) is passed to the renderer. With a plan, the eyes reach their
  target before the head turns.
- **The world bible is checked on real data.** Each shot records the
  transformations the renderer applies and which signature props are visible;
  close-ups may crop a prop. Before rendering, every action image's character
  region is checked against the bible palette, which rejects wrong or swapped art.
- **Repair is a bounded loop.** Visual QA → Repair → Visual QA runs up to
  `NOVA_MAX_REPAIR_ATTEMPTS` (default 2) times. Motion failures rerender only
  the failing shots; technical failures (file, codec, audio) rerender in full.
  Hosted reels regenerate only the failing shots on Runway, then re-cut.
- **The Editor owns the cut.** For hosted shot clips it assembles the reel:
  9:16 crop, planned shot lengths, captions and disclosure at planned times,
  local voice/SFX track. For the local renderer, which already composes the
  reel, it conforms the result to the edit plan (format, duration, shot count,
  cut points, hook timing) and fails the run on drift.
- **Learning reaches the next run.** The learning stage writes
  `artifacts/cartoon/next_loop_plan.json`; every new run starts with it. Its
  preferences boost format ranking and the jury score (by at most 0.12, never
  rescuing a failed gate). Each run draws explore or exploit from
  `exploration_rate`; exploring runs favour formats the policy is not already
  reinforcing.

## Deliberate differences from the diagram

- **Video QA runs after the Editor**, not before Audio and Editor. QA inspects
  the exact reel that will be published. Final QA is evaluated at QA and again
  before the publish queue.
- **Voice and SFX are produced during animation.** Acting is timed to measured
  speech (onset, energy, peak), so audio is generated before frames. The Audio
  stage records and checks that track.
- **Publishing is approval-first.** The pipeline queues `ready_for_review` items.
  A person approves with `python -m app.publish_approved --approve`, which also
  binds the platform content IDs to the run for analytics.

## External services (all off by default)

| Service | Switch | Needs |
|---|---|---|
| Runway video | `NOVA_CARTOON_ANIMATION_BACKEND=runway`, `NOVA_RUNWAY_EXECUTE=true` | `RUNWAYML_API_SECRET`; `NOVA_RUNWAY_MAX_CREDITS` (per episode, repairs included) and `NOVA_RUNWAY_DAILY_CREDITS` |
| Runway voice | `NOVA_VOICE_BACKEND=runway` | `RUNWAYML_API_SECRET`, `RUNWAY_AUDIO_CONFIG_ID`; voices in `config/character_voices.json` |
| YouTube / Instagram / TikTok posting | `NOVA_ENABLE_REAL_PUBLISH=true` + `--approve` + `--made-for-kids` | `YOUTUBE_ACCESS_TOKEN`; `INSTAGRAM_ACCESS_TOKEN` + `INSTAGRAM_USER_ID`; `TIKTOK_ACCESS_TOKEN` |
| YouTube / TikTok / Instagram trends | key present | `YOUTUBE_API_KEY`; `TIKTOK_RESEARCH_ACCESS_TOKEN`; Instagram Graph credentials + Public Content Access |

Check which trend sources are live with `python3 -m app.agents.trend_intelligence_v2`.

## Runway credit safeguards

Runway credits are spent only when every check passes, in this order:

1. Execution is enabled and the API key is set.
2. The request fits the episode cap including what the episode already spent
   (repairs count) and the daily cap including everything spent today.
3. A no-cost preflight passes: every start image exists, every model maps to a
   Runway API id, every prompt fits Runway's limit, and the API and clip
   download hosts are reachable, so no clip is generated that can't be used.

Every submitted shot is recorded in `artifacts/cartoon/runway_credit_ledger.json`.
A Runway-made reel that fails video QA stops and reports instead of
regenerating shots, unless `NOVA_RUNWAY_PAID_REPAIRS=true`. To see what an
episode would cost, and whether it could succeed, without spending anything:

```bash
python3 -m app.providers.runway_video quote artifacts/cartoon/<episode>.runway_plan.json --run-id <run_id>
```
