# NOVA — Free-First Automated Cartoon Content Factory

NOVA is an original fictional-character short-form content system designed so the core production loop can run without a paid AI video API.

## Production loop

Internet research → trend intelligence → format intelligence → idea generation → independent idea jury → cheap animatic simulation → animatic sanity judge → story → script → storyboard (Director AI) → asset + character-bible gate → local/optional animation → audio → captions + AI disclosure → editor → visual QA with targeted repair → final QA → approval-first publish queue → platform analytics → learning strategy → next experiment.

`docs/factory_architecture.md` maps each stage of the NOVA AI Factory design to the code.

The key design rule is **do not spend final-render time on weak ideas**. Format intelligence extracts reusable hook/escalation/payoff structures from research, the jury uses separate hook/comedy/retention/visual/safety rubrics, and the cheap animatic gate catches dead or visually repetitive concepts before final production.

Cloud image/video providers are optional adapters, not hard dependencies. A provider can be added or removed without changing the research, scripting, QA, rendering, publishing queue, or learning phases.

## Free-first stack

- Public Google News RSS + GDELT for research signals.
- Ollama with a small Qwen3 model for optional local script improvement. The current 1.7B quantized model is about 1.4 GB; Qwen3 0.6B is also available at about 523 MB if a lighter footprint is preferred.
- Pillow + FFmpeg for vertical rendering and motion interpolation.
- macOS `say` (or eSpeak NG on Linux) for local narration with a distinct voice per character, plus generated WAV SFX.
- JSON outbox as the default safe publishing boundary.

## Animation Engine 2.0

The local production path now uses `app/providers/cartoon_animation.py`. Approved action images act as deterministic keyframes, and the renderer adds beat-specific anticipation, push-in, recoil/shake, settle, squash/stretch proxy motion, impact graphics, progressive captions, and AI disclosure. The control plane renders locally at 24 fps without a GPU-heavy generative-video dependency. The Director's per-shot acting plan (anticipation, eye lead, head follow, impact, overshoot, recoil, secondary motion, settle) drives the timing of each shot.

This is deliberately a motion/compositing engine, not a claim of full skeletal character animation. The next quality jump is richer pose/action coverage and, later, an optional hosted motion adapter.

## Run the cartoon factory

Setup: Python 3.13, `pip install -r requirements.txt`, and FFmpeg (`brew install ffmpeg` on macOS, `sudo apt-get install ffmpeg espeak-ng` on Linux). macOS uses the built-in `say` voice.

```bash
python3 -m app.cartoon_pipeline
```

Run the autonomous research-to-queue loop:

```bash
python3 -m app.cartoon_autopilot
```

Run the durable 16-stage control plane. With no command it starts a new run; `--help` only prints usage:

```bash
python3 -m app.cartoon_factory_orchestrator            # run (default)
python3 -m app.cartoon_factory_orchestrator resume <run_id>
python3 -m app.cartoon_factory_orchestrator repair <run_id> 9_animation_engine
python3 -m app.cartoon_factory_orchestrator status <run_id>
python3 -m app.cartoon_factory_orchestrator metrics    # every published run, daily
```

Exit codes: `0` completed or success, `1` the run failed or a metrics refresh errored, `2` unknown run or usage error.

Each run is checkpointed under `artifacts/cartoon/runs/<run_id>/run.json`, with file paths stored relative to the repository. A failed run can be resumed, and a failed stage can be repaired without redoing successful earlier stages. The publish stage is intentionally an approval-first outbox.

After reviewing a queued episode, publish it (dry run unless `NOVA_ENABLE_REAL_PUBLISH=true` and platform credentials are set):

```bash
python3 -m app.publish_approved --run-id <run_id> --platforms youtube,instagram,tiktok --made-for-kids no --approve
```

Publishing is safe to repeat: a platform that already has a published id for the run is skipped, and each upload is recorded as soon as it succeeds. Uploads are private by default (`YOUTUBE_PRIVACY_STATUS=private`, `TIKTOK_PRIVACY_LEVEL=SELF_ONLY`). Credentials: YouTube uses `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET` and `YOUTUBE_REFRESH_TOKEN`, TikTok uses `TIKTOK_CLIENT_KEY`, `TIKTOK_CLIENT_SECRET` and `TIKTOK_REFRESH_TOKEN`, and Instagram uses a non-expiring `INSTAGRAM_ACCESS_TOKEN` with `INSTAGRAM_USER_ID`. Public YouTube view counts need only `YOUTUBE_API_KEY`. The CLI entry points read `.env` from the repository root. Values already set in the shell take precedence.

To approve an episode made by the GitHub daily run, download its artifact into a separate folder and copy only the episode and its run record into this checkout, so nothing local is overwritten: `gh run download <run_id> -n nova-episode-<run_id> -D /tmp/ep`, then copy `/tmp/ep/runs/<run_id>/` to `artifacts/cartoon/runs/<run_id>/` and the episode `.mp4` to `artifacts/cartoon/`.

Check which trend sources are live, and rebuild the MOMO/BUBU action packs after a character sheet changes:

```bash
python3 -m app.agents.trend_intelligence_v2
python3 tools/cartoon/build_action_packs.py
```

The autopilot writes research and episode metadata under `artifacts/cartoon/` and queues only media that passes both content and media QA.

## Tests

```bash
python3 -m pytest -q
```

## Current cartoon universe

PIP — tiny sky-blue mischievous robot with oversized expressive eyes and an orange backpack.

MOMO — sleepy orange cartoon cat with oversized expressive eyes and a tiny teal scarf.

BUBU — round blue cartoon bird with oversized expressive eyes and a tiny yellow backpack.

Each character has an action pack (`config/cartoon_action_packs.json`, images in `assets/cartoon/actions/`) with one pose per story role, and a voice identity in `config/character_voices.json`. Ideas only use actions that exist as artwork.

Episodes are designed as original, family-friendly fictional cartoons. Platform audience classification remains an explicit review step rather than being guessed by the generator.

## Provider policy

Runway and other hosted generators may be used when they materially improve an individual asset, but the production architecture does not require them. Runway execution is off by default and capped by `NOVA_RUNWAY_MAX_CREDITS`; see `.env.example` for every switch. The preferred path is to keep the recurring software/API spend at zero and use hosted generation only as an optional quality accelerator.
