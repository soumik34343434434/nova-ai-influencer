# NOVA — Free-First Automated Cartoon Content Factory

NOVA is an original fictional-character short-form content system designed so the core production loop can run without a paid AI video API.

## Production loop

Internet research → trend intelligence → format intelligence → idea generation → independent idea jury → cheap animatic simulation → animatic sanity judge → script → storyboard → asset capability gate → local/optional animation → audio → captions + AI disclosure → final media QA → approval-first publish queue → platform analytics → learning strategy → next experiment.

The key design rule is **do not spend final-render time on weak ideas**. Format intelligence extracts reusable hook/escalation/payoff structures from research, the jury uses separate hook/comedy/retention/visual/safety rubrics, and the cheap animatic gate catches dead or visually repetitive concepts before final production.

Cloud image/video providers are optional adapters, not hard dependencies. A provider can be added or removed without changing the research, scripting, QA, rendering, publishing queue, or learning phases.

## Free-first stack

- Public Google News RSS + GDELT for research signals.
- Ollama with a small Qwen3 model for optional local script improvement. The current 1.7B quantized model is about 1.4 GB; Qwen3 0.6B is also available at about 523 MB if a lighter footprint is preferred.
- Pillow + FFmpeg for vertical rendering and motion interpolation.
- macOS `say` for local narration and generated WAV SFX.
- JSON outbox as the default safe publishing boundary.

## Animation Engine 2.0

The local production path now uses `app/providers/cartoon_animation.py`. Approved action images act as deterministic keyframes, and the renderer adds beat-specific anticipation, push-in, recoil/shake, settle, squash/stretch proxy motion, impact graphics, progressive captions, and AI disclosure. The current control-plane configuration renders locally at 12 fps on the 8 GB M2 so it stays practical without a GPU-heavy generative-video dependency.

This is deliberately a motion/compositing engine, not a claim of full skeletal character animation. The next quality jump is richer pose/action coverage and, later, an optional hosted motion adapter.

## Run the cartoon factory

```bash
python3 -m app.cartoon_pipeline
```

Run the autonomous research-to-queue loop:

```bash
python3 -m app.cartoon_autopilot
```

Run the durable 14-stage control-plane wiring:

```bash
python3 -m app.cartoon_factory_orchestrator
```

Each run is checkpointed under `artifacts/cartoon/runs/<run_id>/run.json`. A failed run can be resumed, and a failed stage can be repaired without redoing successful earlier stages. The publish stage is intentionally an approval-first outbox; real platform publishing is an adapter boundary.

The autopilot writes research and episode metadata under `artifacts/cartoon/` and queues only media that passes both content and media QA.

## Tests

```bash
python3 -m pytest -q
```

## Current cartoon universe

PIP — tiny sky-blue mischievous robot with oversized expressive eyes and an orange backpack.

MOMO — sleepy orange cartoon cat with oversized expressive eyes and a tiny teal scarf.

BUBU — round blue cartoon bird with oversized expressive eyes and a tiny yellow backpack.

Episodes are designed as original, family-friendly fictional cartoons. Platform audience classification remains an explicit review step rather than being guessed by the generator.

## Provider policy

Runway and other hosted generators may be used when they materially improve an individual asset, but the production architecture does not require them. The preferred path is to keep the recurring software/API spend at zero and use hosted generation only as an optional quality accelerator.
