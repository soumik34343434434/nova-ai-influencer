# NOVA — Free-First Automated Cartoon Content Factory

NOVA is an original fictional-character short-form content system designed so the core production loop can run without a paid AI video API.

## Production loop

Internet research → trend scoring → local AI idea/script rewrite → deterministic quality gate → curated character/action asset selection → local 2.5D animation → local TTS + procedural SFX → captions + AI disclosure → media QA → approval-first publish queue → platform analytics → learning strategy.

Cloud image/video providers are optional adapters, not hard dependencies. A provider can be added or removed without changing the research, scripting, QA, rendering, publishing queue, or learning phases.

## Free-first stack

- Public Google News RSS + GDELT for research signals.
- Ollama with a small Qwen3 model for optional local script improvement. The current 1.7B quantized model is about 1.4 GB; Qwen3 0.6B is also available at about 523 MB if a lighter footprint is preferred.
- Pillow + FFmpeg for vertical rendering and motion interpolation.
- macOS `say` for local narration and generated WAV SFX.
- JSON outbox as the default safe publishing boundary.

## Run the cartoon factory

```bash
python3 -m app.cartoon_pipeline
```

Run the autonomous research-to-queue loop:

```bash
python3 -m app.cartoon_autopilot
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

Episodes are designed as original, family-friendly fictional cartoons. Platform audience classification remains an explicit review step rather than being guessed by the generator.

## Provider policy

Runway and other hosted generators may be used when they materially improve an individual asset, but the production architecture does not require them. The preferred path is to keep the recurring software/API spend at zero and use hosted generation only as an optional quality accelerator.
