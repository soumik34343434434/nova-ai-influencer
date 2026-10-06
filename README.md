# NOVA — Zero-Cost Automated AI Influencer

NOVA is an original fictional virtual-influencer automation stack designed to keep the **core software at $0 recurring API spend**. It uses public trend feeds, deterministic/local content generation, local media tooling, approval-first publishing adapters, and analytics storage.

## Current pipeline

Public RSS trends → Trend Scout → Original Content Director → canonical NOVA character asset → storyboard/shot engine → motion provider router → local voice + motion fallback → FFmpeg vertical Reel → approval/publishing adapter → analytics → feedback loop.

The code intentionally does **not** download and repost creators' videos or impersonate real people.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m app.workers.daily_cycle
```

For media generation:

```bash
ffmpeg --version
espeak --version
python scripts/run_media_demo.py
```

Put an approved NOVA reference image at `assets/character/nova.png`.

## API

```bash
uvicorn app.main:app --reload
```

Endpoints include:

- `GET /api/health`
- `POST /api/automation/run`
- `POST /api/content/generate`
- `GET /api/content/queue`
- `POST /api/content/{content_id}/approve`
- `POST /api/media/build-latest`
- `POST /api/metrics`

## Zero-cost principles

- Paid providers are optional, not required for the core loop.
- Local media uses FFmpeg + eSpeak.
- Generated content is original and disclosure-aware.
- Publishing defaults to approval-first.
- Secrets stay in environment variables and are never committed.

## Roadmap

1. Connect vidIQ signals to the trend scorer.
2. Connect Metricool scheduling after account/network verification.
3. Add optional local/open-source image-to-video providers behind the media interface.
4. Ingest platform analytics and automatically select winning formats.
5. Add affiliate/revenue attribution.
