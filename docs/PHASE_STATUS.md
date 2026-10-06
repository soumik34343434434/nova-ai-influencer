# NOVA Phase Validation

Last validated on 2026-10-06 against the local M2 Mac environment.

| Phase | Component | Zero-cost status | Backtest status | Contract |
|---|---|---|---|---|
| 1 | Persona / character | ✅ | ✅ | Stable NOVA identity + disclosure |
| 2 | Trend discovery | ✅ | ✅ | RSS input → ranked Trend objects |
| 3 | Content director | ✅ | ✅ | Trend → original script/caption + scores |
| 4 | Safety gate | ✅ | ✅ | Blocks low-originality/high-risk/reuse signals |
| 5 | Media | ✅ | ✅ | 1080×1920 MP4 + local speech, no paid API |
| 6 | Publishing outbox | ✅ | ✅ | JSON outbox with AI disclosure |
| 7 | Analytics | ✅ | ✅ | Views/likes/comments/shares/retention/revenue |
| 8 | Learning loop | ✅ | ✅ | Winner selection + next strategy |
| 9 | Scheduler / CI | ✅ | ✅ | GitHub Actions scheduled generation |
| 10 | External auto-publishing | ⚠️ provider-dependent | adapter ready | Metricool MCP can schedule on Free; Metricool REST API requires Advanced/Custom |

## Real-machine regression

Validated on an Apple M2 Mac with 8 GB RAM:

- 4 tests passed
- 3 synthetic trends
- 3 drafts
- 3 videos
- 3 publish-outbox items
- 0 safety blocks
- winner selected
- full backtest exited with code 0

## Important zero-cost boundary

The core software does not require OpenAI, Runway, ElevenLabs, or another paid generation API.

The zero-cost media renderer is deliberately conservative: it creates an animated vertical short from a character image using local speech + FFmpeg. It is a real video pipeline, but it is **not** a high-end neural image-to-video model.

For external publishing, Metricool currently offers MCP access on its Free plan, but its REST API is restricted to Advanced/Custom plans. Keep publishing as an outbox unless a supported account/API path is explicitly configured.

See README for operating modes.
