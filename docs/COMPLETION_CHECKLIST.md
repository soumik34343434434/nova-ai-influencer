# NOVA Completion Checklist

## Fully implemented and tested at $0

- Original fictional persona and disclosure policy
- Public RSS trend discovery
- Deterministic/local content generation
- Safety/originality gate
- Local speech synthesis on macOS and Linux, with a distinct voice per character
- Vertical 1080×1920 video rendering
- Caption/disclosure rendering without FFmpeg drawtext
- Publish outbox
- Analytics schema
- Winner/format learning loop that feeds the next run (explore/exploit per run)
- GitHub Actions CI
- Scheduled daily content generation
- End-to-end synthetic backtest
- Cartoon factory runs all 16 stages offline to a reviewable episode (`tests/test_factory_end_to_end.py`)
- Action packs for PIP, MOMO and BUBU; every generated idea is renderable and passes every gate
- Script and Director shot plans drive the local render
- Character/world bible checked per shot and per action image
- Bounded Visual QA → Repair loop
- Editor stage that assembles hosted clips or conforms the local reel

## Optional providers (built, off by default, need credentials)

- Runway video: `NOVA_CARTOON_ANIMATION_BACKEND=runway` with `NOVA_RUNWAY_EXECUTE=true`, `RUNWAYML_API_SECRET` and a `NOVA_RUNWAY_MAX_CREDITS` budget
- Runway voice: `NOVA_VOICE_BACKEND=runway`; Brodie is the current test voice in `config/character_voices.json`
- Live posting to YouTube Shorts, Instagram Reels and TikTok: `NOVA_ENABLE_REAL_PUBLISH=true`, platform tokens, then `python -m app.publish_approved --approve --made-for-kids yes|no`
- Keyed trend sources: YouTube Data API, TikTok Research API, Instagram Graph hashtag search
- vidIQ: trend/outlier/YouTube research from the connected account
- Metricool: connected social publishing and analytics; Free plan currently permits limited scheduling through MCP, while REST API access is a higher-tier feature

These adapters are tested against simulated API responses only; run a first
live check with a private post and a small credit budget.

## Safety defaults

- No celebrity lookalikes
- No private-person source footage
- No face swaps/reposts
- AI disclosure included in media and publishing metadata, and declared to platforms that support it (YouTube `containsSyntheticMedia`, TikTok `is_aigc`)
- Publishing remains outbox-first: live posts need human approval, `NOVA_ENABLE_REAL_PUBLISH=true`, platform credentials and an explicit audience (made-for-kids) decision
- Runway never spends credits unless execution is enabled and the plan fits the credit budget

## Next engineering target

Upgrade cycles use the deterministic selector in `app/agents/subsystem_priority.py`.
It ranks measured quality gaps, failure frequency, user impact, evidence confidence,
and downstream unblock leverage. Missing evidence lowers confidence and falls back
to this safe sequence: runtime/render proof, integrations, performance/animation,
audio, editor, final QA, publishing, analytics. A dependency with incomplete
prerequisites is marked blocked and its prerequisite can be prioritized as an
unblocker. Scores are decision aids; they do not replace human review of creative
quality.

Once runtime/render evidence is available, use the provider interfaces to prioritize
the highest remaining evidenced gap, including:
1. A vidIQ ingestion adapter.
2. A Metricool MCP scheduling bridge outside the GitHub runner.
3. Optional local neural image-to-video backend when hardware supports it.
