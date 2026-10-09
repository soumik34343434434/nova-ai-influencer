# NOVA Completion Checklist

## Fully implemented and tested at $0

- Original fictional persona and disclosure policy
- Public RSS trend discovery
- Deterministic/local content generation
- Safety/originality gate
- Local speech synthesis on macOS and Linux
- Vertical 1080×1920 video rendering
- Caption/disclosure rendering without FFmpeg drawtext
- Publish outbox
- Analytics schema
- Winner/format learning loop
- GitHub Actions CI
- Scheduled daily content generation
- End-to-end synthetic backtest

## Optional providers

- vidIQ: trend/outlier/YouTube research from the connected account
- Runway: higher-quality image/video generation; current connected Free plan gates video
- Metricool: connected social publishing and analytics; Free plan currently permits limited scheduling through MCP, while REST API access is a higher-tier feature

## Safety defaults

- No celebrity lookalikes
- No private-person source footage
- No face swaps/reposts
- AI disclosure included in media and publishing metadata
- Publishing remains outbox-first until a real provider path is explicitly enabled

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
