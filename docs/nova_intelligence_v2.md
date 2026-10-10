# NOVA Intelligence v2

## Goal

Turn the cartoon factory from a pipeline that can produce a video into a closed-loop content decision system that observes trends, infers formats, generates many original concepts, learns from platform performance, and feeds the learned strategy back into the next idea batch.

## Evidence model

Every signal carries its source, timestamp, region, raw popularity when provided, inferred hooks/formats/emotions, freshness, velocity, cross-source support, saturation/novelty proxies, confidence, and explicit capability notes. The engine never labels a proxy as a platform-wide truth.

## Trend sources

Free/public: Google News RSS, GDELT, Reddit public RSS, Google Trends public RSS, and clearly labeled web/RSS social mentions.

Optional direct: YouTube Data API most-popular chart when YOUTUBE_API_KEY is configured; TikTok Research API when an approved TIKTOK_RESEARCH_ACCESS_TOKEN is configured.

Instagram has no public trending endpoint. When INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_USER_ID are set (and the app has Instagram Public Content Access), NOVA reads top media for INSTAGRAM_TREND_HASHTAGS through Graph API hashtag search (limited to 30 unique hashtags per 7 days); otherwise Instagram is covered only by labeled web mentions. Owned-account Instagram media/insights are handled separately through the platform analytics connector. `python3 -m app.agents.trend_intelligence_v2` prints which sources are active and what each missing one needs.

## Trend scoring

opportunity_score combines velocity, cross-platform support, novelty, inverse saturation, format richness, and emotion richness. Historical snapshots in artifacts/cartoon/trend_history.json upgrade velocity and saturation from a one-time proxy to a time-series signal when the same trend reappears.

## Hook intelligence

Detected structures include POV, curiosity question, confession, challenge, reveal, before/after, listicle, reaction, story, and punctuation-based question/exclamation cues.

## Format intelligence

Format patterns include hook pattern, escalation pattern, payoff pattern, visual devices, detected evidence, and learned performance score. Learned format scores come from strategy memory.

## Idea generation

idea_generator_v2 creates 20-100+ cheap candidates from trend, format, character, emotion, and angle. Concepts are recombined rather than copied from trend wording. Diversity is enforced across format, character, and angle. The orchestrator currently requests 100 raw concepts and keeps up to 60 for the jury/animatic gate.

Each idea is four beats (hook → escalation → reaction with the twist → punchline). Beat actions come from the character's action pack, so every idea can be rendered, and each idea carries a `hook_pattern` tag naming its opening style so learning can compare hook styles.

## Learning

learning_v2 stores experiment metadata and normalized metrics. Each tracked feature keeps observation count, posterior mean, and uncertainty-aware UCB score. Dimensions include format, hook, emotion, character, and trend. Portfolio selection mixes exploitation and exploration.

## Platform analytics

platform_analytics.py is read-only and capability-based. Instagram supports recent Reels media plus insights when configured. YouTube supports public statistics and owner Analytics API metrics when OAuth is configured. TikTok supports owner video query data and Research API trend intelligence when approved. Missing credentials never produce fake metrics.

## Closed loop

publish -> experiment metadata -> platform metrics -> normalize -> strategy memory -> next_loop_plan.json -> next run

The publish stage records idea id, trend id, format, hook, emotion, character, angle, target duration, production cost, and whether the run explored or exploited.

Every new run starts with the latest `next_loop_plan.json`. Its preferred formats boost format ranking, and its preferences nudge the jury score of gate-passing ideas by at most 0.12. Each run draws explore or exploit from `exploration_rate`; exploring runs favour formats the policy is not already reinforcing so under-tested options keep receiving evidence.

## Known limits

True platform-wide Instagram trend saturation is not exposed by any public endpoint; hashtag search shows top media for chosen hashtags only, so NOVA uses direct data where available and labeled proxy signals where not.

TikTok Research API access requires approval, and its historical data can lag live numbers. The engine labels it as research data.

## Official API references

https://developers.google.com/youtube/v3/docs/videos/list
https://developers.google.com/youtube/analytics/metrics
https://developers.tiktok.com/doc/research-api-specs-query-videos
https://developers.tiktok.com/doc/display-api-get-startedhttps://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-hashtag-search
https://developers.google.com/youtube/v3/docs/videos/insert
https://developers.facebook.com/docs/instagram-platform/content-publishing
https://developers.tiktok.com/doc/content-posting-api-reference-direct-post
