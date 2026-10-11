# GitHub automation: owner setup

The cartoon factory runs on GitHub Actions. Three workflows share one durable state
branch and one concurrency group, so they never write state at the same time.

| Workflow | Trigger | What it does |
|---|---|---|
| `cartoon-factory.yml` | daily 03:47 UTC, or by hand | Runs all 16 stages, stores memory on `factory-state`, attaches the episode to a **draft** release `episode-<run_id>`, opens a review issue with no episode title |
| `publish-episode.yml` | by hand only | Publishes one reviewed episode (`run_id`, `platforms`, `made_for_kids`) from the draft release, after approval in the `publish` environment |
| `metrics-learning.yml` | daily 06:23 UTC, or by hand | Refreshes platform metrics for published runs and stores the learning update |
| `ci.yml` | every branch push and pull request | Lint (blocking syntax and undefined-name checks, pyflakes report, actionlint), unit tests, and a non-blocking persona backtest |
| `daily-content.yml` | by hand only | Legacy persona content job; its schedule is disabled |

GitHub runs scheduled workflows only from the default branch, so the schedules start
working when the change is merged to `main`.

## Secrets (Settings > Secrets and variables > Actions)

| Name | Used by | Notes |
|---|---|---|
| `YOUTUBE_API_KEY` | factory, metrics | Free YouTube Data API key (public trend chart and public stats) |
| `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_REFRESH_TOKEN` | publish, metrics | Owner analytics and uploads; refresh-token flow |
| `TIKTOK_CLIENT_KEY`, `TIKTOK_CLIENT_SECRET`, `TIKTOK_REFRESH_TOKEN` | publish, metrics | Refresh-token flow |
| `INSTAGRAM_ACCESS_TOKEN`, `INSTAGRAM_USER_ID` | publish, metrics | Must be a non-expiring system-user or page token. A short-lived user token expires within hours |

The factory job deliberately receives no Instagram or TikTok credential: it only reads
public trend data. Runway credentials are never passed to any workflow, so the factory
cannot spend Runway credits.

Optional repository variables: `NOVA_VOICE_BACKEND`, `NOVA_CARTOON_ANIMATION_BACKEND`,
`NOVA_MAX_REPAIR_ATTEMPTS`. An unset variable leaves the code default in place.

## Approval gate

1. Settings > Environments > create `publish`, add yourself as a required reviewer.
   Without this, starting `publish-episode.yml` publishes without a second look.
2. Review the draft release and the review issue. Only repository writers can see a
   draft release.
3. Start `publish-episode.yml` with the run id from the review issue.

## Branch protection on `main` (owner setting)

The overnight upgrade agent (`scripts/n8n_upgrade_server.py`) opens pull requests and
never merges them. Protect `main` so that a PR is the only way in:

- require a pull request before merging,
- require the status checks `NOVA CI / lint` and `NOVA CI / test`,
- block force pushes and branch deletion,
- optionally require one approving review (this stops the agent's work from landing
  without a human).

## Privacy

The repository is public. Public-repository rules apply to:

- the `factory-state` branch (memory and run records, including research headlines),
- review issues (they carry the run id and status only),
- workflow logs and step summaries (the summary omits titles and hooks).

Episode videos are never uploaded as Actions artifacts; they live only in draft
releases. If the research or the memory must stay private, make the repository private
(Settings > General > Change visibility). Private repositories consume Actions minutes
from the plan.

## Keeping the schedule healthy

- Check Settings > Billing and plans if runs are refused for payment reasons.
- A scheduled workflow in a public repository is disabled after 60 days without
  repository activity. The daily factory commits to `factory-state`, which counts as
  activity.
- Dependabot (`.github/dependabot.yml`) proposes updates to the action versions.
