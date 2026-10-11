# NOVA + n8n local integration

n8n acts as the visible workflow/control layer while NOVA keeps durable stage checkpoints and production logic.

## Local topology

- NOVA API: `http://127.0.0.1:8000`
- n8n: `http://localhost:5678`
- n8n reaches the host API as `http://host.docker.internal:8000`
- API authentication uses `X-NOVA-API-Key`

## API surface

- `GET /api/n8n/health`
- `POST /api/n8n/runs`
- `GET /api/n8n/runs/{run_id}`
- `POST /api/n8n/runs/{run_id}/stages/{stage}`
- `POST /api/n8n/runs/{run_id}/resume`
- `POST /api/n8n/runs/{run_id}/repair/{stage}`

## Workflow

`workflows/n8n/nova_live_factory.json` contains the inactive full factory workflow. It has manual and daily triggers followed by one HTTP node per durable NOVA stage, ending with a final run-status read. The current workflow has 16 durable stages, matching `CartoonStage` in `app/cartoon_control.py`; `5 Story Engine` is the hard narrative-quality gate between Idea Jury and Script Engine. `tests/test_n8n_workflows.py` fails if a stage key here drifts from the app. Start the API with `uvicorn app.main:app --port 8000` and set `NOVA_API_TOKEN` (every route returns 503 without it).

The workflow is intentionally imported inactive. Production activation remains a deliberate operator action until the end-to-end quality gates are ready.

## Durable stage order

`Research → Trend Intelligence → Idea Generator → Idea Jury → Story Engine → Script Engine → Storyboard → Asset Engine → Animation → Audio → Captions → Editor → Video QA → Publish → Analytics → Learning`
