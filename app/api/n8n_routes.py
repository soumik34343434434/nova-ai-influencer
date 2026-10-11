"""Authenticated REST adapter for driving NOVA from n8n.

n8n is the visible workflow/control layer; NOVA remains the durable Python engine.
The API is intentionally localhost-friendly and uses a shared header token.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.api.auth import check_token
from app.cartoon_control import CartoonRun, CartoonRunStore, CartoonStage, stage_status
from app.cartoon_factory_orchestrator import CartoonFactoryOrchestrator

router = APIRouter()
factory = CartoonFactoryOrchestrator()


class CreateRunRequest(BaseModel):
    topic: str = ""


def _require_token(x_nova_api_key: str | None) -> None:
    check_token(x_nova_api_key)


def _snapshot(run: CartoonRun) -> dict:
    return {
        "run": asdict(run),
        "run_id": run.run_id,
        "status": run.status,
        "current_stage": run.current_stage,
        "stage_status": stage_status(run),
    }


@router.get("/n8n/health")
def n8n_health():
    return {
        "ok": True,
        "service": "nova",
        "integration": "n8n",
        "api_version": "1",
    }


@router.post("/n8n/runs")
def create_run(
    request: CreateRunRequest,
    x_nova_api_key: str | None = Header(default=None, alias="X-NOVA-API-Key"),
):
    _require_token(x_nova_api_key)
    return _snapshot(factory.create_run(request.topic))


@router.get("/n8n/runs/{run_id}")
def get_run(
    run_id: str,
    x_nova_api_key: str | None = Header(default=None, alias="X-NOVA-API-Key"),
):
    _require_token(x_nova_api_key)
    try:
        run = CartoonRunStore(run_id).load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _snapshot(run)


@router.post("/n8n/runs/{run_id}/stages/{stage}")
def run_stage(
    run_id: str,
    stage: str,
    x_nova_api_key: str | None = Header(default=None, alias="X-NOVA-API-Key"),
):
    _require_token(x_nova_api_key)
    try:
        stage_enum = CartoonStage(stage)
        store = CartoonRunStore(run_id)
        run = store.load()
        factory.plane.store = store
        updated = factory.plane.run_stage(run, stage_enum)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _snapshot(updated)


@router.post("/n8n/runs/{run_id}/resume")
def resume_run(
    run_id: str,
    x_nova_api_key: str | None = Header(default=None, alias="X-NOVA-API-Key"),
):
    _require_token(x_nova_api_key)
    try:
        return _snapshot(factory.resume(run_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/n8n/runs/{run_id}/repair/{stage}")
def repair_run(
    run_id: str,
    stage: str,
    x_nova_api_key: str | None = Header(default=None, alias="X-NOVA-API-Key"),
):
    _require_token(x_nova_api_key)
    try:
        return _snapshot(factory.repair(run_id, CartoonStage(stage)))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
