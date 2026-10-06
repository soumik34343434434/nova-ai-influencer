"""Durable, dependency-free state machine for NOVA pipeline runs."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session

from app.db.models import PipelineRun


class Phase(str, Enum):
    CREATED = "created"
    TREND_VALIDATED = "trend_validated"
    IDEA_CREATED = "idea_created"
    SCRIPT_READY = "script_ready"
    MEDIA_READY = "media_ready"
    MEDIA_QA = "media_qa"
    PUBLISH_READY = "publish_ready"
    PUBLISHED = "published"
    METRICS_CAPTURED = "metrics_captured"
    LEARNING_UPDATED = "learning_updated"
    FAILED = "failed"


_ORDER = [
    Phase.CREATED,
    Phase.TREND_VALIDATED,
    Phase.IDEA_CREATED,
    Phase.SCRIPT_READY,
    Phase.MEDIA_READY,
    Phase.MEDIA_QA,
    Phase.PUBLISH_READY,
    Phase.PUBLISHED,
    Phase.METRICS_CAPTURED,
    Phase.LEARNING_UPDATED,
]


@dataclass(frozen=True)
class RunSnapshot:
    run_id: str
    phase: Phase
    status: str
    retry_count: int


class DurableRun:
    """Persist phase transitions so a failed run can resume without regeneration."""

    def __init__(self, db: Session, run: PipelineRun):
        self.db = db
        self.run = run

    @classmethod
    def start(cls, db: Session, payload: dict | None = None) -> "DurableRun":
        run = PipelineRun(
            id=uuid.uuid4().hex,
            phase=Phase.CREATED.value,
            status="running",
            payload_json=json.dumps(payload or {}),
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return cls(db, run)

    @classmethod
    def resume(cls, db: Session, run_id: str) -> "DurableRun":
        run = db.get(PipelineRun, run_id)
        if not run:
            raise KeyError(f"Unknown pipeline run: {run_id}")
        return cls(db, run)

    @property
    def payload(self) -> dict:
        return json.loads(self.run.payload_json or "{}")

    def update_payload(self, **values: object) -> None:
        payload = self.payload
        payload.update(values)
        self.run.payload_json = json.dumps(payload)
        self.db.commit()

    def transition(self, next_phase: Phase) -> RunSnapshot:
        current = Phase(self.run.phase)
        if next_phase is Phase.FAILED:
            self.run.phase = next_phase.value
            self.run.status = "failed"
        else:
            if current is Phase.FAILED:
                raise RuntimeError("A failed run must be explicitly resumed before transitioning")
            try:
                current_index = _ORDER.index(current)
                next_index = _ORDER.index(next_phase)
            except ValueError as exc:
                raise RuntimeError(f"Unsupported transition: {current} -> {next_phase}") from exc
            if next_index != current_index + 1:
                raise RuntimeError(f"Invalid transition: {current} -> {next_phase}")
            self.run.phase = next_phase.value
            if next_phase is Phase.LEARNING_UPDATED:
                self.run.status = "completed"
        self.db.commit()
        self.db.refresh(self.run)
        return self.snapshot()

    def fail(self, error: str) -> RunSnapshot:
        self.run.last_error = error[:4000]
        self.run.retry_count += 1
        return self.transition(Phase.FAILED)

    def mark_waiting(self, reason: str) -> RunSnapshot:
        """Leave a successful run resumable while an external dependency is pending."""
        self.run.status = "waiting"
        self.run.last_error = reason[:4000]
        self.db.commit()
        self.db.refresh(self.run)
        return self.snapshot()

    def resume_after_failure(self, phase: Phase) -> RunSnapshot:
        if self.run.status != "failed":
            raise RuntimeError("Only failed runs can be resumed")
        if phase not in _ORDER:
            raise ValueError("Invalid resume phase")
        self.run.phase = phase.value
        self.run.status = "running"
        self.run.last_error = ""
        self.db.commit()
        self.db.refresh(self.run)
        return self.snapshot()

    def snapshot(self) -> RunSnapshot:
        return RunSnapshot(self.run.id, Phase(self.run.phase), self.run.status, self.run.retry_count)
