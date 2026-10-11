"""Durable stage orchestration for the cartoon content factory.

This module is the control plane around the production stages. It keeps each
stage idempotent, persists checkpoints to disk, records errors, and supports
repairing only the failed stage instead of regenerating the whole episode.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from app.agent_memory import ensure_experiment_identity, record_failure
from app.paths import ROOT, portable_paths, resolve_stored_paths


RUNS_DIR = ROOT / "artifacts" / "cartoon" / "runs"


class CartoonStage(str, Enum):
    RESEARCH = "1_research"
    TREND_INTELLIGENCE = "2_trend_intelligence"
    IDEA_GENERATOR = "3_idea_generator"
    IDEA_EVALUATOR = "4_idea_evaluator"
    STORY_ENGINE = "5_story_engine"
    SCRIPT_ENGINE = "6_script_engine"
    STORYBOARD_ENGINE = "7_storyboard_engine"
    ASSET_ENGINE = "8_asset_engine"
    ANIMATION_ENGINE = "9_animation_engine"
    AUDIO_ENGINE = "10_audio_engine"
    CAPTION_ENGINE = "11_caption_engine"
    EDITOR_ENGINE = "12_editor_engine"
    VIDEO_QA = "13_video_qa"
    PUBLISH = "14_publish"
    ANALYTICS = "15_analytics"
    LEARNING = "16_learning"


STAGES = tuple(CartoonStage)
_STAGE_BY_NAME = {stage.value.split("_", 1)[1]: stage.value for stage in CartoonStage}


def migrate_stage_key(key: str) -> str:
    """Map a checkpointed stage key from any earlier numbering to the current one.

    Stage values are "<order>_<name>" and names are unique, so a run saved
    before a stage was inserted (story engine, editor) resumes on the right stage.
    """
    if key in _STAGE_BY_NAME.values():
        return key
    prefix, _, name = key.partition("_")
    return _STAGE_BY_NAME.get(name, key) if prefix.isdigit() else key



@dataclass
class StageRecord:
    stage: str
    status: str
    started_at: str
    finished_at: str = ""
    attempts: int = 0
    artifact: dict[str, Any] | None = None
    error: str = ""


@dataclass
class CartoonRun:
    run_id: str
    created_at: str
    current_stage: str
    status: str
    payload: dict[str, Any]
    stages: dict[str, StageRecord]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CartoonRunStore:
    """Atomic-ish JSON checkpoint store; safe to resume after process failure."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self.path = RUNS_DIR / self.run_id / "run.json"

    @staticmethod
    def run_ids() -> list[str]:
        """Every run that has a checkpoint on disk, oldest id first."""
        if not RUNS_DIR.is_dir():
            return []
        return sorted(path.parent.name for path in RUNS_DIR.glob("*/run.json"))

    def create(self, payload: dict[str, Any] | None = None) -> CartoonRun:
        run_payload = dict(payload or {})
        ensure_experiment_identity(run_payload, self.run_id)
        run = CartoonRun(
            run_id=self.run_id,
            created_at=utc_now(),
            current_stage=CartoonStage.RESEARCH.value,
            status="running",
            payload=run_payload,
            stages={},
        )
        self._write(run)
        return run

    def load(self) -> CartoonRun:
        if not self.path.exists():
            raise FileNotFoundError(f"Unknown cartoon run: {self.run_id}")
        raw = json.loads(self.path.read_text())
        raw_stages = raw.get("stages", {})
        # Runs saved before a stage was inserted (Story Engine, Editor) use older
        # numbering. Migrate those checkpoints by stage name so resume/repair
        # stays safe. Current runs map to themselves.
        migrated = {}
        for key, value in raw_stages.items():
            new_key = migrate_stage_key(key)
            row = dict(value)
            row["stage"] = new_key
            row["artifact"] = resolve_stored_paths(row.get("artifact"))
            migrated[new_key] = row
        raw_stages = migrated
        raw["current_stage"] = migrate_stage_key(raw["current_stage"])
        return CartoonRun(
            run_id=raw["run_id"],
            created_at=raw["created_at"],
            current_stage=raw["current_stage"],
            status=raw["status"],
            # Stored paths are repo-relative (or legacy absolute); readers get concrete paths.
            payload=resolve_stored_paths(raw.get("payload", {})),
            stages={
                key: StageRecord(**value)
                for key, value in raw_stages.items()
            },
        )

    def _write(self, run: CartoonRun) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        # Paths inside the repo are written repo-relative so the run survives a move
        # to another checkout (for example, an episode made on a GitHub runner).
        tmp.write_text(json.dumps(portable_paths(asdict(run)), indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    def save(self, run: CartoonRun) -> None:
        self._write(run)

    def set_payload(self, run: CartoonRun, **values: Any) -> None:
        run.payload.update(values)
        self._write(run)

    def begin_stage(self, run: CartoonRun, stage: CartoonStage) -> StageRecord:
        record = run.stages.get(stage.value)
        now = utc_now()
        if record is None:
            record = StageRecord(stage=stage.value, status="running", started_at=now, attempts=1)
        else:
            record.status = "running"
            record.started_at = now
            record.finished_at = ""
            record.error = ""
            record.attempts += 1
        run.stages[stage.value] = record
        run.current_stage = stage.value
        run.status = "running"
        self._write(run)
        return record

    def finish_stage(self, run: CartoonRun, stage: CartoonStage, artifact: dict[str, Any] | None = None) -> None:
        record = run.stages[stage.value]
        record.status = "passed"
        record.finished_at = utc_now()
        record.artifact = artifact or {}
        record.error = ""
        run.current_stage = self._next_stage(stage).value if self._next_stage(stage) else stage.value
        self._write(run)

    def fail_stage(self, run: CartoonRun, stage: CartoonStage, error: str) -> None:
        record = run.stages[stage.value]
        record.status = "failed"
        record.finished_at = utc_now()
        record.error = error[:4000]
        experiment = run.payload.get("experiment", {})
        record_failure(
            run_id=run.run_id,
            experiment_id=str(experiment.get("experiment_id", "")),
            stage=stage.value,
            error_type=error.split(":", 1)[0] if ":" in error else type(error).__name__,
            message=error,
            attempt=record.attempts,
        )
        run.status = "failed"
        run.current_stage = stage.value
        self._write(run)

    def mark_complete(self, run: CartoonRun) -> None:
        run.status = "completed"
        run.current_stage = STAGES[-1].value
        self._write(run)

    def repair_from(self, run: CartoonRun, stage: CartoonStage) -> None:
        """Invalidate stage and everything after it; earlier artifacts remain."""
        order = list(STAGES)
        cut = order.index(stage)
        for later in order[cut:]:
            run.stages.pop(later.value, None)
        run.current_stage = stage.value
        run.status = "running"
        self._write(run)

    @staticmethod
    def _next_stage(stage: CartoonStage) -> CartoonStage | None:
        idx = list(STAGES).index(stage)
        return STAGES[idx + 1] if idx + 1 < len(STAGES) else None


from typing import Optional

Handler = Callable[[CartoonRun], Optional[dict[str, Any]]]


class CartoonControlPlane:
    """Runs registered handlers with checkpointing and stage-local retries."""

    def __init__(self, store: CartoonRunStore | None = None, max_retries: int = 2) -> None:
        self.store = store or CartoonRunStore()
        self.max_retries = max_retries
        self.handlers: dict[CartoonStage, Handler] = {}

    def register(self, stage: CartoonStage, handler: Handler) -> None:
        self.handlers[stage] = handler

    def run(self, run: CartoonRun | None = None, from_stage: CartoonStage | None = None) -> CartoonRun:
        run = run or self.store.create()
        start = from_stage or CartoonStage(run.current_stage)
        for stage in STAGES[list(STAGES).index(start):]:
            existing = run.stages.get(stage.value)
            if existing and existing.status == "passed":
                continue
            handler = self.handlers.get(stage)
            if handler is None:
                self.store.begin_stage(run, stage)
                self.store.finish_stage(run, stage, {"status": "not_implemented"})
                continue
            success = False
            for attempt in range(self.max_retries + 1):
                self.store.begin_stage(run, stage)
                try:
                    artifact = handler(run) or {}
                    self.store.finish_stage(run, stage, artifact)
                    success = True
                    break
                except Exception as exc:
                    self.store.fail_stage(run, stage, f"{type(exc).__name__}: {exc}")
                    if attempt < self.max_retries:
                        time.sleep(min(2 ** attempt, 4))
            if not success:
                return run
        self.store.mark_complete(run)
        return run

    def run_stage(self, run: CartoonRun | None, stage: CartoonStage) -> CartoonRun:
        """Execute exactly one stage, preserving the durable checkpoint contract."""
        run = run or self.store.create()
        index = list(STAGES).index(stage)
        if any(
            run.stages.get(previous.value) is None
            or run.stages[previous.value].status != "passed"
            for previous in STAGES[:index]
        ):
            raise ValueError(f"Cannot run {stage.value} before earlier stages pass")
        existing = run.stages.get(stage.value)
        if existing and existing.status == "passed":
            return run
        handler = self.handlers.get(stage)
        if handler is None:
            raise ValueError(f"No handler registered for {stage.value}")
        for attempt in range(self.max_retries + 1):
            self.store.begin_stage(run, stage)
            try:
                artifact = handler(run) or {}
                self.store.finish_stage(run, stage, artifact)
                if stage is STAGES[-1]:
                    self.store.mark_complete(run)
                return run
            except Exception as exc:
                self.store.fail_stage(run, stage, f"{type(exc).__name__}: {exc}")
                if attempt == self.max_retries:
                    return run
        return run

    def resume(self, run_id: str) -> CartoonRun:
        run = CartoonRunStore(run_id)
        return self.run(run.load())

    def repair(self, run_id: str, stage: CartoonStage) -> CartoonRun:
        store = self.store if self.store.run_id == run_id else CartoonRunStore(run_id)
        run = store.load()
        store.repair_from(run, stage)
        return self.run(run, from_stage=stage)


def stage_status(run: CartoonRun) -> dict[str, str]:
    return {stage: record.status for stage, record in run.stages.items()}


if __name__ == "__main__":
    store = CartoonRunStore()
    run = store.create({"demo": True})
    plane = CartoonControlPlane(store)
    run = plane.run(run)
    print(json.dumps({"run_id": run.run_id, "status": run.status, "stages": stage_status(run)}, indent=2))
