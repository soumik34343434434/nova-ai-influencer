"""Publish a reviewed episode to real platforms after a person approves it.

The factory's publish stage only queues ``ready_for_review`` items. This is the
human-in-the-loop step that follows review:

    python -m app.publish_approved --run-id <run> --platforms youtube,tiktok \
        --made-for-kids no --approve

Without ``--approve``, or unless ``NOVA_ENABLE_REAL_PUBLISH=true`` and the
platform credentials are set, every platform is a dry run and nothing uploads.

Guarantees, so a retry can never double-post:

* A platform that already has a published content id for this run is skipped
  (``already_published``). ``--force-republish`` overrides this on purpose.
* Each successful upload is bound to the run immediately, before the next
  platform starts, so a later network failure cannot lose an earlier success.
* Before each live upload the publish queue row is marked ``publishing``. If a
  previous attempt was left in that state (the process died mid-upload), the
  platform is blocked until a person checks it and passes ``--force-republish``.
* The first approval records the episode's sha256 in the queue. A changed file
  is refused, so the approval always covers the episode that was reviewed.
* Errors are per platform. A connection failure on one platform is recorded as
  ``retryable_error`` and the remaining platforms still run.

The episode is read from the run's repo-relative video path. An episode made by
the GitHub run must be downloaded into the checkout first (see README.md).
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import cartoon_factory_orchestrator as factory
from app.cartoon_control import CartoonRunStore
from app.cartoon_factory_orchestrator import CartoonFactoryOrchestrator
from app.env import load_env
from app.paths import resolve_repo_path
from app.providers.publisher import PublishError, PublishJob, build_publisher, real_publish_enabled

OK_STATUSES = {"published", "already_published", "dry_run"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_queue(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        # Fail closed: overwriting an unreadable queue would lose approval records.
        raise RuntimeError(f"publish queue {path} is not valid JSON; fix or move it before publishing") from exc
    if not isinstance(rows, list):
        raise RuntimeError(f"publish queue {path} must hold a JSON list")
    return rows


def _write_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def _run_rows(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("run_id") == run_id]


def _platform_state(rows: list[dict[str, Any]], run_id: str, platform: str) -> str:
    states = [row.get("platforms", {}).get(platform, {}).get("status", "") for row in _run_rows(rows, run_id)]
    return states[-1] if states else ""


def _mark_platform(path: Path, run_id: str, platform: str, status: str, **fields: Any) -> None:
    rows = _read_queue(path)
    matched = _run_rows(rows, run_id)
    for row in matched:
        entry = row.setdefault("platforms", {}).setdefault(platform, {})
        entry.update(status=status, updated_at=_now(), **fields)
    if matched:
        _write_queue(path, rows)


def _finish_queue_row(path: Path, run_id: str, requested: list[str], results: list[dict[str, Any]]) -> None:
    """Set the queue row's overall status from every platform's outcome."""
    done = {row["platform"] for row in results if row["status"] in {"published", "already_published"}}
    rows = _read_queue(path)
    matched = _run_rows(rows, run_id)
    if not matched:
        return
    if set(requested) <= done:
        overall = "published"
    elif done:
        overall = "partially_published"
    else:
        return
    for row in matched:
        row["status"] = overall
        row["updated_at"] = _now()
    _write_queue(path, rows)


def _blocked_status(exc: Exception) -> str:
    if isinstance(exc, PublishError):
        return "retryable_error" if exc.retryable else "failed"
    if isinstance(exc, (OSError, http.client.HTTPException)) and not isinstance(exc, PermissionError):
        return "retryable_error"  # a transport failure that escaped the adapter is still transient
    return "blocked"


def publish_run(run_id: str, platforms: list[str], *, approved: bool, made_for_kids: bool | None,
                orchestrator: CartoonFactoryOrchestrator | None = None, force_republish: bool = False,
                queue_path: Path | None = None) -> dict[str, Any]:
    store = CartoonRunStore(run_id)
    run = store.load()
    payload = run.payload
    if payload.get("final_qa", {}).get("publish_gate", {}).get("decision") != "allow":
        raise PermissionError("final_qa_publish_gate_not_allow")
    video = resolve_repo_path(payload.get("video", ""))
    if not video.is_file():
        raise FileNotFoundError(
            f"run {run_id} has no episode video; expected {video}. "
            "Download the episode artifact into the checkout at that path, then retry."
        )
    candidate = payload["winner"]["candidate"]
    experiment = payload.get("experiment", {})
    metadata: dict[str, Any] = {"ai_generated": True, "run_id": run_id}
    if made_for_kids is not None:
        metadata["made_for_kids"] = made_for_kids
    live = approved and real_publish_enabled()
    queue = queue_path if queue_path is not None else factory.OUTBOX
    done = {platform: dict(mapping) for platform, mapping in payload.get("published_content_map", {}).items()}
    if live:
        # The first live approval records what was reviewed; a later change to the file is refused.
        digest = _sha256(video)
        rows = _read_queue(queue)
        matched = _run_rows(rows, run_id)
        reviewed = next((row["video_sha256"] for row in matched if row.get("video_sha256")), "")
        if reviewed and reviewed != digest:
            raise PermissionError(f"video_changed_since_review: {video} no longer matches the reviewed episode")
        if matched and not reviewed:
            for row in matched:
                row["video_sha256"] = digest
            _write_queue(queue, rows)

    results, bound = [], {}
    binder = orchestrator
    for platform in platforms:
        if done.get(platform) and not force_republish:
            results.append({"platform": platform, "status": "already_published", "content_id": sorted(done[platform])[0],
                            "error": ""})
            continue
        if live and not force_republish and _platform_state(_read_queue(queue), run_id, platform) == "publishing":
            results.append({
                "platform": platform, "status": "blocked", "content_id": "",
                "error": f"previous_publish_unconfirmed: an earlier {platform} upload for this run did not finish; "
                         "check the platform, then rerun with --force-republish",
            })
            continue
        try:
            job = PublishJob.create(
                video_path=str(video), platform=platform, title=candidate["title"],
                caption=candidate.get("caption") or candidate["title"],
                experiment_id=str(experiment.get("experiment_id", run_id)), metadata=metadata, approved=approved,
            )
            publisher = build_publisher(platform)
            if live:
                _mark_platform(queue, run_id, platform, "publishing")
            result = publisher.publish(job, dry_run=not live)
        except (PermissionError, PublishError, ValueError, OSError, http.client.HTTPException) as exc:
            row = {"platform": platform, "status": _blocked_status(exc), "error": str(exc)}
            results.append(row)
            if live:
                _mark_platform(queue, run_id, platform, row["status"], error=str(exc))
            continue
        row = result.to_dict()
        results.append(row)
        if live:
            _mark_platform(queue, run_id, platform, row["status"], content_id=row["content_id"],
                           published_at=row["published_at"], error=row.get("error", ""))
        if row["status"] == "published":
            entry = dict(payload.get("content_metadata", {}).get(run_id, {}), published_at=row["published_at"])
            binder = binder or CartoonFactoryOrchestrator()
            # Persist now, so this success survives a failure on the next platform.
            binder.bind_published_content(run_id, {platform: {row["content_id"]: entry}})
            bound.setdefault(platform, {})[row["content_id"]] = entry
            done.setdefault(platform, {})[row["content_id"]] = entry
    if live:
        _finish_queue_row(queue, run_id, list(platforms), results)
    return {"run_id": run_id, "live": live, "results": results, "bound": bound}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--platforms", default="youtube,instagram,tiktok")
    parser.add_argument("--approve", action="store_true", help="a person reviewed this episode and approves posting")
    parser.add_argument("--made-for-kids", choices=("yes", "no"), help="explicit audience classification decision")
    parser.add_argument("--force-republish", action="store_true",
                        help="upload again to platforms that already have a published id for this run")
    args = parser.parse_args(argv)
    load_env()
    made_for_kids = None if args.made_for_kids is None else args.made_for_kids == "yes"
    platforms = [p.strip().lower() for p in args.platforms.split(",") if p.strip()]
    try:
        out = publish_run(args.run_id, platforms, approved=args.approve, made_for_kids=made_for_kids,
                          force_republish=args.force_republish)
    except (FileNotFoundError, PermissionError, RuntimeError) as exc:
        print(f"publish_approved: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0 if all(row["status"] in OK_STATUSES for row in out["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
