"""Publish a reviewed episode to real platforms after a person approves it.

The factory's publish stage only queues ``ready_for_review`` items. This is the
human-in-the-loop step that follows review:

    python -m app.publish_approved --run-id <run> --platforms youtube,tiktok \
        --made-for-kids no --approve

Without ``--approve``, or unless ``NOVA_ENABLE_REAL_PUBLISH=true`` and the
platform credentials are set, every platform is a dry run and nothing uploads.
Published content IDs are bound to the run so analytics can collect metrics.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from app.cartoon_control import CartoonRunStore
from app.cartoon_factory_orchestrator import CartoonFactoryOrchestrator
from app.providers.publisher import PublishError, PublishJob, build_publisher, real_publish_enabled


def publish_run(run_id: str, platforms: list[str], *, approved: bool, made_for_kids: bool | None,
                orchestrator: CartoonFactoryOrchestrator | None = None) -> dict[str, Any]:
    run = CartoonRunStore(run_id).load()
    payload = run.payload
    if payload.get("final_qa", {}).get("publish_gate", {}).get("decision") != "allow":
        raise PermissionError("final_qa_publish_gate_not_allow")
    candidate = payload["winner"]["candidate"]
    experiment = payload.get("experiment", {})
    metadata: dict[str, Any] = {"ai_generated": True, "run_id": run_id}
    if made_for_kids is not None:
        metadata["made_for_kids"] = made_for_kids
    live = approved and real_publish_enabled()
    results, bound = [], {}
    for platform in platforms:
        job = PublishJob.create(
            video_path=payload["video"], platform=platform, title=candidate["title"],
            caption=candidate.get("caption") or candidate["title"],
            experiment_id=str(experiment.get("experiment_id", run_id)), metadata=metadata, approved=approved,
        )
        publisher = build_publisher(platform)
        try:
            result = publisher.publish(job, dry_run=not live)
        except (PermissionError, PublishError, ValueError) as exc:
            results.append({"platform": platform, "status": "blocked", "error": str(exc)})
            continue
        row = result.to_dict()
        results.append(row)
        if row["status"] == "published":
            bound.setdefault(platform, {})[row["content_id"]] = dict(
                payload.get("content_metadata", {}).get(run_id, {}), published_at=row["published_at"]
            )
    if bound:
        (orchestrator or CartoonFactoryOrchestrator()).bind_published_content(run_id, bound)
    return {"run_id": run_id, "live": live, "results": results, "bound": bound}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--platforms", default="youtube,instagram,tiktok")
    parser.add_argument("--approve", action="store_true", help="a person reviewed this episode and approves posting")
    parser.add_argument("--made-for-kids", choices=("yes", "no"), help="explicit audience classification decision")
    args = parser.parse_args()
    made_for_kids = None if args.made_for_kids is None else args.made_for_kids == "yes"
    platforms = [p.strip().lower() for p in args.platforms.split(",") if p.strip()]
    print(json.dumps(publish_run(args.run_id, platforms, approved=args.approve, made_for_kids=made_for_kids), indent=2))


if __name__ == "__main__":
    main()
