"""Sync factory memory and run records with the ``factory-state`` git branch.

The cartoon factory runs on ephemeral GitHub runners, so its learning memory
must survive between runs. The ``factory-state`` branch holds it::

    state/cartoon/<memory>.json           <- artifacts/cartoon/<memory>.json
    state/cartoon/runs/<run_id>/run.json  <- artifacts/cartoon/runs/<run_id>/run.json

Per-episode files (anything with ``_factory`` in its name) are not synced: the
episode video travels as a draft GitHub release asset instead.

Commands::

    ensure-branch [--remote R]            create an empty factory-state branch if the remote lacks it
    pull --state-dir DIR                  copy state into artifacts/cartoon (no state is a no-op)
    push --state-dir DIR [--message M]    copy artifacts/cartoon into DIR and commit if changed
    episode-video --state-dir DIR --run-id ID
                                          print the repo-relative mp4 path recorded for a run

Authentication for ensure-branch uses GH_TOKEN or GITHUB_TOKEN when set, passed as
an HTTP header for that command only, so no credential is written to disk.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARTOON_DIR = ROOT / "artifacts" / "cartoon"
STATE_SUBDIR = Path("state") / "cartoon"
EPISODE_MARKER = "_factory"


def memory_files(cartoon_dir: Path) -> list[Path]:
    """Top-level JSON memory: learning, metrics, trend history, experiments, ledgers, queues."""
    return sorted(p for p in cartoon_dir.glob("*.json") if p.is_file() and EPISODE_MARKER not in p.name)


def run_records(cartoon_dir: Path) -> list[Path]:
    return sorted(p for p in (cartoon_dir / "runs").glob("*/run.json") if p.is_file())


def _copy_tree(files: list[Path], source_root: Path, dest_root: Path) -> int:
    for path in files:
        target = dest_root / path.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return len(files)


def _state_files(state_dir: Path) -> tuple[list[Path], list[Path]]:
    base = state_dir / STATE_SUBDIR
    if not base.is_dir():
        return [], []
    return memory_files(base), run_records(base)


def pull(state_dir: Path, cartoon_dir: Path = CARTOON_DIR) -> dict[str, int]:
    base = state_dir / STATE_SUBDIR
    memory, runs = _state_files(state_dir)
    if not memory and not runs:
        print(f"no factory state found under {base}; starting from empty memory")
        return {"memory": 0, "runs": 0}
    counts = {
        "memory": _copy_tree(memory, base, cartoon_dir),
        "runs": _copy_tree(runs, base, cartoon_dir),
    }
    print(f"restored {counts['memory']} memory files and {counts['runs']} run records")
    return counts


_IDENTITY = {
    "GIT_AUTHOR_NAME": "nova-factory",
    "GIT_AUTHOR_EMAIL": "nova-factory@users.noreply.github.com",
    "GIT_COMMITTER_NAME": "nova-factory",
    "GIT_COMMITTER_EMAIL": "nova-factory@users.noreply.github.com",
}


def _git_env() -> dict[str, str]:
    """Commit identity for CI runners that have no global git configuration."""
    return {**_IDENTITY, **os.environ}


def _git(state_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(state_dir), *args], text=True, capture_output=True, check=False, env=_git_env()
    )


def _auth_args() -> list[str]:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        return []
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}"]


def ensure_branch(remote: str = "origin", branch: str = "factory-state") -> bool:
    """Create the state branch as an empty root commit when the remote does not have it yet."""
    listed = subprocess.run(
        ["git", *_auth_args(), "ls-remote", "--exit-code", "--heads", remote, branch],
        text=True,
        capture_output=True,
        check=False,
    )
    if listed.returncode == 0:
        print(f"{branch} exists on {remote}")
        return False
    if listed.returncode != 2:
        raise RuntimeError(f"git ls-remote failed: {listed.stderr.strip()}")
    empty_tree = subprocess.run(
        ["git", "mktree"], input="", text=True, capture_output=True, check=True
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "commit-tree", empty_tree, "-m", f"Create {branch} branch"],
        text=True,
        capture_output=True,
        check=True,
        env=_git_env(),
    ).stdout.strip()
    pushed = subprocess.run(
        ["git", *_auth_args(), "push", remote, f"{commit}:refs/heads/{branch}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if pushed.returncode:
        raise RuntimeError(f"could not create {branch}: {pushed.stderr.strip()}")
    print(f"created {branch} on {remote}")
    return True


def push(state_dir: Path, message: str, cartoon_dir: Path = CARTOON_DIR) -> bool:
    """Copy current memory and run records into the state checkout; commit only real changes."""
    base = state_dir / STATE_SUBDIR
    _copy_tree(memory_files(cartoon_dir), cartoon_dir, base)
    _copy_tree(run_records(cartoon_dir), cartoon_dir, base)
    if not base.is_dir():
        print("no factory state to store yet")
        return False

    added = _git(state_dir, "add", "-A", "--", str(STATE_SUBDIR))
    if added.returncode:
        raise RuntimeError(f"git add failed: {added.stderr.strip()}")
    staged = _git(state_dir, "diff", "--cached", "--quiet", "--", str(STATE_SUBDIR))
    if staged.returncode == 0:
        print("factory state unchanged; nothing to commit")
        return False
    if staged.returncode != 1:
        raise RuntimeError(f"git diff failed: {staged.stderr.strip()}")
    committed = _git(state_dir, "commit", "-m", message)
    if committed.returncode:
        raise RuntimeError(f"git commit failed: {(committed.stdout + committed.stderr).strip()}")
    print("committed factory state")
    return True


def episode_video(state_dir: Path, run_id: str) -> str:
    """Return the repo-relative path where a run's episode video belongs.

    Payloads written before paths became repo-relative hold absolute paths from
    another machine, so only the file name is trusted; episodes always live in
    artifacts/cartoon.
    """
    record = state_dir / STATE_SUBDIR / "runs" / run_id / "run.json"
    if not record.is_file():
        raise SystemExit(f"no run record for {run_id} at {record}")
    video = json.loads(record.read_text(encoding="utf-8")).get("payload", {}).get("video", "")
    if not video:
        raise SystemExit(f"run {run_id} recorded no episode video")
    return (CARTOON_DIR.relative_to(ROOT) / Path(video).name).as_posix()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ensure_cmd = sub.add_parser("ensure-branch")
    ensure_cmd.add_argument("--remote", default="origin")
    pull_cmd = sub.add_parser("pull")
    pull_cmd.add_argument("--state-dir", type=Path, required=True)
    push_cmd = sub.add_parser("push")
    push_cmd.add_argument("--state-dir", type=Path, required=True)
    push_cmd.add_argument("--message", default="Update factory state")
    video_cmd = sub.add_parser("episode-video")
    video_cmd.add_argument("--state-dir", type=Path, required=True)
    video_cmd.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    if args.command == "ensure-branch":
        ensure_branch(args.remote)
    elif args.command == "pull":
        pull(args.state_dir)
    elif args.command == "push":
        push(args.state_dir, args.message)
    else:
        print(episode_video(args.state_dir, args.run_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
