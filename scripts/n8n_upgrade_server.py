from __future__ import annotations
import fnmatch
import hashlib
import hmac
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from app.agents.subsystem_priority import select_subsystem_priority

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPO = Path(__file__).resolve().parents[1]
TOKEN = os.getenv("NOVA_API_TOKEN", "").strip()
LOCK = REPO / ".n8n_upgrade.lock"
MAX_READ = 120_000
ALLOWED_TEST_PREFIX = ".venv/bin/python -m pytest "
CODEX_BIN = os.getenv("CODEX_BIN", "").strip() or "codex"
CODEX_TIMEOUT = 660
CODEX_MAX_PROMPT = 30_000
CODEX_ALLOWED_PREFIXES = ("app/", "tests/", "docs/", "tools/blender/")
CODEX_ALLOWED_FILES = {"pyproject.toml", "pytest.ini", "requirements.txt", "requirements-dev.txt", "config/cartoon_blender_profiles.json"}
# Paths Codex may never change: the approval gate, the platform publishers, the
# Runway spend gate, and the automation itself. Checked before the allow-list.
PROTECTED_PATHS = (
    "app/providers/publisher.py",
    "app/providers/platform_publishers.py",
    "app/publish_approved.py",
    "app/providers/runway_video.py",
    ".github/",
    ".gitignore",
    "scripts/n8n_",
    "workflows/n8n/",
)
# Ignored output that a build or test run legitimately creates. Any other newly
# ignored path (for example a credentials or database file) counts as a violation.
GENERATED_IGNORED_PARTS = (".venv", "artifacts", "__pycache__", ".pytest_cache")
SECRET_FILES = (".env", ".env.local")
BLOCKED_NAMES = {".env", ".env.local", ".git"}

app = FastAPI(title="NOVA n8n Upgrade Executor", version="1.0")

TARGETS = [
    ("performance_animation", 6.8),
    ("audio", 8.5),
    ("editor", 9.0),
    ("final_qa", 9.1),
    ("publishing", None),
    ("analytics", None),
]

QUALITY_PROGRESS = REPO / "docs" / "QUALITY_PROGRESS.json"
ENGINE_ORDER = (
    "performance_animation",
    "audio",
    "editor",
    "final_qa",
    "publishing",
    "analytics",
)


def _load_quality_progress() -> dict:
    """Load the durable engine-by-engine quality scoreboard."""
    default = {
        "version": 1,
        "quality_floor": 8.5,
        "note": "Scores are evidence-backed engineering targets; creative scores remain human-review targets.",
        "engines": {
            "performance_animation": {
                "status": "active",
                "score": 6.8,
                "areas": {
                    "animation_mechanics": 8.2,
                    "anticipation_contact_overshoot": 8.3,
                    "secondary_motion": 7.8,
                    "facial_acting": 7.5,
                    "camera_acting": 6.8,
                    "background_visual_world": 7.5,
                },
            },
            "audio": {"status": "complete", "score": 8.5},
            "editor": {"status": "complete", "score": 9.0},
            "final_qa": {"status": "complete", "score": 9.1},
            "publishing": {"status": "pending", "score": None},
            "analytics": {"status": "pending", "score": None},
        },
    }
    if not QUALITY_PROGRESS.exists():
        return default
    try:
        loaded = json.loads(QUALITY_PROGRESS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    if not isinstance(loaded, dict) or not isinstance(loaded.get("engines"), dict):
        return default
    return loaded


def _current_upgrade_target() -> tuple[str, dict]:
    """Return the evidence-ranked incomplete subsystem and its durable state."""
    progress = _load_quality_progress()
    engines = progress.get("engines", {})
    decision = select_subsystem_priority(progress)
    name = decision["selected"]
    row = engines.get(name, {})
    return name, {**(row if isinstance(row, dict) else {}), "priority": decision}

class Action(BaseModel):
    action: str
    path: str = ""
    content: str = ""
    command: str = ""
    message: str = ""
    branch: str = ""
    pr_number: int | None = None
    limit: int = 80
    pattern: str = ""
    prompt: str = ""

def auth(key: str | None):
    if not TOKEN or not key or not hmac.compare_digest(key, TOKEN):
        raise HTTPException(status_code=401, detail="invalid NOVA API token")

def safe_path(raw: str) -> Path:
    p = (REPO / raw).resolve()
    if p != REPO and REPO not in p.parents:
        raise HTTPException(status_code=400, detail="path escapes repository")
    if p.name in BLOCKED_NAMES or any(part in BLOCKED_NAMES for part in p.relative_to(REPO).parts):
        raise HTTPException(status_code=400, detail="blocked path")
    return p

def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, timeout=300)
    out = (p.stdout + p.stderr).strip()
    if p.returncode:
        raise RuntimeError(out[-12000:] or f"exit={p.returncode}")
    return out[-12000:]

def current_state() -> dict:
    return {
        "branch": run(["git", "branch", "--show-current"]),
        "commit": run(["git", "rev-parse", "HEAD"]),
        "status": run(["git", "status", "--short"]),
        "open_prs": run(["gh", "pr", "list", "--state", "open",
                         "--json", "number,title,headRefName,baseRefName,url"]),
    }



def _repo_status() -> tuple[list[str], set[str]]:
    """Return (changed tracked or untracked paths, ignored paths).

    Ignored files are listed too: a plain ``git status`` never shows them, so an
    edit to an ignored secret file would otherwise go unnoticed.
    """
    # --untracked-files=all lists each new file; by default git collapses a new
    # directory to "app/providers/", which would slip past the protected-path check.
    proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "--ignored=matching", "--untracked-files=all"],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=60,
    )
    if proc.returncode:
        raise RuntimeError((proc.stdout + proc.stderr).strip()[-4000:] or "git status failed")
    changed: list[str] = []
    ignored: set[str] = set()
    for line in proc.stdout.splitlines():
        if not line:
            continue
        entry = line[3:] if len(line) >= 4 else line
        if line.startswith("!! "):
            ignored.add(entry.rstrip("/"))
            continue
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        changed.append(entry)
    return changed, ignored

def _changed_paths() -> list[str]:
    return _repo_status()[0]

def _secret_fingerprint() -> dict[str, str]:
    fingerprint = {}
    for name in SECRET_FILES:
        path = REPO / name
        fingerprint[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
    return fingerprint

def _is_protected(path: str) -> bool:
    return any(path == item or path.startswith(item) for item in PROTECTED_PATHS)

def _is_generated_ignored(path: str) -> bool:
    return any(part in GENERATED_IGNORED_PARTS for part in path.split("/"))

def _ensure_codex_changes_safe(
    ignored_before: set[str] | None = None,
    secrets_before: dict[str, str] | None = None,
) -> list[str]:
    if secrets_before is not None:
        touched = [name for name, digest in _secret_fingerprint().items() if digest != secrets_before.get(name)]
        if touched:
            raise RuntimeError("Codex modified secret files: " + ", ".join(touched))
    paths, ignored = _repo_status()
    if ignored_before is not None:
        new_ignored = sorted(p for p in ignored - ignored_before if not _is_generated_ignored(p))
        if new_ignored:
            raise RuntimeError("Codex created ignored, non-generated files: " + ", ".join(new_ignored))
    protected = [p for p in paths if _is_protected(p)]
    if protected:
        raise RuntimeError("Codex changed protected paths: " + ", ".join(protected))
    unexpected = [p for p in paths if p not in CODEX_ALLOWED_FILES and not p.startswith(CODEX_ALLOWED_PREFIXES)]
    if unexpected:
        raise RuntimeError("Codex changed protected/unexpected paths: " + ", ".join(unexpected))
    return paths

def _codex_exec(prompt: str) -> dict:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt required")
    if len(prompt) > CODEX_MAX_PROMPT:
        raise HTTPException(status_code=400, detail="prompt too large")
    if shutil.which(CODEX_BIN) is None:
        raise HTTPException(status_code=503, detail="Codex CLI not installed")
    lock_path = REPO / ".codex_n8n.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
    except FileExistsError:
        raise HTTPException(status_code=409, detail="Codex cycle already running")
    output_path = Path(f"/tmp/nova-codex-last-message-{os.getpid()}.txt")
    try:
        cmd = [CODEX_BIN, "exec", "--sandbox", "workspace-write", "--color", "never", "--ephemeral", "--output-last-message", str(output_path), prompt]
        proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, timeout=CODEX_TIMEOUT)
        final = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""
        return {"ok": proc.returncode == 0, "exit_code": proc.returncode, "output": final[-20_000:], "stderr": proc.stderr[-12_000:]}
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "exit_code": None, "output": "", "stderr": (str(exc) + "\n" + (exc.stderr or ""))[-12_000:], "timeout": True}
    finally:
        output_path.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)

def _open_pr_for_branch(branch: str) -> dict | None:
    raw = run(["gh", "pr", "list", "--state", "open", "--head", branch, "--json", "number,title,headRefName,baseRefName,url"])
    items = json.loads(raw or "[]")
    return items[0] if items else None

def _pr_checks(pr_number: int) -> tuple[bool, str]:
    """Return (green, evidence) using the actual PR check rollup.

    GitHub can report no required checks even when checks are still running.
    Never interpret that as green; a PR is mergeable only when every reported
    check is completed successfully and at least one check is present.
    """
    proc = subprocess.run(
        ["gh", "pr", "view", str(pr_number), "--json", "statusCheckRollup"],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=120,
    )
    raw = (proc.stdout + proc.stderr).strip()
    if proc.returncode:
        return False, raw or "unable to read PR check rollup"
    try:
        checks = json.loads(proc.stdout).get("statusCheckRollup") or []
    except json.JSONDecodeError:
        return False, raw or "invalid PR check rollup"
    if not checks:
        return False, "no CI checks reported; refusing to assume green"
    pending = [
        f"{check.get('name', 'check')}: {check.get('status', 'UNKNOWN')}"
        for check in checks
        if str(check.get("status", "")).upper() in {"IN_PROGRESS", "QUEUED", "PENDING"}
    ]
    failures = [
        f"{check.get('name', 'check')}: {check.get('conclusion', 'UNKNOWN')}"
        for check in checks
        if str(check.get("status", "")).upper() == "COMPLETED"
        and str(check.get("conclusion", "")).upper() not in {"SUCCESS", "NEUTRAL", "SKIPPED"}
    ]
    if pending:
        return False, "pending checks: " + ", ".join(pending)
    if failures:
        return False, "failed checks: " + ", ".join(failures)
    if all(
        str(check.get("status", "")).upper() == "COMPLETED"
        and str(check.get("conclusion", "")).upper() in {"SUCCESS", "NEUTRAL", "SKIPPED"}
        for check in checks
    ):
        return True, "all reported checks completed successfully: " + ", ".join(
            f"{check.get('name', 'check')}={check.get('conclusion', 'UNKNOWN')}" for check in checks
        )
    return False, raw or "unknown PR check state"

def _codex_cycle() -> dict:
    state = current_state()
    branch = state["branch"]
    open_prs = json.loads(state["open_prs"] or "[]")
    if branch == "main" and open_prs:
        return {"ok": False, "status": "blocked", "reason": "open PR exists; refusing to modify main", "open_prs": open_prs}
    if branch == "main" and state["status"]:
        return {"ok": False, "status": "blocked", "reason": "main working tree is not clean", "changes": state["status"]}
    if branch != "main" and not branch.startswith("codex/overnight-"):
        return {"ok": False, "status": "blocked", "reason": f"refusing to operate on non-Codex branch: {branch}"}

    pr = _open_pr_for_branch(branch) if branch != "main" else None
    if pr:
        green, checks = _pr_checks(int(pr["number"]))
        if green:
            # Never merge from here: a person reviews and merges the PR. Switching
            # back to main keeps the agent blocked until that happens.
            run(["git", "switch", "main"])
            return {"ok": True, "status": "ready_for_review", "pr": pr, "checks": checks, "branch": "main"}
        lowered = checks.lower()
        if any(word in lowered for word in ("pending", "queued", "in progress", "running")):
            return {"ok": True, "status": "ci_pending", "pr": pr, "checks": checks, "branch": branch}

    if branch == "main":
        branch = "codex/overnight-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        run(["git", "switch", "-c", branch])

    target_name, target_state = _current_upgrade_target()
    progress = _load_quality_progress()
    priority = target_state.get("priority", {})
    quality_floor = float(progress.get("quality_floor", 8.5) or 8.5)

    prompt = f'''You are the autonomous senior engineer for the NOVA AI influencer repository.

WORKTREE: {REPO}
BRANCH: {branch}
OPEN PR: {json.dumps(pr) if pr else "none"}

QUALITY CONTROL:
- Quality floor: {quality_floor}
- Active engine: {target_name}
- Active engine progress: {json.dumps(target_state, ensure_ascii=False)}
- Durable scoreboard: docs/QUALITY_PROGRESS.json
- config/cartoon_quality_targets.json is the authoritative definition of quality areas and minimums.
- Deterministic subsystem priority decision: {json.dumps(priority, ensure_ascii=False)}
- Use the selected subsystem as the next target. Prioritize measured gaps, recurring failures, user impact, evidence confidence, and work that unblocks downstream quality. Do not keep optimizing an already-good subsystem while a higher-impact evidenced gap remains. If evidence is missing, gather runtime/render proof or integration evidence using the selector's safe fallback order; do not invent scores.

SAFE FALLBACK ORDER (used only when priority evidence is missing or tied):
1 Runtime/render proof
2 Integrations
3 Performance/Animation AI
4 Audio Engine
5 Editor AI
6 Final QA
7 Publishing
8 Analytics

Script Engine and Story Engine are already frozen/completed and MUST NOT be selected for new work. Do not create structural or regression-only Script Engine changes while the active target is {target_name}. A change is valid only when it materially improves the active target.

Execute exactly ONE upgrade cycle for the selected subsystem: inspect -> identify the highest-priority evidenced gap -> implement -> add regression tests -> run focused tests -> run the full pytest suite when feasible -> update docs/QUALITY_PROGRESS.json with evidence-backed scores/status. A subsystem can be revisited when its current measured gap remains the highest priority, but completion of one subsystem does not override a higher-impact gap or an unblocking dependency.

For Performance/Animation AI, stay inside these quality areas until all are >= {quality_floor}: animation_mechanics, anticipation_contact_overshoot, secondary_motion, facial_acting, camera_acting, background_visual_world. Prefer real character-performance improvements over cosmetic camera-only tricks, and prioritize rig integrity, gaze/facial continuity, hands/arms, anticipation/contact/overshoot/recoil/settle, and targeted motion repair.

IMPORTANT BOUNDARIES:
- Use the ChatGPT-authenticated Codex account; do not use an API key.
- Do NOT use git, gh, or network commands. The host executor handles branch/commit/push/PR/merge.
- Do NOT touch .env, scripts/n8n_*, workflows/n8n/*, or other infrastructure files.
- Do not fabricate scores, metrics, CI results, or credentials. A score update must cite concrete code/test evidence; when evidence is insufficient, keep the engine active.
- Do not spend Runway credits.
- Keep the existing Story Engine freeze intact.
- Stop after one meaningful improvement within the active engine.

At the end, report: active engine, exact quality area improved, concrete changes, evidence/test results, updated scoreboard status, and any blocker.'''
    if pr:
        prompt += "\n\nThe existing PR had failed required checks. Repair only the active engine. Failed-check output:\n" + _pr_checks(int(pr["number"]))[1][-8000:]

    ignored_before = _repo_status()[1]
    secrets_before = _secret_fingerprint()
    codex = _codex_exec(prompt)
    if not codex["ok"]:
        return {"ok": False, "status": "codex_failed", "branch": branch, "codex": codex}
    try:
        changed = _ensure_codex_changes_safe(ignored_before, secrets_before)
    except RuntimeError as exc:
        return {"ok": False, "status": "protected_path_violation", "branch": branch, "codex": codex, "error": str(exc)}
    if not changed:
        return {"ok": True, "status": "no_changes", "branch": branch, "codex": codex}
    try:
        tests = run([".venv/bin/python", "-m", "pytest", "-q"])
    except RuntimeError as exc:
        return {"ok": False, "status": "tests_failed", "branch": branch, "codex": codex, "tests": str(exc)}
    run(["git", "add", "-A"])
    commit = run(["git", "commit", "-m", f"NOVA: autonomous {branch.removeprefix('codex/overnight-')}"])
    push = run(["git", "push", "-u", "origin", "HEAD"])
    pr = _open_pr_for_branch(branch)
    if not pr:
        body = "Autonomous Codex upgrade cycle.\n\n" + codex["output"][-7000:] + "\n\nFull pytest:\n" + tests[-5000:]
        run(["gh", "pr", "create", "--base", "main", "--head", branch, "--title", "NOVA autonomous subsystem upgrade", "--body", body])
        pr = _open_pr_for_branch(branch)
    return {"ok": True, "status": "pr_created_or_updated", "branch": branch, "changed": changed, "tests": tests, "commit": commit, "push": push, "pr": pr, "codex": codex}

@app.get("/health")
def health():
    return {"ok": True, "service": "n8n-upgrade-executor", "repo": str(REPO)}

@app.post("/action")
def action(req: Action, x_nova_api_key: str | None = Header(default=None, alias="X-NOVA-API-Key")):
    auth(x_nova_api_key)
    a = req.action.strip()
    if a == "status":
        return current_state()
    if a == "targets":
        return {"targets": TARGETS}
    if a == "list_files":
        files = []
        for p in REPO.rglob("*"):
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts:
                rel = str(p.relative_to(REPO))
                if not req.pattern or fnmatch.fnmatch(rel, req.pattern):
                    files.append(rel)
        return {"files": sorted(files)[: req.limit]}
    if a == "read_file":
        p = safe_path(req.path)
        if not p.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        return {"path": req.path, "content": p.read_text(encoding="utf-8")[:MAX_READ]}
    if a == "write_file":
        p = safe_path(req.path)
        if _is_protected(req.path.strip()):
            raise HTTPException(status_code=403, detail="protected path: edit it by hand and review the change")
        if not req.path or not p.parent.exists():
            raise HTTPException(status_code=400, detail="invalid file path")
        if len(req.content) > MAX_READ:
            raise HTTPException(status_code=400, detail="content too large")
        p.write_text(req.content, encoding="utf-8")
        return {"ok": True, "path": req.path}

    if a == "run_tests":
        cmd = req.command.strip()
        if not cmd.startswith(ALLOWED_TEST_PREFIX):
            raise HTTPException(status_code=400, detail="only pytest commands are allowed")
        parts = cmd.split()
        return {"ok": True, "output": run(parts)}
    if a == "create_branch":
        name = req.branch.strip()
        if not name or any(c in name for c in " <>~^:?*[]\\"):
            raise HTTPException(status_code=400, detail="invalid branch name")
        if run(["git", "status", "--porcelain"]):
            raise HTTPException(status_code=409, detail="working tree must be clean before branching")
        try:
            run(["git", "switch", name])
        except RuntimeError:
            run(["git", "switch", "-c", name])
        return {"ok": True, "branch": name}
    if a == "commit_push":
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="commit message required")
        if run(["git", "branch", "--show-current"]) == "main":
            raise HTTPException(status_code=409, detail="refusing to commit on main; use a codex/ branch and open a PR")
        run(["git", "add", "-A"])
        diff = run(["git", "status", "--porcelain"])
        if not diff:
            return {"ok": True, "changed": False, "output": "nothing to commit"}
        commit = run(["git", "commit", "-m", req.message.strip()])
        push = run(["git", "push", "-u", "origin", "HEAD"])
        return {"ok": True, "changed": True, "commit": commit, "push": push}
    if a == "create_pr":
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="PR body/summary required")
        title = req.branch.strip() or "NOVA subsystem upgrade"
        out = run(["gh", "pr", "create", "--base", "main", "--head", "HEAD",
                   "--title", title, "--body", req.message])
        return {"ok": True, "output": out}
    if a == "pr_checks":
        if req.pr_number is None:
            raise HTTPException(status_code=400, detail="pr_number required")
        return {"ok": True, "output": run(["gh", "pr", "checks", str(req.pr_number)])}
    if a == "codex_exec":
        return _codex_exec(req.prompt)
    if a == "codex_cycle":
        return _codex_cycle()
    if a == "merge_pr":
        # Disabled on purpose: the agent opens PRs, and a person merges them after
        # review. Branch protection on main (required 'NOVA CI' check, no force
        # push, required review) is set in the repository settings by the owner.
        raise HTTPException(status_code=403, detail="merging is disabled; review and merge the PR on GitHub")
    raise HTTPException(status_code=400, detail=f"unknown action: {a}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8787)
