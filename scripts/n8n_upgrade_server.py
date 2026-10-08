from __future__ import annotations
import fnmatch
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

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPO = Path(__file__).resolve().parents[1]
TOKEN = os.getenv("NOVA_API_TOKEN", "").strip()
LOCK = REPO / ".n8n_upgrade.lock"
MAX_READ = 120_000
ALLOWED_TEST_PREFIX = ".venv/bin/python -m pytest "
CODEX_BIN = shutil.which("codex") or "/Users/sante/.nvm/versions/node/v24.21.0/bin/codex"
CODEX_TIMEOUT = 660
CODEX_MAX_PROMPT = 30_000
CODEX_ALLOWED_PREFIXES = ("app/", "tests/")
CODEX_ALLOWED_FILES = {"pyproject.toml", "pytest.ini", "requirements.txt", "requirements-dev.txt"}
BLOCKED_NAMES = {".env", ".env.local", ".git"}

app = FastAPI(title="NOVA n8n Upgrade Executor", version="1.0")

TARGETS = [
    ("script", 7.5),
    ("performance_animation", 7.0),
    ("audio", 7.0),
    ("editor", 7.0),
    ("final_qa", 7.0),
    ("publishing", 4.0),
    ("analytics", 5.5),
]

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



def _changed_paths() -> list[str]:
    raw = run(["git", "status", "--porcelain=v1"])
    paths: list[str] = []
    for line in raw.splitlines():
        if not line:
            continue
        entry = line[3:] if len(line) >= 4 else line
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        paths.append(entry)
    return paths

def _ensure_codex_changes_safe() -> list[str]:
    paths = _changed_paths()
    unexpected = [p for p in paths if p not in CODEX_ALLOWED_FILES and not p.startswith(CODEX_ALLOWED_PREFIXES)]
    if unexpected:
        raise RuntimeError("Codex changed protected/unexpected paths: " + ", ".join(unexpected))
    return paths

def _codex_exec(prompt: str) -> dict:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt required")
    if len(prompt) > CODEX_MAX_PROMPT:
        raise HTTPException(status_code=400, detail="prompt too large")
    if not shutil.which("codex") and not Path(CODEX_BIN).exists():
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
    proc = subprocess.run(["gh", "pr", "checks", str(pr_number), "--required"], cwd=REPO, text=True, capture_output=True, timeout=120)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()

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
            merged = run(["gh", "pr", "merge", str(pr["number"]), "--squash", "--delete-branch"])
            run(["git", "switch", "main"])
            return {"ok": True, "status": "merged", "pr": pr, "checks": checks, "merge": merged, "branch": "main"}
        lowered = checks.lower()
        if any(word in lowered for word in ("pending", "queued", "in progress", "running")):
            return {"ok": True, "status": "ci_pending", "pr": pr, "checks": checks, "branch": branch}

    if branch == "main":
        branch = "codex/overnight-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        run(["git", "switch", "-c", branch])

    prompt = f'''You are the autonomous senior engineer for the NOVA AI influencer repository.

WORKTREE: {REPO}
BRANCH: {branch}
OPEN PR: {json.dumps(pr) if pr else "none"}

Roadmap, in strict order, with current scores:
1 Script Engine 7.5
2 Performance/Animation AI 7.0
3 Audio Engine 7.0
4 Editor AI 7.0
5 Final QA 7.0
6 Publishing 4.0
7 Analytics 5.5

Execute exactly one subsystem upgrade cycle. Choose the highest-priority subsystem still below 8.5, unless an existing open PR above is the subsystem you must repair. Inspect current code and tests first. Implement a real integrated improvement and add regression tests. Run focused tests and then the full pytest suite when feasible.

IMPORTANT BOUNDARIES:
- Use the ChatGPT-authenticated Codex account; do not use an API key.
- Do NOT use git, gh, or network commands. The host executor handles branch/commit/push/PR/merge.
- Do NOT touch .env, scripts/n8n_*, workflows/n8n/*, or other infrastructure files.
- Do not fabricate scores, metrics, CI results, or credentials.
- Do not spend Runway credits.
- Keep the existing Story Engine freeze intact.
- Stop after one subsystem is actually implemented and tested.

At the end, report the subsystem, concrete changes, focused tests, full-suite result, and any blocker.'''
    if pr:
        prompt += "\n\nThe existing PR had failed required checks. Repair only the relevant implementation and tests. Failed-check output:\n" + _pr_checks(int(pr["number"]))[1][-8000:]

    codex = _codex_exec(prompt)
    if not codex["ok"]:
        return {"ok": False, "status": "codex_failed", "branch": branch, "codex": codex}
    try:
        changed = _ensure_codex_changes_safe()
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
        run(["gh", "pr", "create", "--base", "main", "--head", "HEAD", "--title", "NOVA autonomous subsystem upgrade", "--body", body])
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
        if req.pr_number is None:
            raise HTTPException(status_code=400, detail="pr_number required")
        checks = run(["gh", "pr", "checks", str(req.pr_number), "--required"])
        merged = run(["gh", "pr", "merge", str(req.pr_number), "--squash", "--delete-branch"])
        return {"ok": True, "checks": checks, "merge": merged}
    raise HTTPException(status_code=400, detail=f"unknown action: {a}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8787)
