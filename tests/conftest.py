"""Shared test isolation.

Every test runs with:

* every module-level path under the real ``artifacts/`` directory redirected to
  ``tmp_path`` (run records, learning memory, queues, ledgers, fake renders), so
  no test can change the operator's or CI's learning state;
* operator credentials and switches (``NOVA_*``, ``RUNWAY*``, platform keys)
  removed from the environment, so results never depend on the machine;
* a session-level guard that fails the run if the real ``artifacts/`` tree
  changed while the tests ran, which catches any path built at call time that the
  redirect cannot see.
"""

from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL_ARTIFACTS = ROOT / "artifacts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRUBBED_ENV_PREFIXES = ("NOVA_", "RUNWAY", "YOUTUBE_", "TIKTOK_", "INSTAGRAM_")
_SNAPSHOT_KEY = pytest.StashKey[dict]()


def _artifact_snapshot() -> dict[str, tuple[int, int]]:
    if not REAL_ARTIFACTS.is_dir():
        return {}
    snapshot = {}
    for path in REAL_ARTIFACTS.rglob("*"):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(ROOT))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def pytest_configure(config: pytest.Config) -> None:
    for key in list(os.environ):
        if key.startswith(SCRUBBED_ENV_PREFIXES):
            os.environ.pop(key)
    config.stash[_SNAPSHOT_KEY] = _artifact_snapshot()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    before = session.config.stash.get(_SNAPSHOT_KEY, {})
    after = _artifact_snapshot()
    if after == before:
        return
    changed = {path for path in before.keys() | after.keys() if before.get(path) != after.get(path)}
    print(
        "\nERROR: tests changed the real artifacts/ directory (must stay untouched):\n  "
        + "\n  ".join(sorted(changed)[:50])
    )
    if exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


_APP_MODULES: list | None = None


def _import_app_modules() -> list:
    """Import every app module so each one's path constants are visible to the redirect."""
    global _APP_MODULES
    if _APP_MODULES is None:
        import app

        modules = []
        for info in pkgutil.walk_packages(app.__path__, prefix="app."):
            try:
                modules.append(importlib.import_module(info.name))
            except Exception:  # a module that cannot import is not a path-leak risk
                continue
        _APP_MODULES = modules
    return _APP_MODULES


def _is_real_artifact_path(value: object) -> bool:
    if not isinstance(value, Path):
        return False
    try:
        value.relative_to(REAL_ARTIFACTS)
    except ValueError:
        return False
    return True


def _functions_of(module) -> list:
    functions = []
    for value in list(vars(module).values()):
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            functions.append(value)
        elif inspect.isclass(value) and value.__module__ == module.__name__:
            functions.extend(v for v in vars(value).values() if inspect.isfunction(v))
    return functions


@pytest.fixture(autouse=True)
def _isolate_artifacts_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in list(os.environ):
        if key.startswith(SCRUBBED_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)

    sandbox = tmp_path / "artifacts"

    def redirect(value: object) -> object:
        return sandbox / value.relative_to(REAL_ARTIFACTS) if _is_real_artifact_path(value) else value

    for module in _import_app_modules():
        for name, value in list(vars(module).items()):
            if _is_real_artifact_path(value):
                monkeypatch.setattr(module, name, redirect(value))
        # Default arguments such as ``path: Path = REGISTRY_FILE`` are bound at
        # import time, so the module constant alone does not reach them.
        for func in _functions_of(module):
            if func.__defaults__ and any(_is_real_artifact_path(v) for v in func.__defaults__):
                monkeypatch.setattr(func, "__defaults__", tuple(redirect(v) for v in func.__defaults__))
            if func.__kwdefaults__ and any(_is_real_artifact_path(v) for v in func.__kwdefaults__.values()):
                monkeypatch.setattr(
                    func, "__kwdefaults__", {k: redirect(v) for k, v in func.__kwdefaults__.items()}
                )
