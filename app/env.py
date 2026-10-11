"""Load the operator's ``.env`` file, from command-line entry points only.

Importing a module must never change ``os.environ``: a developer's ``.env`` would
otherwise leak into unrelated code and tests. Entry points call ``load_env()``
once at startup. Values already exported by the shell always win.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from app.paths import ROOT


def load_env(path: str | Path | None = None) -> bool:
    """Set variables from ``.env`` that are not already set. Returns True if the file was read."""
    target = Path(path) if path is not None else ROOT / ".env"
    if not target.is_file():
        return False
    return load_dotenv(target, override=False)
