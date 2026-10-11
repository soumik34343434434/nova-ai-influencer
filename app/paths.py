"""Repo-relative paths for run records, with legacy absolute-path resolution.

Run payloads used to store absolute paths such as
``/home/user/nova-ai-influencer/artifacts/cartoon/x.mp4``. Those break as soon as
the same run is read from another checkout, for example an episode made on a
GitHub runner (``/home/runner/work/...``) and downloaded for approval.

Going forward run records store paths relative to the repository root
(``artifacts/cartoon/x.mp4``). Every reader resolves them through
``resolve_repo_path``, which also re-roots legacy absolute paths at their
``artifacts/`` segment when the file exists under this checkout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_SEGMENT = "artifacts"


def to_repo_relative(path: str | Path) -> str:
    """Repo-relative POSIX path for anything inside ROOT; other values are returned unchanged."""
    text = str(path)
    candidate = Path(text)
    if not candidate.is_absolute():
        return text
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError:
        return text


def resolve_repo_path(value: str | Path) -> Path:
    """Concrete path for a stored value; relative values are relative to ROOT.

    An absolute path that exists is returned as-is. A missing absolute path is
    re-rooted at its ``artifacts/`` segment when that file exists under ROOT,
    which is how a run made on another machine is found after its artifacts are
    downloaded into this checkout. Otherwise the original path is returned so the
    caller reports the path it was given.
    """
    candidate = Path(value)
    if not candidate.is_absolute():
        return ROOT / candidate
    if candidate.exists():
        return candidate
    parts = candidate.parts
    if ARTIFACTS_SEGMENT in parts:
        index = parts.index(ARTIFACTS_SEGMENT)
        rerooted = ROOT.joinpath(*parts[index:])
        if rerooted.exists():
            return rerooted
    return candidate


def map_path_strings(value: Any, convert: Callable[[str], str]) -> Any:
    """Apply ``convert`` to every string inside a JSON-like structure."""
    if isinstance(value, str):
        return convert(value)
    if isinstance(value, dict):
        return {key: map_path_strings(item, convert) for key, item in value.items()}
    if isinstance(value, list):
        return [map_path_strings(item, convert) for item in value]
    return value


def portable_paths(value: Any) -> Any:
    """Form written to disk: absolute paths inside ROOT become repo-relative."""
    return map_path_strings(value, to_repo_relative)


def _resolve_if_artifact_path(text: str) -> str:
    candidate = Path(text)
    if text.startswith(f"{ARTIFACTS_SEGMENT}/") or (candidate.is_absolute() and ARTIFACTS_SEGMENT in candidate.parts):
        return str(resolve_repo_path(text))
    return text


def resolve_stored_paths(value: Any) -> Any:
    """Form read from disk: stored artifact paths resolved for this checkout."""
    return map_path_strings(value, _resolve_if_artifact_path)
