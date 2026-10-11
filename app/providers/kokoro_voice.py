"""Kokoro-82M text-to-speech for cartoon character voices (free, Apache-2.0 weights).

Kokoro runs through ``kokoro-onnx`` (MIT, optional: ``pip install -r
requirements-voice.txt``). Three sharp edges are handled here:

* Kokoro's bundled espeak-ng loader points at a data path that does not exist on
  most machines, and the native library then calls ``exit()``. That would kill the
  orchestrator instead of raising, so synthesis runs in a child process pointed at
  the system ``libespeak-ng`` and ``espeak-ng-data`` (installed by the CI and local
  setup). A crash there is an ordinary ``KokoroUnavailable`` and the caller falls
  back to the next voice backend.
* The model files (about 120 MB) come from the kokoro-onnx GitHub release. They are
  cached under ``NOVA_MODEL_CACHE`` (default ``~/.cache/nova/models``), verified by
  size and SHA-256, and never committed. ``python -m app.providers.kokoro_voice
  fetch`` downloads them ahead of time (CI caches that directory).
* Auto selection only uses Kokoro when the package and verified model files are
  already present. Downloads happen only for an explicit ``NOVA_VOICE_BACKEND=kokoro``
  or the ``fetch`` command, so a render never starts a surprise download.
"""
from __future__ import annotations

import ctypes.util
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
import wave
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
RELEASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
TIMEOUT_SECONDS = 240.0
# Speed 1.0 is about 200-210 words per minute on dialog-style lines, the top of the target range.
NEUTRAL_RATE = 210
SPEED_RANGE = (0.7, 1.4)
ESPEAK_DATA_CANDIDATES = (
    "/usr/lib/x86_64-linux-gnu/espeak-ng-data",
    "/usr/lib/aarch64-linux-gnu/espeak-ng-data",
    "/usr/lib/espeak-ng-data",
    "/usr/share/espeak-ng-data",
    "/opt/homebrew/share/espeak-ng-data",
    "/usr/local/share/espeak-ng-data",
)


class KokoroUnavailable(RuntimeError):
    """Kokoro cannot run here: package, system espeak-ng, model files or synthesis."""


@dataclass(frozen=True)
class ModelFile:
    name: str
    size: int
    sha256: str


MODEL_FILES = (
    ModelFile("kokoro-v1.0.int8.onnx", 92361271, "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb"),
    ModelFile("voices-v1.0.bin", 28214398, "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
)
_verified_cache: dict[str, tuple[int, int, bool]] = {}


@dataclass(frozen=True)
class EspeakLibrary:
    lib_path: str
    data_path: str


def cache_dir() -> Path:
    base = os.getenv("NOVA_MODEL_CACHE", "").strip() or str(Path.home() / ".cache" / "nova" / "models")
    return Path(base).expanduser() / "kokoro"


def system_espeak() -> EspeakLibrary | None:
    """System libespeak-ng plus its data directory, or None when either is missing."""
    lib = os.getenv("NOVA_ESPEAK_LIB", "").strip() or ctypes.util.find_library("espeak-ng") or ""
    data = os.getenv("NOVA_ESPEAK_DATA", "").strip()
    if not data:
        data = next((p for p in ESPEAK_DATA_CANDIDATES if (Path(p) / "phontab").exists()), "")
    if not lib or not data:
        return None
    return EspeakLibrary(lib_path=lib, data_path=data)


def package_available() -> bool:
    return find_spec("kokoro_onnx") is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(path: Path, model: ModelFile) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    key = str(path)
    memo = _verified_cache.get(key)
    if memo and memo[:2] == (stat.st_size, stat.st_mtime_ns):
        return memo[2] and stat.st_size == model.size
    ok = stat.st_size == model.size and _sha256(path) == model.sha256
    _verified_cache[key] = (stat.st_size, stat.st_mtime_ns, ok)
    return ok


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)


def ensure_model(
    directory: Path | None = None,
    *,
    download: bool = True,
    fetch: Callable[[str, Path], None] = _download,
) -> dict[str, Path]:
    """Return verified model paths, downloading missing files only when ``download`` is true."""
    target = directory or cache_dir()
    paths: dict[str, Path] = {}
    for model in MODEL_FILES:
        path = target / model.name
        if not _verified(path, model):
            if not download:
                raise KokoroUnavailable(f"Kokoro model {model.name} is not cached in {target}")
            target.mkdir(parents=True, exist_ok=True)
            part = target / (model.name + ".part")
            try:
                fetch(f"{RELEASE_URL}/{model.name}", part)
            except Exception as exc:  # network errors vary by platform
                part.unlink(missing_ok=True)
                raise KokoroUnavailable(f"could not download {model.name}: {exc}") from exc
            if not _verified(part, model):
                part.unlink(missing_ok=True)
                raise KokoroUnavailable(f"{model.name} failed its size or SHA-256 check")
            os.replace(part, path)
        paths[model.name] = path
    return {"model": paths[MODEL_FILES[0].name], "voices": paths[MODEL_FILES[1].name]}


def ready() -> bool:
    """True when auto selection may use Kokoro: package, system espeak-ng and cached models."""
    if not package_available() or system_espeak() is None:
        return False
    try:
        ensure_model(download=False)
    except KokoroUnavailable:
        return False
    return True


def synthesize(text: str, output: Path, *, voice: str, speed: float = 1.0, lang: str = "en-us") -> dict:
    """Render ``text`` to a WAV at ``output`` in a child process; raises KokoroUnavailable on any failure."""
    if not package_available():
        raise KokoroUnavailable("kokoro-onnx is not installed (pip install -r requirements-voice.txt)")
    espeak = system_espeak()
    if espeak is None:
        raise KokoroUnavailable("system libespeak-ng and espeak-ng-data are required for Kokoro")
    models = ensure_model()
    request = {
        "model": str(models["model"]),
        "voices": str(models["voices"]),
        "espeak_lib": espeak.lib_path,
        "espeak_data": espeak.data_path,
        "text": text,
        "voice": voice,
        "speed": min(SPEED_RANGE[1], max(SPEED_RANGE[0], float(speed))),
        "lang": lang,
        "output": str(output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        done = subprocess.run(
            [sys.executable, "-m", "app.providers.kokoro_voice", "worker"],
            input=json.dumps(request), cwd=str(ROOT), capture_output=True, text=True,
            timeout=TIMEOUT_SECONDS, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise KokoroUnavailable("Kokoro synthesis timed out") from exc
    if done.returncode != 0 or not output.exists():
        raise KokoroUnavailable(f"Kokoro worker failed (exit {done.returncode}): {done.stderr.strip()[-300:]}")
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise KokoroUnavailable("Kokoro worker returned no result") from exc


def _write_wav(path: Path, samples, sample_rate: int) -> None:
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())


def _worker() -> int:
    """Child-process entry point: read one JSON request on stdin, write the WAV, print the result."""
    request = json.loads(sys.stdin.read())
    from kokoro_onnx import EspeakConfig, Kokoro

    kokoro = Kokoro(
        request["model"],
        request["voices"],
        espeak_config=EspeakConfig(lib_path=request["espeak_lib"], data_path=request["espeak_data"]),
    )
    samples, sample_rate = kokoro.create(
        request["text"], voice=request["voice"], speed=float(request["speed"]), lang=request["lang"],
    )
    _write_wav(Path(request["output"]), samples, int(sample_rate))
    print(json.dumps({"sample_rate": int(sample_rate), "seconds": round(len(samples) / sample_rate, 3)}))
    return 0


def main(argv: list[str]) -> int:
    command = argv[0] if argv else ""
    if command == "worker":
        return _worker()
    if command == "fetch":
        paths = ensure_model()
        print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
        return 0
    print("usage: python -m app.providers.kokoro_voice {fetch|worker}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
