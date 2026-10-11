import hashlib
import wave
from pathlib import Path

import pytest

from app.providers import kokoro_voice
from app.providers.kokoro_voice import KokoroUnavailable, ModelFile


def _fake_models(tmp_path: Path, monkeypatch) -> tuple[Path, dict[str, bytes]]:
    """Swap the real 120 MB model list for tiny files so checksum logic runs offline."""
    payloads = {"model.bin": b"abc", "voices.bin": b"xyz123"}
    models = (
        ModelFile("model.bin", len(payloads["model.bin"]), hashlib.sha256(payloads["model.bin"]).hexdigest()),
        ModelFile("voices.bin", len(payloads["voices.bin"]), hashlib.sha256(payloads["voices.bin"]).hexdigest()),
    )
    monkeypatch.setattr(kokoro_voice, "MODEL_FILES", models)
    monkeypatch.setattr(kokoro_voice, "_verified_cache", {})
    return tmp_path / "cache", payloads


def _write_fetch(payloads: dict[str, bytes], calls: list[str]):
    def fetch(url: str, dest: Path) -> None:
        name = url.rsplit("/", 1)[-1]
        calls.append(name)
        dest.write_bytes(payloads[name])
    return fetch


def test_download_is_verified_then_cached(tmp_path, monkeypatch):
    cache, payloads = _fake_models(tmp_path, monkeypatch)
    calls: list[str] = []
    paths = kokoro_voice.ensure_model(cache, fetch=_write_fetch(payloads, calls))
    assert sorted(calls) == ["model.bin", "voices.bin"]
    assert paths["model"].read_bytes() == b"abc"
    # A second call finds verified files and never touches the network.
    kokoro_voice.ensure_model(cache, download=False, fetch=lambda *_: pytest.fail("network used"))


def test_corrupt_download_is_rejected_and_leaves_no_partial_file(tmp_path, monkeypatch):
    cache, payloads = _fake_models(tmp_path, monkeypatch)

    def corrupt(url: str, dest: Path) -> None:
        dest.write_bytes(b"not the model")

    with pytest.raises(KokoroUnavailable, match="SHA-256"):
        kokoro_voice.ensure_model(cache, fetch=corrupt)
    assert not (cache / "model.bin").exists()
    assert not (cache / "model.bin.part").exists()


def test_auto_mode_never_downloads(tmp_path, monkeypatch):
    cache, _ = _fake_models(tmp_path, monkeypatch)
    with pytest.raises(KokoroUnavailable, match="not cached"):
        kokoro_voice.ensure_model(cache, download=False, fetch=lambda *_: pytest.fail("network used"))


def test_cache_directory_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NOVA_MODEL_CACHE", str(tmp_path / "models"))
    assert kokoro_voice.cache_dir() == tmp_path / "models" / "kokoro"


def test_missing_system_espeak_means_kokoro_is_unavailable(monkeypatch):
    import ctypes.util

    monkeypatch.setenv("NOVA_ESPEAK_LIB", "")
    monkeypatch.setenv("NOVA_ESPEAK_DATA", "")
    monkeypatch.setattr(kokoro_voice, "ESPEAK_DATA_CANDIDATES", ())
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
    assert kokoro_voice.system_espeak() is None


def test_worker_failure_is_reported_as_unavailable_not_a_crash(tmp_path, monkeypatch):
    """Regression: an in-process Kokoro with the default espeak loader exits the whole interpreter."""
    cache, payloads = _fake_models(tmp_path, monkeypatch)
    kokoro_voice.ensure_model(cache, fetch=_write_fetch(payloads, []))
    monkeypatch.setattr(kokoro_voice, "package_available", lambda: True)
    monkeypatch.setattr(kokoro_voice, "system_espeak", lambda: kokoro_voice.EspeakLibrary("/nonexistent.so", "/nonexistent"))
    monkeypatch.setattr(kokoro_voice, "ensure_model", lambda **_: {"model": cache / "model.bin", "voices": cache / "voices.bin"})
    with pytest.raises(KokoroUnavailable, match="worker failed"):
        kokoro_voice.synthesize("Hello there.", tmp_path / "out.wav", voice="af_heart")
    assert not (tmp_path / "out.wav").exists()


def test_synthesis_renders_a_wav_when_kokoro_and_models_are_present(tmp_path, monkeypatch):
    pytest.importorskip("kokoro_onnx")
    if kokoro_voice.system_espeak() is None or not kokoro_voice.ready():
        pytest.skip("needs system espeak-ng and cached Kokoro models (python -m app.providers.kokoro_voice fetch)")
    out = tmp_path / "line.wav"
    result = kokoro_voice.synthesize("Pip is trying to help.", out, voice="am_puck", speed=1.0)
    assert result["sample_rate"] == 24000
    with wave.open(str(out)) as wav:
        assert wav.getframerate() == 24000
        seconds = wav.getnframes() / wav.getframerate()
    assert 0.8 < seconds < 4.0


def test_package_is_optional_and_reported_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(kokoro_voice, "package_available", lambda: False)
    with pytest.raises(KokoroUnavailable, match="requirements-voice.txt"):
        kokoro_voice.synthesize("Hello.", tmp_path / "unused.wav", voice="af_heart")
