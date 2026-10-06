from pathlib import Path
from app.providers.motion import LocalMotionProvider, MotionResult, MotionRouter


class StubProvider:
    name = "stub"

    def render(self, image: Path, audio: Path, output: Path, duration: float, shot_index: int) -> MotionResult:
        output.write_bytes(b"ok")
        return MotionResult(output, self.name, "test")


def test_router_accepts_injected_provider(tmp_path):
    image = tmp_path / "image.png"
    audio = tmp_path / "audio.wav"
    output = tmp_path / "out.mp4"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")

    router = MotionRouter(StubProvider())
    result = router.render(image, audio, output, 3.0, 0)

    assert router.provider_name == "stub"
    assert result.output == output
    assert output.read_bytes() == b"ok"


def test_router_defaults_to_local():
    router = MotionRouter()
    assert isinstance(router.provider, LocalMotionProvider)
