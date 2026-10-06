from pathlib import Path

from PIL import Image

from app.providers.cartoon_media import W, H, render_frame


def test_cartoon_frame_is_vertical(tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (768, 1376), "white").save(source)
    frame = render_frame(source, "Test caption", 0.5, 1.08)
    assert frame.size == (W, H)
