from pathlib import Path
from app.providers.local_puppet import _rig_frame
from PIL import Image


def test_puppet_frame_is_vertical_and_dynamic():
    img = Image.new("RGB", (1080, 1920), (21, 23, 28))
    a = _rig_frame(img, 0.1, 0.2, 0)
    b = _rig_frame(img, 0.9, 1.4, 1)
    assert a.size == (540, 960)
    assert b.size == (540, 960)
    assert a.tobytes() != b.tobytes()
