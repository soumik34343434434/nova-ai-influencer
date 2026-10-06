from PIL import Image
from app.providers.local_puppet import _rig_frame


def test_performance_changes_motion_profile_without_face_repainting():
    img = Image.new("RGB", (1080, 1920), (40, 40, 40))
    deadpan = {"expression": "deadpan", "gesture": "freeze", "attention": "camera", "head": 0.15}
    surprised = {"expression": "surprised", "gesture": "phone_glance", "attention": "scene", "head": 0.65}

    a = _rig_frame(img, 0.2, 0.4, 0, deadpan)
    b = _rig_frame(img, 0.8, 1.2, 0, surprised)

    assert a.size == (540, 960)
    assert b.size == (540, 960)
    assert a.tobytes() != b.tobytes()
