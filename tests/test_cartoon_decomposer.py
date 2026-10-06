from pathlib import Path

from PIL import Image

from app.providers.cartoon_decomposer import REGIONS, decompose_action, load_action_layers


def test_known_pip_assets_have_decomposition_regions():
    assets = sorted(Path("assets/cartoon/actions").glob("pip_*.png"))
    assert assets
    assert {p.name for p in assets} <= set(REGIONS)


def test_decompose_action_writes_rgba_layers(tmp_path: Path):
    source = Path("assets/cartoon/actions/pip_press_button.png")
    meta = decompose_action(source, tmp_path)
    assert meta["layers"] == ["full", "body", "torso", "head", "left_arm", "right_arm", "backpack"]
    for name in ("full", "body", "torso", "head", "left_arm", "right_arm", "backpack"): 
        layer = Image.open(tmp_path / f"pip_press_button_{name}.png")
        assert layer.mode == "RGBA"
        assert layer.size == (1080, 1920)
    plate = Image.open(tmp_path / "pip_press_button_body_plate.jpg")
    assert plate.size == (1080, 1920)


def test_load_action_layers_is_cached(tmp_path: Path):
    source = Path("assets/cartoon/actions/pip_press_button.png")
    load_action_layers(source, tmp_path)
    first = load_action_layers(source, tmp_path)
    second = load_action_layers(source, tmp_path)
    assert first["head"] is second["head"]
