from pathlib import Path

def test_character_asset_path_is_documented():
    assert Path("assets/character").exists()

def test_ffmpeg_provider_does_not_require_drawtext():
    source = Path("app/providers/local_media.py").read_text()
    assert "drawtext" not in source
