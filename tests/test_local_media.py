from pathlib import Path

def test_character_asset_path_is_documented():
    assert Path("assets/character").exists()
