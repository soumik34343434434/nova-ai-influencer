"""The n8n factory export must call exactly the stages the cartoon factory runs.

Stage keys are checked against CartoonStage, so renumbering a stage in the app
fails here instead of as an HTTP 400 in a live n8n run.
"""

import json
import re
from pathlib import Path

import pytest

from app.cartoon_control import STAGES, CartoonStage

ROOT = Path(__file__).resolve().parents[1]
N8N_DIR = ROOT / "workflows" / "n8n"
LIVE_FACTORY = N8N_DIR / "nova_live_factory.json"
STAGE_URL = re.compile(r"/stages/([A-Za-z0-9_]+)")
PERSONAL_PATTERNS = [
    re.compile(r"@"),
    re.compile(r"/Users/|/home/[a-z]"),
    re.compile(r"\bsante\b|Soumik", re.IGNORECASE),
    re.compile(r"creatorId|projectId|\"shared\""),
    re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"),
]


def _stage_keys_in_order(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    keys = []
    for node in data["nodes"]:
        url = node.get("parameters", {}).get("url", "")
        match = STAGE_URL.search(url)
        if match:
            keys.append(match.group(1))
    return keys


def test_live_factory_calls_every_stage_once_in_pipeline_order():
    assert _stage_keys_in_order(LIVE_FACTORY) == [stage.value for stage in STAGES]


def test_live_factory_stage_keys_are_valid_cartoon_stages():
    valid = {stage.value for stage in CartoonStage}
    for key in _stage_keys_in_order(LIVE_FACTORY):
        assert key in valid, f"n8n calls unknown stage {key}"


def test_live_factory_has_sixteen_numbered_stage_nodes_including_the_editor():
    data = json.loads(LIVE_FACTORY.read_text(encoding="utf-8"))
    names = [node["name"] for node in data["nodes"] if node["name"][:1].isdigit()]
    assert len(names) == 16
    assert [int(name.split(" ", 1)[0]) for name in names] == list(range(1, 17))
    assert any(name.endswith("Editor") for name in names)


def test_live_factory_chains_stages_in_order():
    data = json.loads(LIVE_FACTORY.read_text(encoding="utf-8"))
    connections = data["connections"]
    names = [node["name"] for node in data["nodes"] if node["name"][:1].isdigit()]
    for upstream, downstream in zip(names, names[1:]):
        assert connections[upstream]["main"][0][0]["node"] == downstream


@pytest.mark.parametrize("path", sorted(N8N_DIR.glob("*.json")), ids=lambda p: p.name)
def test_n8n_exports_contain_no_personal_or_account_data(path):
    text = path.read_text(encoding="utf-8")
    for pattern in PERSONAL_PATTERNS:
        assert not pattern.search(text), f"{path.name} matches {pattern.pattern}"
