"""The autouse fixtures in conftest.py keep tests away from real state and credentials."""

import os
from pathlib import Path

import app.cartoon_factory_orchestrator as orchestrator
import app.experiment_system as experiment_system

REAL_ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def test_module_level_artifact_paths_are_redirected_into_tmp_path(tmp_path):
    assert str(orchestrator.ARTIFACTS).startswith(str(tmp_path))
    assert not str(orchestrator.ARTIFACTS).startswith(str(REAL_ARTIFACTS))
    assert str(orchestrator.OUTBOX).startswith(str(tmp_path))


def test_default_arguments_bound_at_import_time_are_redirected_too(tmp_path):
    default_path = experiment_system._read.__defaults__[0]
    assert str(default_path).startswith(str(tmp_path))
    assert not str(default_path).startswith(str(REAL_ARTIFACTS))


def test_operator_credentials_and_switches_are_not_visible_to_tests():
    leaked = [
        key
        for key in os.environ
        if key.startswith(("NOVA_", "RUNWAY", "YOUTUBE_", "TIKTOK_", "INSTAGRAM_"))
    ]
    assert leaked == []
