"""Local-only defaults for the source snapshot."""

import ast
from pathlib import Path
from urllib.parse import urlparse

from src._config import StudyConfig
from src.api.main import _CORS_ORIGINS


def test_study_listener_defaults_to_loopback():
    assert StudyConfig().fastapi_host == "127.0.0.1"


def test_cors_contains_only_local_origins():
    assert _CORS_ORIGINS
    assert all(urlparse(origin).hostname in {"localhost", "127.0.0.1", "tauri.localhost"} for origin in _CORS_ORIGINS)


def test_loopback_vite_origin_is_allowed():
    assert "http://127.0.0.1:1420" in _CORS_ORIGINS


def test_live_checks_never_shell_into_a_private_deployment():
    tree = ast.parse(Path("tests/phase6/test_e2e_smoke.py").read_text())
    assert not any(isinstance(node, ast.Constant) and node.value == "ssh" for node in ast.walk(tree))


def test_import_pipeline_has_no_personal_database_path_default():
    tree = ast.parse(Path("tools/phase2_pipeline.py").read_text())
    defaults = [
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--hm3-sqlite"
        for keyword in node.keywords
        if keyword.arg == "default" and isinstance(keyword.value, ast.Constant)
    ]
    assert len(defaults) == 1
    assert not Path(defaults[0]).is_absolute()
