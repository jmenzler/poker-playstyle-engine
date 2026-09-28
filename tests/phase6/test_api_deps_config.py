"""Tests for src/api/deps.get_config — fail-loud on malformed study.toml.

A missing config falls back to defaults (dev/test convenience). A *present*
but malformed config must raise ConfigError, not silently revert to defaults.
"""

from __future__ import annotations

import pytest

import src.api.deps as deps
from src._config import StudyConfig
from src._errors import ConfigError


@pytest.fixture(autouse=True)
def reset_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps, "_cached_config", None)
    yield
    deps._cached_config = None


def test_missing_config_falls_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(deps, "_STUDY_CONFIG_PATH", tmp_path / "absent.toml")
    cfg = deps.get_config()
    assert isinstance(cfg, StudyConfig)
    assert cfg.solver_max_concurrent == StudyConfig().solver_max_concurrent


def test_malformed_config_raises_not_masked(tmp_path, monkeypatch):
    bad = tmp_path / "study.toml"
    bad.write_text('solver_max_concurrent = "not-an-int"\n')
    monkeypatch.setattr(deps, "_STUDY_CONFIG_PATH", bad)
    with pytest.raises(ConfigError):
        deps.get_config()
    # Failure must not be cached behind a defaults fallback.
    assert deps._cached_config is None


def test_valid_config_is_cached(tmp_path, monkeypatch):
    good = tmp_path / "study.toml"
    good.write_text("solver_max_concurrent = 3\n")
    monkeypatch.setattr(deps, "_STUDY_CONFIG_PATH", good)
    cfg = deps.get_config()
    assert cfg.solver_max_concurrent == 3
    assert deps.get_config() is cfg  # cached singleton
