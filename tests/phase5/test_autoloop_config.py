"""tests/phase5/test_autoloop_config.py — AutoLoopConfig + AutoLoopAlgoConfig surface tests.

Requirements covered (Plan 02):
- LOOP-04: all auto-loop parameters read from config/autoloop.toml at startup
- OBS-04: ConfigError raised on missing/invalid keys

Tests lock the config surface immediately so downstream plans can rely on these types.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src._config import AutoLoopConfig, load_toml_config
from src._errors import ConfigError

# Path to the shipped default config
_DEFAULT_TOML = Path(__file__).resolve().parent.parent.parent / "config" / "autoloop.toml"


def test_autoloop_config_defaults() -> None:
    """AutoLoopConfig() with no args uses all CONTEXT.md Decision 5 defaults."""
    cfg = AutoLoopConfig()

    assert cfg.enabled is True
    assert cfg.tau_leak == 0.3
    assert cfg.min_observations == 20
    assert cfg.max_patches_per_run == 5
    assert cfg.n_hands_validation == 5000
    assert cfg.bootstrap_resamples == 1000
    assert cfg.ci_alpha == 0.05
    assert cfg.algo.kind == "knn_rebalance"


def test_autoloop_config_is_frozen() -> None:
    """AutoLoopConfig is frozen — assigning to any field raises AttributeError."""
    cfg = AutoLoopConfig()
    with pytest.raises(AttributeError):
        cfg.enabled = False  # type: ignore[misc]


def test_load_toml_config_reads_shipped_defaults() -> None:
    """load_toml_config with the shipped autoloop.toml returns AutoLoopConfig with correct defaults."""
    c = load_toml_config(_DEFAULT_TOML, AutoLoopConfig)

    assert isinstance(c, AutoLoopConfig)
    assert c.tau_leak == 0.3
    assert c.algo.kind == "knn_rebalance"
    assert c.enabled is True
    assert c.min_observations == 20
    assert c.bootstrap_resamples == 1000


def test_load_toml_config_raises_config_error_on_bad_type(tmp_path: Path) -> None:
    """load_toml_config raises ConfigError when a field fails msgspec validation (OBS-04).

    AutoLoopConfig fields are decoded at the top-level of the TOML document
    (not nested under [autoloop]), so the bad value must be a top-level key.
    """
    bad = tmp_path / "bad.toml"
    # tau_leak must be a float; providing a string causes msgspec.ValidationError
    # which load_toml_config wraps in ConfigError
    bad.write_text('tau_leak = "not-a-float"\n')

    with pytest.raises(ConfigError):
        load_toml_config(bad, AutoLoopConfig)


def test_load_toml_config_raises_config_error_on_missing_file() -> None:
    """load_toml_config raises ConfigError with 'config file not found' on missing file."""
    missing = Path("nonexistent_autoloop_config.toml")
    with pytest.raises(ConfigError, match="config file not found"):
        load_toml_config(missing, AutoLoopConfig)
