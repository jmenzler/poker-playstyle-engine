"""AutonomousConfig surface, back-compat, and OBS-04 validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src._config import AutoLoopConfig, AutonomousConfig, load_toml_config
from src._errors import ConfigError

_SHIPPED_TOML = Path(__file__).resolve().parent.parent.parent / "config" / "autoloop.toml"


def test_autonomous_config_defaults() -> None:
    """AutonomousConfig() with no args uses the D-14 defaults."""
    c = AutonomousConfig()

    assert c.max_retries == 3, "max_retries default"
    assert c.backoff_base_s == 2.0, "backoff_base_s default"
    assert c.max_consecutive_failures == 5, "max_consecutive_failures default"
    assert c.heartbeat_every_cycles == 1, "heartbeat_every_cycles default"
    assert c.checkpoint_path == "data/autoloop_checkpoints", "checkpoint_path default"
    assert c.sentinel_path == "data/autoloop_stop", "sentinel_path default"
    assert c.sim_n_hands == 5000, "sim_n_hands default"


def test_autoloop_config_has_autonomous() -> None:
    """AutoLoopConfig().autonomous is an AutonomousConfig with defaults (default_factory wired)."""
    cfg = AutoLoopConfig()

    assert isinstance(cfg.autonomous, AutonomousConfig), "nested autonomous field"
    assert cfg.autonomous.max_retries == 3
    assert cfg.autonomous.sim_n_hands == 5000


def test_load_shipped_toml_parses_autonomous() -> None:
    """load_toml_config(shipped autoloop.toml) reads the [autonomous] section."""
    c = load_toml_config(_SHIPPED_TOML, AutoLoopConfig)

    assert c.autonomous.max_retries == 3
    assert c.autonomous.checkpoint_path == "data/autoloop_checkpoints"
    assert c.autonomous.sim_n_hands == 5000


def test_toml_without_autonomous_section_back_compat(tmp_path: Path) -> None:
    """A TOML without an [autonomous] section still loads (defaults applied) — back-compat."""
    toml = tmp_path / "autoloop.toml"
    toml.write_text(
        "enabled = true\n"
        "tau_leak = 0.3\n"
        "min_observations = 20\n"
        "max_patches_per_run = 5\n"
        "n_hands_validation = 5000\n"
        "bootstrap_resamples = 1000\n"
        "ci_alpha = 0.05\n"
        '\n[algo]\nkind = "knn_rebalance"\n'
    )

    c = load_toml_config(toml, AutoLoopConfig)

    assert isinstance(c.autonomous, AutonomousConfig)
    assert c.autonomous.max_retries == 3, "additive default applied with no [autonomous] section"


def test_bad_autonomous_type_raises_config_error(tmp_path: Path) -> None:
    """A wrong-typed [autonomous] key raises ConfigError before any I/O (OBS-04)."""
    toml = tmp_path / "autoloop.toml"
    toml.write_text('enabled = true\n[autonomous]\nmax_retries = "three"\n')

    with pytest.raises(ConfigError):
        load_toml_config(toml, AutoLoopConfig)


def test_autonomous_config_is_frozen() -> None:
    """AutonomousConfig is frozen — assignment raises AttributeError."""
    c = AutonomousConfig()
    with pytest.raises(AttributeError):
        c.max_retries = 9  # type: ignore[misc]
