"""Phase 9 / EvalConfig — load_eval_config() loads from config/eval.toml."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_load_eval_config_defaults():
    """load_eval_config() with no path reads config/eval.toml, tvd_floor==0.15."""
    from src._config import load_eval_config

    cfg = load_eval_config()
    assert cfg.tvd_floor == pytest.approx(0.15)


def test_eval_config_all_defaults():
    """EvalConfig has the six D-09-3/D-09-12 defaults."""
    from src._config import EvalConfig

    cfg = EvalConfig()
    assert cfg.tvd_floor == pytest.approx(0.15)
    assert cfg.regression_delta == pytest.approx(0.05)
    assert cfg.match_hands_hu == 10000
    assert cfg.match_hands_6max == 20000
    assert cfg.bootstrap_resamples == 1000
    assert cfg.ci_alpha == pytest.approx(0.05)


def test_unknown_key_raises_config_error(tmp_path: Path):
    """An unknown key in the TOML raises ConfigError (forbid_unknown_fields=True)."""
    from src._config import load_eval_config
    from src._errors import ConfigError

    bad_toml = tmp_path / "eval_bad.toml"
    bad_toml.write_text("tvd_floor = 0.15\nunknown_field = 99\n")

    with pytest.raises(ConfigError):
        load_eval_config(bad_toml)
