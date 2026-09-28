"""Phase 6 Wave 0 — StudyConfig acceptance tests.

Pure-Python: no DB, no FastAPI, no network. Safe to run in any environment
including pre-commit fast-feedback loops.
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import pytest

from src._config import StudyConfig, load_study_config
from src._errors import ConfigError


def test_study_config_defaults_match_research() -> None:
    """Defaults match 06-CONTEXT.md addendum #1 D-02 / D-05 / OQ-2 + 06-RESEARCH.md."""
    cfg = StudyConfig()
    assert cfg.fastapi_host == "127.0.0.1", "Must default to loopback, never a wildcard address"
    assert cfg.fastapi_port == 8765
    assert cfg.solver_max_concurrent == 1, "D-05 + OQ-2: sequential Stage B globally"
    assert cfg.solver_timeout_s == 600.0
    assert 0.0 < cfg.eb_default_sigma2 < 1.0, "A5: within-cluster variance must be a valid prior"
    assert cfg.cache_ttl_s == 30
    assert cfg.ascii_sparkline_width == 10
    assert cfg.job_ttl_s == 3600


def test_study_config_host_default_is_never_wildcard_bind() -> None:
    """T-06-02: fastapi_host default MUST NOT be a wildcard bind."""
    cfg = StudyConfig()
    assert cfg.fastapi_host not in ("0.0.0.0", "::", "*"), (
        f"fastapi_host default {cfg.fastapi_host!r} is a wildcard bind — "
        "local-only invariant (D-02) violated"
    )


def test_study_config_frozen() -> None:
    """msgspec.Struct(frozen=True) prevents mutation after construction."""
    cfg = StudyConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.fastapi_port = 9999  # type: ignore[misc]


def test_load_study_config_from_repo_default() -> None:
    """config/study.toml in the repo root loads cleanly with documented defaults."""
    cfg = load_study_config(Path("config/study.toml"))
    assert cfg.fastapi_host == "127.0.0.1"
    assert cfg.fastapi_port == 8765
    assert cfg.solver_max_concurrent == 1


def test_load_study_config_uses_default_path_when_none() -> None:
    """load_study_config(None) resolves config/study.toml relative to CWD."""
    cfg = load_study_config(None)
    assert cfg.fastapi_host == "127.0.0.1"


def test_load_study_config_missing_file_raises_config_error() -> None:
    """Missing file → ConfigError (NOT FileNotFoundError or raw msgspec error)."""
    with pytest.raises(ConfigError, match="not found"):
        load_study_config(Path("/nonexistent/study.toml"))


def test_load_study_config_unknown_key_raises(tmp_path: Path) -> None:
    """Unknown key in TOML → ConfigError (msgspec strict decode)."""
    bad = tmp_path / "study.toml"
    bad.write_text('fastapi_host = "127.0.0.1"\nfastapi_port = 8765\nunknown_key = "value"\n')
    with pytest.raises(ConfigError):
        load_study_config(bad)


def test_load_study_config_wrong_type_raises(tmp_path: Path) -> None:
    """Wrong type in TOML → ConfigError."""
    bad = tmp_path / "study.toml"
    # fastapi_port should be int, not string
    bad.write_text('fastapi_host = "127.0.0.1"\nfastapi_port = "not_an_int"\n')
    with pytest.raises(ConfigError):
        load_study_config(bad)


def test_study_config_constructible_from_explicit_kwargs() -> None:
    """Direct construction works for tests that need custom values."""
    cfg = StudyConfig(
        fastapi_host="127.0.0.1",
        fastapi_port=9000,
        solver_max_concurrent=2,
        solver_timeout_s=30.0,
        eb_default_sigma2=0.1,
        cache_ttl_s=5,
        ascii_sparkline_width=20,
        job_ttl_s=60,
    )
    assert cfg.fastapi_port == 9000
    assert cfg.solver_max_concurrent == 2


def test_study_config_is_msgspec_struct() -> None:
    """Sanity: StudyConfig participates in the existing msgspec ecosystem."""
    assert issubclass(StudyConfig, msgspec.Struct)
