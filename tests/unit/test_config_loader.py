"""OBS-04: startup config validation; missing/invalid keys raise ConfigError."""

from pathlib import Path

import pytest

from src._errors import ConfigError


def _full_tsdb_env() -> dict[str, str]:
    return {
        "TSDB_HOST": "127.0.0.1",
        "TSDB_PORT": "55432",
        "TSDB_DB": "poker_engine",
        "TSDB_USER": "poker",
        "TSDB_PASSWORD": "hunter2",
    }


def _full_milvus_env() -> dict[str, str]:
    return {
        "MILVUS_HOST": "127.0.0.1",
        "MILVUS_PORT": "51530",
        "MILVUS_TOKEN": "root:Milvus",
    }


def test_load_tsdb_env_complete():
    from src._config import load_tsdb_env

    env = load_tsdb_env(_full_tsdb_env())
    assert env.host == "127.0.0.1"
    assert env.port == 55432
    assert env.dbname == "poker_engine"
    assert env.user == "poker"
    assert env.password == "hunter2"


def test_load_tsdb_env_missing_password_raises():
    from src._config import load_tsdb_env

    bad = _full_tsdb_env()
    del bad["TSDB_PASSWORD"]
    with pytest.raises(ConfigError, match="TSDB_PASSWORD"):
        load_tsdb_env(bad)


def test_load_tsdb_env_non_integer_port_raises():
    from src._config import load_tsdb_env

    bad = _full_tsdb_env()
    bad["TSDB_PORT"] = "not-a-number"
    with pytest.raises(ConfigError, match="TSDB_PORT"):
        load_tsdb_env(bad)


def test_load_milvus_env_token_optional():
    from src._config import load_milvus_env

    env = _full_milvus_env()
    del env["MILVUS_TOKEN"]
    result = load_milvus_env(env)
    assert result.token is None


def test_build_tsdb_dsn_kv_form():
    from src._config import build_tsdb_dsn, load_tsdb_env

    env = load_tsdb_env(_full_tsdb_env())
    dsn = build_tsdb_dsn(env)
    # KV form
    assert "host=127.0.0.1" in dsn
    assert "port=55432" in dsn
    assert "dbname=poker_engine" in dsn
    assert "user=poker" in dsn
    assert "password=hunter2" in dsn


def test_load_toml_config_valid(tmp_path: Path):
    import msgspec

    from src._config import load_toml_config

    class Sample(msgspec.Struct, frozen=True, kw_only=True):
        name: str
        count: int

    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text('name = "alpha"\ncount = 7\n')
    cfg = load_toml_config(cfg_file, Sample)
    assert cfg.name == "alpha"
    assert cfg.count == 7


def test_load_toml_config_malformed_raises(tmp_path: Path):
    import msgspec

    from src._config import load_toml_config

    class Sample(msgspec.Struct, frozen=True, kw_only=True):
        name: str
        count: int

    cfg_file = tmp_path / "bad.toml"
    # missing "count" — msgspec.ValidationError -> ConfigError
    cfg_file.write_text('name = "alpha"\n')
    with pytest.raises(ConfigError):
        load_toml_config(cfg_file, Sample)


def test_load_toml_config_missing_file_raises(tmp_path: Path):
    import msgspec

    from src._config import load_toml_config

    class Sample(msgspec.Struct, frozen=True, kw_only=True):
        name: str

    with pytest.raises(ConfigError, match=r"not found|missing"):
        load_toml_config(tmp_path / "does_not_exist.toml", Sample)
