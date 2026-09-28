"""OBS-04: msgspec-based startup config validation.

Two flavors:
- load_*_env(): for Phase-1 use, reads env vars into a typed Struct
- load_toml_config(): general TOML loader (used in Phase 5 for autoloop.toml)

All validation failures raise ConfigError (NOT KeyError, NOT msgspec.ValidationError raw).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

import msgspec

from src._errors import ConfigError


class TsdbEnv(msgspec.Struct, frozen=True, kw_only=True):
    host: str
    port: int
    dbname: str
    user: str
    password: str


class MilvusEnv(msgspec.Struct, frozen=True, kw_only=True):
    host: str
    port: int
    token: str | None = None


class AutoLoopAlgoConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Algorithm-specific sub-config for AutoLoopConfig.

    kind: locked to "knn_rebalance" in Phase 5; future "solver_distillation"
        becomes valid in Phase 5.x once OQ-1 resolves in SOLVER.md.
    """

    kind: str = "knn_rebalance"


class AutonomousConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Durable autonomous-run tuning. Surfaces as the [autonomous] TOML section.

    Per-invocation knobs (run_id, base seed, max_cycles, resume) are CLI flags, not here.
    """

    max_retries: int = 3
    backoff_base_s: float = 2.0
    max_consecutive_failures: int = 5
    heartbeat_every_cycles: int = 1
    checkpoint_path: str = "data/autoloop_checkpoints"
    sentinel_path: str = "data/autoloop_stop"
    sim_n_hands: int = 5000


class SolverQueueConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Batch per-spot solver queue driver config. Surfaces as [solver_queue] in autoloop.toml.

    Distinct from study.toml solver_max_concurrent (interactive path, always 1).
    """

    n_workers: int = 11
    target_exploitability_pct: float = 0.5
    max_iterations: int = 500
    timeout_s: float = 120.0
    checkpoint_every: int = 100
    eval_loo_every: int = 500
    eval_suite_every: int = 5000
    disk_floor_gb: float = 20.0
    retention_days: int = 30
    heartbeat_every: int = 50
    checkpoint_path: str = "data/solver_queue_checkpoints"
    bet_sizes: tuple[str, ...] = ("33%", "66%", "e", "a")
    # Total solver memory pool (MB) under the container mem_limit; per-spot gate budget
    # is this // n_workers (workers solve concurrently). 0 disables the gate.
    memory_budget_mb: int = 9000


class AutoLoopConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Auto-loop runtime configuration loaded from config/autoloop.toml.

    All defaults match 05-CONTEXT.md Decision 5. Validation happens at
    process startup via load_toml_config(path, AutoLoopConfig) — ConfigError
    raised on missing/invalid keys (OBS-04).
    """

    enabled: bool = True
    tau_leak: float = 0.3
    min_observations: int = 20
    max_patches_per_run: int = 5
    n_hands_validation: int = 5000
    bootstrap_resamples: int = 1000
    ci_alpha: float = 0.05
    algo: AutoLoopAlgoConfig = msgspec.field(default_factory=AutoLoopAlgoConfig)
    autonomous: AutonomousConfig = msgspec.field(default_factory=AutonomousConfig)
    solver_queue: SolverQueueConfig = msgspec.field(default_factory=SolverQueueConfig)


class StudyConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """Local study-tool settings; remote exposure needs an external access boundary."""

    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = 8765
    # D-05 + OQ-2: sequential globally — Stage B solver runs one cluster at a time.
    solver_max_concurrent: int = 1
    solver_timeout_s: float = 600.0
    # A5: within-cluster variance default; per-bucket override possible at query time.
    eb_default_sigma2: float = 0.05
    cache_ttl_s: int = 30
    ascii_sparkline_width: int = 10
    # JobRegistry sweeper: prune done/failed jobs older than this from the in-memory
    # registry (Stage B verification jobs, Eval match jobs).
    job_ttl_s: int = 3600


def load_study_config(path: Path | None = None) -> "StudyConfig":
    """Load StudyConfig from a TOML file (defaults to ./config/study.toml).

    Thin wrapper over load_toml_config for caller convenience — equivalent to
    load_toml_config(path, StudyConfig). Raises ConfigError on missing file
    or schema mismatch (OBS-04).
    """
    return load_toml_config(path or Path("config/study.toml"), StudyConfig)


class EvalConfig(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """Phase 9 eval-suite configuration loaded from config/eval.toml."""

    tvd_floor: float = 0.15
    regression_delta: float = 0.05
    match_hands_hu: int = 10000
    match_hands_6max: int = 20000
    bootstrap_resamples: int = 1000
    ci_alpha: float = 0.05


def load_eval_config(path: Path | None = None) -> "EvalConfig":
    """Load EvalConfig from config/eval.toml.

    Raises ConfigError on missing file or schema mismatch (OBS-04).
    """
    return load_toml_config(path or Path("config/eval.toml"), EvalConfig)


T = TypeVar("T", bound=msgspec.Struct)


def _require(env: Mapping[str, str], key: str) -> str:
    try:
        return env[key]
    except KeyError as e:
        raise ConfigError(f"missing required env var: {key}") from e


def _parse_port(env: Mapping[str, str], key: str) -> int:
    raw = _require(env, key)
    try:
        return int(raw)
    except ValueError as e:
        raise ConfigError(f"{key} must be an integer, got: {raw!r}") from e


def load_tsdb_env(env: Mapping[str, str] | None = None) -> TsdbEnv:
    """Read TSDB_* env vars; raise ConfigError on missing or unparseable port."""
    import os

    env = env if env is not None else os.environ
    return TsdbEnv(
        host=_require(env, "TSDB_HOST"),
        port=_parse_port(env, "TSDB_PORT"),
        dbname=_require(env, "TSDB_DB"),
        user=_require(env, "TSDB_USER"),
        password=_require(env, "TSDB_PASSWORD"),
    )


def load_milvus_env(env: Mapping[str, str] | None = None) -> MilvusEnv:
    """Read MILVUS_* env vars; raise ConfigError on missing or unparseable port."""
    import os

    env = env if env is not None else os.environ
    return MilvusEnv(
        host=_require(env, "MILVUS_HOST"),
        port=_parse_port(env, "MILVUS_PORT"),
        token=env.get("MILVUS_TOKEN"),
    )


def load_toml_config(path: Path, schema: type[T]) -> T:
    """Load and validate a TOML file against a msgspec Struct schema.

    Raises ConfigError if the file is missing, unreadable, or fails schema validation.
    """
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise ConfigError(f"config file unreadable: {path}: {e}") from e
    try:
        return msgspec.toml.decode(raw, type=schema)
    except msgspec.ValidationError as e:
        raise ConfigError(f"invalid config at {path}: {e}") from e
    except msgspec.DecodeError as e:
        raise ConfigError(f"malformed TOML at {path}: {e}") from e


def build_tsdb_dsn(env: TsdbEnv) -> str:
    """Construct a psycopg KV-form DSN from TsdbEnv.

    Caller is responsible for redacting before logging — pass through
    src._redact.redact_dsn if the DSN appears in any log line.
    """
    return f"host={env.host} port={env.port} dbname={env.dbname} user={env.user} password={env.password}"
