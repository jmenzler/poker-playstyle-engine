"""src/validation — Sim A/B validation for the auto-loop.

Exports:
    StrategyOverride: In-memory adapter that intercepts decide_with_encoding
        for one cluster_key; delegates for all others. Used by validate() to
        run the candidate engine without touching Milvus or TimescaleDB.
    DryRunObservationWriter: In-memory writer that satisfies the
        ObservationWriter contract without any DB I/O. Used for both
        baseline and candidate sim runs in validate().
    bootstrap_ci: Non-parametric bootstrap CI on the mean of per-hand deltas.
    validate: Paired-seed A/B orchestrator that returns a ValidationResult
        (VALN-01, VALN-02, VALN-03).
"""

from src.validation.override import StrategyOverride
from src.validation.sim_ab import DryRunObservationWriter, bootstrap_ci, validate

__all__ = [
    "DryRunObservationWriter",
    "StrategyOverride",
    "bootstrap_ci",
    "validate",
]
