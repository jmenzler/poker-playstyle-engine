"""src/sim — RLCard adapter + observation writer + harness (Phase 3).

Public exports:
    SimAdapter         — wraps RLCard no-limit-holdem env (ENGN-06)
    PROJECT_TO_RLCARD  — 15-action vocab to RLCard int mapping
    ObservationWriter  — buffered TimescaleDB writer (SIM-03)
    run_record_session — deterministic RECORD-mode loop (SIM-01, SIM-02, SIM-03)
"""

from src.sim.adapter import PROJECT_TO_RLCARD, SimAdapter
from src.sim.harness import run_record_session
from src.sim.observation_writer import ObservationWriter

__all__ = ["PROJECT_TO_RLCARD", "ObservationWriter", "SimAdapter", "run_record_session"]
