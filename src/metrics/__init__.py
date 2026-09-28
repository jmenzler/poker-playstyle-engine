"""src/metrics — session-level metrics computation and persistence (Phase 4 + 5).

Public exports:
    ev_loss            — KL divergence between observed and expected action distributions
    flush_session_metrics — write session metrics rows to the metrics hypertable (METR-01..03)
"""

from src.metrics.ev_loss import ev_loss
from src.metrics.sink import flush_session_metrics

__all__ = ["ev_loss", "flush_session_metrics"]
