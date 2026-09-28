"""Re-export shim: EquityLookup from tools/.

Single source of truth lives in tools/equity_lookup.py.
This module is the src/ import boundary for Phase 3+ consumers.
"""

from tools.equity_lookup import EquityLookup

__all__ = ["EquityLookup"]
