"""Re-export shim: preflop feature extractor from tools/.

Single source of truth lives in tools/feature_extractors/preflop.py.
"""

from tools.feature_extractors.preflop import extract_preflop

__all__ = ["extract_preflop"]
