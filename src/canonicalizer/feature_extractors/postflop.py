"""Re-export shim: postflop feature extractor from tools/.

Single source of truth lives in tools/feature_extractors/postflop.py.
"""

from tools.feature_extractors.postflop import extract_postflop

__all__ = ["extract_postflop"]
