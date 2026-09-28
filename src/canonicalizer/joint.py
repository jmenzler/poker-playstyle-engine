"""Re-export shim: joint suit-iso canonicalization from tools/.

Single source of truth lives in tools/joint_canonicalize.py.
This module is the src/ import boundary for Phase 3+ consumers.
"""

from tools.joint_canonicalize import joint_canonical_key, joint_canonicalize

__all__ = ["joint_canonical_key", "joint_canonicalize"]
