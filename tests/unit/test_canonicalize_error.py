"""Phase 7 / ERR-02 closure — CanonicalizeError raised on unsupported GameState input.

Rewritten from the pre-pivot stub (was importing retired src.canonicalizer.canonicalizer
module). Now imports from current src._errors location.

Cf. .planning/v1.0-MILESTONE-AUDIT.md ERR-02 stale-test row.
"""

import pytest

from src._errors import CanonicalizeError


def test_canonicalize_error_is_subclass_of_value_error():
    assert issubclass(CanonicalizeError, ValueError)


def test_canonicalize_error_raised_for_unsupported_street():
    from tools.build_embedding import encode_spot_to_cluster_key

    with pytest.raises(CanonicalizeError, match="unsupported street"):
        encode_spot_to_cluster_key(
            {
                "street": "fifth_street",
                "hero_hole": ["As", "Ks"],
                "board": [],
            }
        )
