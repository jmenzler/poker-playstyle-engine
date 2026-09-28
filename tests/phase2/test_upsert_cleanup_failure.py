"""tools.upsert_milvus._cleanup_filter_test_residue — fail loud on delete failure.

A swallowed delete leaves filter-test-* rows live in the served index, polluting
kNN neighbors. Cleanup must raise so the upsert aborts.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tools.upsert_milvus import _cleanup_filter_test_residue


def test_cleanup_reraises_on_delete_failure():
    client = MagicMock()
    client.has_collection.return_value = True
    client.delete.side_effect = RuntimeError("milvus timeout")

    with pytest.raises(RuntimeError):
        _cleanup_filter_test_residue(client)


def test_cleanup_succeeds_when_delete_ok():
    client = MagicMock()
    client.has_collection.return_value = True
    client.delete.return_value = {"delete_count": 2}

    total = _cleanup_filter_test_residue(client)
    assert total == 4  # 2 collections x 2 deleted
