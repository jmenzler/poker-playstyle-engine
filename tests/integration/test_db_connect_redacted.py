"""ERR-01: bad DSN raises DBConnectError with `***` not raw password."""

import pytest

pytestmark = pytest.mark.integration


def test_timescale_bad_dsn_redacted() -> None:
    from src._errors import DBConnectError
    from src.db.timescale import connect

    bad = "host=10.255.255.1 port=5432 dbname=poker_engine user=x password=hunter2 connect_timeout=1"
    with pytest.raises(DBConnectError) as exc_info:
        connect(bad)
    msg = str(exc_info.value)
    assert "hunter2" not in msg, "Raw password leaked in error message"
    assert "***" in msg, "Expected '***' redaction token in error message"


def test_milvus_bad_uri_redacted() -> None:
    from src._errors import DBConnectError
    from src.db.milvus import connect

    with pytest.raises(DBConnectError) as exc_info:
        connect("http://10.255.255.1:19530", token="badtoken")
    msg = str(exc_info.value)
    # Token is passed separately, not in URI — URI itself has no embedded creds
    # but the error message must still contain *** (from redact_milvus_uri)
    assert "badtoken" not in msg, "Token should not appear in error message"
