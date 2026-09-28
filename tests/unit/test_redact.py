"""Pitfall 4: DSN/URI redaction edge cases.

These tests run as unit-level (no DB, no Milvus, no network). They are the only
Phase-1 defense against raw passwords leaking into logs and exception messages.
"""


# --- URL form -------------------------------------------------------------------


def test_url_form_simple_password():
    from src._redact import redact_dsn

    out = redact_dsn("postgresql://user:hunter2@host:5432/db")
    assert "hunter2" not in out
    assert "***" in out
    assert "user" in out
    assert "host:5432" in out
    assert "/db" in out


def test_url_form_password_with_at_sign_urlencoded():
    """Password with @ urlencoded (%40) must not be confused with the host @-delimiter."""
    from src._redact import redact_dsn

    # Password is "p@ss" -> "p%40ss" in URL form
    dsn = "postgresql://user:p%40ss@host:5432/db"
    out = redact_dsn(dsn)
    assert "p%40ss" not in out
    assert "p@ss" not in out
    assert "***" in out
    assert "host:5432" in out


def test_url_form_postgres_scheme_alias():
    """Some psycopg builds accept `postgres://` (alias of `postgresql://`)."""
    from src._redact import redact_dsn

    out = redact_dsn("postgres://user:hunter2@host/db")
    assert "hunter2" not in out
    assert "***" in out


# --- KV form --------------------------------------------------------------------


def test_kv_form_simple_password():
    from src._redact import redact_dsn

    out = redact_dsn("host=localhost password=hunter2 dbname=poker_engine")
    assert "hunter2" not in out
    assert "password=***" in out
    assert "host=localhost" in out
    assert "dbname=poker_engine" in out


def test_kv_form_quoted_password_with_colon():
    """KV password may be single-quoted; quotes preserved, contents replaced."""
    from src._redact import redact_dsn

    out = redact_dsn("host=h password='se:cr:et' dbname=p")
    assert "se:cr:et" not in out
    assert "password='***'" in out


def test_kv_form_password_with_percent():
    from src._redact import redact_dsn

    out = redact_dsn("host=h password=p%65rcent dbname=p")
    assert "p%65rcent" not in out
    assert "password=***" in out


def test_kv_form_socket_auth_no_password():
    """No password field (peer/socket auth) — returned unchanged, no exception."""
    from src._redact import redact_dsn

    dsn = "host=/var/run/postgresql user=poker dbname=poker_engine"
    out = redact_dsn(dsn)
    assert out == dsn  # unchanged
    # And idempotent
    assert redact_dsn(out) == dsn


# --- Milvus URI -----------------------------------------------------------------


def test_milvus_uri_with_user_token():
    from src._redact import redact_milvus_uri

    out = redact_milvus_uri("http://user:s3cret@10.0.0.1:19530")
    assert "s3cret" not in out
    assert "***" in out
    assert "10.0.0.1:19530" in out


def test_milvus_uri_without_user():
    from src._redact import redact_milvus_uri

    uri = "http://10.0.0.1:19530"
    assert redact_milvus_uri(uri) == uri


def test_milvus_uri_https():
    from src._redact import redact_milvus_uri

    out = redact_milvus_uri("https://user:s3cret@host:19530")
    assert "s3cret" not in out
    assert "***" in out
    assert out.startswith("https://")


# --- Edge cases -----------------------------------------------------------------


def test_empty_string_returns_empty():
    """redact() must be safe to call on empty input — never raise."""
    from src._redact import redact_dsn, redact_milvus_uri

    assert redact_dsn("") == ""
    assert redact_milvus_uri("") == ""


def test_idempotent():
    """Applying redact twice produces same result as once."""
    from src._redact import redact_dsn

    once = redact_dsn("postgresql://u:p@h/db")
    twice = redact_dsn(once)
    assert once == twice


def test_milvus_idempotent():
    from src._redact import redact_milvus_uri

    once = redact_milvus_uri("http://u:t@h:1")
    twice = redact_milvus_uri(once)
    assert once == twice
