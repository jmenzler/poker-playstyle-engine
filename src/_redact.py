"""DSN / URI redaction utilities (Pattern 2 + Pitfall 4).

These functions MUST be safe to call on any string input and MUST never raise.
Used by every fail-loud connect wrapper before constructing the exception message.
"""

import re
from typing import Final

_REDACTED: Final = "***"

# URL form: scheme://user:password@host[:port][/path]
# Match password as everything between the first ':' after the scheme:// and the LAST '@'
# before the rest of the URI. Using a greedy match on the password group with a possessive
# boundary: the host separator '@' is the LAST '@' in the authority segment.
# We handle passwords containing '@' when URL-encoded as %40 because they don't contain
# literal '@'. Passwords containing literal ':' are also captured since the password group
# matches any non-'@' character.
_URL_RE = re.compile(
    r"""
    ^                            # anchor
    (?P<scheme>postgres(?:ql)?|http|https)
    ://
    (?P<user>[^:/@]+)            # username (no ':', '/', '@')
    :                            # literal ':' separating user and pw
    (?P<pw>[^@]+)                # password — anything not '@' (handles URL-encoded chars)
    @                            # literal '@' separating credentials and host
    (?P<rest>.+)                 # host[:port][/path]
    $
    """,
    re.VERBOSE,
)

# KV form: psycopg key-value DSN. password may be:
#   - bareword:      password=hunter2
#   - single-quoted: password='hun ter2'
#   - double-quoted: password="hun ter2"
# Capture the entire password value and replace it.
_KV_RE = re.compile(
    r"""
    (?P<prefix>password\s*=\s*)              # `password=` with optional whitespace
    (?P<value>
        '(?:[^'\\]|\\.)*'                    # single-quoted (handles escaped \')
      | "(?:[^"\\]|\\.)*"                    # double-quoted (handles escaped \")
      | \S+                                  # bareword (no spaces)
    )
    """,
    re.VERBOSE,
)


def redact_dsn(dsn: str) -> str:
    """Return DSN with password masked. Safe on any input; never raises.

    Handles:
    - URL form (postgresql://user:pw@host:port/db)
    - KV form (host=... password=pw dbname=...)
    - Socket auth (no password field) — returned unchanged
    - Empty string — returned as ""
    """
    if not dsn:
        return dsn

    # Try URL form first
    m = _URL_RE.match(dsn)
    if m:
        return f"{m.group('scheme')}://{m.group('user')}:{_REDACTED}@{m.group('rest')}"

    # KV form: substitute every password=... occurrence
    def _kv_sub(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        value = match.group("value")
        # Preserve quote style if quoted
        if value.startswith("'") and value.endswith("'"):
            return f"{prefix}'{_REDACTED}'"
        if value.startswith('"') and value.endswith('"'):
            return f'{prefix}"{_REDACTED}"'
        return f"{prefix}{_REDACTED}"

    out, _ = _KV_RE.subn(_kv_sub, dsn)
    # If neither form matched, return unchanged (socket auth or unknown format)
    return out


def redact_milvus_uri(uri: str) -> str:
    """Return Milvus URI with token masked. Safe on any input; never raises."""
    if not uri:
        return uri
    m = _URL_RE.match(uri)
    if m and m.group("scheme") in ("http", "https"):
        return f"{m.group('scheme')}://{m.group('user')}:{_REDACTED}@{m.group('rest')}"
    return uri  # No user:token component, or unrecognized form — return unchanged
