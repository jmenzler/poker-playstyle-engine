"""ERR-01: fail-loud Milvus connect wrapper.

All Milvus-touching code MUST go through this function. ``pymilvus.MilvusClient``
construction is lazy and does not always contact the server during ``__init__``,
so this wrapper calls ``has_collection`` on a sentinel name to force the
reachability check. The probe result is discarded; we only care whether the call
raises.

SPIKE A2 (.planning/phases/01-foundation/01-01-SPIKE.md) confirmed that
``MilvusClient(uri=<unreachable>)`` raises ``pymilvus.exceptions.MilvusException``.
We additionally widen the catch to ``Exception`` because pymilvus 3.x may surface
raw gRPC errors (``_InactiveRpcError``, ``RpcError``) that do not always inherit
from ``MilvusException`` on transport-level failures. This is a CONSCIOUS
widening per "fail loudly with redacted URI, never silently" — we re-wrap and
chain via ``from e`` so the original is preserved for debugging.
"""

import os
from typing import Any

from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

from src._errors import DBConnectError
from src._log import get_logger
from src._redact import redact_milvus_uri

log = get_logger("db.milvus")

# Sentinel collection name used to probe server reachability after MilvusClient
# construction. The actual check is whether the call itself raises.
_PROBE_COLLECTION = "__poker_engine_probe__"

# Default per-RPC deadline. Without it a flapping/recovering Milvus can leave a
# gRPC call blocked indefinitely in an uninterruptible syscall (un-killable, hangs
# the whole solver run). A bounded deadline turns a blip into a catchable error.
DEFAULT_RPC_TIMEOUT_S = 30.0

# Data-plane RPCs whose timeout defaults to None in pymilvus — the constructor
# timeout does NOT propagate to these, so the proxy injects the default deadline.
_TIMEOUT_INJECTED_METHODS = frozenset({"search", "query", "upsert", "insert", "get", "delete"})


class _TimeoutInjectingClient:
    """Transparent ``MilvusClient`` proxy that bounds every data-plane RPC.

    pymilvus per-RPC ``timeout`` defaults to ``None`` (the constructor timeout is
    NOT a per-call default), so search/query/upsert/insert/get/delete can block
    indefinitely on a flapping server. This proxy injects ``default_timeout`` into
    those calls unless the caller passes ``timeout=`` explicitly; every other
    attribute delegates straight through.
    """

    __slots__ = ("_client", "_default_timeout")

    def __init__(self, client: MilvusClient, default_timeout: float) -> None:
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_default_timeout", default_timeout)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        if name not in _TIMEOUT_INJECTED_METHODS:
            return attr

        def _bounded(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("timeout", self._default_timeout)
            return attr(*args, **kwargs)

        return _bounded


def connect(uri: str, *, token: str | None = None, timeout: float = DEFAULT_RPC_TIMEOUT_S) -> MilvusClient:
    """Construct a ``MilvusClient`` and verify the server is reachable.

    Args:
        uri: e.g., ``"http://127.0.0.1:51530"`` or ``"http://user:token@host:port"``.
        token: optional explicit token (overrides URI-embedded credentials).
        timeout: default per-RPC deadline (seconds). The returned client is a thin
            proxy that injects this into search/query/upsert/insert/get/delete
            (pymilvus does NOT use the constructor timeout as a per-RPC default),
            unless the caller passes its own ``timeout=``.

    Returns:
        A timeout-injecting proxy over ``MilvusClient`` on success.

    Raises:
        DBConnectError: on any pymilvus / transport failure. Message contains the
            redacted URI (NOT the raw token). Original exception chained via ``from e``.
    """
    try:
        client = (
            MilvusClient(uri=uri, token=token, timeout=timeout)
            if token
            else MilvusClient(uri=uri, timeout=timeout)
        )
        # Force a round-trip to verify reachability (MilvusClient construction is lazy).
        _ = client.has_collection(_PROBE_COLLECTION, timeout=timeout)
        return _TimeoutInjectingClient(client, timeout)
    except MilvusException as e:
        redacted = redact_milvus_uri(uri)
        log.error("db.connect.failed", uri=redacted, error=str(e), error_type=type(e).__name__)
        raise DBConnectError(f"Milvus connect failed ({type(e).__name__}): {redacted}") from e
    except Exception as e:  # pymilvus may surface gRPC errors outside MilvusException
        redacted = redact_milvus_uri(uri)
        log.error("db.connect.failed", uri=redacted, error=str(e), error_type=type(e).__name__)
        raise DBConnectError(f"Milvus connect failed ({type(e).__name__}): {redacted}") from e


def connect_from_env() -> MilvusClient:
    """Build a token-authed MilvusClient from env vars.

    Single source of truth for env-based Milvus auth — every caller (decision
    engine, metrics sink, sim runner) MUST use this so the MILVUS_TOKEN is
    applied uniformly. Reads MILVUS_HOST, MILVUS_PORT, MILVUS_TOKEN; wraps a
    bare token as ``root:<token>`` (auth disabled if MILVUS_TOKEN is unset).
    """
    host = os.environ["MILVUS_HOST"]
    port = os.environ.get("MILVUS_PORT", "51530")
    raw = os.environ.get("MILVUS_TOKEN")
    token = raw if (raw and ":" in raw) else (f"root:{raw}" if raw else None)
    return connect(f"http://{host}:{port}", token=token)
