"""Phase 1 exception hierarchy.

All exceptions are leaf types — no further subclassing in Phase 1.
Future phases may extend (e.g., NoStrategyError in Phase 3, PatchForbiddenError in Phase 5).
"""


class DBConnectError(RuntimeError):
    """ERR-01: database connection failure with redacted DSN/URI in the message.

    Raised by:
    - src/db/timescale.connect on psycopg connect failure
    - src/db/milvus.connect on pymilvus connect failure
    - runner/migrate.run when underlying connect fails

    Message format: f"<store> connect failed: <redacted DSN>"
    Original exception chained via `raise ... from e`.
    """


class ConfigError(ValueError):
    """OBS-04: invalid or missing configuration at process startup.

    Raised by:
    - src/_config.load_* when a required env var is missing
    - src/_config.load_* when a TOML/JSON config fails msgspec validation
    - src/canonicalizer/board_taxonomy on malformed taxonomy.json
    - src/canonicalizer/feature_vector on malformed features.json
    """


class CanonicalizeError(ValueError):
    """ERR-02: a GameState cannot be assigned to a cluster.

    Raised by:
    - src/canonicalizer.Canonicalizer.encode on unsupported street, missing required fields,
      invalid card encoding, or any pre-condition violation.

    The Canonicalizer never returns a partial result; encode either succeeds entirely or
    raises CanonicalizeError with a descriptive message including the failing field.
    """


class NoStrategyError(LookupError):
    """ERR-03: Decision engine found zero Milvus results for a query.

    Raised by:
    - src/decision_engine/engine.KNNDecisionEngine.decide() when Milvus returns
      zero results for the post-encode, post-normalize kNN query.

    The caller is responsible for fallback (e.g., RandomAgent action selection).
    The message MUST include the collection name and the hard_filter dict so the
    caller can decide whether to retry with a relaxed filter or skip the spot.
    """


class PatchForbiddenError(RuntimeError):
    """PTCH-04: raised when apply() targets a strategy_nodes row with source='seed'.

    Message MUST include the cluster_key so the caller can log it.
    Raised by:
    - src/patch_engine.PatchEngine.apply() when the current active node for
      the target cluster_key has source='seed'.
    """


class PatchConflictError(RuntimeError):
    """ERR-04: raised when the target cluster_key was patched between
    prepare-time and apply-time.

    The prev_node_id prepared at candidate-selection time no longer matches
    the current active node_id for the cluster. Caller (auto-loop driver)
    catches and skips the candidate; NO retry in Phase 5.
    Message MUST include cluster_key and both node_ids (expected vs actual).
    """


class SolverParseError(RuntimeError):
    """Phase 6 — OQ-1 resolution: postflop-cli stdout could not be parsed
    or actions could not be mapped to the canonical vocab.

    Raised by:
    - src/solver/postflop_cli.solve() when:
        * subprocess stdout is not valid JSON (wraps json.JSONDecodeError)
        * a parsed action label cannot be mapped to the canonical action set
          (e.g., bet sizes that bucket-snap > 20% off any canonical bin)
        * the solver returns an empty / malformed action_dist payload

    Caller (Stage B verification endpoint in src/api/solver.py) catches and
    surfaces a structured error to the UI; does NOT cache a strategy_nodes
    row with source='solver_verify' for failed parses.
    """


class ValidationError(ValueError):
    """Phase 6 — edit-node + FastAPI POST /api/edit-node: action_dist
    frequencies failed validation.

    Raised by:
    - src/cli/edit_node.py / src/api/edit.py when:
        * sum(frequencies) != 1.0 ± 0.001
        * any frequency is negative
        * an action key is not in the canonical action vocab
        * the JSON payload is structurally invalid (missing required keys)

    Message MUST be operator-readable (rendered verbatim by the CLI / drawer).
    Distinct from ConfigError (process-start config) and PatchConflictError
    (concurrent-write conflict).
    """
