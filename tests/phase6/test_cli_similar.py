"""tests/phase6/test_cli_similar.py — Unit tests for `poker-engine similar` CLI shim (CLI-02)."""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch


def _args(cluster_key="ck-source", *, format="text", k=10):
    return argparse.Namespace(cluster_key=cluster_key, format=format, k=k)


def _canned() -> list[dict]:
    return [
        {"rank": 1, "cluster_key": "neighbor-1", "distance": 0.12, "source": "knn", "n_obs": 87},
        {"rank": 2, "cluster_key": "neighbor-2", "distance": 0.21, "source": "knn", "n_obs": 65},
    ]


def _extract_json_block(out: str):
    """Find the JSON payload in stdout (structlog log lines may surround it).

    Our shims emit JSON via json.dumps(..., indent=2) which always starts on
    column 0 of a line. structlog log lines are timestamp-prefixed (e.g.
    ``2026-05-19 ... [info]``) so do not start with ``{`` or ``[`` at column 0.
    Search line-by-line for the first line whose first character is ``{`` or
    ``[`` and parse from there.
    """
    decoder = json.JSONDecoder()
    lines = out.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line and line[0] in "{[":
            blob = "".join(lines[i:])
            obj, _end = decoder.raw_decode(blob)
            return obj
    raise ValueError(f"no JSON payload found in stdout: {out!r}")


def test_similar_json_format_includes_action_dist(capsys):
    """`similar --format json` passes include_action_dist=True to backend per D-13."""
    canned_with_dist = [
        {
            "rank": 1,
            "cluster_key": "neighbor-1",
            "distance": 0.12,
            "source": "knn",
            "n_obs": 87,
            "action_dist": {"check": 0.5, "bet_50": 0.5},
        }
    ]
    captured_kwargs = {}

    def _capture(*a, **kw):
        captured_kwargs.update(kw)
        return canned_with_dist

    with patch("src.study.similar.find_similar", side_effect=_capture):
        from src.cli.similar import run

        rc = run(_args(format="json"))

    assert rc == 0
    assert captured_kwargs.get("include_action_dist") is True

    payload = _extract_json_block(capsys.readouterr().out)
    assert payload[0]["cluster_key"] == "neighbor-1"
    assert "action_dist" in payload[0]


def test_similar_text_truncates_long_cluster_key(capsys):
    """Text format truncates cluster_key column at ≤62 chars with ellipsis."""
    long_key = "x" * 200
    canned = [{"rank": 1, "cluster_key": long_key, "distance": 0.12, "source": "knn", "n_obs": 87}]
    with patch("src.study.similar.find_similar", return_value=canned):
        from src.cli.similar import run

        rc = run(_args("source", format="text"))

    assert rc == 0
    out = capsys.readouterr().out
    # The full 200-char key should NOT appear in the rendered output
    assert long_key not in out
    # Ellipsis marker present
    assert "…" in out


def test_similar_text_empty_result(capsys):
    """Empty result renders a friendly '(no similar clusters found)' line."""
    with patch("src.study.similar.find_similar", return_value=[]):
        from src.cli.similar import run

        rc = run(_args())

    assert rc == 0
    out = capsys.readouterr().out
    assert "no similar" in out.lower()


def test_similar_exception_returns_exit_2(capsys):
    """Unexpected backend exception → exit 2 + JSON error on stderr."""
    with patch("src.study.similar.find_similar", side_effect=RuntimeError("boom")):
        from src.cli.similar import run

        rc = run(_args())

    assert rc == 2
    err = capsys.readouterr().err
    assert "error" in err.lower()
