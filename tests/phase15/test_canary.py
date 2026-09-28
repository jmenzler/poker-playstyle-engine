# long-ok-file
"""Unit tests for phase15_canary.py assertion logic.  long-ok

Tests the assertion functions with mocks — no live stack required.
The live run itself is the Task 2 checkpoint (human-verify gate).

Covers:
    test_range_assert_*          — A1: placeholder detection
    test_solver_converges_assert_* — A2: exploitability threshold
    test_d16_context_assert_*    — A3: required keys + action_line_full
    test_node_injected_assert_*  — A4: strategy_nodes row count
    test_pot_type_filter_assert_* — A5: encoder propagation + Milvus filter
    test_kill_resume_assert_*    — A6: checkpoint continuity + no-duplicate
    test_disk_mem_assert_*       — A7: disk/mem growth tolerances
    test_compute_sec_per_spot_*  — A8: sec/spot arithmetic + projection
"""

from __future__ import annotations

import pytest

from scripts.phase15_canary import (
    _D16_REQUIRED_KEYS,
    _PLACEHOLDER_RANGE,
    compute_sec_per_spot,
    d16_context_assert,
    disk_mem_assert,
    kill_resume_assert,
    node_injected_assert,
    pot_type_filter_assert,
    project_feasibility,
    range_resolver_assert,
    solver_converges_assert,
)

# ---------------------------------------------------------------------------
# A1: Range resolver
# ---------------------------------------------------------------------------


def test_range_assert_passes_on_real_range():
    """Non-placeholder ranges pass without raising."""
    range_resolver_assert("AA,KK:0.90,AKs:0.75", "22,AQs,KJs")


def test_range_assert_fails_on_placeholder_ip():
    """Assert fails when range_ip is the placeholder string."""
    with pytest.raises(AssertionError, match="placeholder"):
        range_resolver_assert(_PLACEHOLDER_RANGE, "22,AQs")


def test_range_assert_fails_on_placeholder_oop():
    """Assert fails when range_oop is the placeholder string."""
    with pytest.raises(AssertionError, match="placeholder"):
        range_resolver_assert("AA,KK:0.90", _PLACEHOLDER_RANGE)


def test_range_assert_both_placeholder_fails_on_ip_first():
    """When both ranges are placeholder, AssertionError names ip first."""
    with pytest.raises(AssertionError, match="range_ip"):
        range_resolver_assert(_PLACEHOLDER_RANGE, _PLACEHOLDER_RANGE)


# ---------------------------------------------------------------------------
# A2: Solver convergence
# ---------------------------------------------------------------------------


def test_solver_converges_assert_at_threshold():
    """Exactly 0.5 is within the threshold."""
    solver_converges_assert(0.5)


def test_solver_converges_assert_below_threshold():
    """0.1 easily passes."""
    solver_converges_assert(0.1)


def test_solver_converges_assert_fails_above():
    """0.51 exceeds the 0.5 ceiling."""
    with pytest.raises(AssertionError, match=r"exploitability_pct=0\.5100 > 0\.5"):
        solver_converges_assert(0.51)


def test_solver_converges_assert_fails_high():
    """High exploitability (unconverged) always fails."""
    with pytest.raises(AssertionError, match="exploitability"):
        solver_converges_assert(5.0)


# ---------------------------------------------------------------------------
# A3: D-16 full context
# ---------------------------------------------------------------------------


def _make_full_context(**overrides) -> dict:
    """Minimal valid D-16 context dict."""
    ctx = {
        "board": ["Ah", "7c", "2d"],
        "hero_hole": ["As", "Kd"],
        "pot": 1000,
        "effective_stack": 9750,
        "prev_bet": None,
        "action_line_full": ["BTN:raise_half_pot", "BB:call"],
        "preflop_action_seq": ["UTG:fold", "CO:fold", "BTN:raise_2.5", "BB:call"],
        "street": "flop",
        "hero_pos": "BTN",
        "villain_pos": "BB",
        "pot_type": "3bet",
        "n_players_at_street": 2,
        "range_ip": "AA,KK:0.90",
        "range_oop": "22,AQs",
        "solver_settings": {"target_exploitability_pct": 0.5, "max_iterations": 500},
    }
    ctx.update(overrides)
    return ctx


def test_d16_context_assert_passes_complete():
    """All 15 required keys + non-empty action_line_full → passes."""
    d16_context_assert(_make_full_context())


def test_d16_context_assert_fails_missing_key():
    """A missing required key triggers AssertionError naming the key."""
    ctx = _make_full_context()
    del ctx["action_line_full"]
    with pytest.raises(AssertionError, match="action_line_full"):
        d16_context_assert(ctx)


def test_d16_context_assert_fails_empty_action_line():
    """Empty action_line_full triggers the irreversible-context guard."""
    ctx = _make_full_context(action_line_full=[])
    with pytest.raises(AssertionError, match="action_line_full is empty"):
        d16_context_assert(ctx)


def test_d16_context_assert_fails_null_action_line():
    """None action_line_full also triggers the guard."""
    ctx = _make_full_context(action_line_full=None)
    with pytest.raises(AssertionError, match="action_line_full is empty"):
        d16_context_assert(ctx)


def test_d16_context_assert_fails_multiple_missing():
    """Multiple missing keys are all listed in the AssertionError."""
    ctx = _make_full_context()
    del ctx["board"]
    del ctx["range_ip"]
    with pytest.raises(AssertionError) as exc_info:
        d16_context_assert(ctx)
    msg = str(exc_info.value)
    assert "board" in msg or "range_ip" in msg


def test_d16_required_keys_count():
    """Exactly 15 D-16 required keys are defined (matches plan spec)."""
    assert len(_D16_REQUIRED_KEYS) == 15


# ---------------------------------------------------------------------------
# A4: Node injected
# ---------------------------------------------------------------------------


def test_node_injected_assert_passes_one_row():
    """1 row is sufficient."""
    node_injected_assert(1, "street=flop|pot_type=3bet|hero_position=BTN|n_players_active=2")


def test_node_injected_assert_passes_multiple_rows():
    """Multiple rows also pass (e.g. idempotent re-upserts)."""
    node_injected_assert(3, "some-decision-id")


def test_node_injected_assert_fails_zero():
    """0 rows means the per-DP Milvus upsert did not happen."""
    with pytest.raises(AssertionError, match="no Milvus row"):
        node_injected_assert(0, "dp-uuid-xyz")


# ---------------------------------------------------------------------------
# A5: pot_type filter
# ---------------------------------------------------------------------------


def test_pot_type_filter_assert_passes():
    """correct hard_filter + Milvus hit → passes."""
    pot_type_filter_assert("3bet", 1)


def test_pot_type_filter_assert_fails_wrong_encoder_value():
    """Encoder returning 'srp' instead of '3bet' fails."""
    with pytest.raises(AssertionError, match="hard_filter"):
        pot_type_filter_assert("srp", 1)


def test_pot_type_filter_assert_fails_empty_milvus():
    """Correct encoder value but Milvus returns 0 results fails."""
    with pytest.raises(AssertionError, match="Milvus query"):
        pot_type_filter_assert("3bet", 0)


def test_pot_type_filter_assert_fails_both():
    """Wrong encoder AND empty Milvus → encoder error raised first."""
    with pytest.raises(AssertionError, match="hard_filter"):
        pot_type_filter_assert("limp", 0)


# ---------------------------------------------------------------------------
# A6: Kill + resume
# ---------------------------------------------------------------------------


def test_kill_resume_assert_passes():
    """spots_after_resume >= spots_before_kill and row count unchanged → passes."""
    kill_resume_assert(5, 5, 3, 3)


def test_kill_resume_assert_passes_more_spots():
    """Resume can report more spots than the checkpoint value."""
    kill_resume_assert(5, 7, 10, 10)


def test_kill_resume_assert_fails_on_restart_from_zero():
    """If spots_after_resume < spots_before_kill, driver ignored the checkpoint."""
    with pytest.raises(AssertionError, match="resume started from 0"):
        kill_resume_assert(5, 0, 3, 3)


def test_kill_resume_assert_fails_on_duplicate_rows():
    """Duplicate solver_cache rows after resume trigger assertion."""
    with pytest.raises(AssertionError, match="duplicates"):
        kill_resume_assert(5, 5, 3, 5)


# ---------------------------------------------------------------------------
# A7: Disk + memory
# ---------------------------------------------------------------------------


def test_disk_mem_assert_passes_flat():
    """No change in disk or memory → passes."""
    disk_mem_assert(50.0, 50.0, 200.0, 200.0)


def test_disk_mem_assert_passes_small_disk_use():
    """Tiny disk usage (a few MB from solver_cache write) is fine."""
    disk_mem_assert(50.0, 49.99, 200.0, 200.0)


def test_disk_mem_assert_fails_large_disk_growth():
    """1+ GB disk drop triggers alert."""
    with pytest.raises(AssertionError, match="disk used"):
        disk_mem_assert(50.0, 48.9, 200.0, 200.0)


def test_disk_mem_assert_fails_memory_leak():
    """RSS growth > 500 MB indicates leak."""
    with pytest.raises(AssertionError, match="RSS grew"):
        disk_mem_assert(50.0, 50.0, 200.0, 750.0)


def test_disk_mem_assert_custom_thresholds():
    """Custom thresholds are honoured."""
    with pytest.raises(AssertionError, match="disk used"):
        disk_mem_assert(50.0, 49.8, 200.0, 200.0, disk_growth_floor_gb=0.1)


# ---------------------------------------------------------------------------
# A8: sec/spot + projection
# ---------------------------------------------------------------------------


def test_compute_sec_per_spot_single():
    """100s for 1 spot = 100 sec/spot."""
    assert compute_sec_per_spot(100.0, 1) == pytest.approx(100.0)


def test_compute_sec_per_spot_multiple():
    """1100s for 11 spots = 100 sec/spot."""
    assert compute_sec_per_spot(1100.0, 11) == pytest.approx(100.0)


def test_compute_sec_per_spot_zero_guard():
    """0 spots solved returns inf (no ZeroDivisionError)."""
    assert compute_sec_per_spot(100.0, 0) == float("inf")


def test_project_feasibility_fits():
    """10 sec/spot x 73,570 spots = 204.4h → exceeds 168h; fits=False."""
    p = project_feasibility(10.0, total_spots=73_570, budget_hours=168.0)
    assert p["fits_budget"] is False
    assert p["recommended_cap"] < 73_570
    assert p["projected_total_hours"] == pytest.approx(73_570 * 10.0 / 3600.0, rel=1e-3)


def test_project_feasibility_tight_budget():
    """2 sec/spot x 73,570 = 40.9h → fits."""
    p = project_feasibility(2.0, total_spots=73_570, budget_hours=168.0)
    assert p["fits_budget"] is True
    assert p["recommended_cap"] == 73_570


def test_project_feasibility_exact_keys():
    """Return dict has all expected keys."""
    p = project_feasibility(5.0)
    expected_keys = {
        "sec_per_spot",
        "total_spots",
        "budget_hours",
        "projected_total_hours",
        "fits_budget",
        "recommended_cap",
    }
    assert expected_keys <= set(p.keys())


def test_project_feasibility_cap_is_reasonable():
    """recommended_cap when over budget = floor(budget * 3600 / sec_per_spot)."""
    sps = 10.0
    budget_h = 168.0
    p = project_feasibility(sps, budget_hours=budget_h)
    expected_cap = int((budget_h * 3600.0) / sps)
    assert p["recommended_cap"] == expected_cap
