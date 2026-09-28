"""ASCII sparkline utility for the CLI dashboard (CLI-06, D-15).

Pure function. No I/O. Used by src/cli/dashboard.py to render the
'ev_loss trend last 10 sessions' line in the dashboard text panel.

Pattern reference: 06-RESEARCH.md Pattern 7.

Example:
    >>> sparkline([0.84, 0.71, 0.62, 0.55, 0.48, 0.42, 0.39, 0.41, 0.40, 0.39])
    '█▆▅▃▂▁▁▁▁▁'
"""

from __future__ import annotations

# 8 unicode block characters from lowest to tallest (U+2581..U+2588).
_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    """Render a list of floats as a unicode block sparkline.

    Args:
        values: Non-empty list of floats to plot; empty returns "".

    Returns:
        A unicode string the same length as ``values``. Each char is one of the
        8 blocks in ``_BLOCKS`` (U+2581..U+2588), scaled linearly to ``values``'s
        min..max range. Degenerate flat-line (min == max) renders as all lowest
        blocks (``_BLOCKS[0]``) — never NaN, never empty (unless input is empty).
    """
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return _BLOCKS[0] * len(values)
    span = hi - lo
    return "".join(_BLOCKS[min(7, int((v - lo) / span * 8))] for v in values)
