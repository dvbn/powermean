"""Internal resource policy for normal joint WZ covariance only.

The N10/N20, T3/K2 index/profiled cases have conservative plans of
7,493,632/31,319,552 bytes and 16,704,576,000/71,942,937,600 estimated work units.
The base tier is 64MiB and 2**37 work units: coarse headroom over the larger plan.

The existing optional max_dense_bytes controls a finite resource tier. For a
positive exact byte limit B, work tiers are max(1, B //64MiB): only a COMPLETE
additional64MiB raises work. None selects the base tier. Lower byte limits and
values below128MiB keep the base work allowance;128MiB grants two,192MiB three.
This monotone, whole-tier rule is independent of incoming geometry/work estimates.
It avoids an absolute practical geometry ceiling without a new work-count API.
Both unchanged Reservation.require guards still apply afresh on every call.

These bounds cover their existing conservative numeric-buffer/array-work scopes,
not total process memory, cumulative fit work or wall time.
Cancelled inference and coefficient correction retain their separate paths.
"""

from __future__ import annotations

from powermean.solve.cancelled_operator import AttemptBounds

_JOINT_BASE_BYTES = 64 * 1024 * 1024
_JOINT_BASE_WORK = 2**37


def joint_bounds(max_dense_bytes: int | None = None) -> AttemptBounds:
    """Return fresh finite limits from the saved optional byte choice.

    Whole-tier floor rounding never supplies the current plan as its own limit.
    Increasing a byte limit cannot decrease either budget; no prior success,
    source/rank/point/weight/request or numerical check is reused or waived.
    """
    byte_limit = _JOINT_BASE_BYTES if max_dense_bytes is None else max_dense_bytes
    if type(byte_limit) is not int or byte_limit <= 0:
        raise ValueError("attempt bounds must be positive Python integers")
    tiers = max(1, byte_limit // _JOINT_BASE_BYTES)
    return AttemptBounds(byte_limit, _JOINT_BASE_WORK * tiers)
