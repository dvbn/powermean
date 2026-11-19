"""Resource choices for joint WZ covariance and coefficient correction.

An omitted max_dense_bytes uses machine-sized bounds instead of a provisional
buffer/work tier. Explicit positive Python integers retain the byte limit and
the work mapping max(1, B // 64MiB) * 2**37. Reservations still run on each call;
these limits cover modeled numeric buffers and array work, not process memory,
wall time or solver/certificate iterations. Allocation, representation and
numerical failures retain their existing behavior. Cancelled inference retains
its separate explicit controls.
"""

from __future__ import annotations

import sys

from powermean.solve.cancelled_operator import AttemptBounds

_JOINT_BASE_BYTES = 64 * 1024 * 1024
_JOINT_BASE_WORK = 2**37


def joint_bounds(max_dense_bytes: int | None = None) -> AttemptBounds:
    """Return fresh bounds, applying the work-tier policy only to explicit limits.

    Whole-tier floor rounding never supplies the current plan as its own limit.
    Increasing a byte limit cannot decrease either budget; no prior success,
    source/rank/point/weight/request or numerical check is reused or waived.
    """
    if max_dense_bytes is None:
        return AttemptBounds(sys.maxsize, sys.maxsize)
    byte_limit = max_dense_bytes
    if type(byte_limit) is not int or byte_limit <= 0:
        raise ValueError("attempt bounds must be positive Python integers")
    tiers = max(1, byte_limit // _JOINT_BASE_BYTES)
    return AttemptBounds(byte_limit, _JOINT_BASE_WORK * tiers)
