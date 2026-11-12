"""Fresh native-index pair layout for observed geometry, without cached maps.

Integer lexicographic sorting changes only the pair index, never the curvature row
order.
"""

from __future__ import annotations

import numpy as np

from powermean.fe.groups import Block
from powermean.model.order import IndexArray


def _native_vectors(left: object, right: object) -> bool:
    """Inspect only exact array metadata, with no subclass or object callbacks."""
    return (
        type(left) is np.ndarray
        and type(right) is np.ndarray
        and left.dtype == right.dtype == np.dtype(np.intp)
        and left.ndim == right.ndim == 1
        and left.flags.c_contiguous
        and right.flags.c_contiguous
        and left.size == right.size
        and left.size > 0
    )


def _stored_codes(record: Block) -> object:
    """The codes stored in an exact record's plain dict, or ``None`` to defer.

    Only an exact dict whose keys are all exact strings is read. A dict subclass can
    override the lookup that ordinary attribute access bypasses, and a key of another
    type can run its own equality during a lookup; reading either adds callbacks.
    """
    stored = record.__dict__
    if type(stored) is not dict or any(type(key) is not str for key in stored):
        return None
    return stored.get("codes")


def native_pair_inputs(blocks: tuple[Block, ...], first: Block, second: int) -> bool:
    """Fresh eligibility only for plain records with callback-free stored vectors.

    Do not retain generic code-property results across the caller's column_stack.
    Missing fields also defer to the original attribute access and its refusal.
    """
    if not (
        type(blocks) is tuple and type(first) is Block and type(blocks[second]) is Block
    ):
        return False
    return _native_vectors(_stored_codes(first), _stored_codes(blocks[second]))


def native_pair_layout(
    keys: IndexArray, left: IndexArray, right: IndexArray
) -> tuple[IndexArray, IndexArray] | None:
    """Return freshly sorted pairs and the original-row inverse, or defer to unique.

    Only exact, contiguous, native-index vectors and their ordinary stacked keys
    take this path. Other inputs retain the original NumPy unique call, including
    its subclass/object dispatch and refusals. No coordinate products are formed.
    """
    if not (
        _native_vectors(left, right)
        and type(keys) is np.ndarray
        and keys.dtype == np.dtype(np.intp)
        and keys.shape == (left.size, 2)
    ):
        return None
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    ordered = keys[order]
    start = np.ones(order.size, dtype=bool)
    start[1:] = (ordered[1:, 0] != ordered[:-1, 0]) | (
        ordered[1:, 1] != ordered[:-1, 1]
    )
    inverse = np.empty(order.size, dtype=np.intp)
    inverse[order] = np.cumsum(start, dtype=np.intp) - 1
    return ordered[start], inverse
