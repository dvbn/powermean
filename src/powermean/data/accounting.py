"""The accounting of the sample: reasons, codes, masks and the diagnostics of step 2.

Every exclusion of a row or a column is counted and explained in the result.  The
records of :mod:`powermean.data.sample` and :mod:`powermean.data.retained` share what is
here: the four reasons of exclusion in the order of the steps, the code of each (one
plus its index; zero where a row is retained), the seven variable groups of step 2 in
the order of the design, the owned read-only copies of masks, codes and index arrays,
and :class:`MissingAccount`, the masks of the rows that each group explains, which
overlap where a row is invalid in several groups, beside their union, the rows that step
2 excludes once each.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.data.raw import _OwnedMapping
from powermean.model.order import BoolArray, IndexArray

CodeArray = NDArray[np.int8]
# The reasons of exclusion, in the order of the steps of the sample construction; a
# row's code is the index of its reason plus one, and zero where the row is retained.
EXCLUSION_REASONS: tuple[str, ...] = (
    "subset",
    "missing",
    "zero_outcome",
    "zero_weight",
)
RETAINED, SUBSET, MISSING, ZERO_OUTCOME, ZERO_WEIGHT = 0, 1, 2, 3, 4
# The variable groups of step 2, in the order of the design.
MISSING_GROUPS: tuple[str, ...] = (
    "outcome",
    "regressors",
    "effects",
    "instruments",
    "weights",
    "offset",
    "clusters",
)


def bool_mask(mask: ArrayLike, rows: int, name: str) -> BoolArray:
    """A Boolean array with one value per input row, as given.

    Parameters
    ----------
    mask : array_like
        The mask, of Boolean type and shape ``(rows,)``.
    rows : int
        The number of input rows.
    name : str
        The name of the mask in the message of a refusal.

    Raises
    ------
    ValueError
        Another type or shape: nothing is converted.
    """
    array = np.asarray(mask)
    if array.dtype != np.bool_ or array.shape != (rows,):
        raise ValueError(f"{name} must be a Boolean array with one value per input row")
    return cast("BoolArray", array)


def frozen_codes(codes: CodeArray) -> CodeArray:
    """An owned read-only copy of the codes of the rows."""
    stored: CodeArray = np.frombuffer(codes.tobytes(), dtype=np.int8)
    return stored


def frozen_index(index: IndexArray) -> IndexArray:
    """An owned read-only copy of an index array."""
    stored: IndexArray = np.frombuffer(index.tobytes(), dtype=np.intp)
    return stored


def reason_of(code: int) -> str | None:
    """The reason of a code: ``None`` where the row is retained."""
    return None if code == RETAINED else EXCLUSION_REASONS[code - 1]


def counts_of(codes: CodeArray) -> Mapping[str, int]:
    """The rows excluded by each reason, in the order of the steps, and retained."""
    found = {
        reason: int(np.count_nonzero(codes == code))
        for code, reason in enumerate(EXCLUSION_REASONS, start=1)
    }
    found["retained"] = int(np.count_nonzero(codes == RETAINED))
    return MappingProxyType(found)


@dataclass(frozen=True, eq=False)
class MissingAccount:
    """Step 2: the rows that each variable group explains, and the rows excluded.

    Attributes
    ----------
    groups : mapping of str to array of bool
        For each group of :data:`MISSING_GROUPS`, the rows not excluded by the subset
        whose value in that group is missing or non-finite; read-only.  Groups overlap
        where a row is invalid in several of them.
    excluded : array of bool
        The union of the groups: the rows that step 2 excludes, each once.

    Notes
    -----
    The constructor keeps the mapping it is given as it is, unread: a caller's own
    mapping or a plain dict stays the caller's, read when :attr:`groups` or
    :attr:`counts` is read.  The accounts that the package forms (step 2, the isolated
    and owned copies, the snapshot's materialization) come from :meth:`_owned`, whose
    exact ``dict`` of masks is carried as the record's current data by a
    :class:`~powermean.data.raw._OwnedMapping` and handed out read-only.
    """

    groups: Mapping[str, BoolArray]
    excluded: BoolArray

    @classmethod
    def _owned(
        cls, groups: dict[str, BoolArray], excluded: BoolArray
    ) -> MissingAccount:
        """The account of masks the package formed, ``groups`` an exact owned ``dict``.

        The internal construction route: ``groups`` is a ``dict`` of the package's
        own, which the record carries as it is (``TypeError`` for another type) and
        presents read-only as :attr:`groups`.
        """
        return cls(groups=_OwnedMapping(groups), excluded=excluded)

    @property
    def counts(self) -> Mapping[str, int]:
        """The number of rows that each group explains, in the order of the design."""
        return MappingProxyType(
            {group: int(np.count_nonzero(mask)) for group, mask in self.groups.items()}
        )

    @property
    def count(self) -> int:
        """The number of rows excluded by step 2, overlapping groups counted once."""
        return int(np.count_nonzero(self.excluded))
