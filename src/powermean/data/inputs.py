"""The owned record of the arrays of an estimation: outcome, design, offset or exposure.

``exposure=e`` is ``offset=log(e)``.

A :class:`RegressionData` is an already valid numeric record, not the raw input of the
sample construction: every row given is kept at its position, a row of zero weight, of
zero outcome or of missing label included, and nothing is selected, dropped, filled or
normalized.  Its arrays are owned read-only copies of finite real values
(:mod:`powermean.data._arrays`), aligned on their rows; the outcome ``y`` is the outcome
as supplied, nonnegative, whose domain at a mean order the order layer decides
(``powermean.model.order``).

Offset and exposure are mutually exclusive.  The offset ``o`` has unit coefficient on
the log power mean; an exposure ``e``, strictly positive and aligned with ``y``, enters
as the offset ``log e``, formed once here, and is kept as supplied for the provenance
of the record.  The multiplier of the mean order, ``omega o`` on the transformed index,
belongs to the order layer alone and is never applied here.

Row identity is positional: the ``k``-th row of every array is observation ``k``.  The
labels of the rows (``row_labels``: integers or strings, the positions by default) are
carried for the accounting and the output and need not be unique, as the labels of a
pandas index need not be.  The column names of the design are unique nonempty strings
(``x0``, ``x1``, ... by default).  The weights are an
:class:`~powermean.data.weights.ObservationWeights` of an explicit kind, aligned with
``y``; without one every row has weight one.  The record carries no fingerprint: the
fingerprint of the estimation sample is formed on the sample that the sample
construction builds, not on the input.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import repeat
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from powermean.data._arrays import frozen_array
from powermean.data.weights import ObservationWeights
from powermean.model.variance import FloatArray

Label = int | str


def _labels(labels: Sequence[Label] | None, rows: int) -> tuple[Label, ...]:
    """The row labels as a tuple: integers or strings, one per row; duplicates kept.

    The type of each label is tested exactly: a boolean or a NumPy integer is not a
    label. The ordinary positional-int run is checked by identity in bulk; every other
    kind keeps the same original ordered rejection.
    """
    found = tuple(range(rows)) if labels is None else tuple(labels)
    if len(found) == rows and all(map(operator.is_, map(type, found), repeat(int))):
        return found
    if len(found) != rows or any(type(label) not in (int, str) for label in found):
        raise ValueError("row_labels must be aligned integer or string labels")
    return found


def _column_names(names: Sequence[str] | None, columns: int) -> tuple[str, ...]:
    """The column names as a tuple: unique nonempty strings, one per column."""
    found = tuple(f"x{j}" for j in range(columns)) if names is None else tuple(names)
    if (
        len(found) != columns
        or len(set(found)) != len(found)
        or any(not isinstance(name, str) or not name for name in found)
    ):
        raise ValueError(
            "column_names must be unique nonempty strings aligned with design"
        )
    return found


@dataclass(frozen=True, eq=False, init=False)
class RegressionData:
    """The arrays of an estimation, validated and owned.

    Parameters
    ----------
    y : array_like
        The outcome as supplied, one finite nonnegative value per row.
    design : array_like
        The design, one row per observation and at least one column, finite.
    offset : array_like, optional
        The offset ``o`` of the log power mean, one finite value per row.  Zero for
        every row if omitted.  Not together with ``exposure``.
    exposure : array_like, optional
        A strictly positive exposure, one finite value per row, whose logarithm is
        the offset.  Not together with ``offset``.
    weights : ObservationWeights, optional
        The weights of the rows, of an explicit kind, aligned with ``y``.
    row_labels : sequence of int or str, optional
        The labels of the rows, one per row, duplicates allowed; the positions if
        omitted.
    column_names : sequence of str, optional
        The names of the columns, unique and nonempty; ``x0``, ``x1``, ... if omitted.

    Raises
    ------
    ValueError
        Arrays that are not real, finite or aligned; a negative outcome; no row or no
        column; both an offset and an exposure; an exposure that is not strictly
        positive; weights not aligned; malformed labels or column names.
    TypeError
        ``weights`` that is not an ``ObservationWeights``.

    Notes
    -----
    Stored: ``y``, ``design`` and ``offset`` as owned read-only arrays, ``offset``
    being ``log(exposure)`` when an exposure is given; ``exposure`` as an owned array or
    ``None``; ``weights`` as given; ``row_labels`` and ``column_names`` as tuples.  The
    fields declare what is stored, which the explicit constructor forms from what is
    given, as :class:`powermean.systems.joint.JointSystem` does: ``offset``, the labels
    and the names are never ``None`` on a record.  The record is frozen and compares by
    identity.
    """

    y: FloatArray
    design: FloatArray
    offset: FloatArray
    exposure: FloatArray | None
    weights: ObservationWeights | None
    row_labels: tuple[Label, ...]
    column_names: tuple[str, ...]

    def __init__(
        self,
        y: FloatArray,
        design: FloatArray,
        *,
        offset: FloatArray | None = None,
        exposure: FloatArray | None = None,
        weights: ObservationWeights | None = None,
        row_labels: Sequence[Label] | None = None,
        column_names: Sequence[str] | None = None,
    ) -> None:
        """Validate every input and store it owned."""
        outcome = frozen_array(y, ndim=1, name="y")
        matrix = frozen_array(design, ndim=2, name="design")
        if outcome.size == 0 or matrix.shape[0] != outcome.size or matrix.shape[1] == 0:
            raise ValueError(
                "y and design require matching nonempty rows and at least one column"
            )
        if np.any(outcome < 0):
            raise ValueError("outcome must be nonnegative")
        if offset is not None and exposure is not None:
            raise ValueError("specify offset or exposure, not both")
        supplied: ArrayLike = np.zeros(outcome.size) if offset is None else offset
        owned_exposure: FloatArray | None = None
        if exposure is not None:
            owned_exposure = frozen_array(exposure, ndim=1, name="exposure")
            if owned_exposure.shape != outcome.shape or np.any(owned_exposure <= 0):
                raise ValueError(
                    "exposure must be strictly positive and aligned with y"
                )
            supplied = np.log(owned_exposure)  # formed once; finite for a double > 0
        owned_offset = frozen_array(supplied, ndim=1, name="offset")
        if owned_offset.shape != outcome.shape:
            raise ValueError("offset must be aligned with y")
        if weights is not None:
            if not isinstance(weights, ObservationWeights):
                raise TypeError("weights must be an ObservationWeights object")
            if weights.values.shape != outcome.shape:
                raise ValueError("weights must be aligned with y")
        labels = _labels(row_labels, outcome.size)
        names = _column_names(column_names, matrix.shape[1])
        stored: dict[str, object] = {
            "y": outcome,
            "design": matrix,
            "offset": owned_offset,
            "exposure": owned_exposure,
            "weights": weights,
            "row_labels": labels,
            "column_names": names,
        }
        for key, value in stored.items():
            object.__setattr__(self, key, value)

    @property
    def nobs(self) -> int:
        """The number of rows, every row given."""
        return int(self.y.size)

    @property
    def nparams(self) -> int:
        """The number of columns of the design."""
        return int(self.design.shape[1])

    @property
    def weight_values(self) -> FloatArray:
        """The raw weights, one for every row without weights."""
        if self.weights is None:
            return frozen_array(np.ones(self.nobs))
        return self.weights.values

    @property
    def effective_n(self) -> int:
        """``N``: the exact total of frequency counts, else the rows of positive weight.

        Every row without weights.
        """
        if self.weights is None:
            return self.nobs
        return self.weights.effective_n

    def to_dict(self) -> dict[str, object]:
        """Lossless JSON-compatible input record; no inferred or deleted rows.

        With an exposure, the offset is written as ``None`` and formed again from the
        exposure when the record is read back.
        """
        return {
            "type": "RegressionData",
            "y": self.y.tolist(),
            "design": self.design.tolist(),
            "offset": None if self.exposure is not None else self.offset.tolist(),
            "exposure": None if self.exposure is None else self.exposure.tolist(),
            "weights": None if self.weights is None else self.weights.to_dict(),
            "row_labels": list(self.row_labels),
            "column_names": list(self.column_names),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> RegressionData:
        """The record of :meth:`to_dict`, validated as a fresh construction is.

        Raises ``ValueError`` for a record, or a weight record, of another type.
        """
        # Values of any type, by field: the construction of the record validates them.
        values: dict[str, Any] = dict(record)
        if values.pop("type", None) != "RegressionData":
            raise ValueError("not a RegressionData record")
        weights = values.get("weights")
        if weights is not None:
            values["weights"] = ObservationWeights.from_dict(weights)
        return cls(**values)
