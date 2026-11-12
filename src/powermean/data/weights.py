"""Observation weights of the three kinds: analytic, frequency and probability.

The kind is stated by the caller and never inferred from the values: ``[2.0, 3.0]`` are
analytic weights unless the caller says ``"frequency"``.  The kinds differ in what a
later stage does with them, not here.  Analytic and probability weights multiply the
contributions of their rows; frequency weights are replicated rows, in the counts as in
the contributions (:mod:`powermean.inference.convention`).  A row of zero weight is kept
at its position: nothing here selects a row, which is the sample construction (its step
4).  The raw values are stored as given, never normalized.

Frequency weights are exact counts.  Each is validated as supplied, before any
conversion to doubles, in this order: a real number that is not a boolean, from 0 to
``2^53``; finite; equal to its integer.  The total of the counts, summed in integer
arithmetic, must not exceed ``2^53``.  A count of ``2^53 + 1`` is refused although it
rounds to the accepted double ``2^53``, and so is the pair ``[2^53, 1]``, whose total in
doubles rounds to ``2^53`` while its exact total is ``2^53 + 1``: a conversion before
the check could not be repaired after it.  Serialized records take the same path, since
:meth:`ObservationWeights.from_dict` constructs the record.  A second check after the
conversion is implied by the first and is not made.

The values of every kind are an owned read-only array of finite nonnegative doubles with
a positive total that is a double (:mod:`powermean.data._arrays`).  The effective count
``N`` of the ``"stata"`` factors is the exact total of the frequency counts, else the
number of rows of positive weight (:attr:`ObservationWeights.effective_n`).
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    Protocol,
    SupportsFloat,
    SupportsInt,
    TypeGuard,
    get_args,
)

import numpy as np
from numpy.typing import ArrayLike

from powermean.data._arrays import frozen_array
from powermean.model.variance import FloatArray

WeightKind = Literal["analytic", "frequency", "probability"]
WEIGHT_KINDS: tuple[WeightKind, ...] = get_args(WeightKind)
# The largest count below which every integer is a double: a larger count may be the
# rounding of another, so it is refused, each count and their total.
MAXIMUM_COUNT = 2**53


class _RealNumber(SupportsFloat, SupportsInt, Protocol):
    """A real number as :func:`_check_counts` uses it: ordered, ``float``, ``int``."""

    def __lt__(self, other: int, /) -> bool:
        """``self < other``."""
        ...

    def __gt__(self, other: int, /) -> bool:
        """``self > other``."""
        ...


def _is_real_number(value: object) -> TypeGuard[_RealNumber]:
    """``numbers.Real`` and not a boolean: the type test on a count.

    ``numbers.Real`` documents ``<``, ``>`` and ``float()``, and every real number of
    Python, ``fractions`` and NumPy also has ``int()``; the typing stubs of
    ``numbers.Real`` declare neither ``>`` nor ``int()``.  ``_RealNumber`` states what
    the check of a count uses, and this test establishes it.
    """
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, numbers.Real)


def _check_counts(values: ArrayLike) -> None:
    """Frequency weights are exact counts, each checked before its conversion.

    The values are read into an array of objects, so that a Python integer keeps every
    digit, and each is tested as it was supplied, in this order: its type and range, its
    finiteness, then its equality with its integer.  The total of the integers must not
    exceed ``2^53``.
    """
    original = np.asarray(values, dtype=object)
    if original.ndim != 1:
        raise ValueError("frequency weights must be one-dimensional")
    counts: list[int] = []
    for value in original:
        if not _is_real_number(value) or value < 0 or value > MAXIMUM_COUNT:
            raise ValueError("frequency weights must be exactly representable counts")
        if not np.isfinite(float(value)):
            raise ValueError("frequency weights must be finite integer counts")
        count = int(value)
        if value != count or count < 0 or count > MAXIMUM_COUNT:
            raise ValueError("frequency weights must be exactly representable counts")
        counts.append(count)
    if sum(counts) > MAXIMUM_COUNT:
        raise ValueError("total frequency count must not exceed 2**53")


@dataclass(frozen=True, eq=False)
class ObservationWeights:
    """The weights of the observations, of one explicit kind.

    Parameters
    ----------
    values : array_like
        One weight per observation, real, finite and nonnegative, with a positive
        total; stored as an owned read-only array of doubles.  Frequency weights are
        exact integer counts of at most ``2^53``, with a total of at most ``2^53``,
        checked as supplied before any conversion.
    kind : {"analytic", "frequency", "probability"}, default "analytic"
        The kind of the weights, as the caller states it; never inferred.

    Raises
    ------
    ValueError
        An unknown kind; a value that is complex, not one-dimensional, not finite or
        negative; no positive weight; a frequency weight that is not an exact count.
    FloatingPointError
        A total of the weights beyond the doubles.

    Notes
    -----
    The record is frozen and compares by identity: two records with equal values are
    two inputs.  Rows of zero weight are kept at their positions.
    """

    values: FloatArray
    kind: WeightKind = "analytic"

    def __post_init__(self) -> None:
        """Validate the kind and the values, and store the values owned.

        The kind is read as ``object``: this is the run-time check of what the field
        declares.
        """
        kind: object = self.kind
        if kind not in WEIGHT_KINDS:
            raise ValueError("weight kind must be analytic, frequency or probability")
        if kind == "frequency":
            _check_counts(self.values)
        values = frozen_array(self.values, ndim=1, name="weights")
        if values.size == 0 or np.any(values < 0) or not np.any(values > 0):
            raise ValueError("weights must be nonnegative with positive total mass")
        with np.errstate(over="raise", invalid="raise"):
            values.sum()  # a total beyond the doubles raises FloatingPointError
        object.__setattr__(self, "values", values)

    @property
    def nobs(self) -> int:
        """The number of observations, rows of zero weight included."""
        return int(self.values.size)

    @property
    def frequency(self) -> bool:
        """Whether the weights are frequency weights: the flag that consumers take.

        :func:`powermean.inference.contributions.joint_influence` and the certificate
        receive this flag; it is the stated kind, never a reading of the values.
        """
        return self.kind == "frequency"

    @property
    def effective_n(self) -> int:
        """``N``: the exact total of frequency counts, else the rows of positive weight.

        The counts are summed as integers; their total is at most ``2^53``, so it is
        also the exact sum of the stored doubles.
        """
        if self.frequency:
            return sum(int(value) for value in self.values.tolist())
        return int(np.count_nonzero(self.values))

    def to_dict(self) -> dict[str, object]:
        """The kind and the values, as JSON holds them, losslessly."""
        return {
            "type": "ObservationWeights",
            "kind": self.kind,
            "values": self.values.tolist(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> ObservationWeights:
        """The record of :meth:`to_dict`, validated as a fresh construction is.

        Raises ``ValueError`` for a record of another type; a frequency count is
        checked as supplied, so a serialized ``2^53 + 1`` is refused.
        """
        # Values of any type, by field: the construction of the record validates them.
        values: dict[str, Any] = dict(record)
        if values.pop("type", None) != "ObservationWeights":
            raise ValueError("not an ObservationWeights record")
        return cls(**values)
