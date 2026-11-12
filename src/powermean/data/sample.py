"""Sample construction, steps 1 to 4: the order-neutral base.

The interface evaluates the formula on every input row and feeds this module arrays,
codes and records (:class:`~powermean.data.raw.RawInputs`); nothing here reads pandas or
Formulaic.  Every input row has the stable positional identifier ``0 .. n_input - 1``
before any step, every later selection is composed through those positions, and user
labels are identity metadata that the interface keeps apart.  The steps run in this
order, and a row excluded at an earlier step keeps that reason:

1. ``subset``: a row whose subset value is false leaves with the reason ``"subset"``,
   whatever else is wrong with it.
2. missing: a row with a missing or non-finite value in a variable that it uses leaves
   with the one reason ``"missing"``; the groups (outcome, regressors, effects,
   instruments, weights, offset, clusters) keep their own masks of the rows that they
   explain, as diagnostics (:class:`~powermean.data.accounting.MissingAccount`), so a
   row missing in two groups is counted once among the exclusions and once in each
   group.  An invalid value of an unused column marks no row, because no mask names it.
3. outcome domain, on the complete rows of step 2 and before any weight rule is
   applied: a negative outcome is an error at every order, refused here once; a zero
   outcome at ``omega <= 0`` is an error unless ``positive_only``, which removes the
   zero outcomes at every order (:mod:`powermean.data.retained`, for each order).  A
   row of zero weight is a complete row here, so a zero weight hides neither a negative
   outcome nor an invalid zero; nor does it hide a nonpositive exposure, the domain of
   the typed record, refused here with its message.
4. weights, on the rows that step 3 leaves: a negative weight is an error, refused by
   :func:`powermean.data.retained.derive_order` on the rows of its order, so that a
   zero outcome of negative weight that ``positive_only`` removes at step 3 is never
   read at step 4, while a negative weight on any row that step 3 keeps is refused
   whatever the order; a remaining row of zero weight leaves with the reason
   ``"zero_weight"``.  The masks of the negative and of the zero weights among the
   complete rows are formed here, since neither depends on the order, and the rule is
   applied for each order.

The values that the masks of step 2 promise are checked here: an outcome, design
entry, offset, exposure or weight that a mask does not name must be finite, and a
weight held as an object must be a real number, since step 2 is where a missing or
non-finite value leaves; a value that breaks that promise is the interface's error,
not a step of the sample.

:class:`SampleBase`, built once by :func:`build_sample`, holds the raw inputs (the raw
outcome, the unmultiplied offset or exposure, the trained design and the original weight
objects), the masks of steps 1, 2 and 4, and one code per input row.
:meth:`SampleBase.for_order` derives the sample and the problem of one mean order
(:func:`powermean.data.retained.derive_order`) without changing the base or training
anything again.  The messages are ``sample.negative_outcome`` and ``sample.empty`` (no
complete row), with the number of input rows, here; ``sample.negative_weight`` for each
order.  All arrays are owned and read-only; the inputs are not modified.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from powermean.data.accounting import (
    MISSING,
    MISSING_GROUPS,
    SUBSET,
    ZERO_WEIGHT,
    CodeArray,
    MissingAccount,
    bool_mask,
    frozen_codes,
)
from powermean.data.raw import AnyArray, RawInputs
from powermean.data.retained import OrderSample, derive_order
from powermean.errors import SampleError
from powermean.model.order import BoolArray, _flags
from powermean.model.variance import FloatArray


def _complete_finite(values: FloatArray | None, complete: BoolArray, name: str) -> None:
    """The values of the complete rows are finite: the masks of the caller agree."""
    if values is not None and not np.isfinite(values[complete]).all():
        raise ValueError(f"{name} must be finite on the rows that the masks keep")


def _object_weight_flags(
    values: AnyArray, complete: BoolArray
) -> tuple[BoolArray, BoolArray]:
    """The negative mask and the zero mask of weights held as objects.

    An integer is finite; another real number (a float, a fraction) is tested as a
    double, since a non-finite value belongs to step 2 and must have been masked.
    """
    negative = np.zeros(complete.shape, dtype=np.bool_)
    zero = np.zeros(complete.shape, dtype=np.bool_)
    for position in np.flatnonzero(complete).tolist():
        value = values[position]
        if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
            raise ValueError(
                "weights must be real numbers on the rows that the masks keep"
            )
        if not isinstance(value, numbers.Integral) and not math.isfinite(float(value)):
            raise ValueError("weights must be finite on the rows that the masks keep")
        if value < 0:
            negative[position] = True
        elif value == 0:
            zero[position] = True
    return negative, zero


def _weight_flags(values: AnyArray, complete: BoolArray) -> tuple[BoolArray, BoolArray]:
    """The masks of the negative and of the zero weights among the complete rows.

    Weights held as objects (exact counts, fractions) are compared one by one, in
    Python's own arithmetic, never converted to doubles.
    """
    if values.dtype.kind == "O":
        return _object_weight_flags(values, complete)
    selected = values[complete]
    if not np.isfinite(selected).all():
        raise ValueError("weights must be finite on the rows that the masks keep")
    negative = np.zeros(complete.shape, dtype=np.bool_)
    zero = np.zeros(complete.shape, dtype=np.bool_)
    negative[complete] = selected < 0
    zero[complete] = selected == 0
    return negative, zero


def _missing_account(
    missing: Mapping[str, ArrayLike] | None, rows: int, kept: BoolArray
) -> MissingAccount:
    """The masks of the groups on the rows that the subset kept, and their union."""
    given: dict[str, ArrayLike] = {} if missing is None else dict(missing)
    unknown = sorted(set(given) - set(MISSING_GROUPS))
    if unknown:
        raise ValueError(
            f"unknown missing groups {', '.join(map(repr, unknown))}; "
            f"the groups are {', '.join(map(repr, MISSING_GROUPS))}"
        )
    excluded = np.zeros(rows, dtype=np.bool_)
    groups: dict[str, BoolArray] = {}
    for group in MISSING_GROUPS:
        mask = given.get(group)
        if mask is None:
            explained = np.zeros(rows, dtype=np.bool_)
        else:
            explained = bool_mask(mask, rows, f"missing[{group!r}]") & kept
        groups[group] = _flags(explained)
        excluded |= explained
    return MissingAccount._owned(groups, _flags(excluded))


@dataclass(frozen=True, eq=False)
class SampleBase:
    """The order-neutral state of the sample: steps 1 and 2, the weights, the inputs.

    Attributes
    ----------
    inputs : RawInputs
        The raw arrays of every input row, as given.
    subset : array of bool or None
        The subset mask as given, or ``None`` without a subset.
    missing : MissingAccount
        The masks of step 2.
    complete : array of bool
        The rows that steps 1 and 2 keep: the rows on which the domain is validated.
    negative_weight : array of bool
        The complete rows of negative weight, which step 4 refuses where step 3 keeps
        them (:meth:`for_order`).
    zero_weight : array of bool
        The complete rows of zero weight, which step 4 excludes.
    codes : array of int8
        One code per input row: ``0`` where the row is complete and not of zero
        weight, else the step that excluded it (``1`` subset, ``2`` missing, ``4`` zero
        weight).  Step 3 adds its code in :meth:`for_order`, which refuses a negative
        weight among the rows that it keeps.

    Notes
    -----
    Read-only arrays.  The base is reused by every mean order: :meth:`for_order`
    derives a record of its own and changes nothing here.
    """

    inputs: RawInputs
    subset: BoolArray | None
    missing: MissingAccount
    complete: BoolArray
    negative_weight: BoolArray
    zero_weight: BoolArray
    codes: CodeArray

    @property
    def nobs(self) -> int:
        """The number of input rows."""
        return int(self.codes.size)

    def for_order(self, omega: float, *, positive_only: bool = False) -> OrderSample:
        """The sample and the transformed problem of the mean order ``omega``.

        :func:`powermean.data.retained.derive_order` with this base.

        Parameters
        ----------
        omega : float
            The mean order, finite and real; zero for log OLS.
        positive_only : bool, default False
            Whether the sample keeps the rows of positive outcome only.

        Returns
        -------
        OrderSample
            The retained rows with their accounting, the numeric record, the
            transformed problem, the scaled design and the fingerprint.
        """
        return derive_order(self, omega, positive_only=positive_only)


def build_sample(
    inputs: RawInputs,
    *,
    subset: ArrayLike | None = None,
    missing: Mapping[str, ArrayLike] | None = None,
) -> SampleBase:
    """Steps 1, 2 and the order-neutral parts of steps 3 and 4, on every input row.

    Parameters
    ----------
    inputs : RawInputs
        The raw arrays of every input row.
    subset : array_like of bool, optional
        One value per input row, true where the row is in the subset; every row
        without one.  The interface evaluates the expression and resolves a missing
        condition before this step.
    missing : mapping of str to array_like of bool, optional
        For any of :data:`~powermean.data.accounting.MISSING_GROUPS`, the rows whose
        value in that group is missing or non-finite; a group not named marks no row.

    Returns
    -------
    SampleBase
        The order-neutral state of the sample.

    Raises
    ------
    SampleError
        A negative outcome among the complete rows (``sample.negative_outcome``); no
        complete row (``sample.empty``).  A negative weight is refused by
        :meth:`SampleBase.for_order`, on the rows that step 3 keeps.
    ValueError
        A mask of another shape or type; an unknown group; a value that the masks keep
        but that is not finite or not a real number; a nonpositive exposure among the
        complete rows, the domain of the typed record.
    TypeError
        ``inputs`` that is not a ``RawInputs``.
    """
    if type(inputs) is not RawInputs:
        raise TypeError("inputs must be a RawInputs record")
    rows = inputs.nobs
    codes = np.zeros(rows, dtype=np.int8)
    kept = np.ones(rows, dtype=np.bool_)
    subset_mask = None
    if subset is not None:
        subset_mask = bool_mask(subset, rows, "subset")
        codes[~subset_mask] = SUBSET
        kept &= subset_mask
    account = _missing_account(missing, rows, kept)
    codes[account.excluded] = MISSING
    kept &= ~account.excluded
    if not kept.any():
        raise SampleError("sample.empty", {"count": rows})
    _complete_finite(inputs.y, kept, "y")
    if not np.isfinite(inputs.design[kept]).all():
        raise ValueError("design must be finite on the rows that the masks keep")
    _complete_finite(inputs.offset, kept, "offset")
    _complete_finite(inputs.exposure, kept, "exposure")
    negative = int(np.count_nonzero(inputs.y[kept] < 0))
    if negative:
        raise SampleError("sample.negative_outcome", {"count": negative})
    if inputs.exposure is not None and np.any(inputs.exposure[kept] <= 0):
        raise ValueError("exposure must be strictly positive and aligned with y")
    negative_weight = np.zeros(rows, dtype=np.bool_)
    zero_weight = np.zeros(rows, dtype=np.bool_)
    if inputs.weights is not None:
        negative_weight, zero_weight = _weight_flags(inputs.weights.values, kept)
    codes[zero_weight] = ZERO_WEIGHT
    return SampleBase(
        inputs=inputs,
        subset=None if subset_mask is None else _flags(subset_mask),
        missing=account,
        complete=_flags(kept),
        negative_weight=_flags(negative_weight),
        zero_weight=_flags(zero_weight),
        codes=frozen_codes(codes),
    )
