"""The estimation sample of one mean order: steps 3, 4 and 9 on the order-neutral base.

The prepared problem is reusable across ``(omega, tau)``.  :func:`derive_order` takes
the :class:`~powermean.data.sample.SampleBase` of steps 1 and 2 and derives the sample
of one mean order without changing the base or training anything again:

- step 3, the domain rule of :mod:`powermean.model.order`
  (``_selection``), on every complete row, zero and negative weights included, so that
  no weight hides an invalid zero at a nonpositive order; with ``positive_only`` the
  zero outcomes leave at every order with the reason ``"zero_outcome"``, which
  precedes the weight of the same row, and the conditioning is recorded;
- step 4, on the rows that step 3 keeps: a negative weight among them is refused
  (``sample.negative_weight``, with their number), whatever the order, while a negative
  weight on a zero outcome that ``positive_only`` removed is never read; then the zero
  weights of the base leave with the reason ``"zero_weight"``;
- the validated numeric record of the final rows alone
  (:class:`~powermean.data.inputs.RegressionData`): the raw outcome, the unscaled
  trained design, the offset or the exposure (whose logarithm the record forms once),
  the original weight objects, checked exactly there (a frequency count on an excluded
  row is never seen), and the original positions as integer labels;
- the numerical transform once, on the final rows
  (:func:`~powermean.model.order.transform_problem`): a row that step 4 excludes is
  never raised to its power nor has its offset scaled, so a zero-weight row with
  ``y = 1e200`` at ``omega = 2``, or an extreme offset, does not fail on a value that
  is never used, while an invalid domain on such a row still raises at step 3;
- step 9, the scaled design (:mod:`powermean.data.design`), every column kept;
- the aligned metadata of the final rows and the fingerprint of the sample
  (:mod:`powermean.data.fingerprint`).

:class:`OrderSample` keeps one code per input row (the codes of the base, with the
zero outcomes of this order), the positions of the retained rows in increasing order
and the inverse map from an input row to its position or ``-1``; the counts by reason
sum to the input rows.  A sample left without a row raises ``sample.empty`` with the
number of input rows.  All arrays are owned and read-only, the metadata through the
owned columns of :class:`~powermean.data.raw.AlignedColumns`, which hand out a fresh
array of fresh copies of the elements of an object column on every access, so that
no element handed out is a retained one.

The boundary of steps 3 and 4.  :func:`select_order` runs steps 3 and 4 alone, by the
same lines and with the same refusals in the same order as :func:`derive_order`, and
returns their rows (:class:`OrderRows`) without forming the numeric record, the
transform or the scaled design: the ordered composition of
:mod:`powermean.data.sample_fe` and :mod:`powermean.interface.sample` validates the
weights of those rows exactly (:func:`_regression_data`) before any fixed-effect
exclusion, transforms only after its deterministic closure and scales only the final
active columns of the final rows.  :func:`derive_order` itself is unchanged: the same
steps, then the record, the transform and step 9 on every column.  An :class:`OrderRows`
owns exact arrays, also when constructed or replaced directly (:func:`_exact_index`);
the ledger formed from it checks the codes' relation to the rows, so that the
perturbations of steps 3 and 4 keep their detections.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from powermean.data.accounting import (
    EXCLUSION_REASONS,
    ZERO_OUTCOME,
    CodeArray,
    MissingAccount,
    counts_of,
    frozen_codes,
    frozen_index,
    reason_of,
)
from powermean.data.design import ScaledDesign, scale_columns
from powermean.data.fingerprint import sample_fingerprint
from powermean.data.inputs import RegressionData
from powermean.data.raw import AlignedColumns, AnyArray, RawInputs
from powermean.data.weights import ObservationWeights
from powermean.errors import SampleError
from powermean.model.order import (
    IndexArray,
    TransformedProblem,
    _flag,
    _positions,
    _selection,
    transform_problem,
)
from powermean.model.order_scale import OrderScale, _mean_order
from powermean.model.variance import FloatArray

if TYPE_CHECKING:
    from powermean.data.sample import SampleBase


@dataclass(frozen=True, eq=False)
class OrderSample:
    """The estimation sample of one mean order, with its accounting and its problem.

    Attributes
    ----------
    omega : float
        The mean order.
    positive_only : bool
        Whether the caller asked for the positive outcomes only.
    conditioned : bool
        Whether a zero outcome was removed: the sample conditions on positive
        outcomes, which the output states.
    positions : array of int
        The original positions of the retained rows, increasing: the ``k``-th row of
        the sample is the input row ``positions[k]``.
    retained_index : array of int
        For each input row, its position in the sample, or ``-1`` where excluded: the
        inverse of ``positions``.
    codes : array of int8
        One code per input row: ``0`` retained, else one plus the index of its reason
        in :data:`~powermean.data.accounting.EXCLUSION_REASONS`.
    missing : MissingAccount
        The diagnostics of step 2.
    data : RegressionData
        The validated numeric record of the retained rows: raw outcome, unscaled
        trained design, offset or exposure, typed weights, the positions as labels.
    problem : TransformedProblem
        The accepted transform of the retained rows at ``omega``; its selection is
        every retained row.
    design : ScaledDesign
        The design of the retained rows, scaled, with its scale.
    metadata : AlignedColumns
        The aligned metadata of the retained rows, by the same positions: a mapping of
        name to owned array, each handed out read-only.
    fingerprint : str
        The fingerprint of the sample (:mod:`powermean.data.fingerprint`).
    """

    omega: float
    positive_only: bool
    conditioned: bool
    positions: IndexArray
    retained_index: IndexArray
    codes: CodeArray
    missing: MissingAccount
    data: RegressionData
    problem: TransformedProblem
    design: ScaledDesign
    metadata: AlignedColumns
    fingerprint: str

    @property
    def nobs(self) -> int:
        """The number of input rows."""
        return int(self.codes.size)

    @property
    def count(self) -> int:
        """The number of retained rows, which enter the estimating equation."""
        return int(self.positions.size)

    @property
    def effective_n(self) -> int:
        """``N``: the exact total of frequency counts, else the retained rows."""
        return self.data.effective_n

    @property
    def order_scale(self) -> OrderScale:
        """The accepted map from the solver's coordinates to the reported scale."""
        return self.design.order_scale(self.omega)

    @property
    def counts(self) -> Mapping[str, int]:
        """The rows excluded by each reason, in the order of the steps, and retained."""
        return counts_of(self.codes)

    def reason(self, position: int) -> str | None:
        """The reason that excluded the input row ``position``, or ``None`` if kept."""
        return reason_of(int(self.codes[position]))

    def table(self) -> tuple[tuple[str, int], ...]:
        """The counts in the order of the steps: each reason, each group, kept.

        Returns
        -------
        tuple of (str, int)
            ``subset``, ``missing``, then ``missing: <group>`` for each group (which
            overlap and need not sum to ``missing``), ``zero_outcome``, ``zero_weight``
            and ``retained``.  The reasons and ``retained`` sum to the input rows.
        """
        counts = self.counts
        groups = [(f"missing: {group}", n) for group, n in self.missing.counts.items()]
        return (
            ("subset", counts["subset"]),
            ("missing", counts["missing"]),
            *groups,
            ("zero_outcome", counts["zero_outcome"]),
            ("zero_weight", counts["zero_weight"]),
            ("retained", counts["retained"]),
        )


@dataclass(frozen=True, eq=False)
class OrderRows:
    """Steps 3 and 4 of one mean order: the rows that they keep, nothing formed yet.

    Attributes
    ----------
    omega : float
        The mean order.
    positive_only : bool
        Whether the caller asked for the positive outcomes only.
    conditioned : bool
        Whether a zero outcome was removed.
    positions : array of int
        The original positions of the rows that steps 3 and 4 keep, increasing.
    retained_index : array of int
        For each input row, its position among them, or ``-1``.
    codes : array of int8
        One code per input row, as :attr:`OrderSample.codes`.
    """

    omega: float
    positive_only: bool
    conditioned: bool
    positions: IndexArray
    retained_index: IndexArray
    codes: CodeArray

    def __post_init__(self) -> None:
        """Own exact arrays: codes of steps 1 to 4, increasing rows, their inverse."""
        object.__setattr__(self, "omega", _mean_order(self.omega))
        if not isinstance(self.positive_only, bool) or not isinstance(
            self.conditioned, bool
        ):
            raise ValueError("positive_only and conditioned must be bools")
        codes = _exact_index(self.codes, "codes", upper=len(EXCLUSION_REASONS) + 1)
        positions = _exact_index(self.positions, "positions", upper=codes.size)
        index = _exact_index(
            self.retained_index, "retained_index", lower=-1, upper=codes.size
        )
        if not np.array_equal(index, _inverse(positions, codes.size)):
            raise ValueError("retained_index must be the inverse of the positions")
        object.__setattr__(self, "codes", frozen_codes(codes.astype(np.int8)))
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "retained_index", index)

    @property
    def nobs(self) -> int:
        """The number of input rows."""
        return int(self.codes.size)


def _exact_index(
    values: object, what: str, *, upper: int, lower: int = 0
) -> IndexArray:
    """An owned read-only vector of exact integers in ``[lower, upper)``.

    Checked before any conversion to ``np.intp``: a list or tuple of Python or NumPy
    integers (no Boolean, no float), or a one-dimensional array of an integer kind (an
    empty one of any kind).  Nothing is truncated, read as a Boolean or wrapped.
    """
    array: AnyArray
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
            raise ValueError(f"{what} must be a vector of exact integers")
        array = values
        exact = [int(values.min()), int(values.max())] if values.size else []
    elif isinstance(values, tuple | list) and all(
        isinstance(v, int | np.integer) and not isinstance(v, bool) for v in values
    ):
        exact = [int(v) for v in values]
        array = np.array(exact, dtype=object)
    else:
        raise ValueError(f"{what} must be a vector of exact integers")
    if exact and (min(exact) < lower or max(exact) >= upper):
        raise ValueError(f"{what} must lie in [{lower}, {upper})")
    return frozen_index(array.astype(np.intp))


def _inverse(positions: IndexArray, rows: int) -> IndexArray:
    """Each input row's place among ``positions``, else ``-1``; they must increase."""
    if (positions[1:] <= positions[:-1]).any():
        raise ValueError("positions must increase strictly")
    inverse = np.full(rows, -1, dtype=np.intp)
    inverse[positions] = np.arange(positions.size, dtype=np.intp)
    return frozen_index(inverse)


def _complete_outcome(base: SampleBase) -> tuple[IndexArray, FloatArray]:
    """The complete rows of steps 1 and 2, where step 3 reads the domain; their outcome.

    Every complete row, zero and negative weights included.
    """
    complete_positions = _positions(base.complete)
    return complete_positions, base.inputs.y[complete_positions]


def _rows(
    base: SampleBase,
    complete_positions: IndexArray,
    outcome: FloatArray,
    order: float,
    flag: bool,
) -> OrderRows:
    """Steps 3 and 4 on the complete rows and their outcome (module docstring)."""
    rows = base.nobs
    if flag and not np.any(outcome != 0):
        raise SampleError("sample.empty", {"count": rows})
    # The accepted domain rule, on every complete row, whatever its weight.
    selection = _selection(outcome, order, flag)
    removed = complete_positions[selection.removed]
    codes = base.codes.copy()
    codes[removed] = ZERO_OUTCOME
    surviving = base.complete.copy()
    surviving[removed] = False
    # Step 4 on the rows that step 3 keeps: the negative weights first, then the zeros.
    negative = int(np.count_nonzero(base.negative_weight & surviving))
    if negative:
        raise SampleError("sample.negative_weight", {"count": negative})
    kept = surviving & ~base.zero_weight
    positions = _positions(kept)
    if not positions.size:
        raise SampleError("sample.empty", {"count": rows})
    retained_index = np.full(rows, -1, dtype=np.intp)
    retained_index[positions] = np.arange(positions.size, dtype=np.intp)
    return OrderRows(
        omega=order,
        positive_only=flag,
        conditioned=selection.conditioned,
        positions=positions,
        retained_index=frozen_index(retained_index),
        codes=frozen_codes(codes),
    )


def select_order(
    base: SampleBase, omega: float, *, positive_only: bool = False
) -> OrderRows:
    """Steps 3 and 4 of the mean order ``omega`` alone: the rows, nothing formed.

    Parameters
    ----------
    base : SampleBase
        The order-neutral state of :func:`powermean.data.sample.build_sample`.
    omega : float
        The mean order, finite and real; zero for log OLS.
    positive_only : bool, default False
        Whether the sample keeps the rows of positive outcome only.

    Returns
    -------
    OrderRows
        The rows, codes and conditioning that :func:`derive_order` derives, by the
        same steps; no numeric record, transform or scaled design.

    Raises
    ------
    SampleError
        The refusals of steps 3 and 4 of :func:`derive_order`, in its order.
    ValueError, TypeError
        A malformed order or flag.
    """
    order, flag = _mean_order(omega), _flag(positive_only)
    complete_positions, outcome = _complete_outcome(base)
    return _rows(base, complete_positions, outcome, order, flag)


def _regression_data(inputs: RawInputs, positions: IndexArray) -> RegressionData:
    """The validated numeric record of the retained rows, labeled by their positions.

    The weights are the original objects of those rows, so that frequency counts are
    checked exactly there and only there.
    """
    weights = None
    if inputs.weights is not None:
        values = cast("FloatArray", inputs.weights.values[positions])
        weights = ObservationWeights(values, inputs.weights.kind)
    offset = None if inputs.offset is None else inputs.offset[positions]
    exposure = None if inputs.exposure is None else inputs.exposure[positions]
    return RegressionData(
        inputs.y[positions],
        inputs.design[positions],
        offset=offset,
        exposure=exposure,
        weights=weights,
        row_labels=positions.tolist(),
        column_names=inputs.column_names,
    )


def derive_order(
    base: SampleBase, omega: float, *, positive_only: bool = False
) -> OrderSample:
    """The sample and the transformed problem of the mean order ``omega``.

    Parameters
    ----------
    base : SampleBase
        The order-neutral state of :func:`powermean.data.sample.build_sample`.
    omega : float
        The mean order, finite and real; zero for log OLS.
    positive_only : bool, default False
        Whether the sample keeps the rows of positive outcome only.

    Returns
    -------
    OrderSample
        The retained rows with their accounting, the numeric record, the transformed
        problem, the scaled design, the metadata and the fingerprint.

    Raises
    ------
    SampleError
        A zero outcome at ``omega <= 0`` without ``positive_only``
        (``sample.zero_outcome_at_order``); a negative weight among the rows that step
        3 keeps (``sample.negative_weight``); no row left (``sample.empty``).
    ValueError, TypeError
        A malformed order or flag; a weight that the numeric record refuses on the
        retained rows.
    FloatingPointError
        A power, a transformed offset or a scaled entry that is not representable.
    """
    return _derive_order_parts(base, omega, positive_only=positive_only)[0]


def _derive_order_parts(
    base: SampleBase, omega: float, *, positive_only: bool = False
) -> tuple[OrderSample, OrderRows]:
    """Return one actual producer's sample and already checked row selection.

    Public derive_order returns the identical sample. Private owned composition
    carries both fresh parts forward, never a caller-supplied cached order.
    """
    order, flag = _mean_order(omega), _flag(positive_only)
    # Steps 3 and 4 read the complete rows and their outcome; the transform below
    # reads the final rows only.
    complete_positions, outcome = _complete_outcome(base)
    selected = _rows(base, complete_positions, outcome, order, flag)
    positions = selected.positions
    data = _regression_data(base.inputs, positions)
    # The numerical transform once, on the final rows: the domain was applied at step
    # 3, so nothing is removed here and the selection is every final row.
    problem = transform_problem(data.y, order, offset=data.offset, positive_only=False)
    return OrderSample(
        omega=order,
        positive_only=flag,
        conditioned=selected.conditioned,
        positions=positions,
        retained_index=selected.retained_index,
        codes=selected.codes,
        missing=base.missing,
        data=data,
        problem=problem,
        design=scale_columns(data.design, data.column_names),
        metadata=base.inputs.metadata.take(positions),
        fingerprint=sample_fingerprint(data),
    ), selected
