"""The binding of a sample to the preparation and the estimator that it came from.

The following information is bound to the same coefficient point, equation, sample and
coordinates: the sample fingerprint and exclusion accounting, the original positional
IDs, the typed weights with their kind and values, the requested classifications.  The
handoff of a fit (:mod:`powermean.results.handoff`) carries a prepared frame and a
sample derived from it, as two records: the sample holds the retained rows with their
positions and accounting, the prepared frame the owned frame and the order-neutral base
of the construction (:class:`~powermean.data.sample.SampleBase`), with the raw inputs of
every input row.  Neither record names the other, and the retained rows of the frame's
design and outcome are not enough to tell them apart: two preparations of the same
regressors and outcome under another offset, another exposure, other weights, another
subset or other classifications give samples whose design and outcome agree row for row
while their equation, their counts or their covariances differ.  So :func:`check_sample`
compares the sample with the owned inputs of the preparation itself, by position and
without fitting, selecting or training anything:

- the counts: the sample has as many input rows as the frame and the base;
- the estimator: the mean order and the route of the sample are the estimator's, and
  those of its transformed problem, whose selection is every retained row;
- the frame: the columns of the sample are those of the fitted design, and the
  retained rows of the evaluated design and outcome are the sample's numeric record;
- the accounting: the code of every input row is the base's, except where the order
  removed a zero outcome from a row that the base had kept complete, and that row's
  raw outcome is zero; the conditioning flag is that of the codes, under
  ``positive_only``; the retained positions are the rows of code zero, in order, with
  their inverse map and the positional labels of the numeric record; the masks of
  step 2 are the base's, group by group;
- the values: the raw outcome, the trained design, the offset or the exposure (the
  raw column, before any logarithm or order multiplier) and the weights, with their
  kind, of the retained rows are the base's at the retained positions, as the base
  holds them (frequency counts as the exact objects the base holds, compared with the
  exact counts of the record and never after a conversion to doubles, which rounds a
  foreign ``2^53 + 1`` to the ``2^53`` that the record holds), and the columns that
  the preparation read agree with what the base holds (weights, offset, exposure, the
  requested cluster);
- the classifications: the metadata retained with the sample are the base's columns,
  by name and in order, with the same type and, at every retained row, the same
  value, a missing marker of the same kind included, without conversion;
- the scaled design: the retained design over its column scale, bit for bit.

Each check refuses with ``ValueError`` and the reason; nothing is modified.
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np

from powermean._prepare.compose import Columns, PreparedFrame
from powermean.data.accounting import RETAINED, ZERO_OUTCOME
from powermean.data.raw import AlignedColumns, Column, RawInputs
from powermean.data.retained import OrderSample
from powermean.data.sample import SampleBase
from powermean.data.specs import Estimator
from powermean.formula.frame import ModelFrame
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.results.covariance import _missing


def refuse(detail: str) -> ValueError:
    """The refusal of a handoff whose records are not bound to one another."""
    return ValueError(f"the handoff is not bound to its inputs: {detail}")


def _check_order(estimator: Estimator, sample: OrderSample) -> None:
    """The order and route of the sample, and of its problem, are the estimator's."""
    if estimator.mean.omega != sample.omega or estimator.route != sample.problem.route:
        raise refuse("the estimator's mean order is not the sample's")
    problem = sample.problem
    if problem.omega != sample.omega or problem.selection.nobs != sample.count:
        raise refuse("the transformed problem is not that of the sample's rows")
    if problem.selection.removed.size or not np.array_equal(
        problem.selection.positions, np.arange(sample.count)
    ):
        raise refuse("the transformed problem does not retain every row of the sample")


def _check_frame(sample: OrderSample, frame: ModelFrame) -> None:
    """The retained rows of the frame's design and outcome are the sample's record."""
    if sample.nobs != frame.outcome.size:
        raise refuse("the sample has another number of input rows than the frame")
    design = frame.design
    if sample.data.column_names != design.columns:
        raise refuse("the columns of the sample are not those of the fitted design")
    matrix = design.matrix.to_numpy(dtype=np.float64, na_value=np.nan)
    outcome = frame.outcome.to_numpy(dtype=np.float64, na_value=np.nan)
    rows = sample.positions
    if not np.array_equal(matrix[rows], sample.data.design) or not np.array_equal(
        outcome[rows], sample.data.y
    ):
        raise refuse("the retained rows are not those of the frame")


def _check_accounting(base: SampleBase, sample: OrderSample) -> None:
    """The codes, positions, inverse map, labels and masks are the preparation's."""
    if sample.nobs != base.nobs:
        raise refuse("the sample has another number of input rows than the preparation")
    codes, given = sample.codes, base.codes
    removed = codes == ZERO_OUTCOME
    if not np.array_equal(codes[~removed], given[~removed]):
        raise refuse("the exclusions of the sample are not those of the preparation")
    if not base.complete[removed].all() or np.any(base.inputs.y[removed] != 0):
        raise refuse("a zero outcome was removed from a row the preparation excluded")
    conditioned = bool(removed.any())
    if sample.conditioned != conditioned or (conditioned and not sample.positive_only):
        raise refuse("the conditioning of the sample is not that of its accounting")
    positions = np.flatnonzero(codes == RETAINED)
    inverse = np.full(sample.nobs, -1, dtype=np.intp)
    inverse[positions] = np.arange(positions.size, dtype=np.intp)
    if (
        not np.array_equal(positions, sample.positions)
        or not np.array_equal(inverse, sample.retained_index)
        or sample.data.row_labels != tuple(positions.tolist())
    ):
        raise refuse("the retained positions are not those of the accounting")
    groups, account = sample.missing.groups, base.missing
    if tuple(groups) != tuple(account.groups) or not all(
        np.array_equal(mask, account.groups[group]) for group, mask in groups.items()
    ):
        raise refuse("the missing-value masks are not those of the preparation")
    if not np.array_equal(sample.missing.excluded, account.excluded):
        raise refuse("the rows excluded as missing are not those of the preparation")


def _doubles(values: Any, rows: IndexArray) -> FloatArray:
    """The raw weights of the retained rows as doubles, as the record forms them.

    ``values`` is the owned column of the raw weights, numeric or of objects.
    """
    return np.asarray(values[rows], dtype=np.float64)


def _is_count(original: object, count: int) -> bool:
    """Whether a raw weight is the exact count ``count``, by its own comparison.

    The raw weight is the object the preparation holds (a Python or NumPy integer, a
    double, a fraction); a real number that is not a Boolean, as the count check of
    :class:`~powermean.data.weights.ObservationWeights` has it, compared with the
    Python integer exactly and never as a double.
    """
    if isinstance(original, bool) or not isinstance(original, numbers.Real):
        return False
    return bool(original == count)


def _same_counts(values: Any, rows: IndexArray, counts: FloatArray) -> bool:
    """Whether the raw frequency counts of the retained rows are the record's, exactly.

    ``counts`` are the record's weights: integer-valued doubles of at most ``2^53``,
    each checked as an exact count before its conversion, so each is the Python
    integer it stands for.  A raw count that rounds to it is not it: ``2^53 + 1``
    rounds to the double ``2^53`` and would otherwise bind to the record of another
    preparation, whose own count check refuses it.
    """
    originals = values[rows].tolist()
    return all(
        _is_count(original, int(count))
        for original, count in zip(originals, counts.tolist(), strict=True)
    )


def _check_values(inputs: RawInputs, columns: Columns, sample: OrderSample) -> None:
    """The outcome, design, offset or exposure and weights are the preparation's."""
    rows, data = sample.positions, sample.data
    if not np.array_equal(inputs.y[rows], data.y) or not np.array_equal(
        inputs.design[rows], data.design
    ):
        raise refuse("the numeric record of the sample is not that of the preparation")
    if (columns.offset is None) != (inputs.offset is None) or (
        columns.exposure is None
    ) != (inputs.exposure is None):
        raise refuse("the offset or exposure read is not the one the preparation holds")
    exposure = data.exposure
    if inputs.exposure is not None:
        if exposure is None or not np.array_equal(inputs.exposure[rows], exposure):
            raise refuse("the exposure of the sample is not that of the preparation")
    else:
        if exposure is not None:
            raise refuse("the sample holds an exposure that the preparation did not")
        offset = np.zeros(rows.size) if inputs.offset is None else inputs.offset[rows]
        if not np.array_equal(offset, data.offset):
            raise refuse("the offset of the sample is not that of the preparation")
    weights = data.weights
    if (columns.weights is None) != (weights is None) or (inputs.weights is None) != (
        weights is None
    ):
        raise refuse("the weights of the sample are not those of the columns read")
    if weights is not None and inputs.weights is not None:
        if weights.kind != columns.weight_kind or weights.kind != inputs.weights.kind:
            raise refuse("the kind of the weights is not the one the preparation read")
        if not np.array_equal(_doubles(inputs.weights.values, rows), weights.values):
            raise refuse("the weights of the sample are not those of the preparation")
        if weights.frequency and not _same_counts(
            inputs.weights.values, rows, weights.values
        ):
            raise refuse(
                "the frequency counts of the sample are not those of the preparation"
            )


def _same_element(left: object, right: object) -> bool:
    """Whether two values of a classification are the same, exactly.

    The same type; a missing marker of that type on both sides; else equal by the
    values' own comparison, which must answer with a Boolean.
    """
    if type(left) is not type(right):
        return False
    if _missing(left):
        return _missing(right)
    equal = left == right
    if not isinstance(equal, bool | np.bool_):
        raise TypeError(f"a value of type {type(left).__name__} cannot be compared")
    return bool(equal)


def _same_column(given: Column, retained: Column) -> bool:
    """Whether ``retained`` holds the values of ``given``, type and markers included."""
    left, right = given.values, retained.values
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype.kind != "O":
        return bool(left.tobytes() == right.tobytes())
    return all(
        _same_element(x, y) for x, y in zip(left.tolist(), right.tolist(), strict=True)
    )


def _check_metadata(
    inputs: RawInputs, columns: Columns, sample: OrderSample, rows: IndexArray
) -> None:
    """The classifications retained with the sample are the preparation's columns."""
    given: AlignedColumns = inputs.metadata
    retained = sample.metadata
    if tuple(retained) != tuple(given) or retained.rows != rows.size:
        raise refuse("the classifications of the sample are not the preparation's")
    if columns.cluster is not None and columns.cluster not in retained:
        raise refuse("the requested cluster column was not retained with the sample")
    if any(name not in retained for name in columns.metadata):
        raise refuse("a metadata column read was not retained with the sample")
    for name, column in given.columns.items():
        if not _same_column(column.take(rows), retained.columns[name]):
            raise refuse(f"the classification {name!r} is not the preparation's")


def _check_scaled_design(sample: OrderSample) -> None:
    """The design of the solver is the retained design over its scale, bit for bit."""
    with np.errstate(under="ignore"):
        scaled: FloatArray = sample.data.design / sample.design.scale
    if not np.array_equal(scaled, sample.design.matrix):
        raise refuse("the scaled design is not the retained design over its scale")


def check_sample(
    estimator: Estimator,
    prepared: PreparedFrame,
    sample: OrderSample,
    frame: ModelFrame,
) -> None:
    """The sample is the frame's, the preparation's and the estimator's.

    The checks of the module docstring, in its order.

    Parameters
    ----------
    estimator : Estimator
        The estimator record of the handoff.
    prepared : PreparedFrame
        The prepared frame, read for its columns and its base; not retained.
    sample : OrderSample
        The sample of the mean order.
    frame : ModelFrame
        The owned copy of the prepared frame handed out for this binding.

    Raises
    ------
    ValueError
        A sample that is not bound to the frame, the preparation or the estimator.
    TypeError
        A value of a classification that cannot be compared.
    """
    _check_order(estimator, sample)
    _check_frame(sample, frame)
    base = prepared.base
    _check_accounting(base, sample)
    _check_values(base.inputs, prepared.columns, sample)
    _check_metadata(base.inputs, prepared.columns, sample, sample.positions)
    _check_scaled_design(sample)
