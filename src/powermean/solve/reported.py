"""The coefficients of a candidate on the reported scale, and the coefficient gate.

The coefficient correction is measured in units of its standard error and must be at
most ``coef_tol``; zero standard errors follow the gate below.

The coefficient correction of the certificate is the Newton correction ``p = C^-1 S``
of a candidate, mapped to the reported coefficients and compared with their standard
errors.  Both belong to later stages: the standard errors to the covariance of the
inference layer, the reported scale to the order layer and the column
scaling.  A stage does not import a later one, so the caller supplies them, at
the candidate, through a :class:`ReportedScaleProvider`.  Called with the fresh
evaluation that the certificate made at the candidate, it returns a
:class:`ReportedScale`: that candidate, a matrix ``M`` that maps a correction of
``gamma`` to the reported coefficients, and their standard errors at the candidate, on
the reported scale.  The certificate forms the reported correction ``M p`` from its own
correction and refuses a record that names another candidate; it forms no covariance
and no standard error.  Without effects, a caller whose coefficients are
``beta = D^-1 gamma / omega`` (column scale ``D``, mean order ``omega``) supplies
``M = diag(1 / (omega d))`` and the standard errors ``sqrt(diag(M V M'))`` of the
covariance ``V`` of ``gamma``: one map for the correction and for the covariance, so
that their ratio does not depend on the scale.

Two forms of the map.  The generic form is the dense matrix ``M``, which
the certificate applies as the product ``M p``.  The direct form is the accepted
:class:`~powermean.model.order_scale.OrderScale` itself, whose ``coefficients`` maps a
correction entry by entry with exact exponents and whose ``covariance`` and
``standard_errors`` map the covariance the same way: where the column scales differ
by hundreds of octaves the dense ``M`` can be unrepresentable (``omega = d = 2^-600``
gives ``1 / (omega d) = 2^1200``) while every reported quantity is a double, and the
product ``M V M'`` can lose an off-diagonal entry that the direct map keeps.  A
provider supplies one form or the other; the record applies the one it holds, and
``transform`` materializes the dense ``M`` of a direct map on request only, raising
``FloatingPointError`` where an entry is not representable.  A reported correction
that is not representable, or formed from a correction that is not finite, is NaN,
which the gate fails: nothing is floored or replaced by a zero.

The gate.  For each reported coefficient ``j``, with the correction ``c_j`` and
the standard error ``se_j``:

    passed     c_j and se_j finite, se_j >= 0, and |c_j| <= coef_tol * se_j
    ratio      |c_j| / se_j where se_j is finite and positive; undefined (NaN) otherwise

The product form decides, not the ratio: a correction equal to
``coef_tol * se_j`` passes, and at ``se_j = 0`` (``-0.0`` included, which equals zero)
only an exactly zero correction passes.  The ratio of a zero standard error is
undefined, and the record flags inference as degenerate
(:attr:`CoefficientCheck.degenerate`) even when every gate of the certificate passes:
a certified root does not make degenerate inference usable.  A missing (NaN), negative
or infinite standard error does not pass, and neither does a correction that is not
finite.  The finiteness and the sign of ``se_j`` are tested as such, not inferred from
the inequality: at the smallest negative subnormal ``se_j`` and ``coef_tol`` below one,
``coef_tol * se_j`` rounds to ``-0.0``, which a zero correction meets.  Nothing is
floored, and no covariance is substituted.  Where the product ``coef_tol * se_j``
exceeds the largest double it is infinite, and every finite correction is below it,
which is the exact verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.model.order_scale import OrderScale
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_array
from powermean.systems.joint import JointEvaluation

BoolArray = NDArray[np.bool_]


def _real_array(value: ArrayLike, name: str) -> FloatArray:
    """A real array of doubles, which may hold NaN or infinite entries."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    return np.asarray(value, dtype=np.float64)


def _dense_map(transform: ArrayLike, size: int) -> FloatArray:
    """``M``: a finite matrix with a row and with one column per entry of ``gamma``."""
    matrix = _finite_array(transform, "transform")
    if matrix.ndim != 2 or not matrix.shape[0]:
        raise ValueError("transform must be a matrix with a reported coefficient")
    if matrix.shape[1] != size:
        raise ValueError("transform must have one column per entry of gamma")
    return _immutable(matrix)


def _direct_map(order: object, size: int) -> OrderScale:
    """The accepted order scale of the candidate's coordinates, by exact type.

    Typed as ``object`` because it is the run-time check of what a provider passes.
    """
    if type(order) is not OrderScale:
        raise TypeError("order must be an OrderScale")
    if order.dimension != size:
        raise ValueError("order must have one scale per entry of gamma")
    return order


def _direct_corrections(order: OrderScale, correction: FloatArray) -> FloatArray:
    """``p`` mapped by the direct map, NaN where an entry cannot be reported.

    The whole vector is mapped at once where every entry is finite and
    representable; otherwise each finite entry is mapped on its own, in a vector that
    is zero elsewhere, and an entry that is not finite or whose reported value is not
    representable is NaN.  A zero entry stays zero.
    """
    finite = np.isfinite(correction)
    if finite.all():
        try:
            return order.coefficients(correction)
        except FloatingPointError:
            pass
    mapped = np.full(correction.shape, np.nan)
    for position in np.flatnonzero(finite).tolist():
        alone = np.zeros(correction.shape)
        alone[position] = correction[position]
        try:
            mapped[position] = order.coefficients(alone)[position]
        except FloatingPointError:
            mapped[position] = np.nan
    return mapped


@dataclass(frozen=True, eq=False, init=False)
class ReportedScale:
    """The reported coefficients at one candidate: a map of corrections, their SEs.

    Parameters
    ----------
    gamma : array
        The candidate at which the record was formed: the coefficients of the
        evaluation that the provider received.
    transform : array, optional
        The dense matrix ``M``, one row per reported coefficient and one column per
        coefficient of the system: a correction ``p`` of ``gamma`` is ``M p`` on the
        reported scale.  The generic form of the map; not together with ``order``.
    standard_errors : array
        The standard error of each reported coefficient at ``gamma``, on the reported
        scale, as computed: a missing one is NaN, and none is floored or replaced.
    order : OrderScale, optional
        The direct form of the map, the accepted order scale of the candidate's
        coordinates, with one scale per entry of ``gamma``; not together with
        ``transform``.

    Attributes
    ----------
    gamma, standard_errors : array
        As given, owned and read-only.
    order : OrderScale or None
        The direct map, or ``None`` for a dense one.
    transform : array
        The dense ``M`` as given, or that of the direct map, materialized on each
        access, which raises ``FloatingPointError`` where an entry is not
        representable.

    Notes
    -----
    Stored as owned read-only arrays.  ``gamma`` must be a nonempty finite vector;
    with a dense map, ``transform`` must be a finite matrix with a row and with one
    column per entry of ``gamma`` and ``standard_errors`` one real value per row of
    it; with a direct map, ``standard_errors`` one real value per entry of ``gamma``.
    Exactly one of the two maps is given.  Otherwise ``ValueError`` (``TypeError``
    for an ``order`` of another type).
    """

    gamma: FloatArray
    standard_errors: FloatArray
    order: OrderScale | None
    _transform: FloatArray | None

    def __init__(
        self,
        *,
        gamma: FloatArray,
        transform: FloatArray | None = None,
        standard_errors: FloatArray,
        order: OrderScale | None = None,
    ) -> None:
        """Check the shapes and the one map given, and store the arrays read-only."""
        candidate = _finite_array(gamma, "gamma")
        if candidate.ndim != 1 or not candidate.size:
            raise ValueError("gamma must be a nonempty vector")
        if (transform is None) == (order is None):
            raise ValueError("exactly one of transform and order must be given")
        errors = _real_array(standard_errors, "standard_errors")
        dense: FloatArray | None = None
        direct: OrderScale | None = None
        if transform is not None:
            dense = _dense_map(transform, candidate.size)
            if errors.shape != (dense.shape[0],):
                raise ValueError(
                    "standard_errors must have one value per row of transform"
                )
        else:
            direct = _direct_map(order, candidate.size)
            if errors.shape != (candidate.size,):
                raise ValueError(
                    "standard_errors must have one value per entry of gamma"
                )
        object.__setattr__(self, "gamma", _immutable(candidate))
        object.__setattr__(self, "standard_errors", _immutable(errors))
        object.__setattr__(self, "order", direct)
        object.__setattr__(self, "_transform", dense)

    @property
    def transform(self) -> FloatArray:
        """The dense ``M``: as given, or materialized from the direct map."""
        if self._transform is not None:
            return self._transform
        assert self.order is not None  # exactly one map is held
        return self.order.transform

    def corrections(self, correction: FloatArray) -> FloatArray:
        """The reported corrections of ``correction``, by the map that is held.

        Parameters
        ----------
        correction : array
            The Newton correction ``p`` of ``gamma``, which may hold entries that
            are not finite.

        Returns
        -------
        array
            ``M p`` for a dense map, formed without a report of overflow; the direct
            map entry by entry otherwise, NaN where an entry is not finite or its
            reported value is not representable.
        """
        if correction.shape != self.gamma.shape:
            raise ValueError("correction must have one value per entry of gamma")
        if self.order is None:
            with np.errstate(over="ignore", invalid="ignore"):
                reported: FloatArray = self.transform @ correction
            return reported
        return _direct_corrections(self.order, correction)


class ReportedScaleProvider(Protocol):
    """The reported scale of the coefficients at the candidate of an evaluation."""

    def __call__(self, evaluation: JointEvaluation, /) -> ReportedScale:
        """The record at ``evaluation.gamma``, formed from that candidate only."""
        ...


@dataclass(frozen=True, eq=False)
class CoefficientCheck:
    """The coefficient gate, coefficient by coefficient, on the reported scale.

    ``corrections`` are the reported corrections ``M p`` and ``standard_errors`` those
    supplied; ``passed`` says whether each coefficient passes at ``coef_tol``, and
    ``ratios`` holds ``|c_j| / se_j``, NaN where it is undefined.  Read-only arrays.
    """

    corrections: FloatArray
    standard_errors: FloatArray
    coef_tol: float
    passed: BoolArray
    ratios: FloatArray

    @property
    def accepted(self) -> bool:
        """Whether every reported coefficient passes the gate."""
        return bool(self.passed.all())

    @property
    def degenerate(self) -> bool:
        """Whether a standard error is zero: its ratio is undefined."""
        return bool(np.any(self.standard_errors == 0))

    @property
    def largest_ratio(self) -> float:
        """The largest ratio, or NaN when a ratio is undefined."""
        if np.isnan(self.ratios).any():
            return math.nan
        return float(np.max(self.ratios))


def coefficient_check(
    corrections: FloatArray, standard_errors: FloatArray, coef_tol: float
) -> CoefficientCheck:
    """The coefficient gate: ``|c_j| <= coef_tol * se_j`` for finite ``se_j >= 0``.

    Parameters
    ----------
    corrections : array
        The reported corrections ``M p``, one per reported coefficient.
    standard_errors : array
        Their standard errors at the same candidate, aligned.
    coef_tol : float
        The option ``coef_tol``.

    Returns
    -------
    CoefficientCheck
        The verdict and the ratio of each coefficient.
    """
    if corrections.ndim != 1 or corrections.shape != standard_errors.shape:
        raise ValueError("corrections and standard errors must be aligned vectors")
    with np.errstate(over="ignore", invalid="ignore"):
        # The gate admits finite nonnegative standard errors only, tested as such: the
        # inequality does not exclude a negative one, since coef_tol below one times
        # the smallest negative subnormal rounds to -0.0, which a zero correction meets.
        valid = np.isfinite(standard_errors) & (standard_errors >= 0)
        eligible = valid & np.isfinite(corrections)
        size = np.abs(corrections)
        bound = coef_tol * standard_errors
        passed = eligible & (size <= bound)
        positive = np.isfinite(standard_errors) & (standard_errors > 0)
        ratios = np.full(corrections.shape, np.nan)
        ratios[positive] = size[positive] / standard_errors[positive]
    return CoefficientCheck(
        corrections=_immutable(corrections),
        standard_errors=_immutable(standard_errors),
        coef_tol=coef_tol,
        passed=_frozen_flags(passed),
        ratios=_immutable(ratios),
    )


def _frozen_flags(flags: BoolArray) -> BoolArray:
    """An independent read-only copy of a boolean array, in its shape.

    As :func:`powermean.model.rows._immutable` stores doubles: the copy holds bytes of
    its own, whose write flag cannot be set again.
    """
    stored = np.frombuffer(flags.tobytes(), dtype=np.bool_)
    copy: BoolArray = stored.reshape(flags.shape)
    return copy
