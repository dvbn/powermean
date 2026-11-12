"""Fresh primal and dual witnesses of a support diagnostic on its current sample.

Internal module: ``powermean`` exports nothing of it.  The sample record, its
binding and its index storage are :mod:`powermean.solve.support_problem`.

Primal witness.  With ``P = {Z > 0}`` and ``O = {Z = 0}``, ``z = X beta + sum_b
a_b[c_b]`` is reconstructed from the proposed coefficients on the actual rows, the
columns in their order and then the lift in block order, every operation range-checked
(:mod:`powermean.fe.absorb_numeric`): a nonzero product rounded to zero refuses rather
than manufacturing an equality.  A nonzero ``z`` is normalized by dividing the
coefficients by ``max |z|`` and reconstructing again; those rows are authoritative.
With the method's tolerance ``e`` and threshold ``d`` it is valid when ``max_P |z| <=
e``, ``min_O z >= -e`` and ``S = {i in O: z_i > d}`` is nonempty.  Equality at ``d`` is
not strict, nothing is clipped, and a valid witness proves its strict set only.  Every
array a witness keeps, the proposed coefficients and a null direction's included, is an
owned read-only copy.

Dual witness.  A proposed ``l`` is constructed by setting its entries on the witnessed
set ``S`` to zero, no other entry edited, normalized by its largest absolute zero-row
entry and checked afresh: ``l_i > d`` on every zero row outside ``S``, and for every
slope column and every level of every block ``|sum b_i l_i| / sum |b_i l_i| <= e``, an
exact zero denominator requiring an exact zero numerator.  With primal witnesses for
``S`` it is a numerical completeness diagnostic, not an exact theorem; a
nonfinite or range-lost quantity fails it.  A required ratio is a range-checked
quotient (:func:`relative`): a nonzero numerator whose quotient would underflow, to zero
or to an inexact subnormal, is range-lost rather than an exact zero ratio.

Binding.  A record states the sample, the method and the ``e`` and ``d`` it was checked
with; its flags were computed then.  :func:`require_primal` and :func:`require_dual`
bind a record to the current sample, the witnessed set and the active controls and
repeat its check afresh there, so that no stored flag stands for a fresh check.  A
primal record is checked again from its proposed coefficients, by the same
reconstruction, division and reconstruction, and must be that check's record bit for
bit: its divided coefficients are neither trusted nor divided again, so that rescaled
coefficients cannot evade the division by ``max |raw|``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike

from powermean.fe.absorb_numeric import (
    checked_product,
    checked_quotient,
    checked_sum,
    group_sums,
    lift,
)
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.support_problem import (
    NONE,
    SupportProblem,
    indices,
    require_same,
)

Effects = tuple[FloatArray, ...]

SHAPE = "coefficients must match the slopes and the levels of every block"
CONTROLS = "the evidence was checked by another method or under other controls"


def _none() -> IndexArray:
    """The shared read-only empty index array, the default of an empty set."""
    return NONE


# ------------------------------------------------------------------ reconstruction


def coefficients(
    problem: SupportProblem, slopes: ArrayLike, effects: Sequence[ArrayLike]
) -> tuple[FloatArray, Effects]:
    """Owned real copies of slope and block coefficients; ``ValueError`` on a shape."""
    beta: FloatArray = np.array(slopes, dtype=np.float64, copy=True)
    blocks = () if problem.grouping is None else problem.grouping.blocks
    owned = tuple(np.array(a, dtype=np.float64, copy=True) for a in effects)
    if beta.shape != (problem.n_slopes,) or len(owned) != len(blocks):
        raise ValueError(SHAPE)
    if any(a.shape != (b.n_levels,) for a, b in zip(owned, blocks, strict=True)):
        raise ValueError(SHAPE)
    return beta, owned


def reconstruct(
    problem: SupportProblem, slopes: FloatArray, effects: Effects
) -> FloatArray:
    """``X beta + sum_b a_b[c_b]`` on the current rows, every operation range-checked.

    Columns are accumulated in their order, then the lift in block order.
    ``FloatingPointError`` for a coefficient that is not finite, or a product, sum or
    lift that is not representable, a nonzero product underflowing included.
    """
    if not np.isfinite(slopes).all() or not all(np.isfinite(a).all() for a in effects):
        raise FloatingPointError("a witness coefficient is not finite")
    z: FloatArray = np.zeros(problem.n_rows)
    for j in range(problem.n_slopes):
        term = checked_product(problem.design[:, j], slopes[j], "witness product")
        z = checked_sum(z, term, "witness sum")
    if problem.grouping is not None:
        lifted = lift(tuple(a[:, None] for a in effects), problem.grouping)
        z = checked_sum(z, lifted[:, 0], "witness lift")
    return z


# ------------------------------------------------------------------ primal witness


@dataclass(frozen=True, eq=False)
class PrimalWitness:
    """A fresh primal check of one cone direction on the current rows.

    ``proposed_slopes`` and ``proposed_effects`` are the coefficients as proposed;
    ``raw`` and ``scale`` their first reconstruction and ``max |raw|``; ``slopes``,
    ``effects`` and ``z`` the normalized coefficients and their reconstruction, the
    authoritative rows.  ``positive_max`` is ``max_P |z|`` (0 without positive rows),
    ``zero_min`` is ``min_O z``; ``strict`` the zero rows above ``threshold`` and
    ``unresolved`` those in ``(tolerance, threshold]``.  ``failure`` names the failed
    predicates: ``range``, ``null``, ``positive``, ``negative`` or ``no_strict``.
    """

    problem: SupportProblem
    method: str
    tolerance: float
    threshold: float
    failure: tuple[str, ...]
    proposed_slopes: FloatArray | None = None
    proposed_effects: Effects | None = None
    raw: FloatArray | None = None
    scale: float | None = None
    slopes: FloatArray | None = None
    effects: Effects | None = None
    z: FloatArray | None = None
    positive_max: float | None = None
    zero_min: float | None = None
    strict: IndexArray = field(default_factory=_none)
    unresolved: IndexArray = field(default_factory=_none)

    @property
    def valid(self) -> bool:
        """Whether every predicate holds: a witness of its strict set."""
        return not self.failure

    @property
    def positions(self) -> IndexArray:
        """The original positions of the strict rows."""
        return indices(self.problem.positions[self.strict])


def _normalized(
    problem: SupportProblem, beta: FloatArray, a: Effects
) -> tuple[FloatArray, float, FloatArray, Effects, FloatArray]:
    """The first rows, their scale, the divided coefficients and the authoritative rows.

    Nothing is divided at a zero scale.  ``FloatingPointError`` on a range failure.
    """
    raw = reconstruct(problem, beta, a)
    scale = float(np.max(np.abs(raw)))
    if scale == 0:
        return raw, scale, beta, a, raw
    divisor = np.array([scale])
    beta = checked_quotient(beta, divisor, "normalized slopes")
    a = tuple(checked_quotient(x, divisor, "normalized effects") for x in a)
    return raw, scale, beta, a, reconstruct(problem, beta, a)


def _predicates(
    problem: SupportProblem, z: FloatArray, tolerance: float, threshold: float
) -> tuple[tuple[str, ...], float, float, IndexArray]:
    """The failed predicates, ``max_P |z|``, ``min_O z`` and the strict rows."""
    zero, positive = problem.zero, problem.positive
    positive_max = float(np.max(np.abs(z[positive]))) if positive.size else 0.0
    zero_min = float(np.min(z[zero])) if zero.size else 0.0
    strict = indices(zero[z[zero] > threshold])
    checks = (
        ("positive", positive_max > tolerance),
        ("negative", zero_min < -tolerance),
        ("no_strict", not strict.size),
    )
    return tuple(n for n, failed in checks if failed), positive_max, zero_min, strict


def primal_witness(
    problem: SupportProblem,
    slopes: ArrayLike,
    effects: Sequence[ArrayLike],
    *,
    method: str,
    tolerance: float,
    threshold: float,
) -> PrimalWitness:
    """Reconstruct, normalize, reconstruct again and check one direction.

    ``method`` names the step that proposed the coefficients (``fe``, ``lp``, ``ir``);
    ``tolerance`` and ``threshold`` are that method's ``e`` and ``d``.  The proposed
    coefficients are kept with the record, so that the check can be repeated from them.
    """
    beta, a = coefficients(problem, slopes, effects)
    record = PrimalWitness(
        problem,
        method,
        tolerance,
        threshold,
        ("range",),
        proposed_slopes=immutable_array(beta),
        proposed_effects=tuple(immutable_array(x) for x in a),
    )
    try:
        raw, scale, beta, a, z = _normalized(problem, beta, a)
    except FloatingPointError:
        return record
    if scale == 0:
        return replace(record, failure=("null",), raw=immutable_array(raw), scale=0.0)
    failure, positive_max, zero_min, strict = _predicates(
        problem, z, tolerance, threshold
    )
    zero = problem.zero
    return replace(
        record,
        failure=failure,
        raw=immutable_array(raw),
        scale=scale,
        slopes=immutable_array(beta),
        effects=tuple(immutable_array(x) for x in a),
        z=immutable_array(z),
        positive_max=positive_max,
        zero_min=zero_min,
        strict=strict,
        unresolved=indices(zero[(z[zero] > tolerance) & (z[zero] <= threshold)]),
    )


def union(witnesses: Sequence[PrimalWitness]) -> IndexArray:
    """The rows covered by the strict sets of the valid witnesses, increasing."""
    rows = [w.strict for w in witnesses if w.valid]
    return indices(np.unique(np.concatenate(rows))) if rows else NONE


# -------------------------------------------------------------------- dual witness


def relative(magnitudes: FloatArray, denominators: FloatArray) -> FloatArray:
    """The ratios ``|numerator| / denominator`` of a required relative check.

    An exact zero numerator has ratio 0 and a nonzero one over an exact zero
    denominator an infinite, failing ratio.  Otherwise the quotient is range-checked
    (:func:`powermean.fe.absorb_numeric.checked_quotient`): a nonzero numerator whose
    quotient underflows inexactly, to zero or to an inexact subnormal, or overflows
    raises ``FloatingPointError``; it is a range-lost check, never an exact zero.
    """
    ratios = np.where(magnitudes == 0, 0.0, np.inf)
    divided = denominators > 0
    ratios[divided] = checked_quotient(
        magnitudes[divided], denominators[divided], "relative check"
    )
    return ratios


@dataclass(frozen=True, eq=False)
class ColumnSums:
    """``sum b_i l_i`` and ``sum |b_i l_i|`` of a set of columns, and their ratios.

    A ratio is ``|numerator| / denominator`` (:func:`relative`); an exact zero
    denominator has ratio 0 with an exact zero numerator and infinity otherwise.
    """

    numerators: FloatArray
    denominators: FloatArray
    ratios: FloatArray

    def passing(self, tolerance: float) -> bool:
        """Whether every ratio is at most ``tolerance``."""
        return bool((self.ratios <= tolerance).all())


def column_sums(products: FloatArray, codes: IndexArray, count: int) -> ColumnSums:
    """The row-order sums of ``products`` and of their magnitudes, by code, checked.

    ``products`` is a vector or rows by columns (:func:`powermean.fe.absorb_numeric
    .group_sums`).  ``FloatingPointError`` when a sum or a ratio is not representable.
    """
    numerators = group_sums(products, codes, count)
    denominators = group_sums(np.abs(products), codes, count)
    ratios = relative(np.abs(numerators), denominators)
    return ColumnSums(numerators, denominators, immutable_array(ratios))


@dataclass(frozen=True, eq=False)
class DualWitness:
    """A fresh dual check of a proposed ``l`` with its entries on ``separated`` zeroed.

    ``raw`` is the proposal and ``constructed`` the copy with zeros on ``separated``;
    ``normalizer`` is ``max_O |constructed|``, ``normalized`` the divided rows and
    ``margin`` their minimum over the zero rows outside ``separated``.  ``slopes``
    (``None`` without slopes) and ``levels`` (one per block) hold the column sums.
    ``failure`` names the failed checks: ``range``, ``zero``, ``margin``, ``slopes``
    or ``levels``.
    """

    problem: SupportProblem
    tolerance: float
    threshold: float
    raw: FloatArray
    separated: IndexArray
    constructed: FloatArray
    failure: tuple[str, ...]
    normalizer: float | None = None
    normalized: FloatArray | None = None
    margin: float | None = None
    slopes: ColumnSums | None = None
    levels: tuple[ColumnSums, ...] = ()

    @property
    def passed(self) -> bool:
        """Whether every check holds."""
        return not self.failure


def _sums(
    problem: SupportProblem, dual: FloatArray
) -> tuple[ColumnSums | None, tuple[ColumnSums, ...]]:
    """The sums of every slope column (``None`` without slopes) and of every level."""
    slopes = None
    if problem.n_slopes:
        products = checked_product(problem.design, dual[:, None], "dual product")
        found = column_sums(products, indices(np.zeros(problem.n_rows)), 1)
        slopes = ColumnSums(found.numerators[0], found.denominators[0], found.ratios[0])
    blocks = () if problem.grouping is None else problem.grouping.blocks
    return slopes, tuple(column_sums(dual, b.codes, b.n_levels) for b in blocks)


def dual_witness(
    problem: SupportProblem,
    raw: ArrayLike,
    separated: IndexArray,
    *,
    tolerance: float,
    threshold: float,
) -> DualWitness:
    """Construct the zero-on-``separated`` candidate and check it afresh.

    ``ValueError`` when ``raw`` is not one value per row, ``separated`` is not a set of
    zero rows, or every zero row is in ``separated`` (no dual is then needed).
    """
    proposal: FloatArray = np.array(raw, dtype=np.float64, copy=True)
    zero = problem.zero
    remaining = np.setdiff1d(zero, separated)
    if proposal.shape != (problem.n_rows,) or not np.isin(separated, zero).all():
        raise ValueError("a dual needs one value per row and S among the zero rows")
    if not remaining.size:
        raise ValueError("no dual is needed when every zero row is separated")
    constructed = proposal.copy()
    constructed[separated] = 0.0
    record = DualWitness(
        problem,
        tolerance,
        threshold,
        immutable_array(proposal),
        indices(separated),
        immutable_array(constructed),
        ("range",),
    )
    if not np.isfinite(constructed).all():
        return record
    normalizer = float(np.max(np.abs(constructed[zero])))
    if normalizer == 0:
        return replace(record, failure=("zero",), normalizer=0.0)
    try:
        dual = checked_quotient(constructed, np.array([normalizer]), "normalized dual")
        slopes, levels = _sums(problem, dual)
    except FloatingPointError:
        return replace(record, normalizer=normalizer)
    margin = float(np.min(dual[remaining]))
    checks = (
        ("margin", not margin > threshold),
        ("slopes", slopes is not None and not slopes.passing(tolerance)),
        ("levels", not all(s.passing(tolerance) for s in levels)),
    )
    return replace(
        record,
        failure=tuple(name for name, failed in checks if failed),
        normalizer=normalizer,
        normalized=immutable_array(dual),
        margin=margin,
        slopes=slopes,
        levels=levels,
    )


# ----------------------------------------------------------------------- binding


def _bits(values: FloatArray | IndexArray | None) -> tuple[object, ...] | None:
    """The shape and the bytes of an array, ``None`` without one."""
    return None if values is None else (values.shape, values.tobytes())


def _evidence(witness: PrimalWitness) -> tuple[object, ...]:
    """What a primal record states that its proposal determines, compared bit for bit.

    The first rows and their scale, the divided coefficients, the authoritative rows
    and the outcome of every predicate: all but the binding and the proposal itself.
    """
    arrays = (
        witness.raw,
        witness.slopes,
        witness.z,
        witness.strict,
        witness.unresolved,
    )
    effects = None if witness.effects is None else tuple(map(_bits, witness.effects))
    scalars = (witness.scale, witness.positive_max, witness.zero_min)
    return (*map(_bits, arrays), effects, *scalars)


def require_primal(
    witness: PrimalWitness,
    problem: SupportProblem,
    *,
    methods: tuple[str, ...],
    tolerance: float,
    threshold: float,
) -> None:
    """``ValueError`` unless ``witness`` passes its fresh check on ``problem``.

    The witness must be bound to the sample, proposed by one of ``methods`` and
    checked with the active ``e`` and ``d``.  Its retained proposal is checked again on
    the current rows by :func:`primal_witness`, the same reconstruction, division by
    ``max |raw|``, reconstruction and predicates, and the record must be that check's
    in every field the proposal determines, bit for bit, and valid: neither its stored
    flag nor its stored divided coefficients are the check.
    """
    require_same(witness.problem, problem)
    if witness.method not in methods or (witness.tolerance, witness.threshold) != (
        tolerance,
        threshold,
    ):
        raise ValueError(CONTROLS)
    invalid = "a primal witness must be valid under its fresh check on the current rows"
    slopes, effects = witness.proposed_slopes, witness.proposed_effects
    if witness.failure or slopes is None or effects is None:
        raise ValueError(invalid)
    again = primal_witness(
        problem,
        slopes,
        effects,
        method=witness.method,
        tolerance=tolerance,
        threshold=threshold,
    )
    if again.failure or _evidence(again) != _evidence(witness):
        raise ValueError(invalid)


def require_dual(
    dual: DualWitness,
    problem: SupportProblem,
    separated: IndexArray,
    *,
    tolerance: float,
    threshold: float,
) -> None:
    """``ValueError`` unless ``dual`` is bound to ``problem``, ``S`` and the controls.

    The record must be bound to the sample, checked with the witnessed set
    ``separated`` and the active ``e`` and ``d``, and its retained proposal must give
    its recorded verdict when checked afresh: its stored flag is not the check.
    """
    require_same(dual.problem, problem)
    if not np.array_equal(dual.separated, separated):
        raise ValueError("a dual must be checked with the witnessed set of the result")
    if (dual.tolerance, dual.threshold) != (tolerance, threshold):
        raise ValueError(CONTROLS)
    again = dual_witness(
        problem, dual.raw, separated, tolerance=tolerance, threshold=threshold
    )
    if again.failure != dual.failure:
        raise ValueError("a dual must give its recorded verdict when checked afresh")
