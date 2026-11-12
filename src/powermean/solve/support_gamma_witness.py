"""Fresh checks of the GPML recession programs: original-proposal witnesses and duals.

Internal module.  Everything is recomputed from the
bound current sample (:func:`powermean.solve.support_gamma_lp.prepare`), never taken
from the optimizer's sparse product; every product, sum and quotient is range-checked,
so that an unrepresentable quantity fails as ``range`` instead of manufacturing an
equality.

Primal witness (:func:`primal_witness`).  After the raw check
(:func:`powermean.solve.support_gamma_lp.raw_check`), from the OWNED ORIGINAL boxed
``v``: the mapped coefficients ``beta = v_X / s`` and ``alpha_b = v_b``; the raw rows
``X beta + lift(alpha)`` (columns in order, then the lift in block order);
``m = max |raw|`` (``null`` at ``m = 0``: a null direction is no strict witness); the
coefficients divided by ``m``; the authoritative rows reconstructed AGAIN from them on
every original row; ``sigma = sum rho_i h_i``, ``min_P h`` and ``max_P h``.  LP1 gives a
``negative_sum_recession`` when ``min_P h >= -e`` (vacuous without positive rows) and
``sigma < -d``; LP2 a ``zero_sum_recession`` when ``min_P h >= -e``, ``|sigma| <= e``
and ``max_P h > d``.  Failures: ``range``, ``null``, ``positive``, ``total``,
``strict``.  Equality at ``d`` is not strict; nothing is clipped, floored or divided
twice, and ``max |h| = 1`` is not required.

Dual check (:func:`dual_check`).  ``lambda = -m_I``, ``nu = m_E`` (LP2; 0 in LP1),
``l = m_L``, ``u = m_U``: SciPy's ``c = A_P' lambda + t nu + l + u``.  Exact signs
``lambda >= 0``, ``l >= 0``, ``u <= 0`` (``sign``); ``b = -sum l + sum u``;
``|c'v - b|`` (``gap``) and ``|b|`` (``bound``) at most ``e sum |c_j|``.  On every
original slope and every level of every block, from ``X``, the codes, ``rho`` and
``rhoP``: ``c^o`` (``X' rho`` or ``-X_P' rhoP``), the cone term ``X_P' lambda``, the
equality term ``nu X' rho`` and the box terms ``s l`` and ``s u`` (unit for levels).
The cone identity ``c^o - cone - eq`` and the stationarity identity (also minus the box
terms) must each be at most ``e`` times the sum of the absolute values of every
expanded per-row product and scalar term of the component (``cone``,
``stationarity``), with the exact-zero branches of
:func:`~powermean.solve.support_witness.relative`.  Declared predicates, not a
rounding bound.

Replay.  A record is bound to its sample, controls and program, and is evidence only
when the same function on the current inputs, from its retained boxed point, reported
objective and marginals, gives the same record in every field: arrays bit for bit,
floats by their bytes (:meth:`GammaWitness.replay`, :meth:`GammaDual.replay`).  A fresh
replay of the same bytes does not make a caller's array the record's own: every record
(:class:`GammaWitness`, :class:`GammaDual` with its :class:`Multipliers` and
:class:`Identities`, whose two column sums are copied too) keeps owned read-only copies
of each array it retains, block tuples included, made at construction, so a direct
construction or ``dataclasses.replace`` admits no array that its caller can still
write or make writable (:func:`powermean.solve.support_gamma_lp.own`).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from powermean.fe.absorb_numeric import (
    checked_difference,
    checked_product,
    checked_quotient,
    checked_sum,
)
from powermean.fe.groups import Block
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray
from powermean.solve.support_gamma_lp import (
    Effects,
    LPData,
    Marginal,
    RawCheck,
    Stage,
    dot,
    own,
    prepare,
    ratio,
    raw_check,
    rows,
    single,
    split,
    sums,
    total,
)
from powermean.solve.support_gamma_problem import GammaProblem
from powermean.solve.support_lp import LPControls
from powermean.solve.support_witness import ColumnSums, relative

Kind = Literal["negative_sum_recession", "zero_sum_recession"]
KINDS: dict[Stage, Kind] = {
    "negative": "negative_sum_recession",
    "zero_sum": "zero_sum_recession",
}
BINDING = "the evidence was checked under other controls or for another program"
REPLAY = "the evidence is not the fresh check of its proposal on the current sample"
INVALID = "the evidence is not a valid strict recession witness"


def _proposal(
    problem: GammaProblem, boxed: ArrayLike, reported: float
) -> tuple[FloatArray, float]:
    """An owned read-only ``v`` and the reported objective; ``ValueError`` if bad."""
    v = immutable_array(boxed)
    k = problem.n_parameters
    if not k or v.shape != (k,) or not np.isfinite(v).all():
        raise ValueError("a proposal needs one finite coordinate per parameter")
    if isinstance(reported, bool) or not isinstance(reported, float | int):
        raise ValueError("the reported objective must be a finite number")
    if not np.isfinite(reported):
        raise ValueError("the reported objective must be a finite number")
    return v, float(reported)


def _bits(value: object) -> object:
    """Arrays by shape and bytes, floats by bytes, records and tuples field by field."""
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, float):
        return np.float64(value).tobytes()
    if isinstance(value, tuple):
        return tuple(_bits(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(_bits(getattr(value, f.name)) for f in fields(value))
    return value


def _evidence(record: GammaWitness | GammaDual) -> tuple[object, ...]:
    """Every field but the bound sample and controls, compared bit for bit."""
    names = [f.name for f in fields(record) if f.name not in ("problem", "controls")]
    return tuple(_bits(getattr(record, name)) for name in names)


def _bound(
    record: GammaWitness | GammaDual,
    problem: GammaProblem,
    controls: LPControls,
    stage: Stage,
) -> None:
    """``ValueError`` unless the record's sample, controls and program are these."""
    record.problem.require_same(problem)
    if record.controls != controls or record.stage != stage:
        raise ValueError(BINDING)


# ---------------------------------------------------------------- primal witness


@dataclass(frozen=True, eq=False)
class GammaWitness:
    """The fresh primal check of one returned point of a program, from its proposal.

    ``boxed`` and ``reported`` are the optimizer's ``v`` and ``fun`` as returned (owned,
    read-only); ``check`` the raw check; ``mapped_slopes`` and ``mapped_effects`` the
    original mapped proposal; ``raw`` and ``scale`` its rows and ``max |raw|``;
    ``slopes``, ``effects`` and ``h`` the divided coefficients and the authoritative
    rows; ``total`` is ``sigma``; ``positive_min`` and ``positive_max`` the extrema of
    ``h`` on ``P`` (``None`` without positive rows).  ``failure`` is empty for a strict
    witness.
    """

    problem: GammaProblem
    controls: LPControls
    stage: Stage
    boxed: FloatArray
    reported: float
    failure: tuple[str, ...]
    check: RawCheck | None = None
    mapped_slopes: FloatArray | None = None
    mapped_effects: Effects | None = None
    raw: FloatArray | None = None
    scale: float | None = None
    slopes: FloatArray | None = None
    effects: Effects | None = None
    h: FloatArray | None = None
    total: float | None = None
    positive_min: float | None = None
    positive_max: float | None = None

    def __post_init__(self) -> None:
        """Own every array and every block's arrays, null and range records included."""
        arrays = ("boxed", "mapped_slopes", "raw", "slopes", "h")
        own(self, arrays, ("mapped_effects", "effects"))
        object.__setattr__(self, "failure", tuple(self.failure))

    @property
    def valid(self) -> bool:
        """Whether every predicate holds: a strict recession witness."""
        return not self.failure

    @property
    def kind(self) -> Kind | None:
        """The recession this valid witness establishes, ``None`` otherwise."""
        return KINDS[self.stage] if self.valid else None

    def replay(
        self, problem: GammaProblem, *, controls: LPControls, stage: Stage
    ) -> None:
        """``ValueError`` unless this is its proposal's fresh check on ``problem``."""
        _bound(self, problem, controls, stage)
        again = primal_witness(problem, controls, stage, self.boxed, self.reported)
        if _evidence(again) != _evidence(self):
            raise ValueError(REPLAY)

    def require(
        self, problem: GammaProblem, *, controls: LPControls, stage: Stage
    ) -> None:
        """:meth:`replay`, and ``ValueError`` unless the witness is valid."""
        self.replay(problem, controls=controls, stage=stage)
        if not self.valid:
            raise ValueError(INVALID)


def _owned(effects: Effects) -> Effects:
    """Owned read-only copies of block coefficients."""
    return tuple(immutable_array(a) for a in effects)


def _chain(record: GammaWitness, data: LPData) -> GammaWitness:
    """Map, reconstruct, divide by ``max |raw|``, reconstruct again, then predicates."""
    problem, found = record.problem, record
    x, grouping = problem.design, problem.grouping
    try:
        slopes, effects = split(problem, record.boxed)
        beta = checked_quotient(slopes, data.scales, "mapped slopes")
        found = replace(found, mapped_slopes=immutable_array(beta))
        found = replace(found, mapped_effects=_owned(effects))
        raw = rows(x, grouping, beta, effects)
        scale = float(np.max(np.abs(raw)))
        found = replace(found, raw=immutable_array(raw), scale=scale)
        if scale == 0:
            return replace(found, failure=("null",))
        divisor = np.array([scale])
        beta = checked_quotient(beta, divisor, "normalized slopes")
        effects = tuple(
            checked_quotient(a, divisor, "normalized effects") for a in effects
        )
        found = replace(found, slopes=immutable_array(beta), effects=_owned(effects))
        h = rows(x, grouping, beta, effects)
        found = replace(found, h=immutable_array(h))
        sigma = dot(data.weights, h)
    except FloatingPointError:
        return replace(found, failure=("range",))
    return _predicates(found, h, sigma)


def _predicates(record: GammaWitness, h: FloatArray, sigma: float) -> GammaWitness:
    """The program's strict predicates on the authoritative rows ``h``."""
    positive = record.problem.positive
    e, d = record.controls.tolerance, record.controls.threshold
    least = float(np.min(h[positive])) if positive.size else None
    most = float(np.max(h[positive])) if positive.size else None
    failed = ["positive"] if least is not None and not least >= -e else []
    if record.stage == "negative":
        failed += [] if sigma < -d else ["total"]
    else:
        failed += [] if abs(sigma) <= e else ["total"]
        failed += [] if most is not None and most > d else ["strict"]
    found = replace(record, total=sigma, positive_min=least, positive_max=most)
    return replace(found, failure=tuple(failed))


def primal_witness(
    problem: GammaProblem,
    controls: LPControls,
    stage: Stage,
    boxed: ArrayLike,
    reported: float,
) -> GammaWitness:
    """The fresh primal check of ``v`` (``boxed``) returned by ``stage``, ``fun``."""
    v, fun = _proposal(problem, boxed, reported)
    record = GammaWitness(problem, controls, stage, v, fun, ("range",))
    try:
        data = prepare(problem)
    except FloatingPointError:
        return record
    check = raw_check(problem, data, stage, (v, fun), controls)
    if check.failure:
        return replace(record, failure=(check.failure,), check=check)
    return _chain(replace(record, check=check), data)


# -------------------------------------------------------------------------- dual


@dataclass(frozen=True, eq=False)
class Identities:
    """The box terms and both identities of one set of columns, with their ratios.

    ``lower`` and ``upper`` are ``s l`` and ``s u`` (``l`` and ``u`` for levels);
    ``cone`` holds ``c^o - cone - eq`` and ``stationarity`` that minus the box terms,
    each with its absolute-term scale and ratio.
    """

    lower: FloatArray
    upper: FloatArray
    cone: ColumnSums
    stationarity: ColumnSums

    def __post_init__(self) -> None:
        """Own the box terms and both identities' arrays, in copies of their sums."""
        own(self, ("lower", "upper"))
        for name in ("cone", "stationarity"):
            found: ColumnSums = getattr(self, name)
            parts = (found.numerators, found.denominators, found.ratios)
            owned = ColumnSums(*(immutable_array(a) for a in parts))
            object.__setattr__(self, name, owned)


@dataclass(frozen=True, eq=False)
class Multipliers:
    """``lambda`` (the rows of ``P``), ``nu`` and the bound marginals ``l``, ``u``."""

    cone: FloatArray
    nu: float
    lower: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        """Own the three multiplier arrays."""
        own(self, ("cone", "lower", "upper"))


Pair = tuple[FloatArray, FloatArray]


def _pair(values: FloatArray, codes: IndexArray, count: int) -> Pair:
    """The row-order sums of ``values`` by code and of their magnitudes."""
    return sums(values, codes, count), sums(np.abs(values), codes, count)


def _sums(numerators: FloatArray, scale: FloatArray) -> ColumnSums:
    """A residual with its scale and their range-checked ratios."""
    ratios = relative(np.abs(numerators), scale)
    return ColumnSums(immutable_array(numerators), immutable_array(scale), ratios)


def _identities(c: Pair, cone: Pair, total_sums: Pair, m: Multipliers) -> Pair:
    """``c^o - cone - eq`` and its absolute-term scale, ``eq = nu X' rho``."""
    nu = np.array([m.nu])
    eq = checked_product(total_sums[0], nu, "equality term")
    eq_scale = checked_product(total_sums[1], np.abs(nu), "equality term")
    residual = checked_difference(checked_difference(c[0], cone[0]), eq)
    return residual, checked_sum(checked_sum(c[1], cone[1]), eq_scale)


def _with_box(cone: Pair, box: Pair) -> Identities:
    """Both identities of one set of columns from the cone residual and box terms."""
    lower, upper = box
    full = checked_difference(checked_difference(cone[0], lower), upper)
    whole = checked_sum(checked_sum(cone[1], np.abs(lower)), np.abs(upper))
    owned = (immutable_array(lower), immutable_array(upper))
    return Identities(*owned, _sums(*cone), _sums(full, whole))


def _slopes(
    problem: GammaProblem, data: LPData, stage: Stage, m: Multipliers
) -> Identities:
    """The identities of every original slope (box terms scaled by ``s``)."""
    p, positive = problem.n_slopes, problem.positive
    x, xp = problem.design, problem.design[positive]
    every, on_p = single(problem.n_rows), single(positive.size)
    t = _pair(checked_product(x, data.weights[:, None]), every, 1)
    negative = checked_product(xp, -data.positive_weights[:, None])
    c = t if stage == "negative" else _pair(negative, on_p, 1)
    cone = _pair(checked_product(xp, m.cone[:, None]), on_p, 1)
    first = (c[0][0], c[1][0]), (cone[0][0], cone[1][0]), (t[0][0], t[1][0])
    lower = checked_product(data.scales, m.lower[:p], "box term")
    upper = checked_product(data.scales, m.upper[:p], "box term")
    return _with_box(_identities(*first, m), (lower, upper))


def _levels(
    problem: GammaProblem, data: LPData, stage: Stage, m: Multipliers, start: int
) -> list[Identities]:
    """The identities of every level of every block (unit box scale), block order."""
    found = []
    for block in () if problem.grouping is None else problem.grouping.blocks:
        found.append(_block(problem, data, stage, m, (block, start)))
        start += block.n_levels
    return found


def _block(
    problem: GammaProblem,
    data: LPData,
    stage: Stage,
    m: Multipliers,
    where: tuple[Block, int],
) -> Identities:
    """The identities of every level of one block starting at column ``start``."""
    block, start = where
    codes, count = block.codes, block.n_levels
    on_p = codes[problem.positive]
    t = _pair(data.weights, codes, count)
    c = t if stage == "negative" else _pair(-data.positive_weights, on_p, count)
    cone = _identities(c, _pair(m.cone, on_p, count), t, m)
    stop = start + count
    return _with_box(cone, (m.lower[start:stop], m.upper[start:stop]))


@dataclass(frozen=True, eq=False)
class GammaDual:
    """The fresh dual check of one returned point and its marginals.

    ``boxed``, ``reported`` and ``marginals`` are the optimizer's, owned; ``check`` the
    raw check; ``multipliers`` ``lambda``, ``nu``, ``l``, ``u``; ``bound`` is ``b``,
    ``gap`` is ``c'v - b`` and ``ratios`` their ratios to ``sum |c_j|``; ``columns`` the
    identities of the slopes (when there are slopes), then of every block.  ``failure``
    is empty when the dual establishes the program's no-recession evidence.
    """

    problem: GammaProblem
    controls: LPControls
    stage: Stage
    boxed: FloatArray
    reported: float
    marginals: Marginal
    failure: tuple[str, ...]
    check: RawCheck | None = None
    multipliers: Multipliers | None = None
    bound: float | None = None
    gap: float | None = None
    ratios: tuple[float, float] | None = None
    columns: tuple[Identities, ...] = ()

    def __post_init__(self) -> None:
        """Own the point and the marginals; multipliers and identities own theirs."""
        own(self, ("boxed",), ("marginals",))
        object.__setattr__(self, "failure", tuple(self.failure))
        object.__setattr__(self, "columns", tuple(self.columns))

    @property
    def passed(self) -> bool:
        """Whether every check holds."""
        return not self.failure

    def replay(
        self, problem: GammaProblem, *, controls: LPControls, stage: Stage
    ) -> None:
        """``ValueError`` unless this is its report's fresh check on ``problem``."""
        _bound(self, problem, controls, stage)
        point = (self.boxed, self.reported)
        again = dual_check(problem, controls, stage, point, self.marginals)
        if _evidence(again) != _evidence(self):
            raise ValueError(REPLAY)


def _marginals(problem: GammaProblem, stage: Stage, given: Marginal) -> Marginal:
    """Owned finite marginals of the program's shapes; ``ValueError`` otherwise."""
    k, equalities = problem.n_parameters, 1 if stage == "zero_sum" else 0
    shapes = [(problem.positive.size,), (equalities,), (k,), (k,)]
    owned = [immutable_array(m) for m in given]
    finite = all(np.isfinite(m).all() for m in owned)
    if [m.shape for m in owned] != shapes or not finite:
        raise ValueError("marginals must be finite, one per constraint and bound")
    return owned[0], owned[1], owned[2], owned[3]


def _signs(m: Multipliers) -> list[str]:
    """``sign`` unless ``lambda >= 0``, ``l >= 0`` and ``u <= 0`` exactly."""
    wrong = (m.cone < 0).any() or (m.lower < 0).any() or (m.upper > 0).any()
    return ["sign"] if wrong else []


def _verdict(record: GammaDual, signs: list[str]) -> tuple[str, ...]:
    """The failed checks, in the order sign, gap, bound, cone, stationarity."""
    e, ratios = record.controls.tolerance, record.ratios or (np.inf, np.inf)
    failed = [*signs, *([] if ratios[0] <= e else ["gap"])]
    failed += [] if ratios[1] <= e else ["bound"]
    failed += [] if all(c.cone.passing(e) for c in record.columns) else ["cone"]
    stationary = all(c.stationarity.passing(e) for c in record.columns)
    return (*failed, *([] if stationary else ["stationarity"]))


def dual_check(
    problem: GammaProblem,
    controls: LPControls,
    stage: Stage,
    point: tuple[ArrayLike, float],
    marginals: Marginal,
) -> GammaDual:
    """The fresh dual check of a returned point ``(v, fun)`` and its marginals."""
    v, fun = _proposal(problem, point[0], point[1])
    owned = _marginals(problem, stage, marginals)
    record = GammaDual(problem, controls, stage, v, fun, owned, ("range",))
    try:
        data = prepare(problem)
    except FloatingPointError:
        return record
    check = raw_check(problem, data, stage, (v, fun), controls)
    if check.failure or check.values is None:
        return replace(record, check=check, failure=(check.failure,))
    nu = float(owned[1][0]) if stage == "zero_sum" else 0.0
    m = Multipliers(immutable_array(-owned[0]), nu, owned[2], owned[3])
    signs = _signs(m)
    record = replace(record, check=check, multipliers=m, failure=(*signs, "range"))
    scale = check.values.scale
    try:
        bound = checked_difference(total(m.upper), total(m.lower), "dual objective")
        gap = checked_difference(np.array([check.values.objective]), bound, "gap")
        found = (float(bound[0]), float(gap[0]))
        ratios = (ratio(found[1], scale), ratio(found[0], scale))
        slopes = [_slopes(problem, data, stage, m)] if problem.n_slopes else []
        columns = (*slopes, *_levels(problem, data, stage, m, problem.n_slopes))
    except FloatingPointError:
        return record
    record = replace(record, bound=found[0], gap=found[1], ratios=ratios)
    record = replace(record, columns=columns)
    return replace(record, failure=_verdict(record, signs))
