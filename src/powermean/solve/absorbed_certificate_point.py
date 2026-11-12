"""Admitted absorbed points: their fresh rows and fixed-effect equations.

Internal module; no history is read.

Admission (:func:`admit_point`) takes a matching pair: ``WholePoint[PointProfile]`` or
``WholePoint[IndexPoint]`` with ``AbsorbedOptions`` of that organization, or
``LinearPoint`` with ``LinearControls`` (``TypeError`` otherwise).  A record proves
nothing by its type: the base ``X theta + o`` (WLS ``o + Xraw beta``, ``beta`` the
sample's order map of ``theta`` once), a profiled index rebuilt from its effects and a
WLS index rebuilt from its effects must equal the record's bit for bit
(``ValueError``).  The supplied ``q`` is authoritative and never re-profiled.

Root points (the selected-root route of :mod:`powermean.solve.absorbed_root`).  A
``WholePoint[RootPoint]`` with ``AbsorbedOptions`` of either organization (the one the
fit requested, whose inner controls are kept as requested) is an EXACT point of the
full joint system: admitted only for a power variance with ``tau`` outside ``[1, 2]``
(:func:`~powermean.solve.certificate.outside_convex`), its slopes and nuisance
coordinates finite and of the sample's ``K`` and ``|J|`` (the sample's own
:func:`~powermean.fe.certificate_basis.owned_basis`), its base
(:func:`root_base`), its effects (:func:`~powermean.fe.certificate_basis.embed` of the
coordinates) and its index (the accepted
:func:`~powermean.fe.profile_arithmetic.reconstruct_index` of those effects on the
accepted :func:`~powermean.fe.profile_arithmetic.prepare_profile` of the target
family at that base) rebuilt and equal to the record's bit for bit, and a recorded
normalization equal to the accepted
:func:`~powermean.solve.absorbed_direction.normalize` of the effects.  Kind ``"root"``.

Fresh rows (:func:`fresh_rows`) on the original sample at ``q``, raw typed weights
``a``: ``b = a r``, ``C = a c`` with the OBSERVED curvature of the accepted row kernel,
or WLS ``b = a (l - q)``, ``C = a``; checked products, ``FloatingPointError`` for a
value not representable or a negative curvature.  A root point's observed curvature
may be zero or negative on a row (the actual moment Jacobian of a
nonconvex equation); every other check stands for it, and the convex kinds keep the
refusal of a negative curvature.  Effect equations (:func:`effect_equations`) are the
accepted raw-weight LEVEL means, never sums over the global weight total; a root
point's are formed on the same accepted prepared problem as its admission. The
certificate keeps that owned preparation locally from admission to the fresh
equation check, then releases it. Standalone :func:`effect_equations` prepares
afresh; neither route reuses evaluated equations or holds cross-call state.

The inner checks of an admitted point live in
:mod:`powermean.solve.absorbed_certificate_inner`, which reads the records formed
here.  :class:`WorkStep` records one certificate operation with the counters an
accepted routine returned, ``None`` when nothing was returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from powermean.data.sample_fe_state import FESample, same_bits
from powermean.data.inputs import RegressionData
from powermean.data.weights import ObservationWeights
from powermean.fe.absorb_numeric import checked_difference, checked_product
from powermean.fe.certificate_basis import embed, owned_basis
from powermean.fe.profile import _bound
from powermean.fe import profile_arithmetic as _profile_rows
from powermean.fe.profile_arithmetic import (
    ProfileProblem,
    fresh_equations,
    prepare_profile,
    reconstruct_index,
)
from powermean.fe.reductions import immutable_array
from powermean.model import rows as _row_kernel
from powermean.model.rows import RowEvaluation, evaluate_rows
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.solve.absorbed_direction import IndexPoint, normalize
from powermean.solve.absorbed_root_records import RootPoint
from powermean.solve.certificate import outside_convex
from powermean.solve.linear_absorbed_numeric import base_index as linear_base
from powermean.solve.linear_absorbed_numeric import reconstruct as linear_rebuilt
from powermean.solve.linear_absorbed_point import (
    FreshEquations,
    LinearControls,
    LinearPoint,
    linear_sample,
)
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.profiled_trigger import WholePoint
from powermean.solve.reduced_nonlinear import ConditionalPoint
from powermean.systems.profiled import (
    PointProfile,
    ProfiledProblem,
    _prepared,
    base_index,
)

PointKind = Literal["profiled", "reduced", "index", "linear", "root"]
Family = PowerVariance | NB2Variance
BOUND = "the point is not bound to its sample, coordinates and effects"
NOT_ROOT = "a root point needs a power variance with tau outside [1, 2]"
StepStatus = Literal["completed", "failed", "refused"]
Counters = tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class WorkStep:
    """One certificate operation: name, outcome, detail and the counters returned.

    ``counters`` is ``None`` when nothing was returned (a refusal before an accepted
    routine counted anything): unavailable, never zero.
    """

    name: str
    status: StepStatus
    detail: str
    counters: Counters | None


# ------------------------------------------------------------------ admission


@dataclass(frozen=True)
class InnerControls:
    """``af``, the sweep budget, the acceleration and (profiled) ``profile_tol``."""

    tolerance: float
    max_sweeps: int
    acceleration: bool
    profile_tol: float | None


@dataclass(frozen=True, eq=False)
class AdmittedPoint:
    """A point admitted afresh; ``theta``, ``base`` and ``q`` owned read-only copies.

    ``record`` is the caller's point, kept for its own evidence (a recovery).
    """

    kind: PointKind
    sample: FESample
    variance: Family | None
    theta: FloatArray
    base: FloatArray
    q: FloatArray
    inner: InnerControls
    record: PointProfile | ConditionalPoint | IndexPoint | LinearPoint | RootPoint
    _numeric_problem: ProfiledProblem | None = None

    @property
    def n_slopes(self) -> int:
        """The number of active slopes ``K``."""
        return int(self.sample.active.size)

    @property
    def selected_root(self) -> bool:
        """Whether the point is an exact root point of a selected-root equation."""
        return self.kind == "root"


def problem_of(sample: FESample, variance: Family) -> ProfiledProblem:
    """The accepted profiled problem of the sample (the solvers' own construction)."""
    grouping, design = sample.grouping, sample.design
    if grouping is None:
        raise ValueError("an absorbed point needs fixed-effect blocks")
    matrix = np.zeros((sample.count, 0)) if design is None else design.matrix
    return ProfiledProblem(
        z=sample.problem.response,
        design=matrix,
        offset=sample.problem.offset,
        weights=sample.data.weight_values,
        grouping=grouping,
        variance=variance,
        columns=() if design is None else design.column_names,
    )


def _same(found: FloatArray, given: object) -> None:
    """``ValueError`` unless ``given`` is ``found`` bit for bit."""
    if not same_bits(np.asarray(given, dtype=np.float64), found):
        raise ValueError(BOUND)


def _nonlinear(
    whole: WholePoint[PointProfile | ConditionalPoint | IndexPoint],
    inner: AbsorbedOptions,
    numeric: ProfiledProblem | None = None,
) -> tuple[AdmittedPoint, ProfileProblem | None]:
    """A profiled or fitted-index whole point, admitted afresh."""
    sample, variance, record = whole.sample, whole.variance, whole.point
    if type(sample) is not FESample or sample.omega == 0:
        raise ValueError("a nonlinear point needs a sample at omega != 0")
    kind: PointKind = (
        "profiled" if type(record) is PointProfile else
        "reduced" if type(record) is ConditionalPoint else "index"
    )
    requested = "profiled" if kind == "reduced" else kind
    if inner.absorbed != requested:
        raise ValueError(f"a {kind} point needs absorbed options of that organization")
    problem = problem_of(sample, variance) if numeric is None else numeric
    q = immutable_array(record.index)
    prepared: ProfileProblem | None = None
    try:
        base = base_index(problem, record.theta)
        _same(base, record.base)
        if type(record) in (PointProfile, ConditionalPoint):
            prepared = _prepared(problem, base)
            state = _bound(record.state, prepared)
            _same(reconstruct_index(prepared, state.coefficients), q)
        elif numeric is not None:
            prepared = _prepared(problem, base)
    except FloatingPointError as error:
        raise ValueError(BOUND) from error
    if q.shape != (sample.count,) or not np.isfinite(q).all():
        raise ValueError(BOUND)
    profile_tol = inner.profile_tol if kind in ("profiled", "reduced") else None
    controls = InnerControls(
        inner.absorb_tol_final,
        inner.absorb_max_sweeps,
        inner.absorb_acceleration,
        profile_tol,
    )
    theta = immutable_array(record.theta)
    point = AdmittedPoint(kind, sample, variance, theta, base, q, controls, record, numeric)
    return point, prepared


def _linear(record: LinearPoint, inner: LinearControls) -> AdmittedPoint:
    """A WLS point, admitted afresh: ``beta``, base and ``q`` rebuilt from ``theta``."""
    sample = linear_sample(record.sample)
    grouping = sample.grouping
    theta, effects = immutable_array(record.theta), record.effects
    if grouping is None or theta.shape != (sample.active.size,):
        raise ValueError(BOUND)
    if len(effects) != len(grouping.blocks) or any(
        np.shape(e) != (b.n_levels,)
        for e, b in zip(effects, grouping.blocks, strict=True)
    ):
        raise ValueError(BOUND)
    order = sample.order_scale
    try:
        beta = np.zeros(0) if order is None else order.coefficients(theta)
        base = immutable_array(linear_base(sample, beta))
        q = immutable_array(linear_rebuilt(base, effects, grouping))
    except FloatingPointError as error:
        raise ValueError(BOUND) from error
    for found, given in ((beta, record.beta), (base, record.base), (q, record.index)):
        _same(found, given)
    controls = InnerControls(
        inner.absorb_tol_final, inner.absorb_max_sweeps, inner.absorb_acceleration, None
    )
    return AdmittedPoint("linear", sample, None, theta, base, q, controls, record)


def root_base(sample: FESample, theta: FloatArray) -> FloatArray:
    """``fl(X theta + o)`` of a root point: the sample's scaled slopes and offset.

    The arithmetic of the convex points' base (``X theta + o``, checked);
    ``FloatingPointError`` for a base that is not representable.
    """
    design = sample.design
    matrix = np.zeros((sample.count, 0)) if design is None else design.matrix
    with np.errstate(over="ignore", invalid="ignore"):
        base: FloatArray = matrix @ theta + sample.problem.offset
    if not np.isfinite(base).all():
        raise FloatingPointError("the base index is not representable in float64")
    return immutable_array(base)


def root_prepared(
    sample: FESample, variance: PowerVariance, base: FloatArray
) -> ProfileProblem:
    """The accepted prepared problem of the target family at ``base``.

    :func:`~powermean.fe.profile_arithmetic.prepare_profile` of the sample's blocks,
    transformed outcome and raw weights: its checks included, nothing profiled.
    """
    if sample.grouping is None:
        raise ValueError("an absorbed point needs fixed-effect blocks")
    return prepare_profile(
        sample.grouping,
        sample.problem.response,
        base,
        sample.data.weight_values,
        variance=variance,
    )


def _same_effects(found: tuple[FloatArray, ...], given: object) -> None:
    """``ValueError`` unless ``given`` holds the vectors of ``found`` bit for bit."""
    if not isinstance(given, tuple) or len(given) != len(found):
        raise ValueError(BOUND)
    for mine, theirs in zip(found, given, strict=True):
        _same(mine, theirs)


def _root_normalization(record: RootPoint, sample: FESample) -> None:
    """A recorded normalization must be the accepted one of the point's effects."""
    recorded = record.normalization
    if recorded is None:
        return
    assert sample.grouping is not None
    fresh = normalize(sample.grouping, record.effects)
    _same_effects(fresh.effects, recorded.effects)
    _same_effects(fresh.shifts, recorded.shifts)
    refs = recorded.references
    if not isinstance(refs, tuple) or len(refs) != len(fresh.references):
        raise ValueError(BOUND)
    for mine, theirs in zip(fresh.references, refs, strict=True):
        if not np.array_equal(mine, theirs):
            raise ValueError(BOUND)


def _root(
    whole: WholePoint[RootPoint], inner: AbsorbedOptions
) -> tuple[AdmittedPoint, ProfileProblem]:
    """An exact root point, admitted afresh (module docstring).

    The inner controls are the requested organization's, kept as requested: no
    absorption, profile or recovery forms any part of the point.
    """
    sample, variance, record = whole.sample, whole.variance, whole.point
    if type(sample) is not FESample or sample.omega == 0:
        raise ValueError("a nonlinear point needs a sample at omega != 0")
    if type(variance) is not PowerVariance or not outside_convex(variance):
        raise ValueError(NOT_ROOT)
    if sample.grouping is None:
        raise ValueError("an absorbed point needs fixed-effect blocks")
    basis = owned_basis(sample)
    theta, alpha = immutable_array(record.theta), immutable_array(record.alpha)
    shapes = theta.shape == (sample.active.size,) and alpha.shape == (basis.n_kept,)
    if not shapes or not (np.isfinite(theta).all() and np.isfinite(alpha).all()):
        raise ValueError(BOUND)
    try:
        base = root_base(sample, theta)
        _same(base, record.base)
        effects = embed(basis, alpha)
        _same_effects(effects, record.effects)
        prepared = root_prepared(sample, variance, base)
        q = immutable_array(reconstruct_index(prepared, effects))
        _same(q, record.index)
        _root_normalization(record, sample)
    except FloatingPointError as error:
        raise ValueError(BOUND) from error
    if not np.isfinite(q).all():
        raise ValueError(BOUND)
    profile_tol = inner.profile_tol if inner.absorbed == "profiled" else None
    controls = InnerControls(
        inner.absorb_tol_final,
        inner.absorb_max_sweeps,
        inner.absorb_acceleration,
        profile_tol,
    )
    point = AdmittedPoint("root", sample, variance, theta, base, q, controls, record)
    return point, prepared


def admit_point(whole: object, inner: object) -> AdmittedPoint:
    """Admit a point with its inner controls (module docstring).

    ``TypeError`` for a pair that is not a matching point and inner-control record;
    ``ValueError`` for a point not bound to its sample, coordinates or effects.
    """
    point, _ = admit_prepared_point(whole, inner)
    return point


def admit_prepared_point(
    whole: object, inner: object
) -> tuple[AdmittedPoint, ProfileProblem | None]:
    """Admit once, keeping the owned preparation used to reconstruct this point.

    The certificate consumes the preparation within this call only. It is not
    attached to a candidate or retained point, and proves no equation: the original
    fixed-effect equations are still evaluated afresh at the admitted ``q``.
    """
    if type(whole) is LinearPoint and type(inner) is LinearControls:
        return _linear(whole, inner), None
    if type(whole) is WholePoint and type(inner) is AbsorbedOptions:
        point: object = whole.point
        if type(point) in (PointProfile, ConditionalPoint, IndexPoint):
            return _nonlinear(whole, inner)
        if type(point) is RootPoint:
            return _root(whole, inner)
    raise TypeError(
        "certify a WholePoint with AbsorbedOptions or a LinearPoint with LinearControls"
    )


def _admit_numeric_point(
    whole: object, inner: object, problem: ProfiledProblem
) -> tuple[AdmittedPoint, ProfileProblem | None]:
    """Fresh base/effects/q admission on an internal operation's numeric owner.

    Public admit_point/admit_prepared_point remain their original checked route.
    This factory is called only by normal NumericFit composition; every original
    point and final-equation check remains, while fixed inputs are not recopied.
    """
    if type(whole) is WholePoint and type(inner) is AbsorbedOptions:
        if type(whole.point) in (PointProfile, ConditionalPoint, IndexPoint):
            return _nonlinear(whole, inner, problem)
    raise TypeError("numeric admission needs a matching nonlinear whole point")


# -------------------------------------------------------------- fresh evidence


@dataclass(frozen=True, eq=False)
class FreshRows:
    """``b`` and ``C`` at the point, the raw weights and whether they are counts."""

    b: FloatArray
    curvature: FloatArray
    weights: FloatArray
    frequency: bool


def fresh_rows(
    point: AdmittedPoint,
    *, _evaluated: list[RowEvaluation | None] | None = None,
) -> FreshRows:
    """The fresh observed rows (module docstring); ``FloatingPointError`` on range.

    A negative observed curvature is refused, except at a root point:
    its rows keep the actual signed curvature of the nonconvex equation.
    """
    sample = point.sample
    numeric = point._numeric_problem
    a = sample.data.weight_values if numeric is None else numeric.weights
    z = sample.problem.response if numeric is None else numeric.z
    weights = sample.data.weights
    evaluation: RowEvaluation | None = None
    if point.variance is None:
        e = checked_difference(z, point.q, "fresh residual")
        b = checked_product(a, e, "weighted residual")
        c = np.array(a, dtype=np.float64)
    else:
        evaluation = evaluate_rows(z, point.q, variance=point.variance)
        b = checked_product(a, evaluation.score, "weighted score")
        c = checked_product(a, evaluation.observed_curvature, "weighted curvature")
    negative = bool((c < 0).any()) and not point.selected_root
    if not (np.isfinite(b).all() and np.isfinite(c).all()) or negative:
        raise FloatingPointError("the fresh curvature is negative or not representable")
    frequency = weights is not None and weights.frequency
    fresh = FreshRows(
        immutable_array(b), immutable_array(c), immutable_array(a), frequency
    )
    if _evaluated is not None:
        _evaluated[0] = evaluation
    return fresh


@dataclass(frozen=True, eq=False)
class EffectEquations:
    """The raw-weight level means of every original block at the point."""

    means: tuple[FloatArray, ...]
    maximum: float


def effect_equations(point: AdmittedPoint) -> EffectEquations:
    """The fresh fixed-effect equations; ``FloatingPointError`` on range."""
    if type(point.record) is LinearPoint:
        equations = FreshEquations(point.record)
        return EffectEquations(equations.level_means, equations.largest_level_mean)
    if point.variance is None:
        raise ValueError(BOUND)
    if point.selected_root:
        if type(point.variance) is not PowerVariance:
            raise ValueError(NOT_ROOT)
        prepared = root_prepared(point.sample, point.variance, point.base)
    else:
        prepared = _prepared(problem_of(point.sample, point.variance), point.base)
    return prepared_effect_equations(prepared, point.q)


def prepared_effect_equations(
    prepared: ProfileProblem, q: FloatArray
) -> EffectEquations:
    """Fresh original level equations on the local admission's owned preparation.

    This reuses only preparation, never a previous score, equation or verdict.
    ``fresh_equations`` keeps its active-row evaluation, required-quantity guards,
    row-order sums and raw-weight level denominators unchanged.
    """
    check = fresh_equations(prepared, q)
    return EffectEquations(check.errors, check.maximum)


def _prepared_effect_equations_from_rows(
    prepared: ProfileProblem, q: FloatArray, rows: RowEvaluation,
) -> EffectEquations:
    """The same level equations from the immediately preceding fresh row consumer."""
    check = _profile_rows._fresh_equations_from_rows(prepared, q, rows)
    return EffectEquations(check.errors, check.maximum)


def _immutable_row_array(value: object, shape: tuple[int, ...]) -> bool:
    """Accept complete immutable bytes storage, not a read-only flag alone."""
    if (type(value) is not np.ndarray or value.dtype != np.float64
            or value.shape != shape or value.flags.writeable
            or not value.flags.c_contiguous):
        return False
    root = value
    while type(root.base) is np.ndarray:
        root = root.base
    return type(root.base) is bytes and len(root.base) == value.nbytes


def _same_row_values(left: object, right: object, shape: tuple[int, ...]) -> bool:
    """Match actual owned float64 row headers and bits, not cached authority."""
    return (
        _immutable_row_array(left, shape) and _immutable_row_array(right, shape)
        and bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))
    )


_ROW_PROFILE = ProfileProblem
_ROW_NUMERIC = ProfiledProblem
_ROW_POWER = PowerVariance
_ROW_NB2 = NB2Variance
_ROW_EVALUATION = RowEvaluation
_ROW_POINT = AdmittedPoint
_ROW_WEIGHTS = _profile_rows.PreparedWeights
_ROW_DATA_TYPES = (FESample, RegressionData, ObservationWeights)
_ROW_FIELDS = tuple(
    (kind, name, vars(kind).get(name)) for kind, names in (
        (ProfileProblem, ("z", "variance", "prepared", "_all_active")),
        (ProfiledProblem, ("z", "weights", "variance")),
        (AdmittedPoint, ("_numeric_problem", "q", "variance", "kind", "sample")),
        (_ROW_WEIGHTS, ("weights",)),
        (FESample, ("data",)), (RegressionData, ("weights",)),
        (ObservationWeights, ("frequency",)),
    ) for name in names
)

_FRESH_ROW_PRODUCERS = (
    fresh_rows, prepared_effect_equations, _prepared_effect_equations_from_rows,
    evaluate_rows, _row_kernel.evaluate_rows, _profile_rows.evaluate_rows,
    _profile_rows._EVALUATE_ROWS, fresh_equations, _profile_rows.fresh_equations,
    _profile_rows._fresh_equations_from_rows, _profile_rows._equations,
    _profile_rows.active_scores, _profile_rows._cached_rows,
    _profile_rows._checked_active_score, _profile_rows._detached_float,
    _immutable_row_array, _same_row_values, checked_product, immutable_array,
    _row_kernel._aligned_rows, _row_kernel._log_components, _row_kernel._exp,
    _row_kernel._signed_sum, _row_kernel._finite_array, _row_kernel._finite_result,
    _row_kernel._immutable, RowEvaluation.__new__, RowEvaluation.__init__, RowEvaluation.__post_init__,
    RowEvaluation.__getattribute__, PowerVariance.__getattribute__,
    getattr(PowerVariance, "__getattr__", None), NB2Variance.__getattribute__,
    getattr(NB2Variance, "__getattr__", None), ProfileProblem.__getattribute__,
    getattr(ProfileProblem, "__getattr__", None), ProfiledProblem.__getattribute__,
    getattr(ProfiledProblem, "__getattr__", None), FreshRows.__init__,
    AdmittedPoint.__getattribute__, getattr(AdmittedPoint, "__getattr__", None),
    _ROW_WEIGHTS.__getattribute__, getattr(_ROW_WEIGHTS, "__getattr__", None),
    ProfileProblem.__dict__["n_rows"],
    PowerVariance.__dict__["tau"], NB2Variance.__dict__["kappa"],
    FESample.__getattribute__, getattr(FESample, "__getattr__", None),
    RegressionData.__getattribute__, getattr(RegressionData, "__getattr__", None),
    ObservationWeights.__getattribute__, getattr(ObservationWeights, "__getattr__", None),
)


def _share_row_evaluation(point: AdmittedPoint, prepared: ProfileProblem) -> bool:
    """Select only unchanged numeric-owned all-active row/equation composition.

    Replaced aliases, kernel helpers, record accessors and foreign/masked/linear
    or selected-root inputs retain the original two consumers. Actual response
    and raw-weight bits are matched once here; no solver evaluation is consumed.
    """
    if (ProfileProblem is not _ROW_PROFILE or ProfiledProblem is not _ROW_NUMERIC
            or PowerVariance is not _ROW_POWER or NB2Variance is not _ROW_NB2
            or RowEvaluation is not _ROW_EVALUATION
            or _row_kernel.RowEvaluation is not _ROW_EVALUATION
            or _row_kernel.PowerVariance is not _ROW_POWER
            or _row_kernel.NB2Variance is not _ROW_NB2
            or _profile_rows.ProfileProblem is not _ROW_PROFILE
            or AdmittedPoint is not _ROW_POINT
            or _profile_rows.PreparedWeights is not _ROW_WEIGHTS
            or FESample is not _ROW_DATA_TYPES[0]
            or RegressionData is not _ROW_DATA_TYPES[1]
            or ObservationWeights is not _ROW_DATA_TYPES[2]):
        return False
    current = (
        fresh_rows, prepared_effect_equations, _prepared_effect_equations_from_rows,
        evaluate_rows, _row_kernel.evaluate_rows, _profile_rows.evaluate_rows,
        _profile_rows._EVALUATE_ROWS, fresh_equations, _profile_rows.fresh_equations,
        _profile_rows._fresh_equations_from_rows, _profile_rows._equations,
        _profile_rows.active_scores, _profile_rows._cached_rows,
        _profile_rows._checked_active_score, _profile_rows._detached_float,
        _immutable_row_array, _same_row_values, checked_product, immutable_array,
        _row_kernel._aligned_rows, _row_kernel._log_components, _row_kernel._exp,
        _row_kernel._signed_sum, _row_kernel._finite_array, _row_kernel._finite_result,
        _row_kernel._immutable, RowEvaluation.__new__, RowEvaluation.__init__, RowEvaluation.__post_init__,
        RowEvaluation.__getattribute__, PowerVariance.__getattribute__,
        getattr(PowerVariance, "__getattr__", None), NB2Variance.__getattribute__,
        getattr(NB2Variance, "__getattr__", None), ProfileProblem.__getattribute__,
        getattr(ProfileProblem, "__getattr__", None), ProfiledProblem.__getattribute__,
        getattr(ProfiledProblem, "__getattr__", None), FreshRows.__init__,
        AdmittedPoint.__getattribute__, getattr(AdmittedPoint, "__getattr__", None),
        _ROW_WEIGHTS.__getattribute__, getattr(_ROW_WEIGHTS, "__getattr__", None),
        ProfileProblem.__dict__["n_rows"],
        PowerVariance.__dict__["tau"], NB2Variance.__dict__["kappa"],
        FESample.__getattribute__, getattr(FESample, "__getattr__", None),
        RegressionData.__getattribute__, getattr(RegressionData, "__getattr__", None),
        ObservationWeights.__getattribute__, getattr(ObservationWeights, "__getattr__", None),
    )
    if not all(a is b for a, b in zip(current, _FRESH_ROW_PRODUCERS, strict=True)):
        return False
    if not all(vars(kind).get(name) is original
               for kind, name, original in _ROW_FIELDS):
        return False
    if type(point) is not AdmittedPoint or type(prepared) is not ProfileProblem:
        return False
    numeric = point._numeric_problem
    family = type(point.variance)
    if (type(numeric) is not ProfiledProblem
            or type(prepared.prepared) is not _ROW_WEIGHTS
            or type(point.sample) is not FESample
            or type(point.sample.data) is not RegressionData
            or (point.sample.data.weights is not None
                and type(point.sample.data.weights) is not ObservationWeights)
            or (family is not PowerVariance and family is not NB2Variance)
            or point.kind == "root" or point.variance is None
            or prepared.variance is not point.variance
            or numeric.variance is not point.variance
            or prepared._all_active is not True
            or type(point.q) is not np.ndarray or point.q.dtype != np.float64
            or point.q.ndim != 1
            or not _immutable_row_array(point.q, point.q.shape)):
        return False
    return (
        _same_row_values(numeric.z, prepared.z, point.q.shape)
        and _same_row_values(numeric.weights, prepared.prepared.weights, point.q.shape)
        and bool((prepared.prepared.weights > 0).all())
    )
