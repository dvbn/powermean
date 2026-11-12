"""The named HC covariances HC0 to HC5 and HC4m: multipliers, meat, composition.

The multipliers are those of sandwich 3.1-1 ``meatHC``, read from its source and
written here from its formulas.  No public name or default here.  The exact strings
of a request without fixed effects
reach these kernels through the fit-time provider of
:mod:`powermean.inference.hc_provider` and the retained adapter of
:mod:`powermean.results.retained_hc`.

Definitions.  With ``h_i`` the per-replica leverage of
:mod:`powermean.inference.leverage`, ``N`` the statistical count,
``p = round(sum_i f_i h_i)`` (the trace over literal replicas, equal to the identified
full-design rank ``K``) and ``r_i = N h_i / p``, the squared-score multipliers are

    HC0   1                         HC1   1, with the method's own factor N/(N-K)
    HC2   (1-h)^-1                  HC3   (1-h)^-2
    HC4   (1-h)^-min(4, r)          HC4m  (1-h)^-(min(1, r) + min(1.5, r))
    HC5   (1-h)^-(min(r, max(4, 0.7 max_j r_j)) / 2),  the half power included.

The constants 4, 1, 1.5, 0.7 and 1/2 are those of the formulas, not options.  The
covariance is the sandwich of the SAME point, ``V = A [sum_i q_i s_i s_i' / f_i] A'``:
``u_i = A s_i`` the influence rows under the bread (the observed Jacobian by default,
the expected information with ``bread="expected"``, both on the full nuisance basis for
an absorbed fit), ``q_i`` the multiplier of the row, ``f_i`` its frequency count (a
replica carries ``s_i / f_i``), so that HC0 is the robust meat with factor one, bit for
bit through the primitive (:func:`powermean.inference.ssc.primitive_estimate`
of the rows ``sqrt(q_i) u_i``).  The nonlinear hats use the expected-information
geometry whatever the bread: with the observed bread the result is a declared
COMPOSITE, not stock GLM covariance; with
``bread="expected"`` it is the stock sandwich.  Both components are recorded.

Ownership.  The scalar is :mod:`powermean.inference.ssc`'s:
``apply_convention(..., kind="robust", method=...)`` gives the method's own factor
(``N/(N-K)`` for HC1, defined for ``N > K`` only, else
``specification.no_residual_dof``; one otherwise), refuses an explicit convention
whose factor is a second correction (``specification.conflicting``) and carries the
reference distribution of the convention in force, inherited or default conventions
never multiplying.  The factor is applied once by
:func:`powermean.inference.ssc.scaled_covariance`.

Refusals before any solve: the preflight (:func:`hc_preflight`).  From
the design, the raw weights, their kind, the method and the convention alone, which a
fit has before it solves and a post-fit request has in its retained state, the
preflight decides the exact full column rank of the design on its active rows
(:func:`powermean.inference.leverage.exact_full_rank`, the rational echelon: a
dependent column refuses with ``ValueError``), the statistical count, the method's
factor (HC1 at ``N = K``: ``specification.no_residual_dof``; an explicit second
correction: ``specification.conflicting``) and, for HC2 to HC5 and HC4m, structural
unit leverage of one replica (``specification.unit_leverage`` with the count of such
observations).  Every entry point runs it before any evaluation, factorization or
solve, so a refusal costs none; the same function serves a fit before its solve
with the same identifiers.  A computed hat within rounding of
one on a row that is not structural is not refused there: its multiplier is formed as
written, and a base ``1 - h`` that is not positive, or a multiplier or product beyond
the doubles, raises ``FloatingPointError`` (no floor, no clamp, no fallback).  Exact
zero variances stay zero, representable tiny values are kept, a negative
variance gives a missing standard error downstream.

Binding.  Each entry point forms every component from ONE owner of the rows at one
point, and forms it again at every call: nothing is cached, aliased or reused across
calls, and no identity, hash, Gram equality, read-only flag or prior success stands
for the owner.  A joint request names the :class:`~powermean.systems.joint.JointSystem`
that owns the rows, the weights, the offset and the variance, and the evaluation it
formed at the point: the system is evaluated afresh at that ``gamma`` and the record
given must equal the fresh one bit for bit in every field (point, rows, weights,
observed and expected geometry); the design is the system's own active design, so
every row is bound, the rows of zero score included, and no caller matrix enters.  A
linear request names the :class:`LinearRequest` that owns the equation as the solve
received it (response, offset, design, raw weights and their kind, every input row in
its order, the rows of zero weight included) together with the candidate that the
solve returned for it (:class:`LinearSolve`, formed by :func:`solve_request`,
the internal request/solve handoff that captures the equation before the solve): the
candidate's units, weights and residuals are checked bit for bit against the owner's
rows at the candidate's coefficients (``e = l - o - X beta``, the solve's own
arithmetic, no second solve), and the design of the bread, of the hats and of the
scores is the owner's own, since a candidate alone carries neither response, offset
nor design.  An absorbed request names the
:class:`~powermean.solve.absorbed_certificate_point.AdmittedPoint` of the
certificate path, which owns the sample (response, offset, raw weights and their
kind, the scaled slopes and the blocks) and the point (``q``, ``theta``, the base and
the working variance): the record is admitted again by
:func:`~powermean.solve.absorbed_certificate_point.admit_point` and must equal the
record given bit for bit, the full nuisance basis is the sample's own
(:func:`~powermean.fe.certificate_basis.owned_basis`), and the observed rows
(:func:`~powermean.solve.absorbed_certificate_point.fresh_rows`) and the expected
curvature are both evaluated at that one ``q`` from that one sample, so that no caller
row, Fisher vector or basis enters; the composition of given rows is an internal
helper of this module, not an entry point.  The root point of a selected-root
absorbed fit outside ``tau in [1, 2]`` is admitted as its kind (its inner controls
those its organization requested) and its observed rows keep their actual signed
curvature: the hats stay those of the positive expected information, the bread
the actual observed Jacobian, as for every nonlinear point.  Records own their
arrays: the leverage record of a composition is copied whole
(:func:`powermean.inference.leverage.detached`), so that no caller array header or
record is retained.  The reporting map is not applied here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from powermean.data.accounting import frozen_index
from powermean.data.inference_requests import HC_METHODS, HCMethod
from powermean.errors import SpecificationError
from powermean.fe.certificate_basis import CertificateBasis, owned_basis
from powermean.inference.absorbed_code_tuples import CachedCodeTuples
from powermean.inference.absorbed_rank import AbsorbedCount, absorbed_count
from powermean.inference.bread import Bread, bread_influence, same_point_influence
from powermean.inference.contributions import (
    Influence,
    linear_influence,
    statistical_count,
)
from powermean.inference.influence import influence_values, moment_influence_map
from powermean.inference.leverage import (
    FullRank,
    Leverage,
    absorbed_leverage,
    dense_leverage,
    detached,
    exact_full_rank,
)
from powermean.inference.ssc import (
    AppliedConvention,
    ReferenceDistribution,
    ResolvedConvention,
    apply_convention,
    convention_counts,
    primitive_estimate,
    scaled_covariance,
)
from powermean.inference.vcov import CovarianceEstimate
from powermean.model.order import IndexArray
from powermean.model.rows import _immutable, evaluate_rows
from powermean.model.variance import FloatArray, _finite_array, _finite_result
from powermean.solve.absorbed_certificate_geometry import hessian
from powermean.solve.absorbed_certificate_point import (
    AdmittedPoint,
    FreshRows,
    InnerControls,
    admit_point,
    fresh_rows,
)
from powermean.solve.linear import LinearCandidate, solve
from powermean.solve.linear import _offset as _linear_offset
from powermean.solve.linear import _response_and_design as _linear_rows
from powermean.solve.linear import _weights as _linear_weights
from powermean.solve.linear_absorbed_point import LinearControls, LinearPoint
from powermean.solve.absorbed import AbsorbedOptions, Organization
from powermean.solve.profiled_trigger import WholePoint
from powermean.systems.joint import JointEvaluation, JointSystem, SamplingUnits

LEVERAGE_METHODS: tuple[str, ...] = ("HC2", "HC3", "HC4", "HC4m", "HC5")
# The constants of the formulas of sandwich's meatHC, not options.
HC4_CAP = 4.0
HC4M_POWERS = (1.0, 1.5)
HC5_FLOOR = 4.0
HC5_SHARE = 0.7
HC5_HALF = 0.5
Layout = Literal["dense", "absorbed"]
NOT_SOLVED = "the candidate is not the solve of this request at its coefficients"
NOT_ADMITTED = "the point is not the admitted point of its record on its sample"


# -------------------------------------------------------------------- records


@dataclass(frozen=True, eq=False)
class Multipliers:
    """The squared-score multipliers ``q_i`` of one method on the active rows.

    Attributes
    ----------
    method : HC method
        The method.
    values : array
        ``q_i`` per active row, owned; ones for HC0 and HC1.
    r : array
        ``r_i = N h_i / p`` per active row, owned.
    exponents : array
        The exponent of ``(1 - h_i)^-e_i`` per active row, owned; zeros for HC0, HC1.
    count : int
        ``N``.
    p : int
        ``round(sum_i f_i h_i)``.
    max_r : float
        ``max_j r_j`` over the replicas (the compressed rows carry it).
    """

    method: HCMethod
    values: FloatArray
    r: FloatArray
    exponents: FloatArray
    count: int
    p: int
    max_r: float


@dataclass(frozen=True, eq=False)
class HCCovariance:
    """A named HC covariance of ``theta`` at one point, the factor applied once.

    Attributes
    ----------
    method : HC method
        The method.
    covariance : array
        ``factor * V`` with ``V`` the adjusted sandwich, exactly symmetric, owned.
    estimate : CovarianceEstimate
        The primitive's estimate of the adjusted rows without the factor.
    applied : AppliedConvention
        The convention in force, the counts, the factor with its owner, the reference:
        a frozen record of integers and strings, without an array.
    leverage : Leverage
        The per-replica hats and their provenance, an owned copy: no array header or
        nested record of the caller's record is retained.
    multipliers : Multipliers
        ``q_i``, ``r_i``, ``p`` and ``max r``.
    bread : {"observed", "expected"}
        The bread of the influence rows.
    layout : {"dense", "absorbed"}
        Slopes alone, or the full nuisance basis.
    influence : array
        ``u_i = A s_i`` on the active rows, owned.
    """

    method: HCMethod
    covariance: FloatArray
    estimate: CovarianceEstimate
    applied: AppliedConvention
    leverage: Leverage
    multipliers: Multipliers
    bread: Bread
    layout: Layout
    influence: FloatArray

    @property
    def factor(self) -> float:
        """The scalar applied, the method's own."""
        found = self.applied.factor
        assert found is not None
        return found.value

    @property
    def reference(self) -> ReferenceDistribution:
        """The reference distribution of the convention in force."""
        return self.applied.reference

    @property
    def hat_geometry(self) -> str:
        """The leverage convention: the expected-information geometry."""
        return self.leverage.geometry


# ----------------------------------------------------------------- multipliers


def _method(method: HCMethod) -> HCMethod:
    """One of the seven exact method strings, checked at run time."""
    if method not in HC_METHODS:
        raise ValueError(f"unknown HC method {method!r}")
    return method


def _exponents(method: HCMethod, r: FloatArray, max_r: float) -> FloatArray:
    """The exponent ``e_i`` of ``(1 - h_i)^-e_i``."""
    if method in ("HC0", "HC1"):
        return np.zeros(r.shape)
    if method == "HC2":
        return np.ones(r.shape)
    if method == "HC3":
        return np.full(r.shape, 2.0)
    if method == "HC4":
        return np.minimum(HC4_CAP, r)
    if method == "HC4m":
        return np.minimum(HC4M_POWERS[0], r) + np.minimum(HC4M_POWERS[1], r)
    return np.minimum(r, max(HC5_FLOOR, HC5_SHARE * max_r)) * HC5_HALF


def multipliers(method: HCMethod, leverage: Leverage, count: int) -> Multipliers:
    """The squared-score multipliers of ``method`` on the active rows of ``leverage``.

    Parameters
    ----------
    method : HC method
        The exact method string.
    leverage : Leverage
        The per-replica hats, with ``p``.
    count : int
        ``N``, the statistical count.

    Returns
    -------
    Multipliers
        ``q_i``, ``r_i``, the exponents, ``N``, ``p`` and ``max r``.

    Raises
    ------
    FloatingPointError
        A base ``1 - h_i`` that is not positive, or a multiplier that is not
        representable, for HC2 to HC5 and HC4m.  Nothing is floored.
    """
    found = _method(method)
    if type(leverage) is not Leverage:
        raise TypeError("leverage must be a Leverage record")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    hat = leverage.hat[leverage.active]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        r: FloatArray = count * hat / leverage.p
    r = _finite_result(r, "r = N h / p")
    max_r = float(np.max(r)) if r.size else 0.0
    exponents = _exponents(found, r, max_r)
    if found in ("HC0", "HC1"):
        values = np.ones(hat.shape)
    else:
        base = 1.0 - hat
        if np.any(base <= 0):
            raise FloatingPointError(
                "1 - h is not positive on a row that is not structural: the HC "
                "multiplier is not representable"
            )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            values = base ** (-exponents)
        values = _finite_result(values, "HC multiplier")
    return Multipliers(
        found,
        _immutable(values),
        _immutable(r),
        _immutable(exponents),
        int(count),
        int(leverage.p),
        max_r,
    )


# ----------------------------------------------------------------------- meat


def adjusted_estimate(
    influence: FloatArray, weights: FloatArray, values: FloatArray, *, frequency: bool
) -> CovarianceEstimate:
    """The primitive estimate of the rows ``sqrt(q_i) u_i``: the adjusted meat.

    Parameters
    ----------
    influence : array
        ``u_i = A s_i`` on the active rows.
    weights : array
        The raw weights of those rows.
    values : array
        ``q_i`` per active row.
    frequency : bool
        Whether the weights are frequency counts.

    Returns
    -------
    CovarianceEstimate
        ``sum_i q_i u_i u_i' / f_i`` through the accepted primitive, no factor; HC0
        (``q = 1``) is the robust primitive bit for bit.
    """
    rows = _finite_array(influence, "influence")
    q = _finite_array(values, "multipliers")
    if rows.ndim != 2 or q.shape != (rows.shape[0],):
        raise ValueError("influence rows and multipliers must be aligned")
    if np.any(q <= 0):
        raise ValueError("every HC multiplier is positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled: FloatArray = np.sqrt(q)[:, None] * rows
    scaled = _finite_result(scaled, "adjusted influence")
    return primitive_estimate(scaled, weights, frequency=frequency)


# -------------------------------------------------------------------- preflight


@dataclass(frozen=True, eq=False)
class HCPreflight:
    """What a named HC request decides before any evaluation, factorization or solve.

    Attributes
    ----------
    method : HC method
        The method.
    applied : AppliedConvention
        The convention in force applied to the counts: the method's own factor (HC1
        ``N/(N-K)``, defined for ``N > K``), the reference distribution.
    exact : FullRank
        The exact rank decision: ``K`` (plus ``|J|``), the active rows, the rows of
        structural unit leverage (empty, or the method is HC0 or HC1).
    count : int
        ``N``, the statistical count of the active rows.
    frequency : bool
        Whether the weights are frequency counts.
    n_rows : int
        The rows of the design given, the rows of zero weight included.
    """

    method: HCMethod
    applied: AppliedConvention
    exact: FullRank
    count: int
    frequency: bool
    n_rows: int
    absorbed: AbsorbedCount | None = None

    @property
    def rank(self) -> int:
        """``K``, the identified full-design rank: slopes plus kept coordinates."""
        return self.exact.rank

    @property
    def active(self) -> IndexArray:
        """The positions of the rows of positive weight."""
        return self.exact.active


def hc_preflight(
    design: FloatArray,
    weights: FloatArray,
    method: HCMethod,
    convention: ResolvedConvention,
    *,
    frequency: bool,
    basis: CertificateBasis | None = None,
) -> HCPreflight:
    """The refusals and counts of a named HC request that need no point or solve.

    Parameters
    ----------
    design : array
        ``X`` on the rows of the request: every input row of a linear solve (rows of
        zero weight included), the active rows of a joint system, every row of the
        grouping of an absorbed fit.
    weights : array
        The raw weights of those rows, nonnegative.
    method : HC method
        The exact method string.
    convention : ResolvedConvention
        The convention in force for this request.
    frequency : bool
        Whether the weights are frequency counts.
    basis : CertificateBasis, optional
        The kept nuisance coordinates of the same rows, for an absorbed design.

    Returns
    -------
    HCPreflight
        The applied convention, the exact rank decision and the count, for the
        composition that follows or for a fit that has not solved yet.

    Raises
    ------
    ValueError
        Misaligned or invalid inputs, an unknown method, or a design that is not of
        full column rank on the rows of positive weight (the exact witness).
    SpecificationError
        ``specification.no_residual_dof`` (HC1 at ``N = K``),
        ``specification.conflicting`` (an explicit second correction) or
        ``specification.unit_leverage`` (HC2 to HC5 and HC4m, with the count of
        observations of structural unit leverage), in that order.
    TypeError
        ``frequency`` is not a ``bool``.
    """
    found = _method(method)
    exact = exact_full_rank(design, weights, frequency=frequency, basis=basis)
    a = _finite_array(weights, "weights")
    count = statistical_count(a[exact.active], frequency=frequency)
    absorbed = (
        None
        if basis is None
        else absorbed_count(
            tuple(
                (block.name, block.codes[exact.active])
                for block in basis.grouping.blocks
            ),
            (),
        )
    )
    applied = apply_convention(
        convention,
        convention_counts(count, exact.n_slopes, exact.rank, absorbed),
        kind="robust",
        method=found,
    )
    if found in LEVERAGE_METHODS and exact.unit.size:
        raise SpecificationError(
            "specification.unit_leverage",
            {"method": found, "count": int(exact.unit.size)},
        )
    return HCPreflight(found, applied, exact, count, frequency, int(a.size), absorbed)


# ------------------------------------------------------------------ composition


def hc_covariance(
    influence: FloatArray,
    weights: FloatArray,
    leverage: Leverage,
    applied: AppliedConvention,
    *,
    frequency: bool,
    bread: Bread,
    layout: Layout,
) -> HCCovariance:
    """Compose the named HC covariance from its components (module docstring).

    Parameters
    ----------
    influence : array
        ``u_i`` on the active rows, under ``bread``.
    weights : array
        The raw weights of the active rows.
    leverage : Leverage
        The hats of the SAME rows (its active rows are these); copied whole, never
        retained.
    applied : AppliedConvention
        The applied convention of a robust request with the method, from
        :func:`powermean.inference.ssc.apply_convention` or :func:`hc_preflight`.
    frequency : bool
        Whether the weights are frequency counts.
    bread, layout
        Recorded.
    """
    method = applied.method
    if method is None or applied.kind != "robust":
        raise ValueError("the applied convention must name an HC method")
    owned = detached(leverage)
    rows = _finite_array(influence, "influence")
    if rows.ndim != 2 or rows.shape[0] != owned.active.size:
        raise ValueError("the influence rows are not the active rows of the leverage")
    if applied.counts.rank != owned.rank:
        raise ValueError("the counts and the leverage disagree on the rank")
    found = multipliers(method, owned, applied.counts.count)
    estimate = adjusted_estimate(rows, weights, found.values, frequency=frequency)
    scaled = scaled_covariance(estimate, applied)
    return HCCovariance(
        method,
        scaled.covariance,
        estimate,
        applied,
        owned,
        found,
        bread,
        layout,
        _immutable(rows),
    )


def _same(found: FloatArray | IndexArray, given: FloatArray | IndexArray) -> bool:
    """Whether two arrays have one shape and the same bytes."""
    return found.shape == given.shape and found.tobytes() == given.tobytes()


# ------------------------------------------------------------------------ joint


def _bound_evaluation(
    system: JointSystem, evaluation: JointEvaluation
) -> JointEvaluation:
    """The system's fresh evaluation at ``evaluation.gamma``, equal to it bit for bit.

    Every field is compared: the point, the row terms, the raw weights, the moment,
    the observed bread, the expected information, the contributions and the units.  A
    record of another system, another point or another sample is refused; the fresh
    record is the one used, so every row it describes is the system's own.
    """
    fresh = system.evaluate(evaluation.gamma)
    names = (
        "gamma",
        "log_mean",
        "residual",
        "curvature",
        "expected_curvature",
        "weights",
        "moment",
        "bread",
        "information",
        "contributions",
    )
    units, given = fresh.units, evaluation.units
    bound = (
        all(_same(getattr(fresh, name), getattr(evaluation, name)) for name in names)
        and _same(units.identifiers, given.identifiers)
        and _same(units.inactive, given.inactive)
        and units.nobs == given.nobs
    )
    if not bound:
        raise ValueError("the evaluation is not this system's evaluation at its point")
    return fresh


def joint_hc(
    system: JointSystem,
    evaluation: JointEvaluation,
    method: HCMethod,
    convention: ResolvedConvention,
    *,
    frequency: bool,
    bread: Bread = "observed",
) -> HCCovariance:
    """The named HC covariance of a joint system without effects at an evaluation.

    Parameters
    ----------
    system : JointSystem
        The system that owns the rows: the active design, the raw weights, the
        offset and the working variance.
    evaluation : JointEvaluation
        The evaluation of that system at the point, the certificate's or a retained
        one; it is formed again from the system and must equal the record given bit
        for bit (binding of point, rows, weights and geometry).
    method : HC method
        The exact method string.
    convention : ResolvedConvention
        The convention in force for this request.
    frequency : bool
        Whether the weights are frequency counts.
    bread : {"observed", "expected"}, optional
        The observed Jacobian (default, the composite) or the expected
        information (the stock sandwich).  The hats use the expected geometry either
        way.

    Raises
    ------
    SpecificationError
        ``specification.no_residual_dof`` (HC1 at ``N = K``),
        ``specification.conflicting`` (a second correction), or
        ``specification.unit_leverage`` (HC2 to HC5, HC4m), each by the preflight,
        before the evaluation and any solve.
    ValueError
        A design that is not of full column rank, an unknown method or bread, or an
        evaluation that is not this system's at its point.
    TypeError
        Another type of system or evaluation, or ``frequency`` not a ``bool``.
    """
    if type(system) is not JointSystem:
        raise TypeError("system must be a JointSystem")
    if type(evaluation) is not JointEvaluation:
        raise TypeError("evaluation must be a JointEvaluation")
    if bread not in ("observed", "expected"):
        raise ValueError("bread must be 'observed' or 'expected'")
    matrix, a = system.active_design, system.active_weights
    preflight = hc_preflight(matrix, a, method, convention, frequency=frequency)
    fresh = _bound_evaluation(system, evaluation)
    return _joint_composition(
        matrix, a, fresh, preflight.applied, frequency=frequency, bread=bread
    )


def _joint_composition(
    matrix: FloatArray,
    a: FloatArray,
    fresh: JointEvaluation,
    applied: AppliedConvention,
    *,
    frequency: bool,
    bread: Bread,
) -> HCCovariance:
    """The composition of a joint evaluation: an internal helper, not an entry point.

    ``matrix`` and ``a`` are the active design and raw weights of the rows that
    ``fresh`` evaluates, ``applied`` the preflight's convention of the request on
    those rows: the expected-information hats of the evaluation's expected curvature,
    the influence rows under the requested bread and the adjusted meat.  Nothing here
    can tell the rows of one design from an evaluation of another: :func:`joint_hc`
    binds the evaluation to its system first, and the fit-time provider of
    :mod:`powermean.inference.hc_provider` binds its scores to its rows.
    """
    leverage = dense_leverage(matrix, a, fresh.expected_curvature, frequency=frequency)
    jacobian = fresh.jacobian if bread == "observed" else fresh.expected_jacobian
    rows = influence_values(fresh.contributions, moment_influence_map(jacobian))
    return hc_covariance(
        rows,
        a,
        leverage,
        applied,
        frequency=frequency,
        bread=bread,
        layout="dense",
    )


# ----------------------------------------------------------------------- linear


@dataclass(frozen=True, eq=False)
class LinearRequest:
    """One linear equation as the solve receives it: the owner of its rows.

    Formed by :func:`linear_request` before the solve, from the arrays the solve is
    given, by the solve's own validation; every array is an owned read-only copy, one
    value per input row in the input order, the rows of zero weight included.

    Attributes
    ----------
    response : array
        The log-response ``l``.
    design : array
        ``X``, one row per input row and one column per coefficient.
    offset : array
        The offset ``o``, zero where none was given.
    weights : array
        The raw weights ``a``, nonnegative, one where none were given.
    frequency : bool
        Whether the weights are frequency counts, as the caller stated.
    """

    response: FloatArray
    design: FloatArray
    offset: FloatArray
    weights: FloatArray
    frequency: bool

    def __post_init__(self) -> None:
        """Validate direct construction and own every array, as the factory does."""
        if not isinstance(self.frequency, bool):
            raise TypeError("frequency must be a bool")
        values, matrix = _linear_rows(self.response, self.design)
        rows = int(values.size)
        object.__setattr__(self, "response", _immutable(values))
        object.__setattr__(self, "design", _immutable(matrix))
        object.__setattr__(
            self, "offset", _immutable(_linear_offset(self.offset, rows))
        )
        object.__setattr__(
            self, "weights", _immutable(_linear_weights(self.weights, rows))
        )

    @property
    def n_rows(self) -> int:
        """The number of input rows."""
        return int(self.response.size)

    @property
    def dimension(self) -> int:
        """The number of coefficients: the columns of the design."""
        return int(self.design.shape[1])

    @property
    def active(self) -> IndexArray:
        """The positions of the rows of positive weight, increasing."""
        return frozen_index(np.flatnonzero(self.weights > 0))

    @property
    def inactive(self) -> IndexArray:
        """The positions of the rows of zero weight, increasing."""
        return frozen_index(np.flatnonzero(~(self.weights > 0)))


@dataclass(frozen=True, eq=False, init=False)
class LinearSolve:
    """One owned equation and the candidate computed from it, captured together.

    Construction copies the request before the accepted solve and retains its
    result.  It never takes a caller candidate: equal residuals at a coefficient
    vector do not establish which design was solved.  The two outward properties
    are detached views, including array headers and nested candidate records.
    :func:`linear_hc` still checks rows and residuals afresh and computes every
    numerical component from the captured equation; no cached geometry is used.

    Attributes
    ----------
    request : LinearRequest
        The owner of the equation.
    candidate : LinearCandidate
        The candidate of :func:`powermean.solve.linear.solve` on that equation.
    """

    _request: LinearRequest
    _candidate: LinearCandidate

    def __init__(self, request: LinearRequest) -> None:
        """Capture the equation before solving it once; no candidate input exists."""
        if type(request) is not LinearRequest:
            raise TypeError("request must be a LinearRequest")
        owned = replace(request)
        candidate = solve(
            owned.response, owned.design, offset=owned.offset, weights=owned.weights
        )
        object.__setattr__(self, "_request", owned)
        object.__setattr__(self, "_candidate", candidate)

    @property
    def request(self) -> LinearRequest:
        """A detached view of the captured input equation."""
        return replace(self._request)

    @property
    def candidate(self) -> LinearCandidate:
        """A detached candidate, with independently owned unit and factor headers."""
        candidate = self._candidate
        units = candidate.units
        detached_units = SamplingUnits(
            frozen_index(units.identifiers), frozen_index(units.inactive), units.nobs
        )
        factor = candidate.factorization
        return replace(
            candidate,
            units=detached_units,
            factorization=None if factor is None else replace(factor),
        )


def linear_request(
    response: ArrayLike,
    design: ArrayLike,
    *,
    offset: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    frequency: bool,
) -> LinearRequest:
    """The owner of one linear equation, captured before its solve.

    Parameters
    ----------
    response, design, offset, weights
        As :func:`powermean.solve.linear.solve` takes them, validated by its own
        checks: a nonempty finite vector ``l``, a finite matrix with one row per value
        and a column, an offset that is omitted, a scalar or aligned, weights that are
        omitted or aligned, nonnegative and not all zero.
    frequency : bool
        Whether the weights are frequency counts; never inferred from their values.

    Returns
    -------
    LinearRequest
        Owned read-only copies of the four arrays and the kind.

    Raises
    ------
    ValueError
        The refusals of the solve's validation.
    TypeError
        ``frequency`` is not a ``bool``.
    """
    if not isinstance(frequency, bool):
        raise TypeError("frequency must be a bool")
    values, matrix = _linear_rows(response, design)
    rows = int(values.size)
    return LinearRequest(
        values,
        matrix,
        _linear_offset(offset, rows),
        _linear_weights(weights, rows),
        frequency,
    )


def solve_request(request: LinearRequest) -> LinearSolve:
    """The accepted least-squares solve of the request's equation, paired with it.

    Parameters
    ----------
    request : LinearRequest
        The owner of the equation.

    Returns
    -------
    LinearSolve
        The request and the candidate that :func:`powermean.solve.linear.solve`
        returned on its arrays, whatever its reason.

    Raises
    ------
    TypeError
        ``request`` is not a :class:`LinearRequest`.
    """
    return LinearSolve(request)


def _bound_candidate(request: LinearRequest, candidate: LinearCandidate) -> FloatArray:
    """The owner's residuals at the candidate's coefficients, the candidate's own.

    The candidate's units are the request's rows of positive and of zero weight, its
    weights the request's on those rows, and its residuals ``e = (l - o) - X beta`` as
    the solve forms them from the request's rows, each bit for bit; otherwise the
    candidate is not the solve of this request (``ValueError``).  No solve is run.
    """
    coefficients, residuals = candidate.coefficients, candidate.residuals
    if coefficients is None or residuals is None:
        raise ValueError(
            f"a candidate that ended with {candidate.reason!r} has no covariance"
        )
    units, active = candidate.units, request.active
    bound = (
        candidate.dimension == request.dimension
        and units.nobs == request.n_rows
        and _same(units.identifiers, active)
        and _same(units.inactive, request.inactive)
        and _same(candidate.weights, request.weights[active])
    )
    if not bound:
        raise ValueError(NOT_SOLVED)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        target: FloatArray = request.response[active] - request.offset[active]
        fresh: FloatArray = target - request.design[active] @ coefficients
    if not np.isfinite(fresh).all() or not _same(fresh, residuals):
        raise ValueError(NOT_SOLVED)
    return fresh


def _corrected_rows(influence: Influence) -> FloatArray:
    """One working-precision residual correction of the linear HC2-to-HC5 rows.

    For the stored Jacobian ``J = -C``, map ``A = C^-1`` and scores ``s``, first
    form ``u0 = s A'`` with the protected primitive. In exact arithmetic,
    ``u0 + (s + u0 J') A' = s A'``. The correction reduces the measured HC4
    cancellation error on the fixed near-unit regression witness; one float64
    step is not a general componentwise backward-error or accuracy guarantee.

    A diagonal power-of-two map ``D`` brings the positive bread diagonal into
    ``[0.5, 2)``. Form the residual in coordinates ``s D``, ``u0 D^-1`` and
    ``D J D``, then apply ``D^-1 A D^-1`` and map back with ``D``. ``ldexp``
    applies the combined exponents directly, without an overflowing reciprocal
    or product of scale factors. This avoids the overflowing opposite-signed
    ``u0 J'`` products in the scaled five-row regression witness. The stored
    bread, map, scores and HC multipliers are unchanged. No factorization,
    precision upgrade, floor or fallback is used; nonfinite intermediates raise.
    """
    scores, mapping = influence.contributions, influence.influence_map
    first = influence_values(scores, mapping)
    _, exponent = np.frexp(-np.diag(influence.jacobian))
    shift = -(exponent // 2)
    pair = shift[:, None] + shift[None, :]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled_scores = np.ldexp(scores, shift)
        scaled_first = np.ldexp(first, -shift)
        scaled_jacobian = np.ldexp(influence.jacobian, pair)
        scaled_mapping = np.ldexp(mapping, -pair)
    for value in (scaled_scores, scaled_first, scaled_jacobian, scaled_mapping):
        _finite_result(value, "influence residual")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        residual: FloatArray = scaled_scores + scaled_first @ scaled_jacobian.T
    residual = _finite_result(residual, "influence residual")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        correction = influence_values(residual, scaled_mapping)
        rows: FloatArray = first + np.ldexp(correction, shift)
    return _immutable(_finite_result(rows, "influence values"))


def linear_hc(
    solved: LinearSolve, method: HCMethod, convention: ResolvedConvention
) -> HCCovariance:
    """The named HC covariance of the linear path at its request's solve.

    Parameters
    ----------
    solved : LinearSolve
        The request that owns the equation and the candidate of its solve, at full
        column rank; read, never modified.
    method : HC method
        The exact method string.
    convention : ResolvedConvention
        The convention in force for this request.

    Raises
    ------
    SpecificationError
        As :func:`joint_hc`, by the preflight on the request's design and weights,
        before the candidate is read and before the bread solve.
    ValueError
        A design that is not of full column rank, an unknown method, a candidate
        without coefficients, or a candidate that is not the solve of this request
        at its coefficients.
    TypeError
        Another type of pair.

    Notes
    -----
    The preflight runs on the owner's design and weights of every input row before
    the bread solve of :func:`~powermean.inference.contributions.linear_influence`,
    as it would before the least-squares solve itself; the binding check follows it
    and precedes that solve.  The least-squares equation has observed bread equal to
    its expected information, so ``bread`` is ``"observed"`` and the hat is the
    weighted-design hat ``a_i x_i' (X'AX)^-1 x_i`` of the owner's design (per replica
    with frequency counts).  The scores are the candidate's units, weights and
    residuals, equal bit for bit to the owner's at its coefficients, with the owner's
    design. HC0 and HC1 retain the protected primitive's influence rows bit for bit.
    HC2 to HC5 and HC4m apply :func:`_corrected_rows` to the record's own map and
    Jacobian: one residual correction in power-of-two scaled coordinates. This
    addresses the measured near-unit cancellation without a universal error bound.
    """
    found = _method(method)
    if type(solved) is not LinearSolve:
        raise TypeError("solved must be a LinearSolve")
    request, candidate = solved._request, solved._candidate
    frequency = request.frequency
    preflight = hc_preflight(
        request.design, request.weights, found, convention, frequency=frequency
    )
    _bound_candidate(request, candidate)
    influence: Influence = linear_influence(
        candidate, request.design, frequency=frequency
    )
    active = request.design[preflight.active]
    a = influence.weights
    leverage = dense_leverage(active, a, np.ones(a.shape), frequency=frequency)
    rows = (
        influence_values(influence.contributions, influence.influence_map)
        if found in ("HC0", "HC1")
        else _corrected_rows(influence)
    )
    return hc_covariance(
        rows,
        a,
        leverage,
        preflight.applied,
        frequency=frequency,
        bread="observed",
        layout="dense",
    )


# --------------------------------------------------------------------- absorbed


@dataclass(frozen=True)
class HCComposition:
    """Structural facts and the current observed influence of one composition."""

    preflight: HCPreflight | None = None
    code_tuples: CachedCodeTuples | None = None
    observed_influence: FloatArray | None = None


def _absorbed_composition(
    design: FloatArray,
    basis: CertificateBasis,
    rows: FreshRows,
    fisher: FloatArray,
    method: HCMethod,
    *,
    convention: ResolvedConvention,
    bread: Bread,
    prepared: HCComposition | None = None,
) -> HCCovariance:
    """The composition of given absorbed rows: an internal helper, not an entry point.

    ``design`` is ``X`` on every row of the grouping, ``basis`` the kept nuisance
    coordinates of those rows, ``rows`` the observed ``b = a r`` and ``C = a c`` with
    the raw weights and their kind, ``fisher`` the expected row curvature ``w_i`` at
    the same point, unweighted (ones for the linear equation).  Nothing here can tell
    rows of one point from a Fisher vector of another: :func:`absorbed_hc` derives
    all four from one admitted point and is the entry point.
    """
    if type(rows) is not FreshRows or type(basis) is not CertificateBasis:
        raise TypeError("rows must be FreshRows and basis a CertificateBasis")
    matrix = _finite_array(design, "design")
    a, frequency = rows.weights, rows.frequency
    n = basis.grouping.n_rows
    if matrix.ndim != 2 or matrix.shape[0] != n or not matrix.shape[1]:
        raise ValueError("design must have one row per row of the grouping, a column")
    if a.shape != (n,) or rows.b.shape != (n,) or rows.curvature.shape != (n,):
        raise ValueError("the rows belong to another grouping")
    preflight = None if prepared is None else prepared.preflight
    code_tuples = None if prepared is None else prepared.code_tuples
    observed_influence = None if prepared is None else prepared.observed_influence
    if preflight is None:
        preflight = hc_preflight(
            matrix, a, method, convention, frequency=frequency, basis=basis
        )
    active = preflight.active
    leverage = absorbed_leverage(
        matrix, basis, a, fisher, frequency=frequency, code_tuples=code_tuples
    )
    if bread == "observed" and observed_influence is not None:
        inf_rows = observed_influence
    elif bread == "expected" and leverage.slope_map is not None:
        inf_rows = bread_influence(matrix, basis, rows.b, leverage.slope_map.columns)
    else:
        curvature = rows.curvature if bread == "observed" else leverage.gram_weights
        inf_rows = same_point_influence(
            matrix, basis, rows.b, hessian(matrix, curvature, basis), bread=bread
        ).influence
    return hc_covariance(
        inf_rows[active],
        a[active],
        leverage,
        preflight.applied,
        frequency=frequency,
        bread=bread,
        layout="absorbed",
    )


def _absorbed_options(kind: Organization, inner: InnerControls) -> AbsorbedOptions:
    """The accepted options record of a nonlinear admission's inner controls."""
    if inner.profile_tol is None:
        return AbsorbedOptions(
            absorbed=kind,
            absorb_tol_final=inner.tolerance,
            absorb_max_sweeps=inner.max_sweeps,
            absorb_acceleration=inner.acceleration,
        )
    return AbsorbedOptions(
        absorbed=kind,
        absorb_tol_final=inner.tolerance,
        absorb_max_sweeps=inner.max_sweeps,
        absorb_acceleration=inner.acceleration,
        profile_tol=inner.profile_tol,
    )


def _readmitted(point: AdmittedPoint) -> AdmittedPoint:
    """The point's record admitted again by the accepted path, equal to the point.

    :func:`~powermean.solve.absorbed_certificate_point.admit_point` of the point's own
    record under its inner controls (with its sample and working variance for a
    nonlinear record) must give its kind, its sample (the same object), its slopes,
    its base and its index bit for bit; otherwise the point is not the admitted point
    of its record on its sample (``ValueError``; ``TypeError`` for a record that is
    not of its kind).  The fresh record is the one used.
    """
    record, inner = point.record, point.inner
    if point.kind == "linear":
        if not isinstance(record, LinearPoint):
            raise ValueError(NOT_ADMITTED)
        controls = LinearControls(inner.tolerance, inner.max_sweeps, inner.acceleration)
        fresh = admit_point(record, controls)
    else:
        if isinstance(record, LinearPoint) or point.variance is None:
            raise ValueError(NOT_ADMITTED)
        kind: Organization = "profiled" if point.kind == "profiled" else "index"
        if point.kind == "root":
            # A root point keeps the requested organization's controls as given.
            kind = "profiled" if inner.profile_tol is not None else "index"
        whole = WholePoint(point.sample, point.variance, record)
        fresh = admit_point(whole, _absorbed_options(kind, inner))
    bound = (
        fresh.kind == point.kind
        and fresh.sample is point.sample
        and _same(fresh.theta, point.theta)
        and _same(fresh.base, point.base)
        and _same(fresh.q, point.q)
    )
    if not bound:
        raise ValueError(NOT_ADMITTED)
    return fresh


def _expected_curvature(point: AdmittedPoint) -> FloatArray:
    """The expected row curvature ``w_i`` at the admitted point, unweighted.

    One for the WLS equation; else the information weight of the accepted row kernel
    at ``q`` on the sample's response, the evaluation whose score and observed
    curvature ``fresh_rows`` weighs.
    """
    sample = point.sample
    if point.variance is None:
        return np.ones(sample.count)
    rows = evaluate_rows(sample.problem.response, point.q, variance=point.variance)
    return rows.fisher_curvature


def absorbed_hc(
    point: AdmittedPoint,
    method: HCMethod,
    convention: ResolvedConvention,
    *,
    bread: Bread = "observed",
) -> HCCovariance:
    """The named HC covariance of the slopes of an absorbed fit at an admitted point.

    Parameters
    ----------
    point : AdmittedPoint
        The admitted point of the accepted certificate path, which owns the sample
        (response, offset, raw weights and their kind, scaled slopes and blocks) and
        the point (``q``, ``theta``, base, working variance or the WLS equation); it
        is admitted again from its record and must equal the record given bit for
        bit.  Every component is derived from it: the full nuisance basis of its
        sample, the observed rows ``b = a r`` and ``C = a c`` and the expected row
        curvature ``w``, both at its ``q``.
    method : HC method
        The exact method string.
    convention : ResolvedConvention
        The convention in force for this request.
    bread : {"observed", "expected"}, optional
        The observed Hessian ``B' diag(C) B`` (default) or the expected one
        ``B' diag(a w) B``, both assembled on the sample's full basis.

    Raises
    ------
    SpecificationError
        As :func:`joint_hc`, by the preflight on the full design, before any
        assembly or solve.
    ValueError
        A point that is not the admitted point of its record on its sample, a fit
        without a slope (whose slope covariance is empty), an unknown method or
        bread, or a design that is not identified.
    TypeError
        Another type of point, or a record that is not of the point's kind.
    """
    found = _method(method)
    if type(point) is not AdmittedPoint:
        raise TypeError("point must be an AdmittedPoint")
    if bread not in ("observed", "expected"):
        raise ValueError("bread must be 'observed' or 'expected'")
    fresh = _readmitted(point)
    sample = fresh.sample
    design = sample.design
    if design is None:
        raise ValueError("a fit without a slope has no slope covariance")
    basis = owned_basis(sample)
    rows = fresh_rows(fresh)
    fisher = _expected_curvature(fresh)
    return _absorbed_composition(
        design.matrix, basis, rows, fisher, found, convention=convention, bread=bread
    )


__all__ = [
    "HC4M_POWERS",
    "HC4_CAP",
    "HC5_FLOOR",
    "HC5_HALF",
    "HC5_SHARE",
    "LEVERAGE_METHODS",
    "HCCovariance",
    "HCPreflight",
    "LinearRequest",
    "LinearSolve",
    "Multipliers",
    "absorbed_hc",
    "adjusted_estimate",
    "hc_covariance",
    "hc_preflight",
    "joint_hc",
    "linear_hc",
    "linear_request",
    "multipliers",
    "solve_request",
]
