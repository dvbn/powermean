"""The named HC covariances HC0 to HC5 and HC4m: multipliers, meat, composition.

``design/04_inference_and_effects.md``, Sections 2.3 and 3 (D55, D57, D60);
``plan/evidence/M1_contract.txt``, rules B1, B2, B5, B6, B12 and Sections 3.1 and 3.2
(the multipliers of sandwich 3.1-1 ``meatHC``, read from its source, written here from
the pinned formulas); ``plan/evidence/M1_validation_plan.txt``, Section 4;
``plan/evidence/D55_nonlinear_HC_reference.txt`` (the composite with the observed
bread); decisions D49, D51, D54, D57, D60 and D65.  Task M1.18: no public name or
default here.  The exact strings of a request without fixed effects (contract, Section
6(a)) reach these kernels through the fit-time provider of
:mod:`powermean.inference.hc_provider` and the retained adapter of
:mod:`powermean.results.retained_hc`.

Definitions (Section 3.2).  With ``h_i`` the per-replica leverage of
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
bit through the accepted primitive (:func:`powermean.inference.ssc.primitive_estimate`
of the rows ``sqrt(q_i) u_i``).  The nonlinear hats use the expected-information
geometry whatever the bread (D57): with the observed bread the result is the declared
COMPOSITE of ``D55_nonlinear_HC_reference.txt``, not stock GLM covariance; with
``bread="expected"`` it is the stock sandwich.  Both components are recorded.

Ownership (D60, Section 3.1).  The scalar is :mod:`powermean.inference.ssc`'s:
``apply_convention(..., kind="robust", method=...)`` gives the method's own factor
(``N/(N-K)`` for HC1, defined for ``N > K`` only, else
``specification.no_residual_dof``; one otherwise), refuses an explicit convention
whose factor is a second correction (``specification.conflicting``) and carries the
reference distribution of the convention in force, inherited or default conventions
never multiplying.  The factor is applied once by
:func:`powermean.inference.ssc.scaled_covariance`.

Refusals before any solve (rule B12(a)): the preflight (:func:`hc_preflight`).  From
the design, the raw weights, their kind, the method and the convention alone, which a
fit has before it solves and a post-fit request has in its retained state, the
preflight decides the exact full column rank of the design on its active rows
(:func:`powermean.inference.leverage.exact_full_rank`, the rational echelon: a
dependent column refuses with ``ValueError``), the statistical count, the method's
factor (HC1 at ``N = K``: ``specification.no_residual_dof``; an explicit second
correction: ``specification.conflicting``).  Every entry point runs it before any
evaluation, factorization or solve, so a refusal costs none; the same function serves a
fit before its solve (tasks M1.22 and M1.45) with the same identifiers.  Unit leverage
is not refused (owner's ruling 74, as on the fixed-effect routes): where a computed hat
is above ``UNSTABLE_ABOVE`` (``sandwich``'s near-one criterion),
:func:`hc_covariance` issues ``covariance.unstable_leverage`` and HC2 to HC5 and HC4m
are missing (:attr:`HCCovariance.undefined`).  A multiplier or product beyond the
doubles raises ``FloatingPointError`` (no floor, no clamp, no fallback).  Exact
zero variances stay zero (D49), representable tiny values are kept (D54), a negative
variance gives a missing standard error downstream.

Fits with fixed effects do not use that preflight: at ``omega != 0`` (owner's rulings
55, 63 and 65) their request is prepared and composed by
:mod:`powermean.inference.absorbed_block_hc` (no exact rank or unit-leverage decision,
singleton rows excluded and counted, sandwich's warning for hat values close to one,
hats by block elimination, none for HC0 and HC1), and fixed-effect log-OLS
(``omega = 0``, ruling 52) by :mod:`powermean.inference.absorbed_linear`, with the
same rules and the leverage of the dummy-variable regression.

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
accepted solve returned for it (:class:`LinearSolve`, formed by :func:`solve_request`,
the private request/solve handoff that captures the equation before the solve): the
candidate's units, weights and residuals are checked bit for bit against the owner's
rows at the candidate's coefficients (``e = l - o - X beta``, the solve's own
arithmetic, no second solve), and the design of the bread, of the hats and of the
scores is the owner's own, since a candidate alone carries neither response, offset
nor design.  Records own their
arrays: the leverage record of a composition is copied whole
(:func:`powermean.inference.leverage.detached`), so that no caller array header or
record is retained.  The reporting map (rule B3) is not applied here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from powermean.data.accounting import frozen_index
from powermean.data.inference_requests import HC_METHODS, HCMethod
from powermean.errors import CovarianceWarning
from powermean.fe.certificate_basis import CertificateBasis
from powermean.inference.absorbed_rank import AbsorbedCount, absorbed_count
from powermean.inference.bread import Bread
from powermean.inference.contributions import (
    Influence,
    linear_influence,
    statistical_count,
)
from powermean.inference.influence import influence_values, moment_influence_map
from powermean.inference.leverage import (
    FullRank,
    Leverage,
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
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_array, _finite_result
from powermean.solve.linear import LinearCandidate, solve
from powermean.solve.linear import _offset as _linear_offset
from powermean.solve.linear import _response_and_design as _linear_rows
from powermean.solve.linear import _weights as _linear_weights
from powermean.systems.joint import JointEvaluation, JointSystem, SamplingUnits

LEVERAGE_METHODS: tuple[str, ...] = ("HC2", "HC3", "HC4", "HC4m", "HC5")
# The constants of the formulas of sandwich's meatHC (Section 3.2), not options.
HC4_CAP = 4.0
HC4M_POWERS = (1.0, 1.5)
HC5_FLOOR = 4.0
HC5_SHARE = 0.7
HC5_HALF = 0.5
# R sandwich 3.1-3 (CRAN, 2026-08-02), file R/vcovHC.R, function meatHC: for every
# type but "const" the hat values are tested by
#     idx <- diaghat > 1 - sqrt(.Machine$double.eps)
# and, for HC2, HC3, HC4, HC4m and HC5, the covariance is computed with
#     warning("%s covariances are numerically unstable for hat values close to 1
#              (and undefined if exactly 1) as for observation(s) %s").
# The same criterion on the same floating-point type is used here (ruling 65): it is
# sandwich's constant, not one of this package.  sandwich also warns for HC0 and HC1
# ("become (close to) singular if hat values are (close to) 1"); that warning has no
# counterpart here (the routes with fixed effects compute no hat value for them).
UNSTABLE_ABOVE = 1.0 - float(np.sqrt(np.finfo(np.float64).eps))
Layout = Literal["dense", "absorbed"]
NOT_SOLVED = "the candidate is not the solve of this request at its coefficients"


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
    leverage : Leverage or None
        The per-replica hats and their provenance, an owned copy: no array header or
        nested record of the caller's record is retained.  ``None`` for HC0 and HC1
        of a joint fixed-effect fit, which compute no hat
        (:mod:`powermean.inference.absorbed_block_hc`).
    multipliers : Multipliers or None
        ``q_i``, ``r_i``, ``p`` and ``max r``; ``None`` with ``leverage``.
    bread : {"observed", "expected"}
        The bread of the influence rows.
    layout : {"dense", "absorbed"}
        Slopes alone, or the full nuisance basis.
    influence : array
        ``u_i = A s_i`` on the active rows, owned.
    dropped : int
        The singleton rows excluded from the request (joint fixed-effect fits; zero
        elsewhere): ``N`` and the rank of ``applied`` do not count them.
    unstable : int
        The observations whose hat value is above sandwich's criterion
        ``1 - sqrt(eps)``, for which the note ``covariance.unstable_leverage`` was
        issued (HC2 to HC5 and HC4m; zero otherwise).
    undefined : int
        The observations above the near-one warning criterion, on every route
        (owner's ruling 74).  Where it is positive the request is undefined:
        ``covariance`` is missing (every entry NaN), ``multipliers`` is ``None``
        and ``estimate`` is the unadjusted primitive of the same influence rows
        (``gate_covariance`` is then the covariance the fit's coefficient gate
        reads; the reported standard errors of the request are missing).
    """

    method: HCMethod
    covariance: FloatArray
    estimate: CovarianceEstimate
    applied: AppliedConvention
    leverage: Leverage | None
    multipliers: Multipliers | None
    bread: Bread
    layout: Layout
    influence: FloatArray
    dropped: int = 0
    unstable: int = 0
    undefined: int = 0

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
    def gate_covariance(self) -> FloatArray:
        """``covariance``, or the unadjusted estimate where the request is undefined."""
        if not self.undefined:
            return self.covariance
        scaled: FloatArray = self.factor * self.estimate.covariance
        return scaled

    @property
    def hat_geometry(self) -> str:
        """The leverage convention: the expected-information geometry (D57)."""
        return "expected" if self.leverage is None else self.leverage.geometry


def missing_covariance(like: FloatArray) -> FloatArray:
    """A covariance of the shape of ``like`` whose every entry is missing (NaN)."""
    return _immutable(np.full(like.shape, np.nan))


def _unstable(method: HCMethod, leverage: Leverage) -> int:
    """Issue sandwich's warning for hat values close to one; their number.

    ``UNSTABLE_ABOVE`` and the wording follow ``sandwich::meatHC`` (see the constant).
    """
    hat = leverage.hat[leverage.active]
    count = int(np.count_nonzero(hat > UNSTABLE_ABOVE))
    if count:
        warnings.warn(
            CovarianceWarning(
                "covariance.unstable_leverage", {"method": method, "count": count}
            ),
            stacklevel=2,
        )
    return count


# ----------------------------------------------------------------- multipliers


def _method(method: HCMethod) -> HCMethod:
    """One of the seven exact method strings, checked at run time."""
    if method not in HC_METHODS:
        raise ValueError(f"unknown HC method {method!r}")
    return method


def _exponents(method: HCMethod, r: FloatArray, max_r: float) -> FloatArray:
    """The exponent ``e_i`` of ``(1 - h_i)^-e_i`` (Section 3.2)."""
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
        ``specification.conflicting`` (an explicit second correction) or
        ``specification.no_residual_dof`` (HC1 at ``N = K``), in that order.
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
    # Owner's ruling 74: a computed hat above the near-one criterion is a hat of one;
    # HC2 to HC5 are then missing after the note, as on the fixed-effect routes.
    undefined = _unstable(method, owned) if method in LEVERAGE_METHODS else 0
    found = None if undefined else multipliers(method, owned, applied.counts.count)
    values = np.ones(rows.shape[0]) if found is None else found.values
    estimate = adjusted_estimate(rows, weights, values, frequency=frequency)
    scaled = scaled_covariance(estimate, applied)
    return HCCovariance(
        method,
        missing_covariance(scaled.covariance) if undefined else scaled.covariance,
        estimate,
        applied,
        owned,
        found,
        bread,
        layout,
        _immutable(rows),
        unstable=undefined,
        undefined=undefined,
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
        The observed Jacobian (default, the D57 composite) or the expected
        information (the stock sandwich).  The hats use the expected geometry either
        way.

    Raises
    ------
    SpecificationError
        ``specification.no_residual_dof`` (HC1 at ``N = K``) or
        ``specification.conflicting`` (a second correction), each by the preflight,
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
    "adjusted_estimate",
    "hc_covariance",
    "hc_preflight",
    "joint_hc",
    "linear_hc",
    "linear_request",
    "missing_covariance",
    "multipliers",
    "solve_request",
]
