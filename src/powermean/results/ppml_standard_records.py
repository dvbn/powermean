"""Records of the internal joint-PPML standard reading, its statuses and its work.

Nothing in the package imports this module but the other ``ppml_standard_*`` modules and
:mod:`powermean.interface.ppml_audit`.

A standard reading follows the policy ``ppml_standard_v1``: the original equations of an
admitted point read afresh (rows, criterion, native-coordinate gradient, fixed-effect
level means, positive-curvature nuisance support and the unchanged inner checks).  It is
not a full-system audit and not a
:class:`~powermean.solve.absorbed_certificate.FEPointCertificate`: it holds no curvature
factor, conditioning, scaled score, index correction or coefficient gate, and its
verdict is qualified by its mode.  ``StandardPointAssessment.accepted`` is the point
predicate of the standard reading and nothing more; ``score`` is the largest absolute
component of the fresh native gradient; ``error`` the largest ratio of a checked value
to one of its bounds, NaN when a value is missing or a stage or structural requirement
failed.  The ratios are retry diagnostics, never a tolerance.

The computations that the full audit runs and standard mode omits are
:data:`OMITTED_CHECKS`: their quantity is ``None`` with the reason
``not_run_standard_mode``, never zero, a pass or a failure, and ``actual_calls`` is zero
because standard mode calls none of their routines (an explicit no-call record).

The opening-retry reading of the internal index route follows the policy
``ppml_standard_index_opening_v2``.  Only the reading that the retry rule requests of
the first attempt's initialized start is of this kind (:data:`OPENING_RETRY_KIND`): it
keeps the admission, the basis, the fresh rows, the fresh native gradient and the
positive-curvature support, and does not run the fixed-effect level means or the inner
checks.  It is never accepted, its error is NaN and its verdict is
``standard_opening_retry_partial``, whatever its data.  Its effects check holds no
value, NaN ratios and the reason ``not_run_opening_retry``, not a measured failure, and
:func:`opening_omitted_checks` lists the operations it does not run, each with no
quantity and zero actual calls.  Every other reading is ``complete``, the standard
reading under ``ppml_standard_v1``; ``complete`` says which policy formed a reading, not
that it passed.  The fit records of the index route carry the opening-retry policy
(:func:`fit_policy`) while its complete readings keep the point tag
``ppml_standard_v1``; a profiled fit keeps ``ppml_standard_v1`` throughout.

The internal equation-aware stop of the index route
(:mod:`powermean.solve.newton_absorbed`) is the fit policy
``ppml_standard_index_equation_stop_v3`` (:data:`INDEX_STOP_POLICY`): the fit, its
selection, envelope and provenance carry it exactly when the fit's solve ran that stop,
as its bound probes show (:func:`fit_policy` with ``stop``), never by the route alone.
Its complete readings keep the point tag ``ppml_standard_v1`` and its opening-retry
readings their partial tag, kind, omissions and refusals.  The probes of the stop are
:class:`RetainedStopProbes`: each fresh-equations probe with its attempt, accepted step,
content bindings, bounds, maximum or refusal, outcome, combined flag and own seconds.  A
probe is never a standard reading, a native stop or inference evidence, and an empty
record is a stop that ran without a probe.  :class:`PPMLWork` counts the probes and
their seconds apart, inside the solve's seconds and never added to them.

The bounded inference-map refinement.  A standard fit's inference records its requested
and executed method, ``full_joint`` (the reference and default) or ``residualized``,
which only the explicit internal alternative entry requests; no record of one method is
relabelled as the other.  The residualized route's scalar evidence is
:class:`ResidualizedEvidence` (algorithm :data:`ALGORITHM`): its controls, the
projection and its reconstruction checks, the K-dimensional ordinary slope factor record
and its one transpose solve, initial G and full H/A backward checks, fixed projection
work and both ledgers with their typed reservations, the failures and the content
digests of the delivered map and its geometry.  Its ordinary full-system factor is
:data:`FULL_NOT_RUN` with zero actual full-factor and full-solve calls, never a
zero-cost factor or a successful LU.  ``passed`` says that the recorded
finite-arithmetic predicates hold, recomputed from their raw values
(:func:`normwise_bound`, :func:`holds`): never a conditioning, forward-accuracy or
inference-validity certificate.  The evidence holds scalars, strings and an owned factor
record only, no map, coordinate, vector, right-hand side or factor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

import numpy as np

from powermean.fe.certificate_basis import CertificateBasis, SupportCheck
from powermean.model.variance import FloatArray
from powermean.results.absorbed_geometry import (
    RetainedGeometrySnapshot,
    RetainedRowsSnapshot,
)
from powermean.results.absorbed_geometry_records import (
    RetainedBasisSnapshot,
    RetainedMeasuresSnapshot,
    _owned_factor_record,
)
from powermean.results.absorbed_owned import (
    RetainedEffectEquations,
    RetainedInnerEvidence,
    _owned_coefficient_check,
)
from powermean.results.absorbed_scale_cache import RetainedInitialScaleCache
from powermean.results.owned import _optional
from powermean.solve.absorbed_certificate import Level
from powermean.solve.absorbed_certificate_factor import FactorRecord
from powermean.solve.absorbed_certificate_geometry import FEGeometry
from powermean.solve.absorbed_certificate_inner import InnerCheck, InnerEvidence
from powermean.solve.absorbed_certificate_point import (
    AdmittedPoint,
    EffectEquations,
    FreshRows,
    WorkStep,
)
from powermean.solve.absorbed_certificate_scale import FEScaleResult
from powermean.solve.assessment import CertificateControls
from powermean.solve.newton_absorbed import EquationProbe
from powermean.solve.newton_trace import Timing
from powermean.solve.reported import CoefficientCheck

POLICY: Policy = "ppml_standard_v1"
INDEX_RETRY_POLICY: Policy = "ppml_standard_index_opening_v2"
INDEX_STOP_POLICY: Policy = "ppml_standard_index_equation_stop_v3"
FULL_POLICY: Policy = "full_source229"
NOT_RUN = "not_run_standard_mode"
NOT_RUN_OPENING_RETRY = "not_run_opening_retry"
NOT_COMPUTED = "not_computed_before_selection"
ACCEPTED = "standard_point_accepted"
REJECTED = "standard_point_rejected"
OPENING_RETRY_VERDICT = "standard_opening_retry_partial"
COMPLETE_KIND = "complete"
OPENING_RETRY_KIND = "opening_retry"
GRADIENT = "native gradient"
EFFECTS = "effects"

Policy = Literal[
    "ppml_standard_v1",
    "ppml_standard_index_opening_v2",
    "ppml_standard_index_equation_stop_v3",
    "full_source229",
]
StandardStatus = Literal["converged", "refused", "unavailable", "not_run"]
InferenceStatus = Literal[
    "available", "degenerate", "refused", "unavailable", "not_applicable"
]
AuditStatus = Literal[
    "not_run", "verified", "verified_secondary", "failed", "unavailable"
]
Role = Literal["first", "opening", "second", "selected_final"]
ROLES: tuple[Role, ...] = ("first", "opening", "second")
InferenceMethod = Literal["full_joint", "residualized"]
METHODS: tuple[InferenceMethod, ...] = ("full_joint", "residualized")
ALGORITHM = "projection_map_v2"
FULL_NOT_RUN = "not_run_residualized"
NUISANCE_SOLVER = "not_run_projection_map_v2"
SLOPE_FACTOR = "ordinary factor of the K-dimensional slope bread G"


@dataclass(frozen=True)
class StandardControls:
    """The original controls that a standard reading applies, read once.

    ``gradient_tol`` and ``tol`` are those of the solver options; the inner controls
    are those of the absorbed options (``profile_tol`` only on the profiled route).
    """

    gradient_tol: float
    tol: float
    absorb_tol_final: float
    absorb_max_sweeps: int
    absorb_acceleration: bool
    profile_tol: float | None


@dataclass(frozen=True)
class StandardCheck:
    """One numerical check: its value, every bound, the ratio to each, the verdict.

    ``ratios`` holds ``value / bound`` for each bound, NaN for a missing value.
    """

    name: str
    value: float | None
    bounds: tuple[float, ...]
    ratios: tuple[float, ...]
    passed: bool
    reason: str


def bounded(
    name: str, value: float | None, bounds: tuple[float, ...], reason: str = ""
) -> StandardCheck:
    """The check that ``value`` is at most every bound; a missing value fails."""
    passed = value is not None and all(value <= bound for bound in bounds)
    ratios = tuple(math.nan if value is None else value / b for b in bounds)
    if not passed and not reason:
        reason = f"{name}: {value} exceeds {bounds} or is unavailable"
    return StandardCheck(name, value, bounds, ratios, passed, "" if passed else reason)


def not_run(name: str, bounds: tuple[float, ...]) -> StandardCheck:
    """A check that an opening-retry reading does not run (module docstring).

    No value, the bounds of the check it stands for, a NaN ratio to each, never passed,
    and the reason ``not_run_opening_retry``: not a measured failure.
    """
    ratios = tuple(math.nan for _ in bounds)
    return StandardCheck(name, None, bounds, ratios, False, NOT_RUN_OPENING_RETRY)


@dataclass(frozen=True)
class StandardFailure:
    """A stage that did not complete, or a structural requirement that does not hold."""

    stage: str
    reason: str


@dataclass(frozen=True)
class OmittedCheck:
    """A full-audit computation that standard mode does not execute (module docstring).

    ``routine`` names the accepted function that is not called.
    """

    name: str
    routine: str
    quantity: None = None
    reason: str = NOT_RUN
    actual_calls: int = 0


_SPARSE = "solve.absorbed_certificate_sparse"
OMITTED_CHECKS: tuple[OmittedCheck, ...] = (
    OmittedCheck(
        "curvature factor of Aplus",
        "solve.absorbed_certificate_factor.factorized(kind='curvature')",
    ),
    OmittedCheck("whitened scaled score", f"{_SPARSE}.curvature_score"),
    OmittedCheck(
        "inverse 1-norm estimate", "solve.absorbed_certificate_factor.inverse_norm"
    ),
    OmittedCheck("rcond audit", f"{_SPARSE}.conditioning"),
    OmittedCheck("correction solve delta = H^-1 S", f"{_SPARSE}._corrected"),
    OmittedCheck("index correction dq and its RMS", f"{_SPARSE}._change"),
    OmittedCheck("correction residual B'(b - C dq)", f"{_SPARSE}._residual"),
    OmittedCheck(
        "coefficient correction in standard errors",
        "solve.reported.coefficient_check",
    ),
)
_INNER = "solve.absorbed_certificate_inner"


def opening_omitted_checks(slopes: int) -> tuple[OmittedCheck, ...]:
    """The operations that an opening-retry reading does not run (module docstring).

    The fresh level means, and the inner checks of an index point: the same-``q``
    recovery of the effects with its reconstruction gaps, normalization and fresh
    check, then, with ``slopes`` active slopes, the cold auxiliary absorption.  Each is
    an explicit no-call record with the reason ``not_run_opening_retry``.
    """
    found = [
        OmittedCheck(
            "fixed-effect level means",
            "solve.absorbed_certificate_point.effect_equations",
            reason=NOT_RUN_OPENING_RETRY,
        ),
        OmittedCheck(
            "same-q recovery, its reconstruction gaps and normalization",
            f"{_INNER}._recovered",
            reason=NOT_RUN_OPENING_RETRY,
        ),
    ]
    if slopes:
        found.append(
            OmittedCheck(
                "auxiliary absorption of the slopes",
                f"{_INNER}._auxiliary",
                reason=NOT_RUN_OPENING_RETRY,
            )
        )
    return tuple(found)


def fit_policy(route: str, *, stop: bool = False) -> Policy:
    """The policy of a standard fit's records (module docstring).

    ``ppml_standard_v1`` off the index route; on it the equation-stop policy when the
    fit's solve ran the private equation-aware stop (``stop``: its probes are bound),
    else the opening-retry policy.  A complete reading keeps its own point tag
    ``ppml_standard_v1`` on either route.
    """
    if route != "index":
        return POLICY
    return INDEX_STOP_POLICY if stop else INDEX_RETRY_POLICY


def _complete(
    policy: str,
    kind: str,
    omitted: tuple[OmittedCheck, ...],
    checks: tuple[StandardCheck, ...],
) -> bool:
    """A complete-policy reading: its policy and kind, no omission, no check not run."""
    return (
        policy == POLICY
        and kind == COMPLETE_KIND
        and not omitted
        and all(check.reason != NOT_RUN_OPENING_RETRY for check in checks)
    )


@dataclass(frozen=True, eq=False)
class StandardPointAssessment:
    """The standard reading of one point (module docstring); not a certificate.

    ``point``, ``basis`` and the formed records are those of the inputs it read;
    ``rows``, ``effects``, ``criterion``, ``gradient``, ``support`` and ``inner`` are
    ``None`` when not formed; ``failures`` name the stages that did not complete and the
    structural requirements that do not hold; ``work`` the operations in order.
    ``kind`` is ``complete`` or ``opening_retry``; ``omitted`` lists the operations an
    opening-retry reading does not run, empty for a complete reading.
    """

    controls: StandardControls
    point: AdmittedPoint
    basis: CertificateBasis
    rows: FreshRows | None
    effects: EffectEquations | None
    criterion: float | None
    gradient: FloatArray | None
    support: SupportCheck | None
    inner: InnerEvidence | None
    checks: tuple[StandardCheck, ...]
    failures: tuple[StandardFailure, ...]
    work: tuple[WorkStep, ...]
    policy: str = POLICY
    kind: str = COMPLETE_KIND
    omitted: tuple[OmittedCheck, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether this is a complete standard reading, whatever its verdict."""
        return _complete(self.policy, self.kind, self.omitted, self.checks)

    @property
    def score(self) -> float:
        """``s``: the largest absolute fresh native-gradient component, else NaN."""
        value = next((c.value for c in self.checks if c.name == GRADIENT), None)
        return math.nan if value is None else value

    @property
    def accepted(self) -> bool:
        """The point predicate: complete, every stage completed, every check passed."""
        return (
            self.complete
            and not self.failures
            and bool(self.checks)
            and all(check.passed for check in self.checks)
        )

    @property
    def error(self) -> float:
        """The largest ratio of a value to a bound, NaN unless complete and formed."""
        ratios = [ratio for check in self.checks for ratio in check.ratios]
        missing = any(check.value is None for check in self.checks)
        if self.failures or not ratios or missing or any(map(math.isnan, ratios)):
            return math.nan
        if not self.complete:
            return math.nan
        return max(ratios)

    @property
    def verdict(self) -> str:
        """The mode-qualified verdict of the point predicate, or the partial one."""
        if self.kind == OPENING_RETRY_KIND:
            return OPENING_RETRY_VERDICT
        return ACCEPTED if self.accepted else REJECTED


@dataclass(frozen=True)
class RetainedStandardReading:
    """The detached summary of one standard reading, as the fit's record keeps it.

    ``role`` is ``first``, ``opening`` or ``second`` for a reading that the retry
    requested and ``selected_final`` for a fresh reading of the selected point;
    ``inference`` is ``not_computed_before_selection`` for the retry's readings and the
    inference status for the selected summary.  ``theta`` and ``q`` are the bytes of
    the point it read.  ``kind`` and ``omitted`` are the reading's; a call refused
    before any reading keeps the kind it requested and holds no point bytes and no
    omission.
    """

    role: Role
    policy: str
    accepted: bool
    score: float
    error: float
    verdict: str
    checks: tuple[StandardCheck, ...]
    failures: tuple[StandardFailure, ...]
    work: tuple[WorkStep, ...]
    theta: bytes | None
    q: bytes | None
    wall: float
    inference: str = NOT_COMPUTED
    kind: str = COMPLETE_KIND
    omitted: tuple[OmittedCheck, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether a complete reading was formed: never a refusal or a partial one."""
        formed = self.q is not None
        return formed and _complete(self.policy, self.kind, self.omitted, self.checks)


def summary(
    reading: StandardPointAssessment | None,
    role: Role,
    wall: float,
    refusal: str = "",
    kind: str = COMPLETE_KIND,
) -> RetainedStandardReading:
    """The detached summary of a reading, or of a call refused before any reading.

    A refused call is summarized with the ``kind`` of reading it requested, that
    kind's policy and verdict, and no point bytes, check, work or omission.
    """
    if reading is None:
        failure = (StandardFailure("ownership", refusal),)
        partial = kind == OPENING_RETRY_KIND
        return RetainedStandardReading(
            role,
            INDEX_RETRY_POLICY if partial else POLICY,
            False,
            math.nan,
            math.nan,
            OPENING_RETRY_VERDICT if partial else REJECTED,
            (),
            failure,
            (),
            None,
            None,
            wall,
            kind=kind,
        )
    point = reading.point
    return RetainedStandardReading(
        role,
        reading.policy,
        reading.accepted,
        reading.score,
        reading.error,
        reading.verdict,
        reading.checks,
        reading.failures,
        reading.work,
        point.theta.tobytes(),
        point.q.tobytes(),
        wall,
        kind=reading.kind,
        omitted=reading.omitted,
    )


# ------------------------------------------------------- residualized evidence

MAP_CHECKS = ("slope initial", "slope solve", "H backward", "A backward")


def _same_float_bits(found: float, given: float) -> bool:
    """Whether two floats have the same binary64 bits."""
    return bool(np.float64(found).tobytes() == np.float64(given).tobytes())


def normwise_bound(tolerance: float, norm: float, solution: float, rhs: float) -> float:
    """``tolerance * (norm * solution + rhs)`` in binary64, every condition raised.

    The bound of one column's normwise backward predicate: the products and the sum
    run under ``numpy.errstate(all="raise")``, so an overflow, an invalid operation or
    an inexact underflow is a ``FloatingPointError``, never a replaced value.
    """
    with np.errstate(all="raise"):
        found = np.float64(tolerance) * (
            np.float64(norm) * np.float64(solution) + np.float64(rhs)
        )
    return float(found)


def holds(residual: float, bound: float) -> bool:
    """``residual <= bound`` of finite values: a zero bound needs an exact zero."""
    return math.isfinite(residual) and math.isfinite(bound) and residual <= bound


@dataclass(frozen=True)
class MapCheck:
    """One column's normwise backward predicate of a map or of a slope solve.

    ``residual`` is the largest absolute entry of the column's residual, ``norm`` the
    operator's ``||M'||_inf``, ``solution`` and ``rhs`` the largest absolute entries of
    its solution and right-hand side, ``bound`` their :func:`normwise_bound` with
    ``tolerance`` and ``passed`` :func:`holds`.  ``name`` is one of :data:`MAP_CHECKS`;
    ``round`` is zero for this simple-projection revision.
    """

    name: str
    round: int
    column: int
    residual: float
    norm: float
    solution: float
    rhs: float
    tolerance: float
    bound: float
    passed: bool

    @property
    def verified(self) -> bool:
        """Whether the bound and the verdict are those of the raw values, recomputed."""
        if not all(
            math.isfinite(v) and v >= 0.0
            for v in (
                self.residual,
                self.norm,
                self.solution,
                self.rhs,
                self.tolerance,
                self.bound,
            )
        ):
            return False
        try:
            bound = normwise_bound(self.tolerance, self.norm, self.solution, self.rhs)
        except FloatingPointError:
            return False
        recomputed = holds(self.residual, bound)
        return _same_float_bits(bound, self.bound) and self.passed is recomputed


@dataclass(frozen=True)
class Reservation:
    """One projection/check action reserved before it started (round is always zero).

    ``s`` and ``e`` are the units reserved, consumed whatever the action's outcome;
    ``attempted_s`` and ``attempted_e`` the units of the work actually begun, by its
    ``products`` (sparse, cross and slope-solve right-hand sides, 1 S and 1 E each),
    ``dots`` (dot evaluations, 1 E) and ``checks`` (check columns and block-mean
    evaluations, 1 E); the absorber's own reserved sweeps and diagnostic passes for
    the absorption. ``round`` is always zero and ``rhs`` is always -1: no
    correction round or nuisance solve executes in this revision.
    """

    phase: str
    action: str
    round: int
    rhs: int
    s: int
    e: int
    attempted_s: int
    attempted_e: int
    products: int
    dots: int
    checks: int
    outcome: str

    @property
    def unused_s(self) -> int:
        """The S units reserved and not begun."""
        return self.s - self.attempted_s

    @property
    def unused_e(self) -> int:
        """The E units reserved and not begun: an explicit unused slot."""
        return self.e - self.attempted_e


@dataclass(frozen=True)
class BudgetRefusal:
    """A reservation that did not fit: what needed it, the ledgers, what was left."""

    phase: str
    action: str
    round: int
    rhs: int
    required_s: int
    required_e: int
    used_s: int
    used_e: int
    remaining_s: int
    remaining_e: int
    last_checked: str
    unperformed: str


@dataclass(frozen=True)
class LedgerRecord:
    """Both shared ledgers: limits, typed reservations in order, a refusal if any.

    ``s_limit`` is ``M``; ``e_limit`` is ``n_blocks (c + 1) + 2K`` with ``c`` the old
    one-call maximum of fresh checks.  The used units are the sums of the reserved.
    """

    s_limit: int
    e_limit: int
    c: int
    reservations: tuple[Reservation, ...]
    refusal: BudgetRefusal | None

    @property
    def s_used(self) -> int:
        """The S units reserved, in total."""
        return sum(r.s for r in self.reservations)

    @property
    def e_used(self) -> int:
        """The E units reserved, in total."""
        return sum(r.e for r in self.reservations)

    @property
    def unused_e(self) -> int:
        """The E units reserved and not begun, in total."""
        return sum(r.unused_e for r in self.reservations)


@dataclass(frozen=True)
class ProjectionRecord:
    """The one cold absorption of the slopes and the canonical coordinates' checks.

    The absorption's status, reason, tolerance ``af``, budget ``M``, acceleration, work
    (reserved sweeps, attempted visits, operators and diagnostic passes) and returned
    group means; the witness terms converted and the ratios ``n/d`` that binary64
    rounds; the checks in order: the returned group means at ``af``, the
    reconstruction gap ``max |R_c - R_absorb|`` at ``af`` and ``tol``, and the fresh
    weighted group means of ``R_c`` at ``af``.
    """

    status: str
    reason: str
    tolerance: float
    max_sweeps: int
    acceleration: bool
    sweeps: int
    visits: int
    operators: int
    diagnostic_passes: int
    column_errors: tuple[float, ...]
    witness_terms: int
    inexact_ratios: int
    checks: tuple[StandardCheck, ...]

    @property
    def passed(self) -> bool:
        """A converged absorption whose three checks hold, recomputed from values."""
        return (
            self.status == "converged"
            and len(self.checks) == 3
            and all(
                check.passed
                and check.value is not None
                and all(check.value <= bound for bound in check.bounds)
                for check in self.checks
            )
        )


@dataclass(frozen=True, eq=False)
class ResidualizedEvidence:
    """The scalar evidence of simple projection ``projection_map_v2``.

    ``tolerance`` is ``af``, ``tol`` the controls' ``tol``;
    ``slopes`` is ``K``, ``kept`` ``|J|``.  ``slope_factor`` is the record of the one
    ordinary factor of the K-dimensional slope bread and ``slope_actions`` its counted
    solves; ``initial`` the checks of ``G' Z = I``, ``backward`` the full H/A checks;
    ``fixed_work`` counts attempted fixed projection routines (T cross in RHS units);
    ``map_digest`` and
    ``geometry_digest`` the SHA-256 content bindings of the delivered map and of its
    geometry, ``None`` when no map was delivered.
    """

    requested: InferenceMethod
    executed: InferenceMethod
    algorithm: str
    tolerance: float
    tol: float
    max_sweeps: int
    acceleration: bool
    n_blocks: int
    slopes: int
    kept: int
    full_system_factor: str
    full_factor_calls: int
    full_solve_calls: int
    nuisance_solver: str
    slope_factor: FactorRecord | None
    slope_actions: Mapping[str, int]
    projection: ProjectionRecord | None
    initial: tuple[MapCheck, ...]
    backward: tuple[MapCheck, ...]
    fixed_work: tuple[tuple[str, int], ...]
    ledger: LedgerRecord
    failures: tuple[StandardFailure, ...]
    map_digest: str | None
    geometry_digest: str | None

    def __post_init__(self) -> None:
        """Own the slope factor record and freeze the solve counts."""
        if self.slope_factor is not None:
            owned = _owned_factor_record(self.slope_factor)
            object.__setattr__(self, "slope_factor", owned)
        object.__setattr__(
            self, "slope_actions", MappingProxyType(dict(self.slope_actions))
        )

    @property
    def passed(self) -> bool:
        """Whether every recorded predicate holds, recomputed: never a certificate."""
        projection = self.projection
        return (
            not self.failures
            and projection is not None
            and projection.passed
            and len(self.initial) == self.slopes
            and all(check.verified and check.passed for check in self.initial)
            and len(self.backward) == 2 * self.slopes
            and all(check.verified and check.passed for check in self.backward)
            and self.map_digest is not None
        )


@dataclass(frozen=True, eq=False)
class StandardInference:
    """The requested inference at the selected standard point, or why not.

    ``geometry``, ``ordinary`` (the ordinary factor's record, never a conditioning
    pass), ``slope_map`` (``W``), ``scale`` (the admitted provider result) and
    ``actions`` (the counted solves) are ``None`` when not formed.  ``requested`` and
    ``executed`` name the inference method; ``residualized`` is the scalar evidence of
    the residualized route, ``None`` on the full-joint route.
    """

    status: InferenceStatus
    reason: str
    geometry: FEGeometry | None = None
    ordinary: FactorRecord | None = None
    slope_map: FloatArray | None = None
    scale: FEScaleResult | None = None
    actions: Mapping[str, int] | None = None
    work: tuple[WorkStep, ...] = ()
    provider_calls: int = 0
    requested: InferenceMethod = "full_joint"
    executed: InferenceMethod = "full_joint"
    residualized: ResidualizedEvidence | None = None


# ------------------------------------------------------------ detached records


@dataclass(frozen=True)
class RetainedStandardSelection:
    """The detached selection: status, the retry's readings, the selected one, why."""

    policy: str
    status: StandardStatus
    readings: tuple[RetainedStandardReading, ...]
    selected: RetainedStandardReading | None
    consumed: bool
    reason: str
    failures: tuple[StandardFailure, ...]


@dataclass(frozen=True)
class RetainedStandardEvidence:
    """The detached standard evidence of the selected point.

    ``rows`` and ``geometry`` are ``None`` after semantic discard or when not formed.
    """

    policy: str
    controls: StandardControls
    criterion: float | None
    _gradient: FloatArray | None
    support: SupportCheck | None
    basis: RetainedBasisSnapshot
    effects: RetainedEffectEquations | None
    inner: RetainedInnerEvidence | None
    rows: RetainedRowsSnapshot | None
    geometry: RetainedGeometrySnapshot | None

    def __post_init__(self) -> None:
        """Own the gradient."""
        object.__setattr__(self, "_gradient", _optional(self._gradient))

    @property
    def gradient(self) -> FloatArray | None:
        """A fresh copy of every component of the fresh native gradient, or ``None``."""
        return _optional(self._gradient)


@dataclass(frozen=True)
class RetainedStandardInference:
    """The detached inference record: status, factor record, solves, omitted checks.

    ``requested`` and ``executed`` name the inference method; ``residualized`` is the
    owned scalar evidence of the residualized route, ``None`` on the full-joint route.
    """

    status: InferenceStatus
    reason: str
    ordinary: FactorRecord | None
    actions: Mapping[str, int] | None
    work: tuple[WorkStep, ...]
    provider_calls: int
    omitted: tuple[OmittedCheck, ...] = OMITTED_CHECKS
    requested: InferenceMethod = "full_joint"
    executed: InferenceMethod = "full_joint"
    residualized: ResidualizedEvidence | None = None

    def __post_init__(self) -> None:
        """Own the factor records and the evidence and freeze the counts."""
        if self.ordinary is not None:
            object.__setattr__(self, "ordinary", _owned_factor_record(self.ordinary))
        if self.actions is not None:
            object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))
        if self.residualized is not None:
            object.__setattr__(self, "residualized", replace(self.residualized))


@dataclass(frozen=True)
class RetainedStopProbes:
    """The detached probes of the private equation-aware stop of one index fit.

    ``policy`` is :data:`INDEX_STOP_POLICY`; ``tol`` and ``gradient_tol`` the fit's
    captured bounds of the fresh equations and of the two native formulas; ``probes``
    every probe of the solve in order, refused ones included, each bound to its attempt,
    accepted step and point
    (:func:`~powermean.results.ppml_standard_retained.detach_probes`).  An empty tuple
    is a stop that ran and whose native formulas never passed.  No probe is a standard
    reading, a native stop or inference evidence.
    """

    policy: str
    tol: float
    gradient_tol: float
    probes: tuple[EquationProbe, ...]

    @property
    def wall(self) -> float:
        """The probes' own wall seconds, which the solve's include."""
        return math.fsum(probe.wall for probe in self.probes)

    @property
    def cpu(self) -> float:
        """The probes' own CPU seconds, which the solve's include."""
        return math.fsum(probe.cpu for probe in self.probes)


@dataclass(frozen=True)
class RetainedSamePointAudit:
    """The raw unchanged full audit of the standard selected point, detached.

    ``origin`` is ``post_standard_same_point``; ``bound`` says that its admitted
    ``theta``, base and ``q`` and its request are the standard selected point's;
    ``raised`` holds the error that ``certify_point`` raised (no certificate field is
    then formed); ``wall`` its own seconds, apart from the standard fit's.
    """

    origin: str
    bound: bool
    raised: str
    controls: CertificateControls | None
    level: Level | None
    accepted: bool | None
    degenerate: bool | None
    strict: tuple[str, ...]
    secondary: tuple[str, ...]
    errors: Mapping[str, str]
    scaled_score: float
    index_correction: float
    effects_maximum: float | None
    inner_checks: tuple[InnerCheck, ...]
    measures: RetainedMeasuresSnapshot | None
    _coefficients: CoefficientCheck | None
    scale: RetainedInitialScaleCache | None
    work: tuple[WorkStep, ...]
    wall: float

    def __post_init__(self) -> None:
        """Own the coefficient check and freeze the errors."""
        object.__setattr__(self, "errors", MappingProxyType(dict(self.errors)))
        found = _owned_coefficient_check(self._coefficients)
        object.__setattr__(self, "_coefficients", found)

    @property
    def coefficients(self) -> CoefficientCheck | None:
        """A fresh owned copy of the coefficient check, or ``None``."""
        return _owned_coefficient_check(self._coefficients)


@dataclass(frozen=True)
class PPMLWork:
    """The actual work and elapsed seconds of one private fit.

    ``total`` excludes ``validation_extra``, the separately measured same-point full
    audit (``None`` when none ran).  ``None`` marks a part not measured apart.
    ``stop_probes`` counts the actual fresh-equations probes of the private
    equation-aware stop and ``stop_probe`` sums their own seconds, which ``solve``
    already includes (a breakdown, never added to it); both ``None`` when that stop
    did not run.
    """

    total: float
    validation_extra: float | None
    eligibility: float
    solve: Timing
    assessments: int
    assessment: Timing
    capture: float | None
    selection: float | None
    readings: int | None
    inference: float | None
    retention: float
    provider_calls: int | None
    actions: Mapping[str, int] | None
    stop_probes: int | None = None
    stop_probe: Timing | None = None


@dataclass(frozen=True)
class PPMLProvenance:
    """What ran: the caller, the policy, the request, the ownership and the selection.

    ``consumed`` says whether the selected point's evidence was a kept within-fit one
    (the standard reading, or the full mode's handoff certificate), ``reason`` why not.
    ``inference_method`` is the method the entry requested: ``full_joint`` but at the
    explicit private alternative entry.
    """

    caller: str
    policy: Policy
    organization: str
    request: tuple[object, ...]
    snapshot_route: str
    ownership: str
    consumed: bool
    reason: str
    provider: str | None
    provider_version: str | None
    retry: int
    inference_method: InferenceMethod = "full_joint"
