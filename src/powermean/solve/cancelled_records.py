"""Private value records and prospective bounds for cancelled certification."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, get_args

from powermean.fe._cancel_storage import FloatStorage
from powermean.fe.geometry_design import NumericalGeometry
from powermean.model.variance import (
    CancelledFamily,
    FloatArray,
    NB2Variance,
    PowerVariance,
    Variance,
)
from powermean.solve.assessment import Assessment, CertificateControls
from powermean.solve.cancelled_operator import AttemptBounds, workspace_plan
from powermean.solve.certificate import GateName, Level
from powermean.solve.absorbed import (
    ABSORBED_DOMAINS,
    POSITIVE_INTEGER,
    POSITIVE_REAL,
    AbsorbedOptions,
    Setting,
    changes,
    resolved,
)
from powermean.solve.newton_retry import Attempts
from powermean.solve.newton_trace import NativeReason
from powermean.solve.options import UNIT_INTERVAL, Choices, RealInterval
from powermean.solve.reported import CoefficientCheck
from powermean.solve.status import NATIVE_STATUS, Status

if TYPE_CHECKING:
    from powermean.solve.cancelled_certificate import CertifiedCancelledPoint
    from powermean.solve.newton_absorbed import IndexCandidate
    from powermean.solve.newton_profiled import ProfiledCandidate

ScoreMetricKind = Literal["current_symmetric_hessian", "fixed_merit", "none"]
RouteKind = Literal["convex_tau2", "absorbed_tau1", "root_newton"]
CoverageKind = Literal["unique_root", "selected_root"]
MeritMetric = Literal["gamma", "gram"]
ContinuationPolicy = Literal["fallback", "never"]

MERIT_METRICS: tuple[MeritMetric, ...] = get_args(MeritMetric)
CONTINUATION_POLICIES: tuple[ContinuationPolicy, ...] = get_args(ContinuationPolicy)
_UNIT_REAL = RealInterval(lower=0.0, upper=1.0, valid=UNIT_INTERVAL)
_CERTIFICATE_DEFAULT_VARIANCE = PowerVariance(2.0)

ROOT_DOMAINS: Mapping[str, Setting] = MappingProxyType(
    {
        "root_armijo": _UNIT_REAL,
        "root_max_backtracks": POSITIVE_INTEGER,
        "root_max_iter": POSITIVE_INTEGER,
        "merit_metric": Choices(values=MERIT_METRICS),
        "continuation": Choices(values=CONTINUATION_POLICIES),
        "continuation_step": POSITIVE_REAL,
    }
)


def has_convex_criterion(variance: Variance) -> bool:
    """Whether ``variance`` belongs to a route with a convex objective criterion."""
    if isinstance(variance, PowerVariance):
        return variance.tau in (1.0, 2.0)
    if isinstance(variance, NB2Variance):
        return variance.kappa == 0.0
    return False


def route_for_variance(variance: CancelledFamily) -> tuple[RouteKind, CoverageKind]:
    """Return the solver route and root-coverage classification for ``variance``."""
    if isinstance(variance, PowerVariance) and variance.tau == 2.0:
        return "convex_tau2", "unique_root"
    if (isinstance(variance, PowerVariance) and variance.tau == 1.0) or (
        isinstance(variance, NB2Variance) and variance.kappa == 0.0
    ):
        return "absorbed_tau1", "unique_root"
    return "root_newton", "selected_root"


def reserve_certificate(geometry: NumericalGeometry, bounds: AttemptBounds) -> None:
    """Reserve current rows, action scratch, dense d-square factors and diagnostics."""
    n, p, d = geometry.structure.n_rows, geometry.n_columns_full, geometry.n_identified
    groups = geometry.structure.n_groups
    slots = 256 * n + 32 * p + 32 * groups + 32 * d * d + 128 * d
    required = 8 * slots + 16 * (n + p + d)
    visits = (d + 4) * workspace_plan(geometry.design, d).work
    if required > bounds.max_bytes or visits > bounds.max_work:
        raise ValueError("cancelled certificate exceeds supplied byte/work bounds")


@dataclass(frozen=True)
class CancelledMeasures:
    """Original sum-equation gate quantities, without joint evaluation records."""

    evaluated: bool = False
    positive_definite: bool = False
    conditioned: bool = False
    rcond: float = float("nan")
    scaled_score: float = float("nan")
    index_correction: float = float("nan")
    coefficients: CoefficientCheck | None = None
    require_positive_definite: bool = True
    score_metric: ScoreMetricKind = "none"


@dataclass(frozen=True, eq=False)
class CancelledCertificate:
    """Immutable diagnostics of one freshly evaluated identified cancelled point."""

    _point: FloatStorage
    controls: CertificateControls
    measures: CancelledMeasures
    _score: FloatStorage | None
    _hessian: FloatStorage | None
    _correction: FloatStorage | None
    errors: Mapping[GateName, str]
    strict: tuple[GateName, ...]
    secondary: tuple[GateName, ...]
    variance: CancelledFamily = _CERTIFICATE_DEFAULT_VARIANCE

    def __post_init__(self) -> None:
        """Detach source failure text from the local construction dictionary."""
        object.__setattr__(self, "errors", MappingProxyType(dict(self.errors)))

    @property
    def gamma(self) -> FloatArray:
        """Identified point with immutable backing and detached shape metadata."""
        return self._point.view()

    @property
    def score(self) -> FloatArray | None:
        """Current unnormalized moment, if its evaluation completed."""
        return None if self._score is None else self._score.view()

    @property
    def hessian(self) -> FloatArray | None:
        """Actual negative Jacobian, if every implicit column completed."""
        return None if self._hessian is None else self._hessian.view()

    @property
    def correction(self) -> FloatArray | None:
        """H-inverse sum-score correction, if available."""
        return None if self._correction is None else self._correction.view()

    @property
    def level(self) -> Level | None:
        """Strict or secondary only when every applicable gate passes."""
        if not self.strict:
            return "strict"
        return "secondary" if not self.secondary else None

    @property
    def accepted(self) -> bool:
        """Whether the original point passes all gates at either level."""
        return self.level is not None

    @property
    def degenerate(self) -> bool:
        """Whether the coefficient check flags a zero standard error."""
        check = self.measures.coefficients
        return check is not None and check.degenerate

    def assessment(self) -> Assessment:
        """The existing retry arithmetic, using fresh original-system quantities."""
        score, index = self.measures.scaled_score, self.measures.index_correction
        error = (
            float("nan")
            if math.isnan(score) or math.isnan(index)
            else max(score, index)
        )
        return Assessment(self.accepted, score, error)

    def status(self, reason: NativeReason) -> Status:
        """Existing status precedence, without an existence claim from a stop."""
        if self.level == "strict":
            return "verified"
        if self.level == "secondary":
            return "verified_secondary"
        if "evaluation" in self.strict:
            return "nonfinite"
        if "curvature" in self.strict or "conditioning" in self.strict:
            return "singular_system"
        return NATIVE_STATUS[reason]


@dataclass(frozen=True)
class RootOptions:
    """Internal controls of the general root-Newton solver."""

    root_armijo: float = 1e-4
    root_max_backtracks: int = 60
    root_max_iter: int = 200
    merit_metric: MeritMetric = "gamma"
    continuation: ContinuationPolicy = "fallback"
    continuation_step: float = 0.5

    def __post_init__(self) -> None:
        """Validate every root-Newton control against :data:`ROOT_DOMAINS`."""
        for option in fields(self):
            value = getattr(self, option.name)
            stored = resolved(ROOT_DOMAINS, option.name, value)
            object.__setattr__(self, option.name, stored)

    def to_dict(self) -> dict[str, object]:
        """Return resolved options by name in field order as a new dictionary."""
        return {option.name: getattr(self, option.name) for option in fields(self)}

    @classmethod
    def from_dict(cls, options: object) -> RootOptions:
        """Build ``RootOptions`` from a flat mapping of option names to values."""
        return cls(**changes(options, tuple(ROOT_DOMAINS)))


def resolve_root_options(
    options: Mapping[str, object] | RootOptions | None = None,
) -> RootOptions:
    """Resolve a mapping or ``RootOptions`` into a validated ``RootOptions``."""
    if options is None:
        return RootOptions()
    if isinstance(options, RootOptions):
        return options
    return RootOptions.from_dict(options)


def resolve_absorbed_options(
    options: AbsorbedOptions | Mapping[str, Any] | None = None,
) -> AbsorbedOptions:
    """Resolve and validate ``AbsorbedOptions`` (default ``absorbed='index'``)."""
    if options is None:
        return AbsorbedOptions(absorbed="index")
    if isinstance(options, AbsorbedOptions):
        return options
    given = changes(options, tuple(ABSORBED_DOMAINS))
    if "absorbed" not in given:
        given["absorbed"] = "index"
    return AbsorbedOptions(**given)


@dataclass(frozen=True)
class RootExecutionBounds:
    """Certificate and initializer attempt bounds for a root-Newton solve."""

    certificate: AttemptBounds
    attempt: AttemptBounds | None = None
    identity_atol: float = 1e-10


@dataclass(frozen=True)
class RootInitializerRecord:
    """Provenance of the starting point used to seed the root-Newton solver."""

    kind: Literal["gamma_tau2", "user_start"]
    certified: bool
    status: Status | None
    _point: FloatStorage
    certificate: CancelledCertificate | None
    attempts: Attempts | None
    iterations: int = 0
    evaluations: int = 0

    @property
    def point(self) -> FloatArray:
        """Detached starting point ``theta_0``."""
        return self._point.view()


@dataclass(frozen=True)
class RootAttemptRecord:
    """Trace of one root-Newton attempt on a single variance stage."""

    _opening: FloatStorage
    _point: FloatStorage
    native: NativeReason
    iterations: int
    evaluations: int
    backtracks: int
    initial_merit: float
    final_merit: float
    message: str

    @property
    def opening(self) -> FloatArray:
        """Detached opening point of the attempt."""
        return self._opening.view()

    @property
    def point(self) -> FloatArray:
        """Detached final accepted point of the attempt."""
        return self._point.view()


@dataclass(frozen=True)
class RootStageRecord:
    """Trace and certification of one variance stage (direct or continuation)."""

    variance: CancelledFamily
    attempts: tuple[RootAttemptRecord, ...]
    retained: int
    certified: CertifiedCancelledPoint
    status: Status
    iterations: int = 0
    evaluations: int = 0


@dataclass(frozen=True)
class RootSolveResult:
    """Complete outcome and provenance of the general root-Newton solver."""

    certified: CertifiedCancelledPoint
    history: tuple[CancelledCertificate, ...]
    status: Status
    merit_metric: MeritMetric
    _root_options: RootOptions
    _merit_cholesky: FloatStorage | None
    initializer: RootInitializerRecord
    stages: tuple[RootStageRecord, ...]
    continuation_used: bool
    total_iterations: int
    total_evaluations: int
    root_iterations: int = 0
    root_evaluations: int = 0

    @property
    def root_options(self) -> RootOptions:
        """Detached actual resolved controls, never reconstructed from defaults."""
        return replace(self._root_options)

    @property
    def merit_cholesky(self) -> FloatArray | None:
        """Detached lower Cholesky factor ``L_M`` of the fixed merit metric ``M``."""
        return None if self._merit_cholesky is None else self._merit_cholesky.view()

    @property
    def final(self) -> CertifiedCancelledPoint:
        """Alias for ``certified`` matching the route result interface."""
        return self.certified


@dataclass(frozen=True)
class Tau1SolveResult:
    """Outcome of the exact tau=1 / NB2(0) joint-absorbed cancellation route."""

    certified: CertifiedCancelledPoint
    history: tuple[CancelledCertificate, ...]
    status: Status
    iterations: int
    evaluations: int
    organization: Literal["index", "profiled"]
    candidate: IndexCandidate | ProfiledCandidate | None
    bridge_rows: int
    message: str = ""
    origin_failure: object | None = None

    @property
    def final(self) -> CertifiedCancelledPoint:
        """Alias for ``certified`` matching the root-Newton result interface."""
        return self.certified


class Tau1OriginError(ValueError):
    """A genuine normal origin was refused; preserve its owned diagnostic snapshot."""

    def __init__(self, message: str, outcome: object) -> None:
        """Retain the actual detached normal outcome, never a fabricated origin."""
        super().__init__(message)
        self.outcome = outcome


def _valid_continuation_step(
    target: float, step: float, max_stages: int, origin: float = 2.0
) -> float:
    """Return the signed continuation step, or ``0.0`` when no schedule applies."""
    if not (math.isfinite(target) and math.isfinite(step)):
        return 0.0
    if step <= 0.0 or max_stages <= 0 or target == origin:
        return 0.0
    signed = -step if target < origin else step
    first = origin + signed
    if not math.isfinite(first) or first == origin or (first - target) * signed >= 0.0:
        return 0.0
    return signed


def iter_continuation_variances(
    target_variance: PowerVariance,
    step: float,
    max_stages: int,
    *,
    origin: float = 2.0,
) -> Iterator[PowerVariance]:
    """Lazily yield bounded continuation variances from ``origin`` to ``target``.

    ``origin`` is the ``tau`` of the certified initializer: ``2`` for the cancelled
    solver (its Gamma start, the default) and for joint fits above ``tau = 2``, ``1``
    for joint fits below ``tau = 1`` (their PPML start).
    """
    target = target_variance.tau
    signed = _valid_continuation_step(target, step, max_stages, origin)
    if signed == 0.0:
        return
    curr, yielded = origin + signed, 0
    while (curr - target) * signed < 0.0:
        if yielded >= max_stages:
            return
        yield PowerVariance(curr)
        yielded += 1
        nxt = curr + signed
        if not math.isfinite(nxt) or nxt == curr:
            return
        curr = nxt
    if yielded < max_stages:
        yield target_variance
