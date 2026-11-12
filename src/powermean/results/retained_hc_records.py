"""Internal retained adapter of the named HC covariances: capability and computation.

Internal module: no public name, option, message, default or dispatch.

What it does.  A named HC request (HC0, HC1, HC2, HC3, HC4, HC4m, HC5) after a fit is
answered from the retained state of that fit alone, by the kernels of
:mod:`powermean.inference.hc`, and the answer is mapped once to the reported scale.
Three routes, chosen by the retained formulation, never by the method name:

- ``"linear"``, a certified no-FE log OLS/WLS :class:`~powermean.results.retained.
  RetainedFit`: :func:`~powermean.inference.hc.linear_hc` of the
  :class:`~powermean.inference.hc.LinearSolve` that the dispatch captured BEFORE its
  one solve and the handoff bound to the retained equation and candidate
  (:func:`powermean.results.handoff.linear_solve_of`).  A fit without a captured
  solve is refused (``specification.unsupported_inference``); no solve is formed
  after the fit;
- ``"joint"``, a certified no-FE nonlinear :class:`RetainedFit`:
  :func:`~powermean.inference.hc.joint_hc` of the retained equation
  (:func:`~powermean.results.handoff.system_of`, the arrays the solve received) at the
  certificate's evaluation, which the kernel evaluates again at the SAME point and
  must find equal bit for bit; expected-information hats with the observed or the
  expected bread;
- ``"absorbed"``, a certified :class:`~powermean.results.absorbed_retained.
  RetainedAbsorbedEnvelope` (profiled, index or WLS): the internal composition of
  :mod:`powermean.inference.hc` on the retained full nuisance basis, the retained
  scaled slopes and the retained observed rows, with the expected curvature of the
  retained point, after the fresh checks below.  The rows are the certificate's own
  rows of the selected point; the composition derives the full-design hats (nuisance
  plus slopes), the bread on the full basis and the slope block.

Fresh checks, at every call, before any covariance arithmetic: the point of every record
is the selected point (bytes); the rows, weights, response and design of every record
are the retained sample's (bytes); on the absorbed route the observed rows ``b = a r``
and ``C = a c`` are evaluated again from the retained response, the point's ``q``, the
raw weights and the working variance by the row kernel and must equal the retained rows
bit for bit, and the observed joint Hessian assembled from them on the retained basis
must equal the certificate's.  No identity, cache, seal, flag or prior success stands
for these checks.

What is never done: no estimator solve, no sample construction or
reselection, no formula materialization, no regeneration of discarded influence from
the certificate, a prediction or the data.  Same-point geometry (an evaluation of
the retained equation at the retained point, a factorization of the bread, the hats)
is formed again for each request: that is inference algebra, not a refit.

Capability and discard.  :func:`hc_capability` reads the retained records only (their
presence and kinds, no array arithmetic) and states the route, the formulation, the
sampling unit, the weight kind, the breads, the layout and, where the request cannot be
answered, why.  :func:`retained_hc` refuses, before any computation: a record of another
type; an uncertified point (``ValueError``, as the robust and cluster adapter does); a
discarded influence (``specification.influence_discarded``; a named HC request is never
the fit-time request, so no cached covariance can answer it; the rows of a discarded
envelope stay marked as such whatever its flag says); a route without its retained owner
(``specification.unsupported_inference``).  The kernels then refuse HC1 at ``N = K``
(``specification.no_residual_dof``), an explicit second correction
(``specification.conflicting``) and structural unit leverage
(``specification.unit_leverage``) in their preflight, before any evaluation or
factorization.  A refusal leaves every retained record as it was; there is nothing to
restore, since nothing is written.

Conventions.  The method owns its factor; an explicit ``ssc`` is checked by the kernel's
equality rule; an omitted ``ssc`` takes the fit's convention in force as inherited (the
retained initial request's ``"stata"`` by its name, or an internal declaration of a
custom record, as in :mod:`powermean.results.retained_inference`), which never
multiplies a named method and supplies its reference distribution.

HAC and Conley.  The typed :class:`~powermean.data.inference_requests.
HacRequest` and :class:`~powermean.data.inference_requests.ConleyRequest` stay
reserved: :func:`retained_request` refuses them with ``specification.unimplemented``
naming the later release, before reading the retained state, with no
substituted covariance and no refit.  Their computation is not implemented here.

Reporting.  The covariance of ``theta`` is the kernel's, the method's factor applied
once by its owner.  The direct standard errors come from its diagonal by the order scale
(zero where zero, NaN where negative; no floor) and the reported covariance by the
range-safe entrywise map, or its recorded range error; each map is applied once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from powermean.data.inference_conventions import ConventionValues
from powermean.data.inference_requests import (
    HCMethod,
    HCRequest,
)
from powermean.data.request_values import OMITTED, Omitted
from powermean.inference.bread import BREADS, Bread
from powermean.inference.hc import (
    HCCovariance,
)
from powermean.inference.ssc import (
    AppliedConvention,
    ReferenceDistribution,
    resolve_convention,
)
from powermean.model.variance import FloatArray
from powermean.results.absorbed_retained import RetainedAbsorbedEnvelope
from powermean.results.cancelled_retained import RetainedCancelledEnvelope
from powermean.results.owned import _optional, owned_array
from powermean.results.retained import RetainedFit

Route = Literal["linear", "joint", "absorbed"]

WeightKind = Literal["none", "analytic", "frequency", "probability"]

Target = RetainedFit | RetainedAbsorbedEnvelope | RetainedCancelledEnvelope


@dataclass(frozen=True)
class HCInferenceRequest:
    """One post-fit named HC request on retained state, in lower-layer terms.

    Parameters
    ----------
    method : {"HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5"}
        The method, an exact string (``specification.invalid_value`` otherwise, as
        :class:`~powermean.data.inference_requests.HCRequest` refuses it).
    ssc : str, ConventionValues or OMITTED
        The explicit convention of this request, or :data:`OMITTED`.  An explicit
        ``None`` or another value is refused (``specification.invalid_value``).
    bread : {"observed", "expected"}
        The bread of the influence rows; the hats use the expected geometry either way.
        The least-squares equation's observed bread is its expected
        information, so both give one answer on the linear routes.
    inherited : ConventionValues, optional
        An internal declaration of the fit's custom record (inherited and custom
        whatever its values); ``None`` reads the retained initial request's
        convention by its recorded name.
    """

    method: HCMethod
    ssc: str | ConventionValues | Omitted = OMITTED
    bread: Bread = "observed"
    inherited: ConventionValues | None = None

    def __post_init__(self) -> None:
        """Check the method, the explicit convention, the bread and the declaration."""
        object.__setattr__(self, "method", HCRequest(self.method).method)
        resolve_convention(self.ssc)  # the domain of an explicit value, nothing kept
        if self.bread not in BREADS:
            raise ValueError("bread must be 'observed' or 'expected'")
        if self.inherited is not None and type(self.inherited) is not ConventionValues:
            raise TypeError("inherited must be a ConventionValues or None")

    @classmethod
    def from_hc_request(
        cls,
        request: HCRequest,
        *,
        ssc: str | ConventionValues | Omitted = OMITTED,
        bread: Bread = "observed",
        inherited: ConventionValues | None = None,
    ) -> HCInferenceRequest:
        """Bind the typed record; the parser owns the spelling."""
        if type(request) is not HCRequest:
            raise TypeError("request must be an HCRequest")
        return cls(request.method, ssc, bread, inherited)

    @property
    def label(self) -> str:
        """The request as a caller would write it, for a refusal."""
        ssc = "" if isinstance(self.ssc, Omitted) else f", ssc={self.ssc!r}"
        return f"vcov({self.method!r}{ssc}, bread={self.bread!r})"


@dataclass(frozen=True)
class HCCapability:
    """What a retained target supports for a named HC request, read from its records.

    Attributes
    ----------
    route : {"linear", "joint", "absorbed"} or None
        The route of the request, ``None`` where the target has no certified point.
    formulation : str
        The retained formulation, in words.
    unit : {"rows"}
        The primitive sampling unit.
    weight_kind : {"none", "analytic", "frequency", "probability"}
        The stated kind of the retained weights.
    layout : {"dense", "absorbed"} or None
        Slopes alone, or the full nuisance basis.
    breads : tuple of str
        The breads the route answers.
    geometry : str
        The retained owner of the rows that the route reads.
    available : bool
        Whether a request can be computed now (subject to the kernels' refusals).
    reason : str or None
        Why not, where it cannot.
    """

    route: Route | None
    formulation: str
    unit: Literal["rows", "groups"]
    weight_kind: WeightKind
    layout: Literal["dense", "absorbed"] | None
    breads: tuple[str, ...]
    geometry: str
    available: bool
    reason: str | None


def _weight_kind(weights: object) -> WeightKind:
    """The stated kind of a retained weights record, ``"none"`` without one."""
    if weights is None:
        return "none"
    kind = getattr(weights, "kind", None)
    if kind not in ("analytic", "frequency", "probability"):
        raise ValueError("the retained weights have no stated kind")
    found: WeightKind = kind
    return found


def _fit_capability(retained: RetainedFit) -> HCCapability:
    """The capability of a no-FE fit, from its records."""
    kind = _weight_kind(retained.sample.data.weights)
    linear = retained.estimator.route == "linear"
    route: Route = "linear" if linear else "joint"
    formulation = "no-FE log OLS/WLS" if linear else "no-FE joint nonlinear"
    geometry = (
        "the equation captured before the solve, with its candidate"
        if linear
        else "the retained equation, evaluated at the certified point"
    )
    reason: str | None = None
    if not retained.certified or retained.theta is None:
        reason = "no selected point was certified for this candidate"
    elif not retained.keep_influence or retained.influence is None:
        reason = "the influence was discarded (keep_influence=False)"
    elif linear and retained.linear_solve is None:
        reason = "the solved linear equation was not captured with the fit"
    elif not linear and (
        retained.certificate is None or retained.certificate.evaluation is None
    ):
        reason = "the certificate retains no evaluation of the point"
    return HCCapability(
        route=None if not retained.certified else route,
        formulation=formulation,
        unit="rows",
        weight_kind=kind,
        layout="dense",
        breads=BREADS,
        geometry=geometry,
        available=reason is None,
        reason=reason,
    )


def _envelope_capability(envelope: RetainedAbsorbedEnvelope) -> HCCapability:
    """The capability of an absorbed envelope, from its records."""
    sample = envelope.sample
    certificate = envelope.certificate
    formulation = f"absorbed joint ({envelope.route})"
    reason: str | None = None
    if envelope.selected_point is None or certificate is None:
        reason = "no selected point was certified for this candidate"
    elif not certificate.accepted:
        reason = "the selected point is not certified"
    elif envelope.empty:
        reason = "the model has no slope parameters (K = 0)"
    elif envelope.influence_discarded or (
        certificate.rows is not None and certificate.rows._influence_discarded
    ):
        reason = "the influence was discarded (keep_influence=False)"
    elif certificate.rows is None or certificate.geometry is None:
        reason = "the certificate retains no rows or geometry of the point"
    elif sample.design is None or envelope.order is None:
        reason = "the retained sample holds no scaled design or order scale"
    certified = (
        envelope.selected_point is not None
        and certificate is not None
        and certificate.accepted
    )
    return HCCapability(
        route="absorbed" if certified else None,
        formulation=formulation,
        unit="rows",
        weight_kind=_weight_kind(sample.data.weights),
        layout="absorbed",
        breads=BREADS,
        geometry="the full nuisance basis, the slopes and the observed rows",
        available=reason is None,
        reason=reason,
    )


def hc_capability(target: object) -> HCCapability:
    """The capability of a retained target for a named HC request (module docstring).

    Reads the presence and kinds of the retained records, never their arrays, and
    computes nothing; :func:`retained_hc` checks the arrays afresh at every call.

    Raises
    ------
    TypeError
        ``target`` is neither a ``RetainedFit`` nor a ``RetainedAbsorbedEnvelope``.
    """
    if type(target) is RetainedCancelledEnvelope:
        return HCCapability(
            route=None,
            formulation="cancelled group equations",
            unit="groups",
            weight_kind="none",
            layout="dense",
            breads=("observed",),
            geometry="retained implicit design and original cancelled point",
            available=False,
            reason="cancelled HC leverage extension remains open",
        )
    if type(target) is RetainedFit:
        return _fit_capability(target)
    if type(target) is RetainedAbsorbedEnvelope:
        return _envelope_capability(target)
    raise TypeError("target must be a RetainedFit or a RetainedAbsorbedEnvelope")


@dataclass(frozen=True, eq=False)
class RetainedHC:
    """The answer to one named HC request on retained state.

    Attributes
    ----------
    request : HCInferenceRequest
        The request.
    capability : HCCapability
        The route and what it read.
    hc : HCCovariance
        The kernel's record: the covariance of ``theta`` with the method's factor
        applied once, the adjusted primitive estimate, the applied convention, the
        per-replica hats, the multipliers, the bread and the layout.
    theta_covariance : array
        The covariance of ``theta``, owned.
    standard_errors : array
        The direct reported standard errors, owned.
    reported_covariance : array or None
        The reported covariance, or ``None`` with ``covariance_error`` set.
    covariance_error : str or None
        The range error of the reported covariance, if any.
    gamma : array
        The selected point ``theta``, owned.
    names : tuple of str
        The names of the reported coefficients, in order.
    count, active_rows, stored_rows : int
        ``N``; the rows of positive weight; the final stored rows of the sample.
    """

    request: HCInferenceRequest
    capability: HCCapability
    hc: HCCovariance
    theta_covariance: FloatArray
    standard_errors: FloatArray
    reported_covariance: FloatArray | None
    covariance_error: str | None
    gamma: FloatArray
    names: tuple[str, ...]
    count: int
    active_rows: int
    stored_rows: int

    def __post_init__(self) -> None:
        """Own the arrays."""
        object.__setattr__(self, "theta_covariance", owned_array(self.theta_covariance))
        object.__setattr__(self, "standard_errors", owned_array(self.standard_errors))
        object.__setattr__(
            self, "reported_covariance", _optional(self.reported_covariance)
        )
        object.__setattr__(self, "gamma", owned_array(self.gamma))

    @property
    def applied(self) -> AppliedConvention:
        """The convention in force, the counts, the factor and owner, the reference."""
        return self.hc.applied

    @property
    def factor(self) -> float:
        """The scalar applied once, the method's own."""
        return self.hc.factor

    @property
    def reference(self) -> ReferenceDistribution:
        """The reference distribution of the convention in force."""
        return self.hc.reference

    @property
    def degenerate(self) -> bool:
        """Whether a reported standard error is exactly zero."""
        return bool((self.standard_errors == 0.0).any())

    @property
    def missing(self) -> bool:
        """Whether a reported standard error is missing (a negative variance)."""
        return bool(np.isnan(self.standard_errors).any())

    def require_reported_covariance(self) -> FloatArray:
        """The reported covariance, or ``FloatingPointError`` with the range error."""
        if self.reported_covariance is None:
            raise FloatingPointError(
                self.covariance_error or "the reported covariance is not representable"
            )
        return owned_array(self.reported_covariance)
