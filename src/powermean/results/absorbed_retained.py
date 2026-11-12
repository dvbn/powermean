"""Internal retained admission receipt and top-level absorbed envelope.

This module defines the detached retained records for original-admission receipts and
the top-level absorbed retained envelope with deliberate ``keep_influence=False``
discard semantics and no ``check_discard`` opt-outs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import numpy as np

from powermean.errors import SpecificationError
from powermean.fe.admission_records import AdmissionWork
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.results.absorbed_geometry import (
    ExpectedInformationInputs,
    RetainedBasisSnapshot,
    RetainedRowsSnapshot,
    reconstruct_expected_information,
)
from powermean.results.absorbed_history import (
    AbsorbedRoute,
    RetainedHistorySnapshot,
    RetainedPointSnapshot,
)
from powermean.results.absorbed_points import Family
from powermean.results.absorbed_attempts import (
    RetainedAbsorbedSchur,
    RetainedAlarm,
    RetainedPointEquations,
    RetainedProfiledAttempt,
    RetainedProfiledFinal,
    RetainedProfiledIteration,
)
from powermean.results.absorbed_owned import (
    RetainedCertificateSnapshot,
    _OperationalCertificate,
)
from powermean.results.absorbed_root import RetainedRootSnapshot
from powermean.results.absorbed_sample import RetainedSampleSnapshot
from powermean.results.absorbed_scale_cache import (
    RetainedCovarianceRequest,
    RetainedInitialScaleCache,
    snapshot_covariance_request,
    snapshot_initial_scale,
)
from powermean.results.absorbed_trace import RetainedOutcomeSnapshot, RetainedTriggerSnapshot
from powermean.results.owned import _optional, owned_array
from powermean.results.retained_inputs import owned_order_scale
from powermean.solve.absorbed_certificate import Level
from powermean.solve.absorbed_certificate_geometry import SparseMatrix
from powermean.solve.absorbed_certificate_point import WorkStep
from powermean.solve.assessment import Assessment
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.newton_profiled import ProfiledReason
from powermean.solve.newton_trace import Timing
from powermean.solve.options import Provider, SolverOptions
from powermean.solve.profiled_loss import SolveWork
from powermean.solve.starts import Start


@dataclass(frozen=True)
class RetainedAdmissionReceipt:
    """Finite value receipt of original-graph admission before detachment.

    Records the verified identity relations and frozen byte/scalar descriptors without
    retaining live caller objects, ``FEScaleInput``, providers, or tracebacks.
    """

    route: AbsorbedRoute
    omega: float
    variance_name: str | None
    nobs: int
    count: int
    n_slopes: int
    n_effects: int | None
    sample_digest: str
    has_selected_point: bool
    has_certificate: bool
    has_initial_scale: bool
    keep_influence: bool
    same_preparation: bool
    same_outcome_sample: bool
    same_candidate_sample: bool
    same_selected_point: bool
    same_scale_input: bool
    theta_bytes: bytes | None
    base_bytes: bytes | None
    q_bytes: bytes | None
    slope_map_bytes: bytes | None
    request_identity: tuple[object, ...] | None
    root_binding_work: AdmissionWork | None = None
    root_binding_steps: tuple[WorkStep, ...] = ()


@dataclass(frozen=True, slots=True)
class _OperationalHistory:
    """Finite profiled/reduced history data with demand-created presentation links."""

    route: AbsorbedRoute
    sample: RetainedSampleSnapshot
    variance: Family | None
    retained: int | None
    attempts: tuple[_ProfiledAttemptData, ...]
    reason: str
    message: str
    native: str | None
    partials: tuple[str, ...]
    unreached: tuple[str, ...]
    alarm: RetainedAlarm | None
    trigger: RetainedTriggerSnapshot | None
    work: SolveWork
    timing: Timing | None
    selected_point: RetainedPointSnapshot | None
    start: Start | None
    provider: Provider | None
    provider_version: str | None
    initial_assessment: Assessment | None
    no_measurable_progress: bool
    assessments: int
    assessment_timing: Timing
    options: SolverOptions
    absorbed_options: AbsorbedOptions

    def materialize(self) -> RetainedHistorySnapshot:
        """Assemble the canonical graph, preserving selected and final identities."""
        attempts = tuple(a.materialize() for a in self.attempts)
        final = None if self.retained is None else attempts[self.retained].final
        return RetainedHistorySnapshot(
            self.route, self.sample.key, self.sample, self.variance, self.retained,
            attempts, self.reason, self.message, self.native, final, self.partials,
            self.unreached, self.alarm, self.trigger, self.work, self.timing,
            self.selected_point, None, self.start, self.provider, self.provider_version,
            self.initial_assessment, self.no_measurable_progress, self.assessments,
            self.assessment_timing, self.options, self.absorbed_options, None, None,
        )


class _ProfiledFinalData(NamedTuple):
    """Owned final values without the final/equation presentation wrappers."""

    point: RetainedPointSnapshot | None
    equations: tuple[float, FloatArray, FloatArray, FloatArray] | None
    absorbed: RetainedAbsorbedSchur | None
    alarm: RetainedAlarm | None
    reason: str

    def materialize(self) -> RetainedProfiledFinal:
        """Link the already detached final point and numerical evidence."""
        equations = (
            None if self.equations is None else RetainedPointEquations(*self.equations)
        )
        return RetainedProfiledFinal(
            self.point, equations, self.absorbed, self.alarm, self.reason
        )


class _ProfiledAttemptData(NamedTuple):
    """Detached attempt values; no candidate, provider, or callback is retained."""

    iteration: RetainedProfiledIteration | None
    reason: ProfiledReason
    message: str
    opening: RetainedPointSnapshot | None
    point: RetainedPointSnapshot | None
    loss: float | None
    final: _ProfiledFinalData | None
    alarm: RetainedAlarm | None
    assessment: Assessment | None

    def materialize(self) -> RetainedProfiledAttempt:
        """Create a canonical attempt on an actual history request."""
        return RetainedProfiledAttempt(
            self.iteration, self.reason, self.message, self.opening, self.point,
            self.loss, None if self.final is None else self.final.materialize(),
            self.alarm, self.assessment,
        )


@dataclass(frozen=True)
class RetainedAbsorbedEnvelope:
    """Top-level immutable retained state for an absorbed fixed-effects handoff.

    ``root`` is the record of a selected-root fit's root solve
    (:mod:`powermean.results.absorbed_root`), ``None`` on every other route.
    """

    route: AbsorbedRoute
    keep_influence: bool
    receipt: RetainedAdmissionReceipt
    sample: RetainedSampleSnapshot
    outcome: RetainedOutcomeSnapshot | None
    history: RetainedHistorySnapshot
    selected_point: RetainedPointSnapshot | None
    certificate: RetainedCertificateSnapshot | None
    _order: OrderScale | None
    initial_scale: RetainedInitialScaleCache | None
    _observed_influence: FloatArray | None = None
    _slope_map: FloatArray | None = None
    root: RetainedRootSnapshot | None = None

    def __getattribute__(self, name: str) -> object:
        """Realize canonical rich views only at their existing access boundary."""
        found: object = object.__getattribute__(self, name)
        if name == "certificate" and type(found) is _OperationalCertificate:
            return found.materialize()
        if name == "history" and type(found) is _OperationalHistory:
            return found.materialize()
        return found

    def _certificate_data(
        self,
    ) -> _OperationalCertificate | None:
        """Read only an exact compact payload; generic getters remain operative."""
        if type(self) is not RetainedAbsorbedEnvelope:
            return None
        values = cast("dict[str, object]", object.__getattribute__(self, "__dict__"))
        found = values.get("certificate")
        return found if type(found) is _OperationalCertificate else None

    def _history_data(self) -> _OperationalHistory | None:
        """Read only compact history data without a generic/foreign getter bypass."""
        if type(self) is not RetainedAbsorbedEnvelope:
            return None
        values = cast("dict[str, object]", object.__getattribute__(self, "__dict__"))
        found = values.get("history")
        return found if type(found) is _OperationalHistory else None

    def __post_init__(self) -> None:
        """Own arrays and enforce discard clearing when ``keep_influence=False``."""
        object.__setattr__(self, "keep_influence", bool(self.keep_influence))
        object.__setattr__(
            self,
            "_order",
            None if self._order is None else owned_order_scale(self._order),
        )
        if not self.keep_influence:
            object.__setattr__(self, "_observed_influence", None)
            object.__setattr__(self, "_slope_map", None)
            object.__setattr__(
                self,
                "initial_scale",
                None
                if self.initial_scale is None
                else self.initial_scale.without_influence(),
            )
            certificate = RetainedAbsorbedEnvelope._certificate_data(self)
            if certificate is None:
                object.__setattr__(
                    self,
                    "certificate",
                    None
                    if self.certificate is None
                    else self.certificate.without_slope_map(),
                )
            else:
                object.__setattr__(self, "certificate", certificate.without_slope_map())
        else:
            object.__setattr__(
                self, "_observed_influence", _optional(self._observed_influence)
            )
            object.__setattr__(self, "_slope_map", _optional(self._slope_map))

    @property
    def order(self) -> OrderScale | None:
        """Return a fresh owned ``OrderScale``, or ``None``."""
        return None if self._order is None else owned_order_scale(self._order)

    @property
    def influence_discarded(self) -> bool:
        """Return True when ``keep_influence=False`` discarded influence state."""
        return not self.keep_influence

    @property
    def empty(self) -> bool:
        """Return True when the sample has no active slope columns (``K = 0``)."""
        return bool(self.sample.n_slopes == 0)

    @property
    def accepted(self) -> bool:
        """Return True when the selected point is certified at strict or secondary."""
        certificate = RetainedAbsorbedEnvelope._certificate_data(self)
        if certificate is not None:
            return certificate.accepted
        return self.certificate is not None and self.certificate.accepted

    @property
    def level(self) -> Level | None:
        """Return the point certificate level (``'strict'``, ``'secondary'``)."""
        certificate = RetainedAbsorbedEnvelope._certificate_data(self)
        if certificate is not None:
            return certificate.level
        return None if self.certificate is None else self.certificate.level

    def coef(self) -> FloatArray:
        """Return a fresh read-only copy of the reported slope coefficients."""
        if self.selected_point is None:
            raise ValueError("no selected point coefficients available")
        if self.empty:
            return owned_array(np.empty(0, dtype=np.float64))
        if self.initial_scale is not None:
            return self.initial_scale.coefficients
        return self.selected_point.beta

    def se(self) -> FloatArray:
        """Return a fresh read-only copy of the initial direct standard errors."""
        if self.selected_point is None:
            raise ValueError("initial scale evidence was not formed for this candidate")
        if self.empty and (
            RetainedAbsorbedEnvelope._certificate_data(self) is not None
            or self.certificate is not None
        ):
            return owned_array(np.empty(0, dtype=np.float64))
        if self.initial_scale is not None:
            return self.initial_scale.standard_errors
        raise ValueError("initial scale evidence was not formed for this candidate")

    def vcov(self) -> FloatArray:
        """Return a fresh read-only copy of the initial reported covariance matrix."""
        if self.selected_point is None:
            raise ValueError("initial scale evidence was not formed for this candidate")
        if self.empty and (
            RetainedAbsorbedEnvelope._certificate_data(self) is not None
            or self.certificate is not None
        ):
            return owned_array(np.empty((0, 0), dtype=np.float64))
        if self.initial_scale is not None:
            return self.initial_scale.require_reported_covariance()
        raise ValueError("initial scale evidence was not formed for this candidate")

    def require_influence(self, request: str) -> None:
        """Raise ``specification.influence_discarded`` when influence was discarded."""
        if not self.keep_influence:
            raise SpecificationError(
                "specification.influence_discarded",
                {"request": request},
            )

    @property
    def observed_influence(self) -> FloatArray | None:
        """Return a fresh owned copy of ``(n, K)`` observed influence rows."""
        self.require_influence("influence")
        return _optional(self._observed_influence)

    def influence(self) -> FloatArray | None:
        """Return a fresh owned copy of ``(n, K)`` observed influence rows."""
        return self.observed_influence

    @property
    def slope_map(self) -> FloatArray | None:
        """Return a fresh owned copy of ``W = [I_K, 0] H^-1 M'``."""
        self.require_influence("slope_map")
        return _optional(self._slope_map)

    def scores(self) -> FloatArray | None:
        """Return a fresh owned copy of the raw weighted score vector ``b``."""
        self.require_influence("scores")
        if self.certificate is None or self.certificate.rows is None:
            return None
        return self.certificate.rows.b

    def expected_inputs(self) -> ExpectedInformationInputs | None:
        """Return lossless same-point inputs for expected information assembly."""
        self.require_influence("expected_information")
        if (
            self.selected_point is None
            or self.certificate is None
            or self.certificate.rows is None
            or self.certificate.geometry is None
        ):
            return None
        return ExpectedInformationInputs(
            sample=self.sample,
            point=self.selected_point,
            basis=self.certificate.basis,
            rows=self.certificate.rows,
            order=self.order,
            design=self.certificate.geometry.design,
        )

    def expected_hessian(self) -> SparseMatrix | None:
        """Assemble the expected joint Hessian ``M' diag(C_expected) M`` on demand."""
        inputs = self.expected_inputs()
        return None if inputs is None else inputs.expected_hessian()

    def expected_information(self) -> SparseMatrix | None:
        """Assemble the expected joint Hessian ``M' diag(C_expected) M`` on demand."""
        return self.expected_hessian()


AbsorbedRetainedState = RetainedAbsorbedEnvelope

__all__ = [
    "AbsorbedRetainedState",
    "ExpectedInformationInputs",
    "RetainedAbsorbedEnvelope",
    "RetainedAdmissionReceipt",
    "RetainedBasisSnapshot",
    "RetainedCovarianceRequest",
    "RetainedInitialScaleCache",
    "RetainedRowsSnapshot",
    "reconstruct_expected_information",
    "snapshot_covariance_request",
    "snapshot_initial_scale",
]
