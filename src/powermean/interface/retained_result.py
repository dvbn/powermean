"""Post-fit retained absorbed FE result facade.

This module defines :class:`RetainedAbsorbedResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from powermean.data.inference_requests import (
    ConleyRequest,
    HacRequest,
)
from powermean.data.request_values import OMITTED, Omitted
from powermean.errors import SpecificationError
from powermean.interface.retained_covariance import (
    PostFitCovarianceSpec,
    recompute_absorbed_scale_for_envelope,
)
from powermean.interface.retained_metadata import BoundDependence
from powermean.interface.retained_space import RetainedSpaceView
from powermean.interface.retained_time import RetainedTimeView
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.results.absorbed_geometry import ExpectedInformationInputs
from powermean.results.absorbed_history import RetainedHistorySnapshot
from powermean.results.absorbed_outcome_records import RetainedOutcomeSnapshot
from powermean.results.absorbed_owned import RetainedCertificateSnapshot
from powermean.results.absorbed_point_records import RetainedPointSnapshot
from powermean.results.absorbed_points import AbsorbedRoute
from powermean.results.absorbed_retained import (
    RetainedAbsorbedEnvelope,
    RetainedAdmissionReceipt,
)
from powermean.results.absorbed_sample import RetainedSampleSnapshot
from powermean.results.absorbed_scale_cache import RetainedInitialScaleCache
from powermean.results.retained_input_values import InputKey, SampleKey
from powermean.solve.absorbed_certificate import Level
from powermean.solve.absorbed_certificate_geometry import SparseMatrix
from powermean.solve.absorbed_certificate_scale import (
    ComponentRecord,
    CovarianceRecord,
    EstimateRecord,
)

__all__ = [
    "RetainedAbsorbedResult",
]


@dataclass(frozen=True, init=False)
class RetainedAbsorbedResult:
    """Post-fit retained absorbed FE facade owning ``envelope`` and ``dependence``."""

    envelope: RetainedAbsorbedEnvelope
    dependence: BoundDependence | None

    def __init__(
        self,
        envelope: RetainedAbsorbedEnvelope,
        *,
        dependence: BoundDependence | None = None,
    ) -> None:
        """Bind a :class:`RetainedAbsorbedEnvelope` and optional ``BoundDependence``."""
        if type(envelope) is not RetainedAbsorbedEnvelope:
            raise TypeError("envelope must be a RetainedAbsorbedEnvelope")
        if dependence is not None and type(dependence) is not BoundDependence:
            raise TypeError("dependence must be a BoundDependence or None")
        bound_dep = (
            None
            if dependence is None
            else dependence.with_keys(
                input_key=envelope.sample.input_key,
                sample_key=envelope.sample.key,
            )
        )
        object.__setattr__(self, "envelope", envelope)
        object.__setattr__(self, "dependence", bound_dep)

    @property
    def retained(self) -> RetainedAbsorbedEnvelope:
        """Return the underlying :class:`RetainedAbsorbedEnvelope`."""
        return self.envelope

    @property
    def input_key(self) -> InputKey | None:
        """Return the result-local ``InputKey``."""
        return self.envelope.sample.input_key

    @property
    def sample_key(self) -> SampleKey:
        """Return the result-local ``SampleKey``."""
        return self.envelope.sample.key

    @property
    def route(self) -> AbsorbedRoute:
        """Return the absorbed candidate route."""
        return self.envelope.route

    @property
    def keep_influence(self) -> bool:
        """Return whether post-fit influence rows are currently retained."""
        return self.envelope.keep_influence

    @property
    def influence_discarded(self) -> bool:
        """Return True if influence rows were discarded."""
        return self.envelope.influence_discarded

    @property
    def empty(self) -> bool:
        """Return True when the model has no slope parameters (K = 0)."""
        return self.envelope.empty

    @property
    def accepted(self) -> bool:
        """Return True if the selected point was certified."""
        return self.envelope.accepted

    @property
    def level(self) -> Level | None:
        """Return the certification gate level."""
        return self.envelope.level

    @property
    def receipt(self) -> RetainedAdmissionReceipt:
        """Return the original-graph admission receipt."""
        return self.envelope.receipt

    @property
    def sample(self) -> RetainedSampleSnapshot:
        """Return the detached final sample snapshot."""
        return self.envelope.sample

    @property
    def outcome(self) -> RetainedOutcomeSnapshot | None:
        """Return the detached coordinator outcome snapshot, if any."""
        return self.envelope.outcome

    @property
    def history(self) -> RetainedHistorySnapshot:
        """Return the detached candidate history snapshot."""
        return self.envelope.history

    @property
    def selected_point(self) -> RetainedPointSnapshot | None:
        """Return the detached selected point snapshot, if any."""
        return self.envelope.selected_point

    @property
    def certificate(self) -> RetainedCertificateSnapshot | None:
        """Return the detached certificate snapshot, if any."""
        return self.envelope.certificate

    @property
    def order(self) -> OrderScale | None:
        """Return a fresh owned ``OrderScale``, if any."""
        return self.envelope.order

    @property
    def initial_scale(self) -> RetainedInitialScaleCache | None:
        """Return the cached initial scale evaluation, if any."""
        return self.envelope.initial_scale

    def coef(self) -> FloatArray:
        """Return a fresh read-only copy of the reported slope coefficients."""
        return self.envelope.coef()

    def scale_evaluation(
        self, vcov: PostFitCovarianceSpec = None
    ) -> RetainedInitialScaleCache:
        """Return the initial or recomputed scale evaluation for ``vcov``."""
        return recompute_absorbed_scale_for_envelope(
            self.envelope, self.dependence, vcov
        )

    def se(self, vcov: PostFitCovarianceSpec = None) -> FloatArray:
        """Return a fresh read-only copy of the direct reported standard errors."""
        if vcov is None and self.envelope.empty:
            return self.envelope.se()
        return self.scale_evaluation(vcov).standard_errors

    def vcov(self, vcov: PostFitCovarianceSpec = None) -> FloatArray:
        """Return a fresh read-only copy of the reported covariance matrix."""
        if vcov is None and self.envelope.empty:
            return self.envelope.vcov()
        return self.scale_evaluation(vcov).require_reported_covariance()

    def theta_covariance(self, vcov: PostFitCovarianceSpec = None) -> CovarianceRecord:
        """Return a fresh owned solver-coordinate :class:`CovarianceRecord`."""
        return self.scale_evaluation(vcov).covariance

    def estimate(self, vcov: PostFitCovarianceSpec = None) -> EstimateRecord:
        """Return a fresh owned solver-coordinate :class:`EstimateRecord`."""
        return self.scale_evaluation(vcov).estimate

    def components(
        self, vcov: PostFitCovarianceSpec = None
    ) -> tuple[ComponentRecord, ...]:
        """Return fresh owned solver-coordinate :class:`ComponentRecord` records."""
        return self.scale_evaluation(vcov).components

    def influence(self) -> FloatArray | None:
        """Return a fresh read-only copy of the observed influence rows ``u_i``."""
        return self.envelope.influence()

    def scores(self) -> FloatArray | None:
        """Return a fresh read-only copy of the score multipliers ``b_i``."""
        return self.envelope.scores()

    def expected_inputs(self) -> ExpectedInformationInputs | None:
        """Return guarded :class:`ExpectedInformationInputs` before discard."""
        return self.envelope.expected_inputs()

    def expected_information(self) -> SparseMatrix | None:
        """Assemble the guarded expected-information Hessian before discard."""
        return self.envelope.expected_information()

    def time_view(
        self,
        request: HacRequest | None = None,
        *,
        time: str | None = None,
        unit: str | None = None,
        step: int = 1,
        lag: int | Omitted = OMITTED,
        check_chronology: bool = False,
    ) -> RetainedTimeView:
        """Build a :class:`RetainedTimeView` from retained dependence metadata."""
        if self.dependence is None:
            raise SpecificationError(
                "specification.metadata_not_retained",
                {"name": time or "time", "retained": "none"},
            )
        return self.dependence.time_view(
            request,
            time=time,
            unit=unit,
            step=step,
            lag=lag,
            check_chronology=check_chronology,
        )

    def space_view(
        self,
        request: ConleyRequest | None = None,
        *,
        latitude: str | None = None,
        longitude: str | None = None,
        pool_times: Literal[True] | Omitted | None = None,
        check_requirements: bool = False,
    ) -> RetainedSpaceView:
        """Build a :class:`RetainedSpaceView` from retained dependence metadata."""
        if self.dependence is None:
            raise SpecificationError(
                "specification.metadata_not_retained",
                {"name": latitude or "latitude", "retained": "none"},
            )
        return self.dependence.space_view(
            request,
            latitude=latitude,
            longitude=longitude,
            pool_times=pool_times,
            check_requirements=check_requirements,
        )
