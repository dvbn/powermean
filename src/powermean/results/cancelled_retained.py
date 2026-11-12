"""Private cancellation envelope on the common retained/inference data contract."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.data.raw import Column
from powermean.errors import SpecificationError
from powermean.fe._cancel_storage import BuilderRecord, FloatStorage
from powermean.fe.admission_incidence import QuotientRank
from powermean.fe.admission_records import AdmissionWork, EvidenceValue, WorkLog
from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.geometry_storage import BoolStorage
from powermean.inference.cancelled import contracted_influence, reserve_inference
from powermean.model.order_scale import OrderScale
from powermean.model.variance import CancelledFamily, FloatArray, PowerVariance
from powermean.results.cancelled_inference import unavailable
from powermean.results.cancelled_provenance import (
    detached_certificate,
    detached_initializer,
    detached_stages,
    detached_tau1,
)
from powermean.results.owned import _owned_attempt, owned_array
from powermean.results.wz_bias import WZBias
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.cancelled_records import (
    CancelledCertificate,
    CoverageKind,
    MeritMetric,
    RootInitializerRecord,
    RootOptions,
    RootStageRecord,
    RouteKind,
    Tau1SolveResult,
)
from powermean.solve.newton_retry import Attempts
from powermean.solve.newton_trace import Timing
from powermean.solve.options import SolverOptions
from powermean.solve.status import Status
from powermean.systems.admitted import CapturedFit
from powermean.systems.cancelled import CancelledAccounting
from powermean.systems.identified import IdentifiedState

if TYPE_CHECKING:
    from powermean.results.retained_inference import InferenceRequest, RetainedInference


@dataclass(frozen=True)
class CancelledReporting:
    """Reporting coordinates and original labels, already aligned by the caller."""

    omega: float
    column_scales: ArrayLike
    user_labels: ArrayLike
    cancelled_name: str = "cancelled"
    retained_names: tuple[str, ...] = ()


def prepare_cancelled_reporting(
    geometry: NumericalGeometry, reporting: CancelledReporting
) -> tuple[OrderScale, Column, FloatStorage]:
    """Validate and bind ``CancelledReporting`` against ``geometry``."""
    if type(reporting) is not CancelledReporting:
        raise TypeError("use a CancelledReporting record")
    full_order = cast("Callable[[float, ArrayLike], OrderScale]", OrderScale)(
        reporting.omega, reporting.column_scales
    )
    if full_order.omega == 0 or full_order.dimension != geometry.n_slopes_full:
        raise ValueError(
            "nonzero omega and one original-column scale per slope required"
        )
    if geometry.n_slopes_identified == 0:
        raise unavailable("inference without a retained slope")
    block_names = (reporting.cancelled_name, *reporting.retained_names)
    if (
        type(reporting.retained_names) is not tuple
        or any(type(name) is not str or not name for name in block_names)
        or len(set(block_names)) != len(block_names)
    ):
        raise ValueError(
            "distinct nonempty cancelled and retained block names required"
        )
    if len(reporting.retained_names) != len(geometry.structure.block_sizes):
        raise ValueError("one reporting name per retained block required")
    labels = Column(reporting.user_labels, name="user labels")
    if labels.values.shape != (geometry.structure.n_rows,):
        raise ValueError("user labels must align with every original positional row")
    order = OrderScale(full_order.omega, full_order.scale[geometry.kept_slope_indices])
    return order, labels, FloatStorage.store(full_order.scale)


def compute_cancelled_accounting(
    capture: CapturedFit, state: IdentifiedState | None
) -> CancelledAccounting:
    """Return full original row and group accounting for ``capture``."""
    if state is not None:
        return state.accounting()
    structure = capture.admission.geometry.structure
    singleton = structure.counts == 1
    positive = np.bincount(
        structure.cancel_codes[capture.z > 0], minlength=structure.n_groups
    )
    all_zero = positive == 0
    return CancelledAccounting(
        structure.n_rows,
        structure.n_groups,
        singleton,
        all_zero,
        int(singleton.sum()),
        int(all_zero.sum()),
        int(structure.counts[singleton].sum()),
        int(structure.counts[all_zero].sum()),
    )


@dataclass(frozen=True, eq=False, init=False)
class RetainedCancelledEnvelope(BuilderRecord):
    """One immutable private cancelled fit, with deliberate influence discard.

    The accepted geometry supplies names, positions, FE labels and the actual
    retained normalization. Cancelled effects are never reported as estimates.
    Kept slopes use omitted-original-slopes = zero normalization; the numerical
    individual identification mask remains diagnostic, not a new claim.
    """

    __slots__ = (
        "_accounting",
        "_attempts",
        "_capture",
        "_certificate",
        "_full_scales",
        "_history",
        "_inference_bounds",
        "_initial",
        "_initializer",
        "_order",
        "_point",
        "_request",
        "_rhs",
        "_root_options",
        "_root_stages",
        "_state",
        "_tau1_result",
        "_user_labels",
        "_wz_bias",
        "_zero_outcomes",
        "admission_work",
        "block_order",
        "cancelled_index",
        "cancelled_name",
        "continuation_used",
        "executed_variance",
        "geometry",
        "keep_influence",
        "merit_metric",
        "options",
        "quotient",
        "retained_names",
        "root_selection",
        "route",
        "selection_mode",
        "slope_evidence",
        "slope_names_full",
        "status",
        "timing",
        "variance",
        "work",
    )

    _accounting: CancelledAccounting
    _full_scales: FloatStorage
    _zero_outcomes: BoolStorage
    cancelled_name: str
    retained_names: tuple[str, ...]
    block_order: tuple[str, ...]
    cancelled_index: int
    selection_mode: Literal["direct_inputs", "named", "most_levels"]
    variance: CancelledFamily
    executed_variance: CancelledFamily
    route: RouteKind
    root_selection: CoverageKind
    merit_metric: MeritMetric | None
    _root_options: RootOptions | None
    _initializer: RootInitializerRecord | None
    continuation_used: bool
    _root_stages: tuple[RootStageRecord, ...]
    _tau1_result: Tau1SolveResult | None
    _inference_bounds: AttemptBounds
    geometry: NumericalGeometry
    slope_names_full: tuple[str, ...]
    quotient: QuotientRank
    slope_evidence: EvidenceValue
    admission_work: AdmissionWork
    options: SolverOptions
    keep_influence: bool
    status: Status
    timing: Timing
    work: AdmissionWork
    _point: FloatStorage
    _order: OrderScale
    _user_labels: Column
    _certificate: CancelledCertificate
    _history: tuple[CancelledCertificate, ...]
    _attempts: Attempts | None
    _request: InferenceRequest
    _initial: RetainedInference | None
    _capture: CapturedFit | None
    _state: IdentifiedState | None
    _rhs: FloatStorage | None
    _wz_bias: WZBias | None

    @property
    def root_options(self) -> RootOptions | None:
        """Actual root controls, detached; None when no root route executed."""
        value = self._root_options
        return None if value is None else replace(value)

    @property
    def initializer(self) -> RootInitializerRecord | None:
        """Detached initial solve provenance; immutable owned storage stays shared."""
        value = self._initializer
        return None if value is None else detached_initializer(value)

    @property
    def root_stages(self) -> tuple[RootStageRecord, ...]:
        """Detached current-family stage diagnostics, without equation-state replay."""
        return detached_stages(self._root_stages)

    @property
    def tau1_result(self) -> Tau1SolveResult | None:
        """Detached optional inner-solver diagnostics, with no refit or outer copy."""
        value = self._tau1_result
        return None if value is None else detached_tau1(value)

    @property
    def equation(self) -> str:
        """Exact equation, normalization and statistical-multiplier provenance."""
        var_label = (
            "PowerVariance(2)"
            if isinstance(self.executed_variance, PowerVariance)
            and self.executed_variance.tau == 2.0
            else repr(self.executed_variance)
        )
        return (
            f"cancelled sum moments; H=-dS/dtheta'; {var_label}; unit multipliers"
        )

    @property
    def column_scales(self) -> FloatArray:
        """All original slope scales, including omitted slopes, in source order."""
        return self._full_scales.view()

    @property
    def zero_outcome_rows(self) -> NDArray[np.bool_]:
        """Zero outcomes in original positional row order, retained after discard."""
        return self._zero_outcomes.view()

    @property
    def accounting(self) -> CancelledAccounting:
        """Detached original zero/singleton/all-zero group accounting."""
        return self._accounting._replace(
            singleton_groups=owned_array(self._accounting.singleton_groups),
            all_zero_groups=owned_array(self._accounting.all_zero_groups),
        )

    @property
    def accepted(self) -> bool:
        """Whether fresh final certification, including initial inference, passed."""
        return (
            self.status in ("verified", "verified_secondary")
            and self._certificate.accepted
        )

    @property
    def level(self) -> str | None:
        """The retained original-equation certificate's strict/secondary level."""
        return self._certificate.level if self.accepted else None

    @property
    def coverage(self) -> None:
        """No characterized existence analysis is claimed by this bounded entry."""
        return None

    @property
    def certificate(self) -> CancelledCertificate:
        """Fresh diagnostic metadata detached from the retained certificate."""
        return detached_certificate(self._certificate)

    @property
    def history(self) -> tuple[CancelledCertificate, ...]:
        """Every fresh retry assessment, followed by the fresh selected certificate."""
        return tuple(detached_certificate(item) for item in self._history)

    @property
    def attempts(self) -> Attempts | None:
        """Actual shared retry/last-state records with detached numeric metadata."""
        if self._attempts is None:
            return None
        return self._attempts._replace(
            attempts=tuple(_owned_attempt(item) for item in self._attempts.attempts)
        )

    @property
    def request(self) -> InferenceRequest:
        """The fit-time request, detached including cluster array metadata."""
        return copy.deepcopy(self._request)

    @property
    def initial_scale(self) -> RetainedInference | None:
        """Common inference record, withheld on failed fits, detached outward."""
        return copy.deepcopy(self._initial) if self.accepted else None

    @property
    def inference_diagnostics(self) -> RetainedInference | None:
        """Computed quantities of a failed scale remain inspectable as diagnostics."""
        return copy.deepcopy(self._initial)

    @property
    def n_rows(self) -> int:
        """Original rows, including zeros, singletons and all-zero groups."""
        return self.geometry.structure.n_rows

    @property
    def count(self) -> int:
        """Primitive sampling units N: cancelled groups, distinct from rows."""
        return self.geometry.structure.n_groups

    @property
    def slope_names(self) -> tuple[str, ...]:
        """Kept original slope names in the geometry's reporting order."""
        return tuple(self.slope_names_full[j] for j in self.geometry.kept_slope_indices)

    @property
    def omitted_slopes(self) -> tuple[tuple[str, str], ...]:
        """Named omitted and annihilated slopes with their accepted reasons."""
        return tuple(
            (self.slope_names_full[j], why) for j, why in self.geometry.omitted_slopes
        )

    @property
    def user_labels(self) -> NDArray[Any]:
        """Original user labels, separate from immutable positional row keys."""
        return self._user_labels.values.copy()

    @property
    def order(self) -> OrderScale:
        """Detached nonzero order and kept original-column scales."""
        return OrderScale(self._order.omega, self._order.scale)

    def coef(self) -> FloatArray:
        """Reported normalized kept slopes, even for a visibly failed point."""
        k = self.geometry.n_slopes_identified
        return self._order.coefficients(self._point.view()[:k])

    def retained_effects(self) -> FloatArray:
        """Estimated retained effects under actual T normalization, in block order."""
        full = self.geometry.expand(self._point.view())[self.geometry.n_slopes_full :]
        if full.size == 0:
            return owned_array(full)
        return OrderScale(self._order.omega, np.ones(full.size)).coefficients(full)

    def se(self) -> FloatArray:
        """Direct initial standard errors, only for a certified point."""
        if not self.accepted or self._initial is None:
            raise ValueError("inference is withheld for an uncertified cancelled fit")
        return owned_array(self._initial.standard_errors)

    def vcov(self) -> FloatArray:
        """Initial reported covariance, or its recorded range refusal."""
        if not self.accepted or self._initial is None:
            raise ValueError("inference is withheld for an uncertified cancelled fit")
        return self._initial.require_reported_covariance()

    def require_influence(self, request: str) -> None:
        """Refuse alternate requests after deliberate influence discard."""
        if not self.keep_influence:
            raise SpecificationError(
                "specification.influence_discarded", {"request": request}
            )

    def influence(self, *, rows: bool = False) -> FloatArray:
        """Same-point group (default) or original-row reported slope influences."""
        return self.influence_with_work(rows=rows)[0]

    def influence_with_work(
        self, *, rows: bool = False
    ) -> tuple[FloatArray, AdmissionWork]:
        """Reconstruct the requested influences and return actual operation counts."""
        self.require_influence("influence")
        if not self.accepted or self._state is None or self._rhs is None:
            raise ValueError("no certified cancelled influence is available")
        reserve_inference(self.geometry, self._inference_bounds)
        work = WorkLog()
        values = contracted_influence(
            self._state, self._rhs.view(), rows=rows, work=work
        )
        mapped = work.run("report_influence", lambda: self._order.influences(values))
        return mapped, work.snapshot()
