"""Thin retained-state adapter of the conventions, counts and breads.

Internal module: it extracts the retained arrays and records of a
:class:`~powermean.results.absorbed_retained.RetainedAbsorbedEnvelope`, checks the
request against the cache and the discard state, and hands arrays to the kernels of
:mod:`powermean.inference.ssc`, :mod:`powermean.inference.absorbed_rank`,
:mod:`powermean.inference.bread`, :mod:`powermean.inference.multiway` and
:mod:`powermean.inference.psd`.  It imports nothing of the interface: a cluster
dimension arrives already bound as :class:`~powermean.inference.convention.ClusterCodes`
(one code per final row of the sample), which the interface forms from its dependence
sidecar.  No public name, parser or export is touched.

Multiway.  A request with two or more dimensions, or with an explicit ``multiway`` form,
takes the multiway path: the raw port forms the signed (``"cgm"``) or DDG sum of the
active rows' influence with every factor off, and the convention's factor is applied
once by its owner (``G_min`` over the requested dimensions, repetitions kept, under
``cluster_count="min"``; each component's own count under ``"each"``).  The ``psd``
adjustment is then a SEPARATE step on the REPORTED factor-scaled covariance of a signed
request with two or more dimensions, resolved from the request's explicit ``psd``, else
the convention's: ``"cgm"`` applies the native single-fit rule of
:func:`powermean.inference.psd.adjust_psd` in the native standardized frame and in the
sequence of the convention in force, keeps the raw matrix, components, frame and
replaced eigenvalues, reads the standard errors from the FINAL reported matrix's
diagonal, and issues the notes ``covariance.psd_adjusted`` (active) and
``covariance.indefinite`` (a final matrix that is not positive semidefinite; no second
repair); ``"none"`` leaves the raw estimate, whose negative variance gives a missing
standard error.  DDG and one-way requests are never adjusted.  The covariance of
``theta`` is never adjusted, and the reported matrix must be representable for an
adjusted request (``FloatingPointError`` otherwise, with nothing replaced).

Frame and sequence.  The frame scales are the native unweighted row-sample standard
deviations, floored at ``1e-3``, of the user's regressor columns as supplied (the
retained unscaled reported columns: the scaled design times its power-of-two column
scale, exactly) on the scale sample, which on the supported domain is the estimation
sample, taken here as the active rows of the final sample (rows of positive weight, one
row per compressed row whatever its frequency count); the reported intercept has scale
exactly one.  The record keeps the scales, the deviations, the floor flags, the row
count and the sample fingerprint (:class:`FrameRecord`).  The sequence is the
convention's, read from the resolved record by its origin
(:func:`powermean.inference.ssc.sequence_of`): ``"legacy"`` under the ``"stata"`` preset
put in force by its name (an explicit ``"stata"``, the default, or inherited from the
retained initial request, whose record holds that name), ``"modern"`` otherwise, every
custom record included whatever its values; the frame record names the convention as it
was put in force.  For the legacy sequence of an absorbed fit, whose reported
coefficients hold no intercept, the internal reporting constant enters the frame matrix:
its influence row is the weighted-mean functional of the retained nuisance geometry at
the same fitted point (:mod:`powermean.inference.absorbed_constant`, the staged weights
being the expected curvature of the retained rows: true weights times the IRLS working
weight, ``mu`` at ``tau = 1``), the raw port forms its variance and cross-covariances
beside the slopes' retained influence rows with the same cluster totals, the
convention's factor is applied once by the same owner, the reporting map once with the
constant at scale one, and the row is retained in the adjustment record, never reported.
The reported intercept is the column of ones of the retained unscaled no-FE design (the
package's own intercept rule, ``powermean.solve.starts``); absorbed fits report none,
since a constant column is absorbed by the effects (``powermean.data.collinearity``).  A
no-FE fit without an intercept under the legacy sequence has no nuisance geometry and
forms no constant row (a held domain, recorded as such).  Nothing of the fit-time gate,
the initial cache or the one-dimension path changes, and a refusal leaves the retained
state as it is.

Warnings at the composition boundary.  The frozen raw port warns about its RAW estimate
as it forms it ("No pseudoinverse, zero fallback, or PSD correction was applied"), which
is accurate for a raw output and false for an adjusted one.  The multiway path therefore
records the port's diagnostic warnings of that one call (:func:`_raw_multiway_estimate`)
and reissues them unchanged whenever the reported matrix is the raw estimate (``"psd":
"none"``, the ``"none"`` preset, DDG, a ``FloatingPointError`` refusal, or a ``"cgm"``
treatment whose adjustment was inactive); when the adjustment was active they are
withheld and the notes alone describe the reported matrix, the port's diagnostics of the
raw estimate staying in the record (``multiway.estimate.diagnostics``).  The port is
unchanged, no warning filter outside that call is touched and no class is suppressed.

What is distinguished: the sample's original ``positions`` (identity of the final
rows), the sampling units (the final rows of positive weight, ``units.identifiers``),
the raw ``weight_values`` and their frequency kind, and the statistical count ``N``
(the exact frequency total of the active rows, else their number, equal to the
sample's ``effective_n``).  The exact full ``K`` is the certificate's
``geometry.dimension``; the convention counts are recomputed for every request from
the grouping's blocks and the requested dimensions on the active rows.

Cache and discard.  A robust or one-cluster request under a convention with the
``"stata"`` values and the observed bread whose exact identity equals the fit-time
request is served from the retained initial cache, before and after
``keep_influence=False``, whatever the origin of that convention: such a request takes
no ``psd`` step, so no sequence enters its answer, and its factor and reference are the
values'.  Every other request first requires the retained influence
(``specification.influence_discarded`` after discard) and reads the retained observed
influence rows, or forms the expected bread from the guarded expected inputs on the same
full basis.  Nothing is solved, reselected or replayed; a refusal leaves the envelope as
it is.  The reporting map is applied once: the direct standard errors by the order
scale, the reported covariance by its range-safe entrywise map or its recorded range
error.  The ``ValueError`` guards on a missing grouping, geometry or order scale, an
inconsistent statistical count or cache count, and absent influence or expected inputs
restate invariants of the retained records, which their builders cannot violate.

The candidate of a fit.  A cluster request of the expanded menu at a fit without fixed
effects is answered by the same composition at the influence record of a certificate's
fresh evaluation, before any result exists (:func:`candidate_nofe_inference`): the
counts at the evaluation's units, the convention of the request at the fit (its explicit
``ssc``, else the default ``"stata"``; nothing is inherited at the fit), the observed
influence rows of that record, the port, the factor once, the reporting map once and the
``psd`` frame and sequence, exactly as :func:`retained_nofe_inference` composes them
after the fit.  The reported scale of a solve reads its standard errors at every
candidate it assesses, with the adjustment's notes withheld (``Reporting.notes``); the
record of the fit forms them once, with its notes.  The port's raw diagnostics keep
their routing at every call.  A request answered from the retained state after the fit
never reads an expanded fit-time record through the default initial-cache rule: that
record answers its own request by its typed request and convention
(:mod:`powermean.results.active_covariance`).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from powermean.data.fingerprint import sample_fingerprint
from powermean.data.inference_conventions import (
    PSD_TREATMENTS,
    ConventionValues,
    PsdTreatment,
)
from powermean.data.inference_requests import (
    MULTIWAY_FORMS,
    ClusterRequest,
    Multiway,
    PairAdjustment,
)
from powermean.data.request_values import OMITTED, Omitted, choice, given
from powermean.data.retained import OrderSample
from powermean.errors import CovarianceWarning, SpecificationError
from powermean.fe.admission_records import AdmissionWork
from powermean.inference.absorbed_constant import ConstantRow, constant_influence
from powermean.inference.absorbed_details import InferenceDetails, typed_details
from powermean.inference.absorbed_rank import AbsorbedCount, absorbed_count
from powermean.inference.bread import BREADS, Bread, same_point_influence
from powermean.inference.contributions import Influence, statistical_count
from powermean.inference.convention import ClusterCodes, Covariance, _active_codes
from powermean.inference.influence import influence_values, moment_influence_map
from powermean.inference.multiway import (
    MultiwayCovariance,
    multiway_covariance,
    multiway_estimate,
)
from powermean.inference.psd import (
    FrameScales,
    PsdAdjustment,
    Sequence,
    adjust_psd,
    column_scales,
    reported_errors,
)
from powermean.inference.ssc import (
    STATA,
    AppliedConvention,
    ConventionCovariance,
    ResolvedConvention,
    apply_convention,
    convention_counts,
    direct_errors,
    primitive_estimate,
    resolve_convention,
    scaled_covariance,
)
from powermean.inference.vcov import CovarianceEstimate
from powermean.inference.vcov import CovarianceWarning as PortWarning
from powermean.inference.wz_covariance import retained_covariance_record
from powermean.inference.wz_operator import PairCovariance
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.results.absorbed_retained import RetainedAbsorbedEnvelope
from powermean.results.absorbed_scale_cache import RetainedInitialScaleCache
from powermean.results.cancelled_inference import retained_cancelled_inference
from powermean.results.cancelled_retained import RetainedCancelledEnvelope
from powermean.results.covariance import cluster_codes, frequency_of
from powermean.results.owned import _optional, owned_array
from powermean.results.retained import CovarianceRecord, RetainedFit
from powermean.solve.absorbed_certificate_scale import CovarianceRequest
from powermean.systems.joint import SamplingUnits

IntegerArray = NDArray[np.integer[Any]]
MeatKind = Literal["robust", "cluster"]
ConstantBuilder = Callable[[Bread], ConstantRow]


@dataclass(frozen=True, eq=False)
class FrameRecord:
    """The frame of one ``psd`` operation, retained internally.

    Attributes
    ----------
    sequence : {"modern", "legacy"}
        The native sequence applied.
    convention : str
        The label of the convention in force that selected it.
    scales : array
        ``T``, one scale per reported coefficient (the intercept's exactly one).
    deviations : array
        The computed standard deviations before the floor (the intercept's zero).
    floored : array of bool
        Whether the floor was active for each reported coefficient.
    count : int
        ``N``, the rows of the scale sample.
    fingerprint : str
        The fingerprint of the sample whose rows entered the scales.
    constant : bool
        Whether the internal reporting constant entered the frame matrix.
    constant_total : float or None
        The weight total of the constant's weighted mean, or ``None``.
    constant_diagnostics : tuple of str
        The port's diagnostics of the auxiliary augmented raw estimate that formed the
        constant's row (never the reported matrix's); empty without a constant.
    """

    sequence: Sequence
    convention: str
    scales: FloatArray
    deviations: FloatArray
    floored: NDArray[np.bool_]
    count: int
    fingerprint: str
    constant: bool
    constant_total: float | None
    constant_diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        """Own the arrays."""
        object.__setattr__(self, "scales", owned_array(self.scales))
        object.__setattr__(self, "deviations", owned_array(self.deviations))
        object.__setattr__(
            self,
            "floored",
            np.frombuffer(np.asarray(self.floored, dtype=np.bool_).tobytes(), np.bool_),
        )


@dataclass(frozen=True)
class Reporting:
    """What the reporting step of a computed request needs, in lower-layer terms.

    ``order`` the order scale; ``intercept`` the index of the reported intercept or
    ``None``; ``columns`` the unscaled reported regressor columns of the active rows
    (the scale sample); ``fingerprint`` the sample's; ``active`` the positions of the
    active rows among the final rows; ``constant`` the builder of the internal constant
    row of an absorbed fit under a bread, or ``None`` for a fit without nuisance
    geometry; ``notes`` whether the ``psd`` notes are issued when the adjustment is
    active or its final matrix indefinite (``False`` for the reported scale of a solve,
    whose fit-time record issues them once).
    """

    order: OrderScale
    intercept: int | None
    columns: FloatArray
    fingerprint: str
    active: IntegerArray
    constant: ConstantBuilder | None = None
    notes: bool = True


@dataclass(frozen=True)
class InferenceRequest:
    """One post-fit request on retained absorbed state, in lower-layer terms.

    Parameters
    ----------
    kind : {"robust", "cluster"}
        The robust estimator or clustering on the bound dimensions.
    clusters : tuple of ClusterCodes
        The bound dimensions, in the order of the request, repetitions kept, one
        integer code per final row of the sample; empty for ``"robust"``.
    ssc : str, ConventionValues or OMITTED
        The explicit convention, or :data:`OMITTED`.
    bread : {"observed", "expected"}
        The bread of the slope influence.
    inherited : ConventionValues, optional
        An internal declaration of the fit's custom record: inherited and custom
        whatever its values, so it never selects the legacy sequence.
        ``None`` reads the retained initial request, whose record holds its
        convention by name, the ``"stata"`` preset.
    multiway : {"cgm", "ddg"} or OMITTED
        The multiway form of a cluster request (the key ``multiway``);
        omitted, the signed form with two or more dimensions.  Given, the request
        takes the multiway path whatever its number of dimensions.
    psd : {"cgm", "none"} or OMITTED
        The ``psd`` treatment for this request (the key ``psd``), valid on
        a signed multiway request only: two or more dimensions, not ``"ddg"``
        (``specification.conflicting`` otherwise, as the parser refuses it).
        Omitted, the convention in force decides.
    """

    kind: MeatKind
    clusters: tuple[ClusterCodes, ...] = ()
    ssc: str | ConventionValues | Omitted = OMITTED
    bread: Bread = "observed"
    inherited: ConventionValues | None = None
    multiway: Multiway | Omitted = OMITTED
    psd: PsdTreatment | Omitted = OMITTED
    adjustment: PairAdjustment | Omitted = OMITTED

    def __post_init__(self) -> None:
        """Check the kind, the bound dimensions, the bread, the form and ``psd``."""
        if self.kind not in ("robust", "cluster"):
            raise ValueError("kind must be 'robust' or 'cluster'")
        clusters = tuple(self.clusters)
        if any(type(c) is not ClusterCodes for c in clusters):
            raise TypeError("clusters must be ClusterCodes records")
        if (self.kind == "cluster") != bool(clusters):
            raise ValueError(
                "a cluster request names its dimensions; a robust one none"
            )
        object.__setattr__(self, "clusters", clusters)
        if self.bread not in BREADS:
            raise ValueError("bread must be 'observed' or 'expected'")
        if self.inherited is not None and type(self.inherited) is not ConventionValues:
            raise TypeError("inherited must be a ConventionValues or None")
        if given(self.multiway):
            if not clusters:
                raise ValueError("multiway belongs to a cluster request")
            form = choice("multiway", self.multiway, MULTIWAY_FORMS)
            object.__setattr__(self, "multiway", form)
        if given(self.psd):
            if not clusters:
                raise ValueError("psd belongs to a cluster request")
            psd = choice("psd", self.psd, PSD_TREATMENTS)
            object.__setattr__(self, "psd", psd)
            if not self.signed:
                second = (
                    "one cluster dimension" if len(clusters) < 2 else "multiway='ddg'"
                )
                raise SpecificationError(
                    "specification.conflicting",
                    {"first": f"psd={psd!r}", "second": second},
                )

        if given(self.adjustment):
            parsed = ClusterRequest(
                tuple(c.name for c in clusters),
                self.multiway,
                self.psd,
                self.adjustment,
            )
            object.__setattr__(self, "adjustment", parsed.adjustment)

    @classmethod
    def from_cluster_request(
        cls,
        request: ClusterRequest,
        clusters: tuple[ClusterCodes, ...],
        *,
        ssc: str | ConventionValues | Omitted = OMITTED,
        bread: Bread = "observed",
        inherited: ConventionValues | None = None,
    ) -> InferenceRequest:
        """Bind the parser's record to dimensions already bound to the rows.

        ``clusters`` names the request's dimensions in its order, repetitions kept;
        another sequence of names is refused.  The parser owns the spellings; this
        binding adds none.
        """
        if type(request) is not ClusterRequest:
            raise TypeError("request must be a ClusterRequest")
        bound = tuple(clusters)
        if any(type(c) is not ClusterCodes for c in bound):
            raise TypeError("clusters must be ClusterCodes records")
        if tuple(c.name for c in bound) != request.dimensions:
            raise ValueError("the bound dimensions must be the request's, in order")
        return cls(
            "cluster",
            bound,
            ssc,
            bread,
            inherited,
            request.multiway,
            request.psd,
            request.adjustment,
        )

    @property
    def form(self) -> Multiway:
        """The multiway form in force: the explicit one, else ``"cgm"``."""
        return "cgm" if isinstance(self.multiway, Omitted) else self.multiway

    @property
    def signed(self) -> bool:
        """Whether the request is signed multiway: two or more dimensions, not DDG."""
        return len(self.clusters) >= 2 and self.form == "cgm"

    @property
    def multiway_path(self) -> bool:
        """Whether the multiway kernel answers: several dimensions, or a given form."""
        return len(self.clusters) >= 2 or given(self.multiway)

    @property
    def label(self) -> str:
        """The request as a caller would write it, for a refusal."""
        if self.kind == "robust":
            head = "vcov('robust'"
        else:
            dims = ", ".join(repr(c.name) for c in self.clusters)
            head = f"vcov(cluster={dims}"
        ssc = "" if isinstance(self.ssc, Omitted) else f", ssc={self.ssc!r}"
        form = "" if isinstance(self.multiway, Omitted) else f", multiway={self.form!r}"
        psd = "" if isinstance(self.psd, Omitted) else f", psd={self.psd!r}"
        adjustment = (
            ""
            if isinstance(self.adjustment, Omitted)
            else f", adjustment={self.adjustment!r}"
        )
        return f"{head}{form}{psd}{adjustment}{ssc}, bread={self.bread!r})"


@dataclass(frozen=True, eq=False)
class RetainedInference:
    """The answer to one request: convention, counts, covariance and reported scale.

    Attributes
    ----------
    request : InferenceRequest
        The request.
    applied : AppliedConvention
        The convention in force, its source, the counts (``N``, ``K``, ``K_reference``,
        ``K_factor``, the nested adjustment, ``G`` per dimension), the factor with its
        owner and the reference.
    absorbed : AbsorbedCount or None
        The per-block convention count of this request, or ``None`` without fixed
        effects.
    bread : {"observed", "expected"}
        The bread used.
    covariance : ConventionCovariance or None
        The covariance of ``theta`` formed here; ``None`` when served from the cache.
    theta_covariance : array
        The covariance of ``theta``, the factor applied once, owned.
    standard_errors : array
        The direct reported standard errors, owned.
    reported_covariance : array or None
        The reported covariance, or ``None`` with ``covariance_error`` set.
    covariance_error : str or None
        The range error of the reported covariance, if any.
    gamma : array
        The selected point ``theta``, owned.
    from_initial_cache : bool
        Whether the retained initial cache answered the request.
    count, active_rows, stored_rows : int
        ``N``; the rows of positive weight; the final stored rows of the sample.
    multiway : MultiwayCovariance or None
        The multiway record (form, dimensions, components, factors, the raw port
        estimate) when the multiway kernel answered; ``None`` otherwise.  Exactly one
        of ``covariance`` and ``multiway`` is set for a computed request.
    psd : {"cgm", "none"} or None
        The ``psd`` treatment resolved for a signed multiway request (the request's
        explicit ``psd``, else the convention's); ``None`` where it does not apply.
    adjustment : PsdAdjustment or None
        The record of the ``psd`` operation when applied (``psd == "cgm"``): the raw
        reported matrix, the frame matrix, the eigenvalues replaced, the final matrices
        and their diagnostics.  ``reported_covariance`` is then its ``adjusted`` matrix
        and ``standard_errors`` are read from that matrix's diagonal;
        ``theta_covariance`` stays the raw factor-scaled covariance of ``theta``.
    intercept : int or None
        The index of the reported intercept used by the intercept-block rule, or
        ``None`` where none is reported.
    work : AdmissionWork or None
        Actual cancelled inference calls through assembly; absent on existing routes.
        The fit envelope carries the complete lifecycle ledger.
    frame : FrameRecord or None
        The frame and sequence of the ``psd`` operation when it was applied: the
        scales, their provenance and the internal constant's record; ``None`` otherwise.
    """

    request: InferenceRequest
    applied: AppliedConvention
    absorbed: AbsorbedCount | None
    bread: Bread
    covariance: ConventionCovariance | None
    theta_covariance: FloatArray
    standard_errors: FloatArray
    reported_covariance: FloatArray | None
    covariance_error: str | None
    gamma: FloatArray
    from_initial_cache: bool
    count: int
    active_rows: int
    stored_rows: int
    multiway: MultiwayCovariance | None = None
    psd: PsdTreatment | None = None
    adjustment: PsdAdjustment | None = None
    intercept: int | None = None
    frame: FrameRecord | None = None
    work: AdmissionWork | None = None
    wz: PairCovariance | None = None

    def __post_init__(self) -> None:
        """Own the arrays."""
        object.__setattr__(self, "theta_covariance", owned_array(self.theta_covariance))
        object.__setattr__(self, "standard_errors", owned_array(self.standard_errors))
        object.__setattr__(
            self, "reported_covariance", _optional(self.reported_covariance)
        )
        object.__setattr__(self, "gamma", owned_array(self.gamma))

    @property
    def degenerate(self) -> bool:
        """Whether a reported standard error is exactly zero."""
        return bool((self.standard_errors == 0.0).any())

    @property
    def adjusted(self) -> bool:
        """Whether the ``psd`` adjustment was active on the reported matrix."""
        return self.adjustment is not None and self.adjustment.active

    def require_reported_covariance(self) -> FloatArray:
        """The reported covariance, or ``FloatingPointError`` with the range error."""
        if self.reported_covariance is None:
            raise FloatingPointError(
                self.covariance_error or "the reported covariance is not representable"
            )
        return owned_array(self.reported_covariance)


# ------------------------------------------------------------------ extraction


def _sampling_units(envelope: RetainedAbsorbedEnvelope) -> SamplingUnits:
    """The sample's sampling units: the final rows of positive weight."""
    return envelope.sample.units


def _active_cluster_codes(
    request: InferenceRequest, units: SamplingUnits
) -> tuple[tuple[str, IntegerArray], ...]:
    """The codes of each bound dimension on the active rows, checked as accepted."""
    return tuple((c.name, _active_codes(c, units)) for c in request.clusters)


def _blocks(
    envelope: RetainedAbsorbedEnvelope, active: IntegerArray
) -> tuple[tuple[str, IntegerArray], ...]:
    """The retained blocks' codes on the active rows, in their order."""
    grouping = envelope.sample.grouping
    if grouping is None:
        raise ValueError("the retained sample holds no fixed-effect grouping")
    return tuple((block.name, block.codes[active]) for block in grouping.blocks)


def _statistical_count(envelope: RetainedAbsorbedEnvelope, active: IntegerArray) -> int:
    """``N`` of the active rows, equal to the sample's ``effective_n``."""
    weights = envelope.sample.weight_values
    count = statistical_count(weights[active], frequency=envelope.sample.frequency)
    if count != envelope.sample.effective_n:
        raise ValueError("the active rows do not give the sample's statistical count")
    return count


def _cache_request(
    request: InferenceRequest, envelope: RetainedAbsorbedEnvelope
) -> CovarianceRequest:
    """The request in the accepted fit-time form, bound to the sample's positions."""
    if request.kind == "robust":
        return CovarianceRequest("robust")
    cluster = request.clusters[0]
    return CovarianceRequest(
        "cluster", cluster.name, cluster.codes, envelope.sample.positions
    )


def _cluster_cache_matches(
    request: InferenceRequest,
    initial: RetainedInitialScaleCache,
    envelope: RetainedAbsorbedEnvelope,
) -> bool:
    """Match exact dimensions, aligned codes and requested multiway form."""
    if request.adjustment != initial.request.adjustment:
        return False
    init_clusters = initial.request.clusters
    if len(request.clusters) != len(init_clusters):
        return False
    if initial.request.positions is None or not np.array_equal(
        initial.request.positions, envelope.sample.positions
    ):
        return False
    for rc, (ic_name, ic_codes) in zip(request.clusters, init_clusters, strict=True):
        if rc.name != ic_name or not np.array_equal(rc.codes, ic_codes):
            return False
    init_uses_mw = len(init_clusters) >= 2 or initial.request.multiway is not OMITTED
    if request.multiway_path != init_uses_mw:
        return False
    if request.multiway_path:
        init_form = (
            "cgm" if initial.request.multiway is OMITTED else initial.request.multiway
        )
        if request.form != init_form:
            return False
    return True


def _applied_cache_matches(
    request: InferenceRequest, applied: AppliedConvention, det: InferenceDetails
) -> bool:
    """The same values, ``psd`` sequence and exact request-specific counts."""
    if applied.convention.values != det.applied.convention.values:
        return False
    psd = _resolved_psd(request, applied)
    if psd != det.psd:
        return False
    if psd == "cgm" and applied.sequence != det.applied.sequence:
        return False
    if det.count != applied.counts.count or det.applied.counts != applied.counts:
        raise ValueError(
            "the retained initial cache does not hold the request's counts"
        )
    return True


def _cache_matches(
    request: InferenceRequest,
    applied: AppliedConvention,
    envelope: RetainedAbsorbedEnvelope,
) -> RetainedInitialScaleCache | None:
    """The initial cache when the request is exactly the fit-time one, else ``None``."""
    initial = envelope.initial_scale
    if initial is None or initial.request.kind != request.kind:
        return None
    if initial.request.bread != request.bread:
        return None
    if initial.details is None:
        if (
            request.bread != "observed"
            or applied.convention.values != STATA
            or request.multiway_path
        ):
            return None
        if _cache_request(request, envelope).identity() != initial.request.identity():
            return None
        cached = initial.covariance
        expected = applied.factor
        assert expected is not None
        if type(cached) is not Covariance or (cached.factor, cached.count) != (
            expected.value,
            applied.counts.count,
        ):
            raise ValueError(
                "the retained initial cache does not hold the request's counts"
            )
        return initial
    det = typed_details(initial.details)
    if request.kind == "cluster" and (
        not _cluster_cache_matches(request, initial, envelope)
    ):
        return None
    return initial if _applied_cache_matches(request, applied, det) else None


def _influence_rows(envelope: RetainedAbsorbedEnvelope, bread: Bread) -> FloatArray:
    """The slope influence of every final row under ``bread`` (after the guard)."""
    if bread == "observed":
        found = envelope.observed_influence
        if found is None:
            raise ValueError(
                "observed influence rows were not formed for this candidate"
            )
        return found
    inputs = envelope.expected_inputs()
    if inputs is None:
        raise ValueError(
            "expected-information inputs were not formed for this candidate"
        )
    computed = same_point_influence(
        inputs.design,
        inputs.basis.basis,
        inputs.rows.b,
        inputs.expected_hessian(),
        bread="expected",
    )
    return computed.influence


# ------------------------------------------------------------ no-FE extraction


def _checked_nofe(
    retained: object, request: object
) -> tuple[RetainedFit, InferenceRequest]:
    """The types, and a certified no-FE point with an evaluation (``K > 0``)."""
    if type(retained) is not RetainedFit:
        raise TypeError("retained must be a RetainedFit")
    if type(request) is not InferenceRequest:
        raise TypeError("request must be an InferenceRequest")
    if not retained.certified or retained.theta is None or retained.certificate is None:
        raise ValueError("no selected point was certified for this candidate")
    evaluation = retained.certificate.evaluation
    if evaluation is None:
        raise ValueError("the certificate retains no evaluation for this candidate")
    n_slopes = retained.sample.data.nparams
    if n_slopes < 1 or evaluation.bread.shape != (n_slopes, n_slopes):
        raise ValueError("the model has no parameters (K = 0)")
    return retained, request


def _nofe_intercept(sample: OrderSample) -> int | None:
    """The reported intercept of a no-FE fit: the column of ones of its design.

    The package's own intercept rule (``powermean.solve.starts``): a column that holds
    one in every retained row of the unscaled design; the first such column where
    several would (collinearity leaves one), ``None`` where there is none.
    """
    design = sample.data.design
    for column in range(design.shape[1]):
        if np.all(design[:, column] == 1.0):
            return column
    return None


def _nofe_counts(
    sample: OrderSample,
    units: SamplingUnits,
    weights: FloatArray,
    request: InferenceRequest,
    inherited: ConventionValues | ResolvedConvention | None,
) -> tuple[AppliedConvention, tuple[tuple[str, IntegerArray], ...], int, bool]:
    """Convention, counts and factors of a no-FE request at the units of its point.

    ``units`` and ``weights`` are those of the evaluation of the point: the
    certificate's after the fit, the candidate's at the fit; ``inherited`` the fit's
    convention in force after the fit, ``None`` at the fit.
    """
    active = units.identifiers
    data = sample.data
    frequency = frequency_of(sample)
    count = statistical_count(data.weight_values[active], frequency=frequency)
    if count != sample.effective_n or not np.array_equal(
        data.weight_values[active], weights
    ):
        raise ValueError("the active rows do not give the sample's statistical count")
    codes = _active_cluster_codes(request, units)
    groups = [int(np.unique(c).size) for _, c in codes]
    counts = convention_counts(count, data.nparams, data.nparams, None, groups)
    resolved = resolve_convention(request.ssc, inherited)
    applied = apply_convention(resolved, counts, kind=request.kind)
    return applied, codes, count, frequency


def _nofe_reporting(
    sample: OrderSample, active: IntegerArray, *, notes: bool = True
) -> Reporting:
    """The reporting map of a no-FE sample: its scale, intercept and scale sample."""
    data = sample.data
    return Reporting(
        sample.order_scale,
        _nofe_intercept(sample),
        owned_array(data.design[active]),
        sample_fingerprint(data),
        active,
        notes=notes,
    )


def _nofe_cache_matches(
    request: InferenceRequest,
    applied: AppliedConvention,
    retained: RetainedFit,
) -> CovarianceRecord | None:
    """The default record when a no-FE request is the fit-time one, else ``None``.

    The default initial-cache rule: a robust or one-way request under the ``"stata"``
    values.  An expanded fit-time record (one that holds its typed answer) is never read
    through it: it answers its own request by its typed request and convention, before
    this adapter is reached.
    """
    default = retained.default
    if (
        default is None
        or default.answer is not None
        or request.bread != "observed"
        or applied.convention.values != STATA
        or request.multiway_path
    ):
        return None
    if request.kind == "robust":
        if default.kind != "robust" or default.cluster is not None:
            return None
    else:
        cluster = request.clusters[0]
        if (
            default.kind != "cluster"
            or default.cluster != cluster.name
            or cluster.name not in retained.sample.metadata
        ):
            return None
        bound = cluster_codes(retained.sample, cluster.name)
        if not np.array_equal(cluster.codes, bound.codes):
            return None
    expected = applied.factor
    assert expected is not None
    expected_clusters = applied.counts.clusters[0] if applied.counts.clusters else None
    if type(default) is not CovarianceRecord or (
        default.factor,
        default.count,
        default.clusters,
    ) != (expected.value, applied.counts.count, expected_clusters):
        raise ValueError(
            "the retained initial cache does not hold the request's counts"
        )
    return default


def _nofe_influence_rows(retained: RetainedFit, bread: Bread) -> FloatArray:
    """The active influence rows of a certified no-FE fit under ``bread``."""
    influence = retained.influence
    certificate = retained.certificate
    if influence is None or certificate is None or certificate.evaluation is None:
        raise ValueError("observed influence rows were not formed for this candidate")
    evaluation = certificate.evaluation
    if not np.array_equal(influence.jacobian, -evaluation.bread):
        raise ValueError("the retained influence jacobian is not -evaluation.bread")
    if bread == "observed":
        return influence_values(influence.contributions, influence.influence_map)
    expected_map = moment_influence_map(-evaluation.information)
    return influence_values(evaluation.contributions, expected_map)


# --------------------------------------------------------------------- service


def _checked(
    envelope: object, request: object
) -> tuple[RetainedAbsorbedEnvelope, InferenceRequest]:
    """The types, and a selected point with slopes (``K > 0``)."""
    if type(envelope) is not RetainedAbsorbedEnvelope:
        raise TypeError("envelope must be a RetainedAbsorbedEnvelope or RetainedFit")
    if type(request) is not InferenceRequest:
        raise TypeError("request must be an InferenceRequest")
    if envelope.selected_point is None or envelope.certificate is None:
        raise ValueError("no selected point was certified for this candidate")
    if envelope.empty:
        raise ValueError("the model has no slope parameters (K = 0)")
    return envelope, request


def _counts(
    envelope: RetainedAbsorbedEnvelope, request: InferenceRequest
) -> tuple[
    AppliedConvention,
    AbsorbedCount,
    SamplingUnits,
    tuple[tuple[str, IntegerArray], ...],
]:
    """Convention, counts and factors of the request, from retained records alone."""
    units = _sampling_units(envelope)
    active = units.identifiers
    certificate = envelope.certificate
    assert certificate is not None
    geometry = certificate.geometry
    if geometry is None:
        raise ValueError("the certificate retains no geometry for this candidate")
    codes = _active_cluster_codes(request, units)
    blocks = _blocks(envelope, active)
    absorbed = absorbed_count(blocks, [c for _, c in codes])
    groups = [int(np.unique(c).size) for _, c in codes]
    counts = convention_counts(
        _statistical_count(envelope, active),
        geometry.n_slopes,
        geometry.dimension,
        absorbed,
        groups,
    )
    inherited: ConventionValues | ResolvedConvention | None = request.inherited
    if inherited is None and envelope.initial_scale is not None:
        if envelope.initial_scale.details is not None:
            inherited = typed_details(envelope.initial_scale.details).applied.convention
        else:
            inherited = resolve_convention(envelope.initial_scale.request.convention)
    resolved = resolve_convention(request.ssc, inherited)
    applied = apply_convention(resolved, counts, kind=request.kind)
    return applied, absorbed, units, codes


def _resolved_psd(
    request: InferenceRequest, applied: AppliedConvention
) -> PsdTreatment | None:
    """The ``psd`` treatment of a signed multiway request: explicit or conventional."""
    if not request.signed:
        return None
    if given(request.psd):
        found: PsdTreatment = "cgm" if request.psd == "cgm" else "none"
        return found
    return applied.psd


def _warn(identifier: str, fields: dict[str, object]) -> None:
    """Issue one covariance note to the caller."""
    warnings.warn(CovarianceWarning(identifier, fields), stacklevel=4)


def _raw_multiway_estimate(
    rows: FloatArray, codes: tuple[tuple[str, IntegerArray], ...], form: Multiway
) -> tuple[CovarianceEstimate, list[warnings.WarningMessage]]:
    """The port's raw estimate, with its diagnostic warnings held for routing.

    The frozen port warns about its RAW estimate ("No pseudoinverse, zero fallback, or
    PSD correction was applied") as it forms it, before the separate ``psd`` step of
    this module decides what the result reports.  The warnings of that one call are
    recorded here instead of being issued at once, and any warning of another class
    raised inside the call is reissued immediately; :func:`_route_port_warnings`
    reissues the port's own diagnostics unchanged once the reported matrix is known to
    be the raw estimate, and withholds them when the ``psd`` adjustment was active, in
    which case they do not describe the reported matrix and the notes
    (``covariance.psd_adjusted``, ``covariance.indefinite``) do.  The port's diagnostics
    themselves stay in the record (``estimate.diagnostics``).  No filter outside this
    call is changed and nothing is suppressed globally.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PortWarning)
        estimate = multiway_estimate(rows, codes, form=form)
    held: list[warnings.WarningMessage] = []
    for item in caught:
        if isinstance(item.message, PortWarning):
            held.append(item)
        else:
            warnings.warn(item.message, stacklevel=3)
    return estimate, held


def _route_port_warnings(
    held: list[warnings.WarningMessage], adjustment: PsdAdjustment | None
) -> None:
    """Reissue the port's raw diagnostics unless the ``psd`` adjustment was active."""
    if adjustment is not None and adjustment.active:
        return  # the reported matrix is adjusted: the notes describe it
    for item in held:
        warnings.warn(item.message, stacklevel=3)


def _frame_scales(reporting: Reporting) -> FrameScales:
    """The frame scales of the reported coefficients: the columns', the intercept's one.

    The native standard deviations of the unscaled reported columns on the scale sample
    (``column_scales``); the reported intercept, when one is reported, takes the scale
    exactly one and is not computed.
    """
    columns, index = reporting.columns, reporting.intercept
    size = columns.shape[1]
    keep = np.array([j != index for j in range(size)])
    found = column_scales(columns[:, keep])
    scales, deviations = np.ones(size), np.zeros(size)
    floored = np.zeros(size, dtype=bool)
    scales[keep], deviations[keep] = found.scales, found.deviations
    floored[keep] = found.floored
    return FrameScales(
        scales=owned_array(scales),
        deviations=owned_array(deviations),
        floored=np.frombuffer(floored.tobytes(), dtype=np.bool_),
        count=found.count,
    )


def _constant_row(
    reporting: Reporting,
    common: tuple[InferenceRequest, AppliedConvention, AbsorbedCount | None, Bread],
    rows: FloatArray,
    codes: tuple[tuple[str, IntegerArray], ...],
) -> tuple[FloatArray, ConstantRow, tuple[str, ...]]:
    """The internal constant's raw reported row beside the slopes.

    The constant's influence row on the active rows is appended to the slopes'
    retained rows; the raw port forms the augmented signed estimate with the same
    cluster totals, the convention's factor is applied once by its owner, and the
    reporting map once with the constant at scale one.  Returns the constant's raw
    reported cross-covariances followed by its variance, the constant's record and the
    port's diagnostics of the auxiliary estimate (which never describe the reported
    matrix and are not issued).
    """
    request, applied, bread = common[0], common[1], common[3]
    assert reporting.constant is not None
    constant = reporting.constant(bread)
    augmented = np.column_stack([rows, constant.influence[reporting.active]])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", PortWarning)
        estimate = multiway_estimate(augmented, codes, form=request.form)
    scaled = multiway_covariance(
        estimate, applied, form=request.form, dimensions=[n for n, _ in codes]
    )
    order = reporting.order
    full = OrderScale(order.omega, np.append(order.scale, 1.0))
    reported = full.covariance(scaled.covariance)
    size = rows.shape[1]
    row = np.append(reported[size, :size], reported[size, size])
    return owned_array(row), constant, tuple(estimate.diagnostics)


def _adjusted(
    reported: FloatArray | None,
    error: str | None,
    reporting: Reporting,
    *,
    scales: FrameScales,
    sequence: Sequence,
    constant: FloatArray | None,
) -> PsdAdjustment:
    """The ``psd`` operation on the reported matrix, which must be representable.

    Its notes are issued where ``reporting`` asks for them (``Reporting.notes``).
    """
    if reported is None:
        raise FloatingPointError(
            (error or "the reported covariance is not representable")
            + "; the eigenvalue adjustment acts on the reported "
            "covariance and nothing replaces it"
        )
    adjustment = adjust_psd(
        reported,
        intercept=reporting.intercept,
        scales=scales.scales,
        sequence=sequence,
        constant=constant,
    )
    if not reporting.notes:
        return adjustment
    if adjustment.active:
        _warn("covariance.psd_adjusted", {"count": adjustment.count})
    if adjustment.final_indefinite:
        _warn(
            "covariance.indefinite",
            {
                "method": "adjusted signed multiway",
                "minimum": adjustment.final_minimum,
            },
        )
    return adjustment


def _frame_step(
    common: tuple[InferenceRequest, AppliedConvention, AbsorbedCount | None, Bread],
    reporting: Reporting,
    reported: FloatArray | None,
    error: str | None,
    *,
    rows: FloatArray,
    codes: tuple[tuple[str, IntegerArray], ...],
) -> tuple[PsdAdjustment, FrameRecord]:
    """The frame, the sequence, the private constant where it belongs, the operation."""
    applied = common[1]
    sequence = applied.sequence
    scales = _frame_scales(reporting)
    constant: FloatArray | None = None
    record: ConstantRow | None = None
    diagnostics: tuple[str, ...] = ()
    if (
        sequence == "legacy"
        and reporting.intercept is None
        and reporting.constant is not None
    ):
        constant, record, diagnostics = _constant_row(reporting, common, rows, codes)
    adjustment = _adjusted(
        reported, error, reporting, scales=scales, sequence=sequence, constant=constant
    )
    frame = FrameRecord(
        sequence=sequence,
        convention=applied.convention.label,
        scales=scales.scales,
        deviations=scales.deviations,
        floored=scales.floored,
        count=scales.count,
        fingerprint=reporting.fingerprint,
        constant=record is not None,
        constant_total=None if record is None else record.total,
        constant_diagnostics=diagnostics,
    )
    return adjustment, frame


def _computed_inference(
    common: tuple[InferenceRequest, AppliedConvention, AbsorbedCount | None, Bread],
    tail: tuple[int, int, int],
    gamma: FloatArray,
    rows: FloatArray,
    weights: FloatArray,
    *,
    frequency: bool,
    codes: tuple[tuple[str, IntegerArray], ...],
    reporting: Reporting,
) -> RetainedInference:
    """Compute the uncached covariance and map it once to the reported scale.

    One dimension without an explicit form takes the one-way composition; the multiway
    path (module docstring) takes the port's raw multiway estimate and the convention's
    factors once, then the separate ``psd`` step, in the frame and the sequence of the
    convention in force, on the reported matrix of a signed request whose resolved
    treatment is ``"cgm"``.
    """
    request, applied = common[0], common[1]
    order, intercept = reporting.order, reporting.intercept
    covariance: ConventionCovariance | None = None
    multiway: MultiwayCovariance | None = None
    held: list[warnings.WarningMessage] = []
    if request.multiway_path:
        estimate, held = _raw_multiway_estimate(rows, codes, request.form)
        multiway = multiway_covariance(
            estimate, applied, form=request.form, dimensions=[n for n, _ in codes]
        )
        theta_covariance = multiway.covariance
    else:
        estimate = primitive_estimate(
            rows,
            weights,
            frequency=frequency,
            cluster=codes[0] if codes else None,
        )
        covariance = scaled_covariance(estimate, applied)
        theta_covariance = covariance.covariance
    errors = direct_errors(theta_covariance, order)
    reported: FloatArray | None = None
    error: str | None = None
    try:
        reported = order.covariance(theta_covariance)
    except FloatingPointError as exc:
        error = str(exc)
    psd = _resolved_psd(request, applied)
    adjustment: PsdAdjustment | None = None
    frame: FrameRecord | None = None
    if psd == "cgm":
        try:
            adjustment, frame = _frame_step(
                common, reporting, reported, error, rows=rows, codes=codes
            )
        except FloatingPointError:
            _route_port_warnings(held, None)  # nothing was adjusted: raw diagnostics
            raise
        reported = adjustment.adjusted
        errors = reported_errors(reported)
    _route_port_warnings(held, adjustment)
    return RetainedInference(
        *common,
        covariance,
        theta_covariance,
        errors,
        reported,
        error,
        gamma,
        False,
        *tail,
        multiway=multiway,
        psd=psd,
        adjustment=adjustment,
        intercept=intercept if psd is not None else None,
        frame=frame,
    )


def retained_nofe_inference(
    retained: RetainedFit, request: InferenceRequest
) -> RetainedInference:
    """Answer ``request`` from the retained state of a certified no-FE fit."""
    retained, request = _checked_nofe(retained, request)
    assert retained.theta is not None
    assert retained.certificate is not None
    evaluation = retained.certificate.evaluation
    assert evaluation is not None
    inherited: ConventionValues | ResolvedConvention | None = request.inherited
    if inherited is None and retained.default is not None:
        # the retained initial request's convention, by its recorded name: a preset
        inherited = resolve_convention(retained.default.convention)
    applied, codes, count, frequency = _nofe_counts(
        retained.sample, evaluation.units, evaluation.weights, request, inherited
    )
    active = evaluation.units.identifiers
    common = (request, applied, None, request.bread)
    tail = (count, int(active.size), retained.sample.count)
    cached = _nofe_cache_matches(request, applied, retained)
    if cached is not None:
        return RetainedInference(
            *common,
            None,
            cached.theta,
            cached.standard_errors,
            cached.reported,
            None,
            retained.theta,
            True,
            *tail,
        )
    if not retained.keep_influence or retained.influence is None:
        raise SpecificationError(
            "specification.influence_discarded", {"request": request.label}
        )
    rows = _nofe_influence_rows(retained, request.bread)
    return _computed_inference(
        common,
        tail,
        retained.theta,
        rows,
        evaluation.weights,
        frequency=frequency,
        codes=codes,
        reporting=_nofe_reporting(retained.sample, active),
    )


def candidate_nofe_inference(
    sample: OrderSample,
    influence: Influence,
    request: InferenceRequest,
    *,
    notes: bool = True,
) -> RetainedInference:
    """Answer a fit-time request at the influence record of one candidate.

    Parameters
    ----------
    sample : OrderSample
        The sample of the fit, whose rows the request's dimensions are bound to.
    influence : Influence
        The influence record of a certificate's fresh evaluation at the candidate
        (the observed bread), joint or linear.
    request : InferenceRequest
        A robust or cluster request with the observed bread, its dimensions bound to
        the sample's final rows; its convention is its explicit ``ssc``, else the
        default ``"stata"``: nothing is inherited at the fit.
    notes : bool, default True
        Whether the ``psd`` notes are issued; ``False`` for the reported scale of a
        solve, which assesses many candidates (module docstring).

    Returns
    -------
    RetainedInference
        The convention, the counts, the covariance, its reported matrix and standard
        errors, and the multiway and ``psd`` records, by the composition of
        :func:`retained_nofe_inference`; no cache answers it.

    Raises
    ------
    TypeError, ValueError
        Another type; the expected bread; an influence record of other rows or
        weights than the sample's; the refusals of the composition.
    SpecificationError
        The refusals of the convention (``specification.no_residual_dof``,
        ``specification.invalid_value``).
    FloatingPointError
        A value that is not representable, the reported matrix of an adjusted request
        included.
    """
    if type(influence) is not Influence:
        raise TypeError("influence must be an Influence record")
    if type(request) is not InferenceRequest:
        raise TypeError("request must be an InferenceRequest")
    if request.bread != "observed" or influence.frequency != frequency_of(sample):
        raise ValueError("the influence record is not the observed one of the sample")
    applied, codes, count, frequency = _nofe_counts(
        sample, influence.units, influence.weights, request, request.inherited
    )
    active = influence.units.identifiers
    return _computed_inference(
        (request, applied, None, request.bread),
        (count, int(active.size), sample.count),
        influence.gamma,
        influence_values(influence.contributions, influence.influence_map),
        influence.weights,
        frequency=frequency,
        codes=codes,
        reporting=_nofe_reporting(sample, active, notes=notes),
    )


def nofe_convention(
    sample: OrderSample,
    units: SamplingUnits,
    weights: FloatArray,
    request: InferenceRequest,
) -> AppliedConvention:
    """The convention of a fit-time request applied to the counts of its rows.

    The active codes, counts and factors that :func:`candidate_nofe_inference` forms,
    at the sample's units and the raw weights of its active rows before any candidate
    exists, so that their refusals cost no solve; nothing is inherited
    at the fit.
    """
    if type(request) is not InferenceRequest:
        raise TypeError("request must be an InferenceRequest")
    return _nofe_counts(sample, units, weights, request, request.inherited)[0]


def _constant_builder(envelope: RetainedAbsorbedEnvelope) -> ConstantBuilder | None:
    """The builder of the private constant row of an absorbed fit, or ``None``.

    Reads the retained geometry under the requested bread (the certificate's observed
    ``H``, or the expected information assembled on the same basis) and the staged
    weights (the expected curvature of the retained rows: true weights times the IRLS
    working weight); ``None`` where the envelope retains no geometry.
    """
    certificate = envelope.certificate
    inputs = envelope.expected_inputs()
    if certificate is None or certificate.geometry is None or inputs is None:
        return None
    geometry = certificate.geometry

    def build(bread: Bread) -> ConstantRow:
        hessian = geometry.hessian if bread == "observed" else inputs.expected_hessian()
        return constant_influence(
            inputs.design,
            inputs.basis.basis,
            inputs.rows.b,
            hessian,
            inputs.expected_curvature(),
        )

    return build


def retained_inference(
    envelope: RetainedAbsorbedEnvelope | RetainedFit | RetainedCancelledEnvelope,
    request: InferenceRequest,
) -> RetainedInference:
    """Answer ``request`` from the retained state of ``envelope`` (module docstring).

    Parameters
    ----------
    envelope : RetainedAbsorbedEnvelope or RetainedFit
        The retained state of an absorbed or no-FE fit.
    request : InferenceRequest
        The request, its cluster dimensions already bound to the sample's final rows.

    Returns
    -------
    RetainedInference
        The convention, counts, covariance and reported scale.

    Raises
    ------
    TypeError, ValueError
        Another type; no selected point, no slopes, a dimension that is not one code
        per final row, a missing label or fewer than two clusters among the active
        rows; an inconsistent count.
    SpecificationError
        ``specification.invalid_value``, ``specification.no_residual_dof`` and
        ``specification.conflicting`` from the convention;
        ``specification.influence_discarded`` for a request that needs the discarded
        influence.
    FloatingPointError
        A value that is not representable, the reported matrix of an adjusted request
        included.

    Warns
    -----
    CovarianceWarning
        The port's diagnostics of a raw estimate; ``covariance.psd_adjusted`` when
        the ``psd`` adjustment is active; ``covariance.indefinite`` when the final
        adjusted matrix is not positive semidefinite.
    """
    if type(request) is InferenceRequest and request.adjustment == "wz":
        raise SpecificationError(
            "specification.unsupported_inference",
            {
                "method": "WZ",
                "case": "normal FE result with declared term provenance required",
            },
        )
    if type(envelope) is RetainedCancelledEnvelope:
        if type(request) is not InferenceRequest:
            raise TypeError("request must be an InferenceRequest")
        return retained_cancelled_inference(envelope, request, RetainedInference)
    if type(envelope) is RetainedFit:
        return retained_nofe_inference(envelope, request)
    envelope, request = _checked(envelope, request)
    applied, absorbed, units, codes = _counts(envelope, request)
    point = envelope.selected_point
    assert point is not None
    active = units.identifiers
    common = (request, applied, absorbed, request.bread)
    tail = (
        _statistical_count(envelope, active),
        int(active.size),
        envelope.sample.count,
    )
    cached = _cache_matches(request, applied, envelope)
    if cached is not None:
        det = None if cached.details is None else typed_details(cached.details)
        if det is None:
            return RetainedInference(
                *common,
                None,
                cached.covariance.covariance,
                cached.standard_errors,
                cached.reported_covariance,
                cached.covariance_error,
                point.theta,
                True,
                *tail,
            )
        if det.adjustment is not None:
            if det.adjustment.active:
                _warn("covariance.psd_adjusted", {"count": det.adjustment.count})
            if det.adjustment.final_indefinite:
                _warn(
                    "covariance.indefinite",
                    {
                        "method": "adjusted signed multiway",
                        "minimum": det.adjustment.final_minimum,
                    },
                )
        mw = (
            None
            if det.multiway is None
            else (
                det.multiway
                if det.multiway.applied == applied
                else MultiwayCovariance(
                    covariance=det.multiway.covariance,
                    estimate=det.multiway.estimate,
                    applied=applied,
                    form=det.multiway.form,
                    dimensions=det.multiway.dimensions,
                    components=det.multiway.components,
                )
            )
        )
        frame_rec = (
            None
            if det.frame is None
            else FrameRecord(
                sequence=det.frame.sequence,
                convention=applied.convention.label,
                scales=det.frame.scales,
                deviations=det.frame.deviations,
                floored=det.frame.floored,
                count=det.frame.count,
                fingerprint=det.frame.fingerprint,
                constant=det.frame.constant,
                constant_total=det.frame.constant_total,
                constant_diagnostics=det.frame.constant_diagnostics,
            )
        )
        return RetainedInference(
            *common,
            None,
            cached.covariance.covariance,
            cached.standard_errors,
            cached.reported_covariance,
            cached.covariance_error,
            point.theta,
            True,
            *tail,
            multiway=mw,
            psd=det.psd,
            adjustment=det.adjustment,
            intercept=None,
            frame=frame_rec,
        )
    envelope.require_influence(request.label)
    order = envelope.order
    if order is None:
        raise ValueError("no order scale was retained for this candidate")
    rows = _influence_rows(envelope, request.bread)
    weights = envelope.sample.weight_values
    design = envelope.sample.design
    if design is None:
        raise ValueError("the retained sample holds no scaled design")
    # An absorbed fit reports no intercept: a constant column is absorbed by the effects
    # (powermean.data.collinearity), so the block rule has none; the legacy sequence
    # takes the internal reporting constant instead.
    reporting = Reporting(
        order,
        None,
        owned_array(design.unscaled()[active]),
        envelope.sample.fingerprint,
        active,
        _constant_builder(envelope),
    )
    return _computed_inference(
        common,
        tail,
        point.theta,
        rows[active],
        weights[active],
        frequency=envelope.sample.frequency,
        codes=codes,
        reporting=reporting,
    )


def initial_inference_from_envelope(
    envelope: RetainedAbsorbedEnvelope,
) -> RetainedInference:
    """Build ``RetainedInference`` directly from ``envelope.initial_scale``."""
    init = envelope.initial_scale
    point = envelope.selected_point
    if init is None or point is None:
        raise ValueError("no initial scale was retained for this candidate")
    req = init.request.request
    clusters = tuple(ClusterCodes(name, codes) for name, codes in req.clusters)
    inherited = (
        req.convention
        if type(req.convention) is ConventionValues and req.ssc_inherited
        else None
    )
    inf_req = InferenceRequest(
        "robust" if req.kind == "robust" else "cluster",
        clusters,
        req.ssc,
        req.bread,
        inherited,
        req.multiway,
        req.psd,
        req.adjustment,
    )
    det = None if init.details is None else typed_details(init.details)
    if det is None:
        return retained_inference(envelope, inf_req)
    if det.adjustment is not None:
        if det.adjustment.active:
            _warn("covariance.psd_adjusted", {"count": det.adjustment.count})
        if det.adjustment.final_indefinite:
            _warn(
                "covariance.indefinite",
                {
                    "method": "adjusted signed multiway",
                    "minimum": det.adjustment.final_minimum,
                },
            )
    frame_rec = (
        None
        if det.frame is None
        else FrameRecord(
            sequence=det.frame.sequence,
            convention=det.frame.convention,
            scales=det.frame.scales,
            deviations=det.frame.deviations,
            floored=det.frame.floored,
            count=det.frame.count,
            fingerprint=det.frame.fingerprint,
            constant=det.frame.constant,
            constant_total=det.frame.constant_total,
            constant_diagnostics=det.frame.constant_diagnostics,
        )
    )
    return RetainedInference(
        inf_req,
        det.applied,
        det.absorbed,
        det.bread,
        None if det.wz is None else retained_covariance_record(det.wz, det.applied),
        init.covariance.covariance,
        init.standard_errors,
        init.reported_covariance,
        init.covariance_error,
        point.theta,
        True,
        det.count,
        int(det.units.identifiers.size),
        envelope.sample.count,
        multiway=det.multiway,
        psd=det.psd,
        adjustment=det.adjustment,
        intercept=None,
        frame=frame_rec,
        wz=det.wz,
    )


__all__ = [
    "FrameRecord",
    "InferenceRequest",
    "Reporting",
    "RetainedInference",
    "candidate_nofe_inference",
    "initial_inference_from_envelope",
    "nofe_convention",
    "retained_inference",
    "retained_nofe_inference",
]
