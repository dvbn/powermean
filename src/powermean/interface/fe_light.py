"""Fixed-effect log-OLS by a dedicated light pipeline (owner's ruling 87).

The regression of ``log y`` on the regressors with the effects absorbed, chosen at
dispatch (:func:`powermean.interface.fe_dispatch.fit_fe_prepared`), doing what
``pyfixest.feols`` does and no more:

1. The rows: steps 1 to 4 of the prepared sample (subset, missing values, the outcome
   domain, zero weights) by the package's own selection
   (:func:`powermean.data.retained.select_order`), so the counts and refusals are the
   package's; then the singleton levels of every block removed until none remains
   (step 5; its fixed point does not depend on the order of the visits).  Step 6
   (levels without a positive outcome) removes nothing at ``omega = 0``.
2. The blocks are numbered from the prepared raw labels (the package's encoder of
   integral keys, ``pandas.factorize`` otherwise; only the partition is read).
3. ``[log y, X]``, with each regressor scaled to raw unit norm through reversible
   power-of-two coordinates, is demeaned once by ``within`` (``within-py``; LSMR,
   relative normal-equation residual at most :data:`TOLERANCE`, the tolerance of the
   start of the exact route).  Every column must report convergence, and every column
   that is read (``log y`` and the regressors not absorbed within the levels of a block)
   must pass a fresh check: the largest group mean of the demeaned column, over every
   block and level, at most :data:`FRESH_TOL` times the demeaned column's root mean
   square.  Otherwise the fit is made by the exact route of
   :mod:`powermean.interface.fe_linear`, unchanged.
4. The collinearity decision of the package
   (:mod:`powermean.data.collinearity`), read on the demeaned columns, without a
   second absorption, where it is clear-cut.  Exact zero and constant columns (the
   formula's intercept among them) and columns exactly constant within the levels of
   a block are absorbed by the effects, as the package decides them; every other
   column must be clearly active: its within norm at least :data:`ACTIVE_RATIO` of
   its uncentered norm and its squared ratio at least :data:`PIVOT_MARGIN` times
   ruling 92's relative cutoff.  The shared relative pivot rule (formula order,
   omitted columns entering no later pivot) then keeps a column whose pivot is at
   least :data:`PIVOT_MARGIN` times its tolerance and :data:`PIVOT_RELATIVE` of the
   column's within squared norm, and omits one whose pivot is at most the tolerance
   over :data:`PIVOT_MARGIN`, its within squared norm at most
   :data:`OMIT_SQUARED_NORM`.  A column near a threshold sends the fit to the exact
   route, which applies the same relative policy.  The omitted regressors are noted
   (``coverage.collinear_omitted``) as the exact route notes them.
5. Least squares on the demeaned columns, the residuals, the slope influence rows of
   the within regression (:func:`~powermean.inference.absorbed_linear.
   within_influence`).  Coefficients and influence rows then undo only the raw-norm
   normalization, retaining the reversible power-of-two working frame for covariance
   diagnostics and HC.  The order scale maps reported coefficients, errors and
   covariance once; D4's reporting columns stay in their original units.
6. The covariance by the package's own functions on these arrays: robust and clustered
   requests by :func:`powermean.results.retained_inference._computed_inference` with
   the counts of ruling 80 (or the exact rank with ``absorbed_count="exact"``), the
   named methods by :func:`~powermean.inference.absorbed_linear.linear_preflight` and
   :func:`~powermean.inference.absorbed_linear.linear_composition` (the exact
   leverage engine, run only when a leverage method is requested).  Their records,
   factors, references and notes are those of the exact route.

A saturated fit (rows at most the slopes plus the levels), a fit without slopes, a
request this module does not form (WZ, an implicit pair cluster, a start, explicit
sample or linear absorption controls other than the defaults, a compiled basis, an
offset; a signed multiway request whose
PSD step reads the legacy frame with the private reporting constant, whose eigenvalue
decisions are decided by rounding on some designs), and any refusal go to the exact
route, which answers as before.  The result (:class:`LightLogBackend`) answers
coefficients, standard errors, covariances, in-sample predictions, tables, the summary,
the provenance and ``with_vcov`` from these records. The exact route's result of the
same call, formed once on first use, serves ``fixef``, ``influence``, the sample
grouping, a post-fit request without retained influence, and prediction near a
float64 representability boundary.
"""

from __future__ import annotations

import copy
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, replace
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Unpack, cast

import numpy as np
import pandas as pd
import within

from powermean._prepare.owned import _owned_index
from powermean.data import retained as _retained_rows
from powermean.data.accounting import counts_of
from powermean.data.collinearity import PIVOT_TOL, _within_levels, relative_pivots
from powermean.data.collinearity_numeric import reversible
from powermean.data.collinearity_records import ABSORBED, COLLINEAR
from powermean.data.fingerprint import sample_fingerprint
from powermean.data.inference_conventions import ConventionValues
from powermean.data.inference_requests import ClusterRequest, HCMethod, HCRequest
from powermean.data.inputs import RegressionData
from powermean.data.raw import Column
from powermean.data.request_values import OMITTED, Omitted
from powermean.data.specs import FixedEffectFormulation
from powermean.effects.targets import ReportingContext, ReportingPoint
from powermean.errors import ConvergenceError
from powermean.inference.absorbed_constant import ConstantRow
from powermean.inference.absorbed_linear import (
    Coordinates,
    exact_coordinates,
    exact_count,
    exact_requested,
    fixest_count,
    fixest_rank,
    linear_composition,
    linear_preflight,
    rank_formed,
    within_constant,
    within_influence,
)
from powermean.inference.absorbed_preflight import _psd_request
from powermean.inference.absorbed_rank import AbsorbedCount
from powermean.inference.bread import BREADS
from powermean.inference.contributions import statistical_count
from powermean.inference.convention import ClusterCodes, _active_codes
from powermean.inference.ssc import (
    AppliedConvention,
    RequestKind,
    ResolvedConvention,
    apply_convention,
    convention_counts,
    direct_errors,
    owned_factor,
    resolve_convention,
)
from powermean.interface.cancelled_selection import default_diagnostic
from powermean.interface.fe_joint_result import (
    ABSORBED_COUNT_NOTES,
    AbsorbedResultBackend,
)
from powermean.interface.fe_result_backends import (
    _fe_summary_lines,
    _joint_block_lines,
    _parse_vcov_spec_for_absorbed,
    _target_reporting_context,
    _tidy_frame,
    _vcov_attrs_from_retained,
    _vcov_frame_from_retained,
)
from powermean.interface.retained_binding import _bind_owned_metadata_parts
from powermean.interface.retained_metadata import BoundDependence
from powermean.interface.sample_records import SampleRequest
from powermean.model.order_scale import OrderScale
from powermean.results import citations, naming
from powermean.results.fe_helpers import (
    normal_quantile,
)
from powermean.results.fe_sample import FESampleView
from powermean.results.prediction import Prediction, PredictionOptions, prediction_kind
from powermean.results.provenance import (
    IMPLEMENTATION,
    Provenance,
    _frozen,
    carried_files,
    versions,
)
from powermean.results.result import Result
from powermean.results.retained_hc import _reported_hc, _request_errors, _resolved
from powermean.results.retained_hc_records import (
    HCCapability,
    HCInferenceRequest,
    RetainedHC,
)
from powermean.results.retained_inference import (
    InferenceRequest,
    Reporting,
    RetainedInference,
    _computed_inference,
    _resolved_psd,
)
from powermean.results.summary import WIDTH, _table
from powermean.solve.absorbed_certificate_scale import CovarianceRequest
from powermean.solve.m1_options import AbsorbedOptions

if TYPE_CHECKING:
    from powermean._prepare.compose import PreparedFrame

TOLERANCE = 1e-12
"""``within``'s relative normal-equation residual (that of the exact route's start)."""

FRESH_TOL = 2e-11
"""The fresh check: the largest group mean over the column's root mean square."""

_NORMAL_LOG_MIN = float(np.log(np.finfo(np.float64).tiny))
_NORMAL_LOG_MAX = float(np.log(np.finfo(np.float64).max))
_PREDICTION_MARGIN = 1e-10
"""Use the exact route within the agreed log-index tolerance of the normal range."""

ACTIVE_RATIO = 1e-5
"""A clearly active column: within norm at least this share of its uncentered norm.

The additional relative-policy margin below must also pass; borderline columns
decline to the exact route rather than being decided by this margin."""

PIVOT_MARGIN = 1e3
"""A clearly kept column: pivot at least this multiple of the pivot rule's tolerance."""

PIVOT_RELATIVE = 1e-8
"""A clearly kept column: pivot at least this share of its within squared norm."""

OMIT_SQUARED_NORM = 1e8
"""A clearly omitted column: pivot at most the rule's tolerance over the margin, and a
standardized within squared norm at most this.  The existing conservative guard is
retained; normalization bounds a true within squared norm by one."""

_LINEAR_REFERENCE = "reghdfe"
_GEOMETRY = "the full nuisance basis, the slopes and the observed rows"


class _Exact(Exception):  # noqa: N818 - a decision, not an error
    """The light pipeline declines: the exact route makes the fit."""


def _decline(reason: str) -> _Exact:
    return _Exact(reason)


# ------------------------------------------------------------------ eligibility


def _eligible(problem: Any, spec: Any) -> str | None:
    """``None`` when the light pipeline may be tried, else the reason it is not."""
    if spec.estimator.mean.omega != 0:
        return "nonzero omega"
    if spec.cancel is not None or spec.bias_correction is not None:
        return "cancel or bias correction"
    columns = problem.frame.columns
    if (
        columns.weights is not None
        or columns.offset is not None
        or columns.exposure is not None
        or problem.offset_design is not None
    ):
        return "weights, offset or exposure"
    if spec.start is not None:
        return "start"
    controls = spec.fe_solver_options
    if controls is None or controls.absorbed_options is None:
        return "controls"
    # The linear solve reads these three controls; within's fixed settings do not
    # implement them. The exact dispatch also resolves an explicit compiled basis
    # before solving. Compare resolved values: flat, nested and typed requests with
    # equivalent defaults have the same semantics and recorded provenance.
    for name in (
        "absorb_tol_final",
        "absorb_max_sweeps",
        "absorb_acceleration",
        "basis_backend",
    ):
        if getattr(controls.absorbed_options, name) != getattr(AbsorbedOptions, name):
            return f"non-default {name}"
    if controls.sample_request is not None:
        default = SampleRequest.default(
            0.0, None, problem.positive_only, default_diagnostic()
        )
        if controls.sample_request != default:
            return "explicit sample request"
    if problem.frame.effects is None or problem.retained_preparation is None:
        return "no effects"
    fe_vcov = spec.fe_vcov
    if isinstance(fe_vcov, ClusterRequest):
        if not isinstance(fe_vcov.adjustment, Omitted):
            return "cluster adjustment"
    elif isinstance(fe_vcov, HCRequest):
        pass
    elif fe_vcov == "robust":
        if not spec.explicit_vcov and len(problem.frame.effects.terms) == 3:
            return "implicit default of three blocks (pair cluster)"
    else:
        return "covariance request"
    return None


# ------------------------------------------------------------------ the rows


def _component(effects: Any, name: str) -> np.ndarray:
    """The raw labels of one component on every input row."""
    native = vars(effects).get("_native_columns")
    column: Column = native[name] if native is not None else effects.columns[name]
    stored = column._stored
    if isinstance(stored, np.ndarray):
        return stored
    return column.values


def _codes_of(values: np.ndarray) -> np.ndarray:
    """Dense codes of the labels ``values`` (only the partition is read).

    Python equality of the labels, as the package's encoder reads them (``1``,
    ``1.0`` and ``True`` are one level); a missing label declines.
    """
    found = pd.factorize(values, sort=False, use_na_sentinel=True)[0]
    if found.size and int(found.min()) < 0:
        raise _decline("missing label")
    return np.asarray(found, dtype=np.intp)


def _joint_codes(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Dense codes of the pairs of two dense code vectors."""
    span = int(right.max()) + 1 if right.size else 1
    key = left.astype(np.int64) * span + right
    return np.asarray(pd.factorize(key, sort=False)[0], dtype=np.intp)


def _block_codes(effects: Any, positions: np.ndarray) -> list[np.ndarray]:
    """Dense codes of every block on the rows ``positions``."""
    blocks: list[np.ndarray] = []
    for term in effects.terms:
        parts = [_codes_of(_component(effects, name)[positions]) for name in term]
        blocks.append(parts[0] if len(parts) == 1 else _joint_codes(*parts))
    return blocks


def _dense(codes: np.ndarray) -> np.ndarray:
    """``codes`` renumbered over the levels present, in increasing order."""
    present = np.bincount(codes) > 0
    if present.all():
        return codes
    return cast("np.ndarray", (np.cumsum(present) - 1)[codes])


def _without_singletons(
    blocks: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rows kept after the singleton levels are removed until none remains."""
    n = blocks[0].size
    keep = np.ones(n, dtype=bool)
    changed = True
    while changed:
        changed = False
        for codes in blocks:
            current = codes[keep]
            counts = np.bincount(current)
            single = counts[current] == 1
            if single.any():
                idx = np.flatnonzero(keep)
                keep[idx[single]] = False
                changed = True
    rows = np.flatnonzero(keep)
    if rows.size == n:
        return rows, blocks
    return rows, [_dense(codes[rows]) for codes in blocks]


# ------------------------------------------------------------------ the counts


def _counts(
    blocks: tuple[tuple[str, np.ndarray], ...],
    clusters: list[np.ndarray],
    n_slopes: int,
    count: int,
    resolved: ResolvedConvention,
    kind: RequestKind,
    memo: dict[bytes, Coordinates],
) -> tuple[AppliedConvention, AbsorbedCount]:
    """The applied convention and absorbed count, as the retained service counts."""
    every = np.arange(count, dtype=np.intp)
    groups = [int(np.unique(c).size) for c in clusters]
    if exact_requested(resolved):
        coordinates = exact_coordinates(
            [values for _, values in blocks], every, count, memo
        )
        dimension = n_slopes + coordinates.rank
        absorbed = exact_count(blocks, clusters, every, coordinates)
    else:
        dimension = n_slopes + fixest_rank(blocks, every)
        absorbed = fixest_count(blocks, clusters, every)
    counts = convention_counts(count, n_slopes, dimension, absorbed, groups)
    return apply_convention(resolved, counts, kind=kind), absorbed


class _Units:
    """Every final row is a sampling unit of weight one (no weights on this route)."""

    def __init__(self, count: int) -> None:
        self.identifiers = np.arange(count, dtype=np.intp)
        self.count = count
        self.nobs = count


class LightCore:
    """Working inference arrays and reported beta, shared by fit/post-fit requests.

    ``matrix``, ``theta`` and ``rows`` share the power-of-two working coordinates.
    ``columns`` and ``beta`` are in original units, for D4 and direct reporting.
    """

    def __init__(
        self,
        *,
        names: tuple[str, ...],
        matrix: np.ndarray,
        columns: np.ndarray,
        scales: np.ndarray,
        beta: np.ndarray,
        rows: np.ndarray,
        residual: np.ndarray,
        log_response: np.ndarray,
        blocks: tuple[tuple[str, np.ndarray], ...],
        fingerprint: Callable[[], str],
    ) -> None:
        self.names = names
        self.matrix = matrix  # X / scales, active columns on final rows
        self.columns = columns  # original-unit columns for the D4 reporting frame
        self.order = OrderScale(0.0, scales)
        self.theta = beta
        self.beta = self.order.coefficients(beta)
        # These are discarded after the initial inference when the fit requests it.
        self.rows: np.ndarray | None = rows
        self.residual: np.ndarray | None = residual
        # The within residual accounts for both slopes and effects. Reuse the
        # private log response buffer for the fitted index.
        self.fitted_index = np.subtract(log_response, residual, out=log_response)
        self.fitted_index.flags.writeable = False
        self.blocks = blocks
        self.count = int(matrix.shape[0])
        self.weights = np.ones(self.count)
        self._fingerprint = fingerprint
        self._fingerprint_value: str | None = None
        self.memo: dict[bytes, Coordinates] = {}

    @property
    def fingerprint(self) -> str:
        """The sample fingerprint, formed on first read."""
        if self._fingerprint_value is None:
            self._fingerprint_value = self._fingerprint()
        return self._fingerprint_value

    def constant(self, bread: str) -> ConstantRow:  # noqa: ARG002 - one curvature
        """The private reporting constant's row (both breads are the same matrix)."""
        return within_constant(
            self.matrix,
            self.weights,
            cast("np.ndarray", self.residual),
            cast("np.ndarray", self.rows),
        )

    def inference(
        self,
        request: InferenceRequest,
        inherited: Any,
    ) -> RetainedInference:
        """A robust or clustered request (``retained_inference`` on these arrays)."""
        units = _Units(self.count)
        codes = tuple(
            (c.name, _active_codes(c, units))  # type: ignore[arg-type]
            for c in request.clusters
        )
        found = request.inherited if request.inherited is not None else inherited
        resolved = resolve_convention(request.ssc, found)
        applied, absorbed = _counts(
            self.blocks,
            [c for _, c in codes],
            len(self.names),
            self.count,
            resolved,
            request.kind,
            self.memo,
        )
        if _resolved_psd(request, applied) == "cgm" and applied.sequence == "legacy":
            # The legacy frame holds the private reporting constant: its eigenvalue
            # decisions are decided by rounding on some designs (a structurally zero
            # eigenvalue), so the exact route forms them as before.
            raise _decline("the legacy PSD frame of a signed multiway request")
        n = statistical_count(self.weights, frequency=False)
        common = (request, applied, absorbed, request.bread)
        tail = (n, self.count, self.count)
        reporting = Reporting(
            self.order,
            None,
            self.columns,
            self.fingerprint if request.signed else "",
            units.identifiers,
            self.constant,
            reference=_LINEAR_REFERENCE,
        )
        return _computed_inference(
            common,
            tail,
            self.theta,
            cast("np.ndarray", self.rows),
            self.weights,
            frequency=False,
            codes=codes,
            reporting=reporting,
        )

    def capability(self, keep_influence: bool) -> HCCapability:
        """The HC capability record of a linear envelope (``_envelope_capability``)."""
        return HCCapability(
            route="absorbed",
            formulation="absorbed joint (linear)",
            unit="rows",
            weight_kind="none",
            layout="absorbed",
            breads=BREADS,
            geometry=_GEOMETRY,
            available=keep_influence,
            reason=None
            if keep_influence
            else ("the influence was discarded (keep_influence=False)"),
        )

    def hc(
        self, request: HCInferenceRequest, resolved: Any, keep_influence: bool
    ) -> RetainedHC:
        """A named method (``retained_hc._linear_hc`` on these arrays)."""
        preflight, coordinates = linear_preflight(
            len(self.names),
            self.blocks,
            self.weights,
            request.method,
            resolved,
            frequency=False,
            memo=self.memo,
        )
        found = linear_composition(
            self.matrix,
            self.weights,
            cast("np.ndarray", self.rows),
            preflight,
            coordinates,
            bread=request.bread,
        )
        errors, reported, error = _reported_hc(found, self.order)
        return RetainedHC(
            request,
            self.capability(keep_influence),
            found,
            found.covariance,
            errors,
            reported,
            error,
            self.theta,
            self.names,
            found.applied.counts.count,
            self.count,
            self.count,
        )


def _fit_hc(
    core: LightCore, request: CovarianceRequest, keep_influence: bool
) -> RetainedHC:
    """The fit-time named method, as ``fe_linear._hc_scale`` with its initial record."""
    inherited = (
        resolve_convention(request.convention) if request.ssc_inherited else None
    )
    resolved = resolve_convention(request.ssc, inherited)
    # The fit calls here only for a validated named covariance request.
    method = cast("HCMethod", request.method)
    preflight, coordinates = linear_preflight(
        len(core.names),
        core.blocks,
        core.weights,
        method,
        resolved,
        frequency=False,
        memo=core.memo,
    )
    found = linear_composition(
        core.matrix,
        core.weights,
        cast("np.ndarray", core.rows),
        preflight,
        coordinates,
        bread=request.bread,
    )
    errors = direct_errors(found.gate_covariance, core.order)
    reported: np.ndarray | None = None
    error: str | None = None
    if found.undefined:
        reported = found.covariance
    else:
        try:
            reported = core.order.covariance(found.covariance)
        except FloatingPointError as failure:
            error = str(failure)
    hc_inherited = (
        request.convention
        if type(request.convention) is ConventionValues and request.ssc_inherited
        else None
    )
    return RetainedHC(
        HCInferenceRequest(method, request.ssc, request.bread, hc_inherited),
        core.capability(keep_influence),
        found,
        found.covariance,
        _request_errors(found, errors),
        reported,
        error,
        core.theta,
        core.names,
        preflight.count,
        core.count,
        core.count,
    )


def _fit_inference(core: LightCore, request: CovarianceRequest) -> RetainedInference:
    """The fit-time robust or clustered request (``fe_linear._robust_scale``)."""
    inherited = (
        request.convention
        if type(request.convention) is ConventionValues and request.ssc_inherited
        else None
    )
    found = InferenceRequest(
        "robust" if request.kind == "robust" else "cluster",
        tuple(ClusterCodes(name, codes) for name, codes in request.clusters),
        request.ssc,
        request.bread,
        inherited,
        request.multiway,
        request.psd,
        request.adjustment,
    )
    return core.inference(found, None)


def _held(compute: Callable[[], Any]) -> tuple[Any, list[warnings.WarningMessage]]:
    """``compute()`` and the warnings it raised, held rather than issued."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = compute()
    return value, list(caught)


def _reissue(caught: list[warnings.WarningMessage]) -> None:
    """Issue held warnings at the place where each was raised, with its registry."""
    if not caught:
        return
    modules = {
        getattr(mod, "__file__", None): mod for mod in list(sys.modules.values())
    }
    for item in caught:
        mod = modules.get(item.filename)
        registry = (
            None if mod is None else vars(mod).setdefault("__warningregistry__", {})
        )
        warnings.warn_explicit(
            item.message,
            item.category,
            item.filename,
            item.lineno,
            registry=registry,
        )


def _replayed(compute: Callable[[], Any]) -> Any:
    """``compute()``, its warnings issued only once it has returned.

    A failure discards them: the exact route then issues its own.
    """
    value, caught = _held(compute)
    _reissue(caught)
    return value


# ------------------------------------------------------------------ the fit


def _request_of(
    spec: Any, positions: np.ndarray, bind: Callable[[str], np.ndarray]
) -> CovarianceRequest:
    """The covariance request (``fe_dispatch._joint_covariance_request``)."""
    fe_vcov, ssc, bread = spec.fe_vcov, spec.ssc, spec.bread
    conv: str | ConventionValues = "stata" if isinstance(ssc, Omitted) else ssc
    if isinstance(fe_vcov, HCRequest):
        return CovarianceRequest(
            kind="hc", convention=conv, method=fe_vcov.method, bread=bread, ssc=ssc
        )
    if isinstance(fe_vcov, ClusterRequest):
        clusters = tuple((name, bind(name)) for name in fe_vcov.dimensions)
        if (
            len(clusters) == 1
            and bread == "observed"
            and isinstance(fe_vcov.multiway, Omitted)
            and isinstance(fe_vcov.psd, Omitted)
            and isinstance(ssc, Omitted)
            and isinstance(fe_vcov.adjustment, Omitted)
        ):
            return CovarianceRequest(
                "cluster", clusters[0][0], clusters[0][1], positions
            )
        return CovarianceRequest(
            kind="cluster",
            name=clusters[0][0] if len(clusters) == 1 else None,
            codes=clusters[0][1] if len(clusters) == 1 else None,
            positions=positions,
            convention=conv,
            clusters=clusters,
            bread=bread,
            multiway=fe_vcov.multiway,
            psd=fe_vcov.psd,
            adjustment=fe_vcov.adjustment,
            ssc=ssc,
        )
    if bread == "observed" and isinstance(ssc, Omitted):
        return CovarianceRequest("robust")
    return CovarianceRequest(kind="robust", convention=conv, bread=bread, ssc=ssc)


def _fresh(values: np.ndarray, blocks: list[np.ndarray]) -> bool:
    """Every group mean of every column at most ``FRESH_TOL`` of its RMS."""
    n: int = values.shape[0]
    sizes = [np.bincount(codes) for codes in blocks]
    for c in range(values.shape[1]):
        column = values[:, c]
        bound = FRESH_TOL * float(np.sqrt(np.mean(column * column)))
        for codes, size in zip(blocks, sizes, strict=True):
            means = np.bincount(codes, weights=column, minlength=size.size) / size
            if float(np.max(np.abs(means))) > bound:
                return False
    return n > 0


def _light_core(problem: Any) -> tuple[LightCore, dict[str, Any]]:
    """The rows, the demeaned columns, the decision and the slopes, or ``_Exact``."""
    frame = problem.frame
    base = frame.base
    selected = _retained_rows.select_order(
        base, 0.0, positive_only=problem.positive_only
    )
    positions0 = np.asarray(selected.positions, dtype=np.intp)
    effects = frame.effects
    blocks0 = _block_codes(effects, positions0)
    kept_rows, codes = _without_singletons(blocks0)
    if not kept_rows.size:
        raise _decline("no row after the singletons")
    positions = positions0[kept_rows]
    count = int(positions.size)
    inputs = base.inputs
    all_names = tuple(inputs.column_names)
    design_all = inputs.design[positions]
    y_raw = inputs.y[positions]
    reasons: list[str | None] = [None] * design_all.shape[1]
    general: list[int] = []
    for j in range(design_all.shape[1]):
        column = design_all[:, j]
        if bool(np.all(column == column[0])):
            # Exact zero and constant columns, decided by their entries: absorbed.
            reasons[j] = ABSORBED
        else:
            general.append(j)
    if not general:
        raise _decline("no slope")
    levels = sum(int(c.max()) + 1 for c in codes)
    if count <= len(general) + levels:
        raise _decline("possibly saturated")
    x_general = np.asarray(design_all[:, general], dtype=np.float64)
    scaled, scales = reversible(x_general, "light columns")
    raw_norms = np.sqrt(np.einsum("ij,ij->j", scaled, scaled))
    with np.errstate(all="raise"):
        t = np.log(y_raw)
    stacked = np.empty((count, 1 + len(general)), dtype=np.float64, order="F")
    stacked[:, 0] = t
    stacked[:, 1:] = scaled / raw_norms
    matrix = np.empty((count, len(codes)), dtype=np.uint32, order="F")
    for b, c in enumerate(codes):
        matrix[:, b] = c
    try:
        found = within.solve_batch(
            matrix, stacked, None, within.LsmrOptions(tol=TOLERANCE)
        )
    except (RuntimeError, ValueError, TypeError) as error:
        raise _decline(f"within: {error}") from None
    if not all(found.converged):
        raise _decline("within did not converge")
    demeaned = np.asarray(found.demeaned, dtype=np.float64)
    if demeaned.ndim == 1:
        demeaned = demeaned[:, None]
    if not np.isfinite(demeaned).all():
        raise _decline("within: a non-finite value")
    wt, w_general = demeaned[:, 0], demeaned[:, 1:]
    within_norms = np.sqrt(np.einsum("ij,ij->j", w_general, w_general))
    candidates: list[int] = []
    for k, j in enumerate(general):
        if any(_within_levels(x_general[:, k], c) for c in codes):
            # Exactly constant within the levels of a block: absorbed (the span).
            reasons[j] = ABSORBED
        elif within_norms[k] >= max(ACTIVE_RATIO, np.sqrt(PIVOT_MARGIN * PIVOT_TOL)):
            candidates.append(k)
        else:
            raise _decline("collinearity: a column near the screen")
    # The fresh check on the columns that are read: log y and the candidates (a
    # column absorbed within the levels of a block is not read).
    if not _fresh(demeaned[:, [0, *(1 + k for k in candidates)]], codes):
        raise _decline("fresh check")
    pivots = relative_pivots(w_general[:, candidates])
    kept: list[int] = []
    for k, pivot in zip(candidates, pivots, strict=True):
        squared = float(within_norms[k]) ** 2
        if pivot >= PIVOT_MARGIN * PIVOT_TOL and pivot >= PIVOT_RELATIVE * squared:
            kept.append(k)
        elif pivot <= PIVOT_TOL / PIVOT_MARGIN and squared <= OMIT_SQUARED_NORM:
            reasons[general[k]] = COLLINEAR  # the pivot rule omits it
        else:
            raise _decline("collinearity: a pivot near the rule")
    if not kept:
        raise _decline("no slope kept")
    active = [general[k] for k in kept]
    x = np.ascontiguousarray(scaled[:, kept])
    # design_all is already fit-owned and retained by fingerprint.  D4 needs its
    # exact values, not contiguous storage; only gaps in the active order need a copy.
    columns = (
        design_all[:, active[0] : active[-1] + 1]
        if active[-1] - active[0] + 1 == len(active)
        else design_all[:, active]
    )
    columns.flags.writeable = False
    wx = np.ascontiguousarray(w_general[:, kept])
    beta = np.linalg.lstsq(wx, wt, rcond=None)[0]
    residual = wt - wx @ beta
    rows, _ = within_influence(wx, np.ones(count), residual)
    # Undo only unit-norm normalization here.  Covariance diagnostics and HC read
    # the same power-of-two frame as the exact route, not the reported units.
    # LightCore's OrderScale owns the remaining reporting conversion.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        beta = beta / raw_norms[kept]
        rows = rows / raw_norms[kept]
    names = tuple(all_names[j] for j in active)
    blocks = tuple(
        ("^".join(term), c) for term, c in zip(effects.terms, codes, strict=True)
    )

    def fingerprint() -> str:
        data = RegressionData(
            y_raw, design_all, row_labels=positions.tolist(), column_names=all_names
        )
        return sample_fingerprint(data)

    core = LightCore(
        names=names,
        matrix=x,
        columns=columns,
        scales=scales[kept],
        beta=beta,
        rows=rows,
        residual=residual,
        log_response=t,
        blocks=blocks,
        fingerprint=fingerprint,
    )
    facts = {
        "selected": selected,
        "positions": positions,
        "singleton": int(positions0.size - count),
        "all_names": all_names,
        "reasons": tuple(reasons),
    }
    return core, facts


def light_fit(
    problem: Any, spec: Any, exact: Callable[[], Result], *, stacklevel: int = 1
) -> Result | None:
    """The light fit of ``spec`` on ``problem``, or ``None`` for the exact route.

    ``exact`` makes the same call by the exact route; the result keeps it for what it
    answers from the exact route's records (module docstring).  ``stacklevel`` is the
    exact route's ``_joint`` stacklevel: the note of an omitted regressor names the
    caller's line as it does there.
    """
    if _eligible(problem, spec) is not None:
        return None
    try:
        core, facts = _light_core(problem)
        positions = facts["positions"]
        bound: list[BoundDependence] = []

        def dependence() -> BoundDependence:
            if not bound:
                target = SimpleNamespace(
                    positions=positions,
                    nobs=int(problem.frame.base.nobs),
                    ledger=SimpleNamespace(nobs=int(problem.frame.base.nobs)),
                    count=int(positions.size),
                )
                bound.append(_bind_owned_metadata_parts(problem.frame, target))
            return bound[0]

        def bind(name: str) -> np.ndarray:
            return np.asarray(dependence().cluster_codes(name).codes)

        request = _request_of(spec, positions, bind)
        if request.kind == "cluster":
            for _, codes in request.clusters:
                if int(np.unique(codes).size) < 2:
                    raise _decline("fewer than two clusters")
        _psd_request(request)
        inherited = (
            resolve_convention(request.convention) if request.ssc_inherited else None
        )
        resolve_convention(request.ssc, inherited)
        keep = bool(spec.keep_influence)

        def covariance() -> RetainedInference | RetainedHC:
            if request.kind == "hc":
                return _fit_hc(core, request, keep)
            return _fit_inference(core, request)

        initial, held = _held(covariance)
        if not keep:
            # keep_influence=False: the rows are discarded once the fit's covariance
            # is formed, as on the exact route; a post-fit request that needs them
            # is answered (refused) by the exact route.
            core.rows = None
            core.residual = None
    except Exception:
        return None
    # The notes in the exact route's order: the omitted regressors (before its
    # solve, from the same frame depth as its ``_joint``), then the covariance's.
    from powermean.interface.fe_dispatch import _note_omitted  # noqa: PLC0415 - a cycle

    _note_omitted(
        facts["all_names"],
        facts["reasons"],
        problem.frame._frame.design,
        stacklevel=stacklevel,
    )
    _reissue(held)
    return Result(
        LightLogBackend(problem, spec, core, facts, initial, keep, dependence, exact)
    )


# ------------------------------------------------------------------ the result


class _LazyRecord(Mapping[str, Any]):
    """A record whose ``fingerprint`` is formed on first read."""

    def __init__(self, fields: dict[str, Any], fingerprint: Callable[[], str]) -> None:
        self._fields = fields
        self._fingerprint = fingerprint

    def __getitem__(self, key: str) -> Any:
        if key == "fingerprint":
            return self._fingerprint()
        return self._fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __repr__(self) -> str:
        return repr(dict(self.items()))


class LightSampleView:
    """The sample view of a light fit; what it does not hold, the exact route's."""

    def __init__(self, backend: LightLogBackend) -> None:
        self._backend = backend

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._backend._exact().sample, name)

    @property
    def fingerprint(self) -> str:
        return self._backend._core.fingerprint

    @property
    def omega(self) -> float:
        return 0.0

    @property
    def positive_only(self) -> bool:
        return bool(self._backend._problem.positive_only)

    @property
    def conditioned(self) -> bool:
        return bool(self._backend._facts["selected"].conditioned)

    @property
    def column_names(self) -> tuple[str, ...]:
        return self._backend._core.names

    @property
    def active_names(self) -> tuple[str, ...]:
        return self.column_names

    @property
    def nobs(self) -> int:
        return int(self._backend._problem.frame.base.nobs)

    @property
    def count(self) -> int:
        return self._backend._core.count

    @property
    def effective_n(self) -> int:
        return self._backend._core.count

    @property
    def zeros(self) -> int:
        return 0

    @property
    def weight_kind(self) -> None:
        return None

    @property
    def weights(self) -> None:
        return None

    @property
    def dropped_singleton(self) -> int:
        return int(self._backend._facts["singleton"])

    @property
    def dropped_separated(self) -> int:
        return 0

    @property
    def dropped_separated_by_fixed_effect(self) -> int:
        return 0

    @property
    def dropped_collinear(self) -> tuple[str, ...]:
        active = set(self.column_names)
        return tuple(n for n in self._backend._facts["all_names"] if n not in active)

    @property
    def omitted_slopes(self) -> tuple[tuple[str, str], ...]:
        return tuple((name, "collinear") for name in self.dropped_collinear)

    @property
    def counts(self) -> Mapping[str, int]:
        selected = self._backend._facts["selected"]
        found: dict[str, int] = dict(counts_of(selected.codes))
        found["singleton"] = self.dropped_singleton
        found["separated"] = 0
        found["separated_by_fixed_effect"] = 0
        found["retained"] = self.count
        return MappingProxyType(found)

    @property
    def missing(self) -> Mapping[str, int]:
        frame: PreparedFrame = self._backend._problem.frame
        return frame.base.missing.counts

    def table(self) -> tuple[tuple[str, int], ...]:
        return tuple(self.counts.items())

    @property
    def positions(self) -> np.ndarray:
        found = np.array(self._backend._facts["positions"], dtype=np.intp)
        found.flags.writeable = False
        return found

    @property
    def labels(self) -> pd.Index[Any]:
        """Return detached labels in the retained positional order."""
        frame: PreparedFrame = self._backend._problem.frame
        return _owned_index(frame._labels.take(self._backend._facts["positions"]))

    @property
    def cluster_names(self) -> tuple[str, ...]:
        return self._backend._retained_names()


class LightLogBackend:
    """The result backend of a light fixed-effect log-OLS fit (module docstring)."""

    organization = "linear"
    status = "verified"
    certified = True
    _bias = None

    def _indefinite_root(self) -> bool:
        """Whether the certified root's curvature is not positive definite (ruling 78).

        Never on this route: the light fit solves least squares, whose curvature is the
        demeaned Gram matrix, positive semidefinite, and its result is not a root found
        by the nonlinear solver whose certificate can read the fixed merit metric.
        """
        return False

    def __init__(
        self,
        problem: Any,
        spec: Any,
        core: LightCore,
        facts: dict[str, Any],
        initial: RetainedInference | RetainedHC,
        keep_influence: bool,
        dependence: Callable[[], BoundDependence],
        exact: Callable[[], Result],
    ) -> None:
        self._problem = problem
        self._spec = spec
        self._core = core
        self._facts = facts
        self.initial_inference = initial
        self._inference_override: RetainedInference | RetainedHC | None = None
        self._keep = keep_influence
        self._dependence = dependence
        self._exact_call = exact
        self._exact_box: list[AbsorbedResultBackend] = []
        model = problem.frame._frame
        self.parts = model.parts
        self.design = model.design
        self.estimator = replace(
            spec.estimator,
            fixed_effects=FixedEffectFormulation(
                tuple("^".join(term) for term in model.parts.fixed_effects)
            ),
        )
        self.fit_vcov_spec = (spec.fe_vcov, spec.ssc, spec.bread)
        self._provenance = self._build_provenance()

    # -------------------------------------------------------------- the exact route

    def _exact(self) -> AbsorbedResultBackend:
        """The exact route's backend of the same call, formed once, silently."""
        if not self._exact_box:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = self._exact_call()
            # The fallback of an ordinary absorbed log-OLS fit uses this backend.
            self._exact_box.append(cast("AbsorbedResultBackend", result._fe))
        return self._exact_box[0]

    def _retained_names(self) -> tuple[str, ...]:
        frame = self._problem.frame
        dep = frame.dependence
        names = [] if dep is None else list(dep.columns)
        for name in frame.base.inputs.metadata.columns:
            if name not in names:
                names.append(name)
        return tuple(names)

    # -------------------------------------------------------------- attributes

    @property
    def certificate(self) -> None:
        """None: least squares by absorption holds no certificate (ruling 52)."""
        return None

    @property
    def sample(self) -> FESampleView:
        """The public sample view, backed by the exact view for any deferred field."""
        # Result owns the view. Its backend must not retain a view that owns it.
        return cast("FESampleView", LightSampleView(self))

    @property
    def provenance(self) -> Provenance:
        """What ran."""
        return self._provenance

    def _names(self) -> pd.Index[str]:
        return pd.Index(list(self._core.names), name="term")

    def _withheld(self) -> ConvergenceError:
        return self._exact()._withheld()

    def coef(self) -> pd.Series[float]:
        """The estimates, by name."""
        return pd.Series(
            np.array(self._core.beta, dtype=np.float64, copy=True),
            index=self._names(),
            name="estimate",
        )

    def _reported_inference(self) -> RetainedInference | RetainedHC:
        return (
            self.initial_inference
            if self._inference_override is None
            else self._inference_override
        )

    def se(self) -> pd.Series[float]:
        """The standard errors of the active covariance, by name."""
        record = self._reported_inference()
        return pd.Series(
            np.array(record.standard_errors, dtype=np.float64, copy=True),
            index=self._names(),
            name="std_error",
        )

    def _reporting_context(self) -> ReportingContext:
        return _target_reporting_context(
            self._core.names,
            np.asarray(self.coef().to_numpy(), dtype=np.float64),
            self._reported_inference(),
            self.estimator.mean.omega,
            ReportingPoint(self.status, "certified uncorrected root"),
        )

    # -------------------------------------------------------------- covariances

    def _matches_initial(self, fe_vcov: Any, ssc_val: Any, bread_val: Any) -> bool:
        init_vcov, init_ssc, init_bread = self.fit_vcov_spec
        return cast(
            "bool",
            (
                fe_vcov == init_vcov
                and (isinstance(ssc_val, Omitted) or ssc_val == init_ssc)
                and bread_val == init_bread
            ),
        )

    def _inherited(self) -> Any:
        record = self.initial_inference
        if type(record) is RetainedHC:
            return record.hc.applied.convention
        return record.applied.convention

    def _cluster_codes(self, request: ClusterRequest) -> tuple[ClusterCodes, ...]:
        """``fe_result_backends._bind_absorbed_cluster_codes`` on the light sample."""
        dependence = self._dependence()
        block_map = dict(self._core.blocks)
        found: list[ClusterCodes] = []
        for dim in request.dimensions:
            if dim in dependence.columns:
                found.append(dependence.cluster_codes(dim))
            elif dim in block_map:
                found.append(ClusterCodes(dim, block_map[dim]))
            else:
                found.append(dependence.cluster_codes(dim))
        return tuple(found)

    def _postfit_hc(self, request: HCInferenceRequest) -> RetainedHC:
        """``retained_hc._envelope_hc`` of a linear envelope, on the light arrays."""
        init = self.initial_inference
        inherited = self._inherited()
        if (
            type(init) is RetainedHC
            and init.request.method == request.method
            and init.request.bread == request.bread
        ):
            applied = init.hc.applied
            resolved_init = _resolved(request, applied.convention)
            owned_factor(
                resolved_init, applied.counts, kind="robust", method=request.method
            )
            if resolved_init.values == applied.convention.values:
                hc_rec = init.hc
                if resolved_init != applied.convention:
                    hc_rec = replace(
                        hc_rec,
                        applied=apply_convention(
                            resolved_init,
                            applied.counts,
                            kind="robust",
                            method=request.method,
                        ),
                    )
                return RetainedHC(
                    request,
                    self._core.capability(self._keep),
                    hc_rec,
                    init.theta_covariance,
                    _request_errors(hc_rec, init.standard_errors),
                    init.reported_covariance,
                    init.covariance_error,
                    self._core.theta,
                    self._core.names,
                    init.count,
                    self._core.count,
                    self._core.count,
                )
        return self._core.hc(request, _resolved(request, inherited), self._keep)

    def _inference(self, spec: object = None) -> Any:
        init_bread = self.fit_vcov_spec[2]
        if spec is None and self._inference_override is not None:
            return self._inference_override
        if spec is None:
            return self.initial_inference
        fe_vcov, ssc_val, bread_val = _parse_vcov_spec_for_absorbed(
            spec, OMITTED, init_bread
        )
        if self._matches_initial(fe_vcov, ssc_val, bread_val):
            return self.initial_inference
        if not self._keep or (
            isinstance(fe_vcov, ClusterRequest)
            and not isinstance(fe_vcov.adjustment, Omitted)
        ):
            return self._exact()._inference(spec)
        if isinstance(fe_vcov, HCRequest):
            hc_req = HCInferenceRequest.from_hc_request(
                fe_vcov, ssc=ssc_val, bread=bread_val
            )
            return _replayed(lambda: self._postfit_hc(hc_req))
        if isinstance(fe_vcov, ClusterRequest):
            clusters = self._cluster_codes(fe_vcov)
            inf_req = InferenceRequest.from_cluster_request(
                fe_vcov, clusters, ssc=ssc_val, bread=bread_val
            )
        else:
            inf_req = InferenceRequest("robust", (), ssc_val, bread_val)
        if inf_req.adjustment == "wz":
            return self._exact()._inference(spec)
        inherited = self._inherited()
        try:
            return _replayed(lambda: self._core.inference(inf_req, inherited))
        except _Exact:
            return self._exact()._inference(spec)

    def vcov(self, spec: object = None) -> pd.DataFrame:
        """The active covariance, or that of ``spec``, as a frame with its attrs."""
        record = self._inference(spec)
        if isinstance(record, pd.DataFrame):
            return record
        frame = _vcov_frame_from_retained(record, self._core.names)
        frame.attrs.update(self._count_fields(record))
        return frame

    def with_vcov(self, spec: object) -> LightLogBackend:
        """A covariance view without a new fit."""
        record = self._inference(spec)
        result = copy.copy(self)
        result._inference_override = record
        result._provenance = result._build_provenance()
        return result

    def tidy(self, level: float = 0.95) -> pd.DataFrame:
        """The table of estimates, standard errors and intervals."""
        normal_quantile(level)
        record = self._reported_inference()
        return _tidy_frame(
            self._core.names,
            self._core.beta,
            record.standard_errors,
            level,
            record.applied.reference,
        )

    def confint(self, level: float = 0.95) -> pd.DataFrame:
        """The intervals of the active covariance."""
        table = self.tidy(level)
        arr = table[["conf_low", "conf_high"]].to_numpy(dtype=np.float64)
        return pd.DataFrame(arr, index=self._names(), columns=["lower", "upper"])

    # -------------------------------------------------------------- exact route

    def influence(self) -> pd.DataFrame:
        """The influence rows, from the exact route's result."""
        return self._exact().influence()

    def influence_rows(self) -> pd.DataFrame:
        """The influence by row, from the exact route's result."""
        return self._exact().influence_rows()

    def fixef(self) -> dict[str, pd.Series[float]]:
        """The normalized fixed effects, from the exact route's result."""
        return self._exact().fixef()

    def predict(
        self, data: pd.DataFrame | None = None, **options: Unpack[PredictionOptions]
    ) -> Prediction:
        """Return the fitted power mean on the retained rows."""
        kind = prediction_kind(options)
        if data is not None or kind != "power_mean":
            return self._exact().predict(data, **options)
        index = self._core.fitted_index
        if not (
            float(index.min()) > _NORMAL_LOG_MIN + _PREDICTION_MARGIN
            and float(index.max()) < _NORMAL_LOG_MAX - _PREDICTION_MARGIN
        ):
            return self._exact().predict(**options)
        series = pd.Series(np.exp(index), index=self.sample.labels, name=kind)
        series.attrs["label"] = naming.prediction_label(0.0)
        series.attrs["omega"] = 0.0
        return series

    # -------------------------------------------------------------- reports

    def _absorbed_count(self, record: object) -> str | None:
        """The count rule only where a factor or reference reads a parameter count."""
        applied = getattr(record, "applied", None)
        if applied is None or not applied.reads_count:
            return None
        return str(applied.convention.values.absorbed_count)

    def _count_fields(self, record: object) -> dict[str, object]:
        """The exact route's rank/K metadata, read without forming another rank."""
        applied = getattr(record, "applied", None)
        if applied is None:
            return {}
        method = record.request.method if isinstance(record, RetainedHC) else None
        counts = applied.counts
        exact = rank_formed(method, applied.convention)
        found: dict[str, object] = {"rank": int(counts.rank) if exact else None}
        if applied.reads_count:
            found["K"] = int(counts.k_factor)
        return found

    def summary(self, level: float = 0.95) -> str:
        """The text summary, as the exact route's result prints it."""
        normal_quantile(level)
        # The retained provenance counts avoid recounting the light sample rows.
        effects_line, observations_line, coverage_line = _fe_summary_lines(
            self.estimator,
            cast(Mapping[str, int], self.provenance.sample["counts"]),
            self.sample.nobs,
            self.provenance.coverage,
        )
        lines = [
            "powermean",
            "=" * WIDTH,
            naming.estimator_name(self.estimator),
            effects_line,
            observations_line,
            f"Status: {self.status}; certified: "
            "not applicable (least squares by absorption holds no certificate)",
            coverage_line,
        ]
        lines.extend(_table(self.tidy(level), level))
        record = self._reported_inference()
        clusters = (
            record.request.clusters if isinstance(record, RetainedInference) else ()
        )
        lines.extend(_joint_block_lines(self._core.blocks, clusters))
        counted = self._absorbed_count(record)
        if counted is not None:
            lines.append(ABSORBED_COUNT_NOTES[counted])
        return "\n".join(lines)

    def _build_provenance(self) -> Provenance:
        spec = self._spec
        controls = spec.fe_solver_options
        cov_label = naming.coverage(self.estimator)
        request = SampleRequest.default(
            0.0, None, self._problem.positive_only, default_diagnostic()
        )
        options = _frozen(
            {
                "sample_request": asdict(request),
                "solver_options": spec.options.to_dict(),
                "absorbed": "linear",
                "absorbed_options": controls.absorbed_options.to_dict(),
                "max_dense_bytes": controls.max_dense_bytes,
            }
        )
        record = self._reported_inference()
        attrs = {str(k): v for k, v in _vcov_attrs_from_retained(record).items()}
        attrs.update(self._count_fields(record))
        counted = self._absorbed_count(record)
        if counted is not None:
            attrs["absorbed_count"] = counted
        view = self.sample
        columns = self._problem.frame.columns
        fields = {
            "fingerprint": None,
            "nobs": view.nobs,
            "count": view.count,
            "effective_n": view.effective_n,
            "counts": dict(view.counts),
            "missing": dict(view.missing),
            "subset_missing": int(self._problem.frame.subset_missing),
            "positive_only": view.positive_only,
            "conditioned": view.conditioned,
            "weight_kind": None,
            "weights_column": columns.weights,
            "offset_column": columns.offset,
            "exposure_column": columns.exposure,
            "offset_expression": False,
            "retained_classifications": view.cluster_names,
            "dropped_singleton": view.dropped_singleton,
            "dropped_separated": 0,
            "dropped_collinear": view.dropped_collinear,
        }
        core = self._core
        frozen = dict(_frozen(fields))
        sample_rec = MappingProxyType(_LazyRecord(frozen, lambda: core.fingerprint))
        formula_rec = _frozen(
            {
                "terms": tuple((term, tuple(cols)) for term, cols in self.design.terms),
                "variables": tuple(self.design.variables),
                "formulaic": self.design.formulaic_version,
                "names": core.names,
                "fixed_effects": tuple(
                    "^".join(term) for term in self.parts.fixed_effects
                ),
            }
        )
        trace = _frozen(
            {"organization": "linear", "n_attempts": 0, "reason": "complete"}
        )
        return Provenance(
            versions=versions(),
            implementation=IMPLEMENTATION,
            solver="linear absorbed least squares",
            carried_files=carried_files(None),
            options=options,
            trace=trace,
            linear=None,
            status=self.status,
            level="strict",
            certified=True,
            degenerate=bool(np.any(record.standard_errors == 0)),
            coverage=_frozen({"label": cov_label.label, "basis": cov_label.basis}),
            estimator=_frozen(self.estimator.to_dict()),
            sample=sample_rec,
            formula=formula_rec,
            covariance=_frozen(attrs),
            references=citations.references(self.estimator),
            requested_start=None,
            failure=None,
        )


__all__ = ["LightLogBackend", "light_fit"]
