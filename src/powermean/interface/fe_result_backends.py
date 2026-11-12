"""Result backends for joint absorbed fixed-effects and selected-cancellation fits."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd

from powermean import messages
from powermean._prepare.compose import Columns
from powermean.data.inference_conventions import ConventionValues
from powermean.data.inference_requests import (
    ClusterRequest,
    HCRequest,
)
from powermean.data.request_values import Omitted
from powermean.data.retained import OrderSample
from powermean.data.specs import Estimator, Inference
from powermean.effects.targets import ReportingContext, ReportingPoint
from powermean.errors import SpecificationError
from powermean.fe.groups import Grouping
from powermean.formula.frame import Design
from powermean.formula.grammar import FormulaParts
from powermean.inference.convention import ClusterCodes
from powermean.inference.ssc import ReferenceDistribution
from powermean.interface.cancelled_selection import CancelledFitControls
from powermean.interface.fe_options import (
    FEBread,
    FEVcovRequest,
    check_deferred_vcov,
    resolve_fe_vcov,
)
from powermean.interface.retained_metadata import BoundDependence
from powermean.interface.retained_result import RetainedAbsorbedResult
from powermean.model.variance import FloatArray
from powermean.results.absorbed_outcome_records import RetainedOutcomeSnapshot
from powermean.results.cancelled_bias import CancelledBias
from powermean.results.cancelled_inference import (
    bind_cluster as bind_cancelled_cluster,
)
from powermean.results.cancelled_retained import (
    RetainedCancelledEnvelope,
)
from powermean.results.effect_reporting import context_snapshot
from powermean.results.fe_helpers import (
    normal_quantile,
    two_sided_p,
)
from powermean.results.retained_hc import (
    RetainedHC,
)
from powermean.results.retained_inference import (
    RetainedInference,
)
from powermean.results.wz_bias import WZBias
from powermean.solve.cancelled_records import RootOptions
from powermean.solve.absorbed import AbsorbedOptions
from powermean.solve.no_point import StartFailure
from powermean.solve.options import SolverOptions


class _StudentT(Protocol):
    """The existing SciPy Student-t distribution operations used by FE output."""

    def ppf(self, quantile: float, df: int) -> float:
        """The Student-t quantile."""
        ...

    def sf(self, statistic: FloatArray, df: int) -> FloatArray:
        """The Student-t upper tail."""
        ...


def _reference_quantile_and_p(
    level: float,
    statistic: FloatArray,
    reference: ReferenceDistribution | None,
) -> tuple[float, FloatArray]:
    """Return quantile and two-sided p-values under ``reference`` (normal or t)."""
    if reference is None or reference.kind == "normal":
        return normal_quantile(level), two_sided_p(statistic)
    normal_quantile(level)
    if reference.kind != "t" or reference.df is None or reference.df <= 0:
        raise ValueError("a Student-t reference needs positive degrees of freedom")
    student_t = cast("_StudentT", importlib.import_module("scipy.stats").t)
    quantile = float(student_t.ppf((1.0 + level) / 2.0, reference.df))
    p_values: FloatArray = 2.0 * student_t.sf(np.abs(statistic), reference.df)
    return quantile, p_values


def _tidy_frame(
    names: tuple[str, ...],
    estimate: FloatArray,
    errors: FloatArray,
    level: float,
    reference: ReferenceDistribution | None,
) -> pd.DataFrame:
    """Build the 8-column tidy DataFrame for ``Result.tidy``."""
    normal_quantile(level)
    if len(names) == 0:
        return pd.DataFrame(
            {
                "term": pd.Series([], dtype=object),
                "estimate": pd.Series([], dtype=np.float64),
                "std_error": pd.Series([], dtype=np.float64),
                "statistic": pd.Series([], dtype=np.float64),
                "p_value": pd.Series([], dtype=np.float64),
                "conf_low": pd.Series([], dtype=np.float64),
                "conf_high": pd.Series([], dtype=np.float64),
                "inference": pd.Series([], dtype=object),
            }
        )
    est = np.array(estimate, dtype=np.float64, copy=True)
    err = np.array(errors, dtype=np.float64, copy=True)
    statistic = np.full(est.shape, np.nan, dtype=np.float64)
    usable = np.isfinite(err) & (err > 0.0)
    quantile, _ = _reference_quantile_and_p(level, statistic, reference)
    with np.errstate(over="ignore", invalid="ignore"):
        statistic[usable] = est[usable] / err[usable]
        half = np.where(usable, quantile * err, np.nan)
    _, p_values = _reference_quantile_and_p(level, statistic, reference)
    status = np.where(usable, "ok", np.where(err == 0.0, "degenerate", "missing"))
    return pd.DataFrame(
        {
            "term": list(names),
            "estimate": est,
            "std_error": err,
            "statistic": statistic,
            "p_value": p_values,
            "conf_low": est - half,
            "conf_high": est + half,
            "inference": status.astype(object),
        }
    )


def _covariance_diagnostics(record: RetainedInference) -> tuple[str, ...]:
    """Final adjustment disposition outward; untouched raw diagnostics stay retained."""
    if record.adjustment is not None and record.adjustment.active:
        return ("covariance.psd_adjusted",) + (
            ("covariance.indefinite",) if record.adjustment.final_indefinite else ()
        )
    if record.covariance is not None:
        return tuple(record.covariance.estimate.diagnostics)
    if record.multiway is not None:
        return tuple(record.multiway.estimate.diagnostics)
    return ()


def _vcov_attrs_from_retained(
    record: RetainedInference | RetainedHC,
) -> dict[str, object]:
    """Build the ``DataFrame.attrs`` dictionary for a retained covariance record."""
    if isinstance(record, RetainedHC):
        return {
            "method": record.request.method,
            "cluster": None,
            "factor": float(record.factor),
            "N": int(record.applied.counts.count),
            "G": None,
            "rank": int(record.applied.counts.rank),
            "diagnostics": (),
            "bread": record.request.bread,
            "coordinate": "reported",
            "convention": record.applied.convention.preset,
        }
    cluster_dims = tuple(c.name for c in record.request.clusters)
    cluster_attr: str | tuple[str, ...] | None
    if len(cluster_dims) == 0:
        cluster_attr = None
    elif len(cluster_dims) == 1:
        cluster_attr = cluster_dims[0]
    else:
        cluster_attr = cluster_dims
    cluster_counts = record.applied.counts.clusters
    g_attr: int | tuple[int, ...] | None
    if len(cluster_counts) == 0:
        g_attr = None
    elif len(cluster_counts) == 1:
        g_attr = int(cluster_counts[0])
    else:
        g_attr = tuple(int(c) for c in cluster_counts)
    factor_val: float | tuple[float, ...]
    if record.applied.factor is not None:
        factor_val = float(record.applied.factor.value)
    else:
        factor_val = tuple(float(f.value) for f in record.applied.components)
    attributes: dict[str, object] = {
        "method": record.request.kind,
        "cluster": cluster_attr,
        "factor": factor_val,
        "N": int(record.applied.counts.count),
        "G": g_attr,
        "rank": int(record.applied.counts.rank),
        "diagnostics": _covariance_diagnostics(record),
        "bread": record.bread,
        "coordinate": "reported",
        "convention": record.applied.convention.preset,
    }
    if record.wz is not None:
        attributes.update(
            adjustment="wz",
            factor_owner="published_WZ_only",
            coefficient_correction=False,
            covariance_point="uncorrected root",
            reference=record.applied.reference.kind,
            reference_df=record.applied.reference.df,
        )
    return attributes


def _vcov_frame_from_retained(
    record: RetainedInference | RetainedHC,
    names: tuple[str, ...],
) -> pd.DataFrame:
    """Convert a ``RetainedInference`` or ``RetainedHC`` into a DataFrame with attrs."""
    idx = pd.Index(list(names), name="term")
    matrix = np.array(record.require_reported_covariance(), dtype=np.float64, copy=True)
    frame = pd.DataFrame(matrix, index=idx, columns=idx.copy())
    frame.attrs.update(_vcov_attrs_from_retained(record))
    return frame


def _target_reporting_context(
    names: tuple[str, ...],
    beta: FloatArray,
    record: RetainedInference | RetainedHC,
    omega: float,
    point: ReportingPoint,
) -> ReportingContext:
    """Compose an FE numeric snapshot from its already selected reporting record."""
    metadata = _vcov_attrs_from_retained(record)
    applied = record.applied
    metadata.update(
        ssc=applied.convention.values.to_dict(),
        counts=asdict(applied.counts),
        factors=(
            [asdict(applied.factor)]
            if applied.factor is not None
            else [asdict(factor) for factor in applied.components]
        ),
    )
    return context_snapshot(
        names,
        beta,
        record.require_reported_covariance(),
        applied.reference,
        omega,
        point=point,
        metadata=metadata,
    )


def _bind_absorbed_cluster_codes(
    retained_result: RetainedAbsorbedResult,
    request: ClusterRequest,
) -> tuple[ClusterCodes, ...]:
    """Bind ``ClusterCodes`` from ``retained_result.dependence`` for ``vcov(spec)``."""
    dependence = retained_result.dependence
    grouping = retained_result.sample.grouping
    block_map = {} if grouping is None else {b.name: b.codes for b in grouping.blocks}
    codes_list: list[ClusterCodes] = []
    for dim in request.dimensions:
        if dependence is not None and dim in dependence.columns:
            codes_list.append(dependence.cluster_codes(dim))
        elif dim in block_map:
            codes_list.append(ClusterCodes(dim, block_map[dim]))
        elif dependence is not None:
            codes_list.append(dependence.cluster_codes(dim))
        else:
            raise SpecificationError(
                "specification.cluster_not_retained",
                {"name": dim, "retained": "none"},
            )
    return tuple(codes_list)


def _bind_cancelled_cluster_codes(
    envelope: RetainedCancelledEnvelope,
    dependence: BoundDependence,
    grouping: Grouping,
    request: ClusterRequest,
) -> tuple[ClusterCodes, ...]:
    """Bind ``ClusterCodes`` for a post-fit ``ClusterRequest`` on a cancelled fit."""
    block_map = {b.name: b.codes for b in grouping.blocks}
    positions = np.arange(envelope.n_rows, dtype=np.intp)
    codes_list: list[ClusterCodes] = []
    for dim in request.dimensions:
        if dim == envelope.cancelled_name:
            codes_list.append(
                ClusterCodes(
                    envelope.cancelled_name,
                    envelope.geometry.structure.cancel_codes,
                )
            )
        elif dim in dependence.columns:
            meta = dependence.columns[dim]
            if meta.sample_missing_count > 0:
                raise SpecificationError(
                    "specification.cluster_missing_values",
                    {"name": dim, "count": meta.sample_missing_count},
                )
            codes_list.append(
                bind_cancelled_cluster(envelope, dim, meta.values, positions)
            )
        elif dim in block_map:
            codes_list.append(ClusterCodes(dim, block_map[dim]))
        else:
            dependence.cluster_codes(dim)
    return tuple(codes_list)


def _parse_vcov_spec_for_absorbed(
    spec: object,
    default_ssc: str | ConventionValues | Omitted,
    default_bread: FEBread,
) -> tuple[FEVcovRequest, str | ConventionValues | Omitted, FEBread]:
    """Parse ``Result.vcov(spec)`` argument for an absorbed or cancelled FE result."""
    check_deferred_vcov(spec)
    if isinstance(spec, Inference):
        # A robust record with an explicit ssc is refused below, never downgraded.
        if spec.covariance == "robust" and isinstance(spec.ssc, Omitted):
            return "robust", default_ssc, default_bread
        if (
            spec.covariance != "cluster"
            or spec.clusters is not None
            or spec.cluster is None
        ):
            raise SpecificationError(
                "specification.unsupported_inference",
                {
                    "method": "An ordinary Inference record for this covariance",
                    "case": "fixed-effect results; use a supported vcov mapping "
                    "or the existing typed HCRequest/ClusterRequest",
                },
            )
        return ClusterRequest(dimensions=(spec.cluster,)), default_ssc, default_bread
    if isinstance(spec, (HCRequest, ClusterRequest)):
        return spec, default_ssc, default_bread
    if isinstance(spec, str):
        fe_vcov, ssc_v, bread_v, _ = resolve_fe_vcov(
            spec, None, ssc=default_ssc, bread=default_bread
        )
        return fe_vcov, ssc_v, bread_v
    if isinstance(spec, Mapping):
        allowed = {
            "cluster",
            "multiway",
            "psd",
            "ssc",
            "bread",
            "method",
            "vcov",
            "adjustment",
        }
        for k in spec:
            if k not in allowed:
                raise SpecificationError(
                    "specification.unknown_option",
                    {
                        "kind": "vcov key",
                        "key": k,
                        "valid": messages.format_keys(["cluster", "multiway", "psd"]),
                    },
                )
        ssc_in = spec.get("ssc", default_ssc)
        bread_in = spec.get("bread", default_bread)
        if "method" in spec or "vcov" in spec:
            m = spec.get("method", spec.get("vcov"))
            fe_vcov, ssc_v, bread_v, _ = resolve_fe_vcov(
                m, spec.get("cluster"), ssc=ssc_in, bread=bread_in
            )
            return fe_vcov, ssc_v, bread_v
        cluster_map = {
            k: v
            for k, v in spec.items()
            if k in ("cluster", "multiway", "psd", "adjustment")
        }
        fe_vcov, ssc_v, bread_v, _ = resolve_fe_vcov(
            cluster_map, None, ssc=ssc_in, bread=bread_in
        )
        return fe_vcov, ssc_v, bread_v
    raise SpecificationError(
        "specification.invalid_value",
        {
            "argument": "vcov",
            "value": spec,
            "valid": "'robust', 'HC0'..'HC5', 'HC4m', or a mapping with 'cluster'",
        },
    )


@dataclass(frozen=True)
class FEView:
    """Owned normal-call reporting and provenance inputs."""

    estimator: Estimator
    order_sample: OrderSample
    input_labels: pd.Index[Any]
    columns: Columns
    subset_missing: int
    parts: FormulaParts
    design: Design
    offset_design: Design | None
    solver_options: SolverOptions
    fit_vcov_spec: tuple[FEVcovRequest, str | ConventionValues | Omitted, FEBread]
    requested_start: FloatArray | None


@dataclass(frozen=True)
class AbsorbedState:
    """Actual retained solver state and its initial requested scale.

    ``organization`` is the organization requested at the fit; ``operation`` the
    actual operation of a selected-root fit's root stages (its convex initializer
    ran the requested organization), ``None`` on the convex and linear routes, and
    ``root_options`` the resolved root-Newton options with which it ran.
    """

    retained_result: RetainedAbsorbedResult | None
    initial_inference: RetainedInference | RetainedHC | None
    unsolved_outcome: RetainedOutcomeSnapshot | None
    start_failure: StartFailure | None
    status: str
    absorbed_options: AbsorbedOptions
    organization: Literal["index", "profiled", "linear"]
    max_dense_bytes: int | None = None
    bias_correction: WZBias | None = None
    operation: str | None = None
    root_options: RootOptions | None = None


@dataclass(frozen=True)
class CancelledState:
    """Actual complete-fit cancellation envelope and aligned dependence."""

    envelope: RetainedCancelledEnvelope
    dependence: BoundDependence
    grouping: Grouping
    controls: CancelledFitControls
    requested_absorbed_options: AbsorbedOptions
    bias_correction: CancelledBias | WZBias | None = None
