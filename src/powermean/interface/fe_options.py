"""Option resolution for fixed-effects and selected-cancellation routes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, cast

from powermean import messages
from powermean.data.inference_conventions import ConventionValues, RetainedColumns
from powermean.data.inference_requests import (
    HC_METHODS,
    ClusterRequest,
    ConleyRequest,
    HacRequest,
    HCRequest,
)
from powermean.data.request_values import OMITTED, Omitted
from powermean.errors import SpecificationError
from powermean.inference.bread import BREADS, Bread
from powermean.inference.ssc import resolve_convention
from powermean.interface.cancelled_selection import CancelledFitControls
from powermean.interface.sample_records import SampleRequest
from powermean.model.variance import (
    CancelledFamily,
    NB2Variance,
    PowerVariance,
    resolve_cancelled_variance,
)
from powermean.results.retained_hc import deferred_vcov
from powermean.solve.cancelled_records import (
    RootOptions,
    resolve_root_options,
)
from powermean.solve.absorbed import ABSORBED_DOMAINS, AbsorbedOptions
from powermean.solve.options import (
    OPTION_NAMES,
    SolverOptions,
    resolve_options,
)

FEBread = Bread
FEVcovRequest = (
    Literal["robust", "cluster_cancelled"]
    | HCRequest
    | ClusterRequest
    | HacRequest
    | ConleyRequest
)

_CANCELLED_CONTROL_FIELDS = frozenset(
    {
        "gram_rtol",
        "gap_min",
        "geometry_bytes",
        "admission_bytes",
        "identity_atol",
        "attempt",
        "certificate",
        "inference",
        "diagnostic",
        "collinear",
    }
)

_FE_SOLVER_WRAPPER_KEYS = frozenset(
    {
        "controls",
        "cancelled_controls",
        "solver_options",
        "absorbed_options",
        "root_options",
    }
)


@dataclass(frozen=True)
class ResolvedFESolverOptions:
    """Validated solver, absorbed, and cancelled control options for FE fits."""

    solver_options: SolverOptions
    absorbed_options: AbsorbedOptions | None
    root_options: RootOptions | None
    explicit_absorbed: bool
    cancelled_controls: CancelledFitControls | None
    sample_request: SampleRequest | None
    max_dense_bytes: int | None = None


def _unknown(kind: str, key: object, valid: Sequence[str]) -> SpecificationError:
    return SpecificationError(
        "specification.unknown_option",
        {"kind": kind, "key": key, "valid": messages.format_keys(valid)},
    )


def _invalid(argument: str, value: object, valid: str) -> SpecificationError:
    return SpecificationError(
        "specification.invalid_value",
        {"argument": argument, "value": value, "valid": valid},
    )


def _conflicting(first: str, second: str) -> SpecificationError:
    return SpecificationError(
        "specification.conflicting", {"first": first, "second": second}
    )


def _unimplemented(feature: str) -> SpecificationError:
    return SpecificationError("specification.unimplemented", {"feature": feature})


def check_deferred_vcov(vcov: object, cluster: object = None) -> None:
    """Refuse reserved HAC and Conley covariance requests before solve.

    The one check of :func:`powermean.results.retained_hc.deferred_vcov`, which the
    requests on a result share.
    """
    deferred_vcov(vcov, cluster)


def resolve_cancel(
    cancel: object, *, has_fixed_effects: bool
) -> str | Literal[True] | None:
    """Validate ``cancel`` against formula structure."""
    if cancel is None:
        return None
    if not has_fixed_effects:
        raise _invalid("cancel", cancel, "None for models without fixed effects")
    if isinstance(cancel, bool):
        if not cancel:
            raise _invalid(
                "cancel", cancel, "None, True, or a fixed-effect column name"
            )
        return True
    if isinstance(cancel, str):
        stripped = cancel.strip()
        if not stripped:
            raise _invalid("cancel", cancel, "a non-empty fixed-effect column name")
        return stripped
    raise _invalid("cancel", cancel, "None, True, or a fixed-effect column name")


def resolve_retain(retain: object) -> RetainedColumns:
    """Validate ``retain`` into :class:`RetainedColumns`."""
    if retain is None:
        return RetainedColumns(())
    if isinstance(retain, RetainedColumns):
        return retain
    if isinstance(retain, str):
        return RetainedColumns((retain,))
    if isinstance(retain, Sequence) and not isinstance(retain, (str, bytes)):
        return RetainedColumns(tuple(retain))
    raise _invalid("retain", retain, "column names")


def resolve_fe_variance(
    omega: float,
    tau: object | None,
    variance: object | None,
    *,
    cancel: str | Literal[True] | None,
) -> PowerVariance | NB2Variance | CancelledFamily:
    """Resolve working variance for joint absorbed or cancelled FE.

    A joint power outside ``[1, 2]`` at ``omega != 0`` is the selected-root route of
    :mod:`powermean.solve.absorbed_root` (the nearest convex start on the same
    sample, then sparse root Newton); at ``omega = 0``, where no root route exists,
    it stays refused.
    """
    if tau is not None and variance is not None:
        raise _conflicting(f"tau={tau!r}", f"variance={variance!r}")
    if omega == 0.0 and cancel is not None:
        raise _conflicting("omega=0", f"cancel={cancel!r}")
    if cancel is None:
        family = resolve_cancelled_variance(
            variance=variance, tau=1.0 if tau is None and variance is None else tau
        )
        if (
            omega == 0.0
            and isinstance(family, PowerVariance)
            and not 1 <= family.tau <= 2
        ):
            raise _invalid("tau", family.tau, "a joint power in [1, 2]")
        return family
    return resolve_cancelled_variance(variance=variance, tau=tau)


def resolve_fe_solver(value: object | None) -> Literal["newton"]:
    """Validate solver for FE models."""
    if value is None or value == "newton":
        return "newton"
    if value == "iols":
        raise _unimplemented('solver="iols"')
    raise _invalid("solver", value, messages.format_keys(["newton"]))


def _cluster_dimensions(cluster: object) -> tuple[str, ...]:
    if isinstance(cluster, str):
        if "+" in cluster:
            parts = tuple(p.strip() for p in cluster.split("+"))
            return ClusterRequest(dimensions=parts).dimensions
        return ClusterRequest(dimensions=(cluster,)).dimensions
    if isinstance(cluster, Sequence) and not isinstance(cluster, (str, bytes)):
        return ClusterRequest(dimensions=tuple(cluster)).dimensions
    raise _invalid("cluster", cluster, "column names")


def _cov_string(
    value: str, dims: tuple[str, ...] | None, cancel: object
) -> FEVcovRequest:
    """Normalize named robust/HC or selected-group covariance."""
    if value in HC_METHODS:
        if dims is not None:
            raise _conflicting("cluster", value)
        return HCRequest(
            cast("Literal['HC0', 'HC1', 'HC2', 'HC3', 'HC4', 'HC4m', 'HC5']", value)
        )
    if value == "robust":
        if dims is not None:
            raise _conflicting("cluster", value)
        return "robust"
    if value == "cluster":
        if dims is not None:
            return ClusterRequest(dimensions=dims)
        if cancel is not None:
            return "cluster_cancelled"
    raise _invalid("vcov", value, "robust, HC0..HC5/HC4m, or a cluster mapping")


def _cov_mapping(value: Mapping[object, object]) -> ClusterRequest:
    """Validate all mapping keys before constructing the actual request."""
    for key in value:
        if key not in ("cluster", "multiway", "psd", "adjustment"):
            raise _unknown(
                "vcov key", key, ["cluster", "multiway", "psd", "adjustment"]
            )
    if "cluster" not in value:
        raise _invalid("vcov", value, "a mapping with cluster")
    return ClusterRequest.from_dict(
        {
            "type": "ClusterRequest",
            "dimensions": _cluster_dimensions(value["cluster"]),
            **{
                key: value[key]
                for key in ("multiway", "psd", "adjustment")
                if key in value
            },
        }
    )


def resolve_fe_vcov(
    vcov: object | None,
    cluster: object | None,
    *,
    ssc: object = OMITTED,
    bread: object = "observed",
    cancel: str | Literal[True] | None = None,
) -> tuple[FEVcovRequest, str | ConventionValues | Omitted, Bread, bool]:
    """Normalize an FE covariance request without computing or changing its gate."""
    check_deferred_vcov(vcov, cluster)
    if not isinstance(ssc, Omitted):
        resolve_convention(ssc)
    if not isinstance(ssc, (str, ConventionValues, Omitted)):
        raise _invalid("ssc", ssc, "a convention name or ConventionValues")
    if bread not in BREADS:
        raise _invalid("bread", bread, messages.format_keys(BREADS))
    resolved_bread: Bread = "observed" if bread == "observed" else "expected"
    dims = None if cluster is None else _cluster_dimensions(cluster)
    result: FEVcovRequest
    if vcov is None:
        result = "robust" if dims is None else ClusterRequest(dimensions=dims)
    elif isinstance(vcov, str):
        result = _cov_string(vcov, dims, cancel)
    elif isinstance(vcov, (HCRequest, ClusterRequest)):
        result = vcov
    elif isinstance(vcov, Mapping):
        result = _cov_mapping(vcov)
    else:
        raise _invalid("vcov", vcov, "robust, HC0..HC5/HC4m, or a cluster mapping")
    if dims is not None and (
        not isinstance(result, ClusterRequest) or result.dimensions != dims
    ):
        raise _conflicting("cluster", "vcov")
    if result is vcov and isinstance(result, (HCRequest, ClusterRequest)):
        result = replace(result)
    if isinstance(ssc, ConventionValues):
        ssc = replace(ssc)
    return result, ssc, resolved_bread, vcov is not None or dims is not None


def wz_coefficient_vcov(
    request: FEVcovRequest, *, explicit: bool, pair_name: str
) -> ClusterRequest:
    """Settle WZ's default before constructing the current-point assessor.

    Explicit adjustment='none' remains ordinary. A declared other sampling unit
    is refused rather than silently replaced by the published pair unit.
    """
    if not explicit or request == "cluster_cancelled":
        return ClusterRequest((pair_name,), adjustment="wz")
    if not isinstance(request, ClusterRequest) or request.dimensions != (pair_name,):
        raise SpecificationError(
            "specification.unsupported_inference",
            {
                "method": "WZ coefficient",
                "case": "the actual declared pair sampling unit required",
            },
        )
    return (
        replace(request, adjustment="wz")
        if isinstance(request.adjustment, Omitted)
        else request
    )


def _option_mapping(
    value: object, valid: Sequence[str], label: str
) -> dict[str, object]:
    """Own a closed option mapping and reject every unknown key."""
    if not isinstance(value, Mapping):
        raise _invalid(label, value, "a mapping")
    found: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or key not in valid:
            raise _unknown(label, key, valid)
        found[key] = item
    return found


def _merged(
    base: dict[str, object], raw: dict[str, object], valid: Sequence[str]
) -> dict[str, object]:
    """Merge explicit equivalent controls, refusing contradictory snapshots."""
    for key in valid:
        if key in raw:
            if key in base and raw[key] != base[key]:
                raise _conflicting(key, "nested option snapshot")
            base[key] = raw[key]
    return base


def _absorbed_options(raw: dict[str, object]) -> AbsorbedOptions:
    """Require the fit's explicit organization, never choose one."""
    nested = raw.get("absorbed_options")
    if isinstance(nested, AbsorbedOptions):
        values: dict[str, object] = dict(nested.to_dict())
    else:
        values = (
            {}
            if nested is None
            else _option_mapping(nested, tuple(ABSORBED_DOMAINS), "absorbed_options")
        )
    values = _merged(values, raw, tuple(ABSORBED_DOMAINS))
    if "absorbed" not in values:
        raise _invalid("absorbed", None, "explicit 'index' or 'profiled' at fit")
    return AbsorbedOptions.from_dict(values)


def _solver_options(raw: dict[str, object], tol: object | None) -> SolverOptions:
    """Preserve existing Newton options and the top-level tolerance conflict."""
    nested = raw.get("solver_options")
    if isinstance(nested, SolverOptions):
        values: dict[str, object] = dict(nested.to_dict())
    else:
        values = (
            {}
            if nested is None
            else _option_mapping(nested, OPTION_NAMES, "solver_options")
        )
    values = _merged(values, raw, OPTION_NAMES)
    if tol is not None:
        if "tol" in values and values["tol"] != tol:
            raise _conflicting("tol", "solver_options tol")
        values["tol"] = tol
    return resolve_options(values)


def _dense_limit(value: object) -> int | None:
    """Validate the optional joint WZ numeric-buffer ceiling at the typed seam."""
    if value is None:
        return None
    if not isinstance(value, int) or type(value) is not int or value <= 0:
        raise _invalid("max_dense_bytes", value, "a positive Python integer")
    return value


def resolve_fe_solver_options(
    solver_options: object | None,
    tol: object | None,
    *,
    cancel: str | Literal[True] | None,
    selected_root: bool = False,
) -> ResolvedFESolverOptions:
    """Resolve explicit FE organization and existing numerical control records.

    Callers supply SampleRequest for a joint fit and actual
    CancelledFitControls for cancellation. The optional joint byte limit
    applies to WZ numeric-buffer reservations, including later covariance calls.
    A joint fit outside ``tau in [1, 2]`` (``selected_root``) reads the existing
    ``root_options`` key, a :class:`RootOptions` or its mapping, with the defaults
    where omitted; any other joint fit refuses it.
    """
    allowed = tuple(
        set(OPTION_NAMES)
        | set(ABSORBED_DOMAINS)
        | {
            "solver_options",
            "absorbed_options",
            "root_options",
            "controls",
            "sample_request",
            "max_dense_bytes",
        }
    )
    raw = _option_mapping(solver_options, allowed, "solver_options")
    absorbed = _absorbed_options(raw)
    solver = _solver_options(raw, tol)
    controls = raw.get("controls")
    sample_request = raw.get("sample_request")
    max_dense_bytes = _dense_limit(raw.get("max_dense_bytes"))
    if cancel is not None:
        if "max_dense_bytes" in raw:
            raise _conflicting("max_dense_bytes", "cancel")
        if not isinstance(controls, CancelledFitControls):
            raise _invalid("controls", controls, "explicit CancelledFitControls")
        if sample_request is not None:
            raise _conflicting("sample_request", "cancel")
    else:
        if controls is not None:
            raise _conflicting("controls", "joint")
        if not isinstance(sample_request, SampleRequest):
            raise _invalid("sample_request", sample_request, "explicit SampleRequest")
    root = raw.get("root_options")
    if root is not None and not isinstance(root, (RootOptions, Mapping)):
        raise _invalid("root_options", root, "RootOptions or a mapping")
    if cancel is None and root is not None and not selected_root:
        raise _conflicting("root_options", "joint")
    resolved = None if root is None else resolve_root_options(root)
    if cancel is None and selected_root and resolved is None:
        resolved = RootOptions()
    return ResolvedFESolverOptions(
        solver,
        absorbed,
        resolved,
        True,
        controls,
        sample_request,
        max_dense_bytes,
    )
