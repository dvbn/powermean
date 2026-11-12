"""Same-point cancelled slope sandwiches from sum scores and the actual H."""

from __future__ import annotations

import importlib
import warnings
from typing import Protocol, cast

import numpy as np

from powermean.errors import CoverageWarning
from powermean.fe._cancel_storage import finite_product
from powermean.fe.admission_records import WorkLog
from powermean.fe.geometry_design import NumericalGeometry
from powermean.inference.absorbed_rank import nested_in
from powermean.inference.convention import ClusterCodes
from powermean.inference.ssc import (
    AppliedConvention,
    ConventionCovariance,
    direct_errors,
    primitive_estimate,
    scaled_covariance,
)
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.solve.cancelled_operator import AttemptBounds, workspace_plan
from powermean.solve.factorization import Factor, PivotArray
from powermean.systems.identified import IdentifiedState


class _TransposeSolve(Protocol):
    """Supported SciPy transpose solve on the existing equilibrated LU."""

    def __call__(
        self,
        lu_and_piv: tuple[FloatArray, PivotArray],
        rhs: FloatArray,
        /,
        *,
        trans: int,
    ) -> FloatArray:
        """Solve with the transposed factorized matrix, without another LU."""
        ...


_lu_solve = cast("_TransposeSolve", importlib.import_module("scipy.linalg").lu_solve)


def transpose_rhs(factor: Factor, projection: FloatArray) -> FloatArray:
    """W = H^(-T) L' from the CURRENT factor, for a generic nonsymmetric H.

    H = Ds Hs Ds implies H' = Ds Hs' Ds. This additive private wrapper changes
    neither the ported Factor.solve nor factorize. It forms no inverse.
    """
    d = factor.ds.size
    if projection.ndim != 2 or projection.shape[1] != d:
        raise ValueError("slope projection must have one column per equation")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        rhs = finite_product(projection.T / factor.ds[:, None])
        return finite_product(_lu_solve(factor.lu, rhs, trans=1) / factor.ds[:, None])


def reserve_inference(geometry: NumericalGeometry, bounds: AttemptBounds) -> None:
    """Prospectively reserve W, one group OR row matrix, and covariance workspace.

    This deliberately covers the more expensive crossing request even for a group
    consumer. No default builds duplicate n-by-K and G-by-K influence matrices.
    Shared point/geometry storage, Python objects, label copies and vendor-internal
    workspaces are excluded; array visits, not native scalar work/time, are bounded.
    """
    n, d, k = (
        geometry.structure.n_rows,
        geometry.n_identified,
        geometry.n_slopes_identified,
    )
    groups = geometry.structure.n_groups
    slots = 8 * d * k + 12 * n * k + 32 * k * k + 128 * (n + groups + d)
    required = 8 * slots + 32 * (n + groups + k)
    visits = (2 * k + 2) * workspace_plan(geometry.design, d).work
    visits += 16 * n * k * k + 32 * k * k * k
    if required > bounds.max_bytes or visits > bounds.max_work:
        raise ValueError("cancelled inference exceeds supplied byte/work bounds")


def slope_rhs(state: IdentifiedState, factor: Factor, work: WorkLog) -> FloatArray:
    """Compute only the kept-slope RHS columns, using the actual same-point factor."""
    k, d = state.design.n_slopes_identified, state.design.n_identified
    projection = np.zeros((k, d))
    projection[:, :k] = np.eye(k)
    return work.run("transpose_rhs", lambda: transpose_rhs(factor, projection))


def contracted_influence(
    state: IdentifiedState, rhs: FloatArray, *, rows: bool, work: WorkLog
) -> FloatArray:
    """Contract implicit group or original-row scores on W, one column at a time."""
    size = state.design.structure.n_rows if rows else state.design.structure.n_groups
    if rhs.shape != (state.design.n_identified, state.design.n_slopes_identified):
        raise ValueError("influence RHS does not match the retained geometry")
    work.begin("row_materialization" if rows else "group_materialization")
    result = np.empty((size, rhs.shape[1]))
    action = state.row_term_dot if rows else state.group_score_dot
    for j in range(rhs.shape[1]):

        def column_product(column: int = j) -> FloatArray:
            return action(rhs[:, column])

        result[:, j] = work.run(
            "row_term_dot" if rows else "group_score_dot", column_product
        )
    work.complete("row_materialization" if rows else "group_materialization")
    return finite_product(result)


def _units(
    state: IdentifiedState, rhs: FloatArray, cluster: ClusterCodes | None, work: WorkLog
) -> tuple[FloatArray, ClusterCodes | None]:
    structure = state.design.structure
    if cluster is None:
        return contracted_influence(state, rhs, rows=False, work=work), None
    codes = cluster.codes
    if codes.shape != (structure.n_rows,) or np.any(codes < 0):
        raise ValueError("cluster codes must align with every original positional row")
    if nested_in(structure.cancel_codes, codes):
        first = structure.order[structure.starts]
        group_cluster = ClusterCodes(cluster.name, codes[first])
        return contracted_influence(state, rhs, rows=False, work=work), group_cluster
    warnings.warn(
        CoverageWarning(
            "coverage.outside_bbp",
            {"case": "clustering with no dimension nesting the cancelled groups"},
        ),
        stacklevel=3,
    )
    return contracted_influence(state, rhs, rows=True, work=work), cluster


def cancelled_covariance(
    state: IdentifiedState,
    rhs: FloatArray,
    order: OrderScale,
    applied: AppliedConvention,
    cluster: ClusterCodes | None,
    *,
    work: WorkLog,
) -> tuple[ConventionCovariance, FloatArray, FloatArray | None, str | None]:
    """Group sampling-unit sandwich, or a one-way original-row cluster sandwich.

    Contributions and H are sums. The existing primitive's source-average adapter
    handles its own normalization. Neither weights nor n/G divisions are reapplied.
    The existing convention owns the single scalar correction. Direct SEs use the
    accepted range-safe map independently of reported covariance representability.
    """
    influence, bound = _units(state, rhs, cluster, work)
    cluster_arg = None if bound is None else (bound.name, bound.codes)
    estimate = work.run(
        "sandwich_primitive",
        lambda: primitive_estimate(
            influence, np.ones(influence.shape[0]), frequency=False, cluster=cluster_arg
        ),
    )
    covariance = scaled_covariance(estimate, applied)
    errors = work.run(
        "direct_errors", lambda: direct_errors(covariance.covariance, order)
    )
    reported: FloatArray | None = None
    failure: str | None = None
    try:
        reported = work.run(
            "report_covariance", lambda: order.covariance(covariance.covariance)
        )
    except FloatingPointError as error:
        failure = str(error)
    return covariance, errors, reported, failure
