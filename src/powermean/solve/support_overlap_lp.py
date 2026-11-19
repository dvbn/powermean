"""A single auxiliary LP for an empty separation set on the current FE rows."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from powermean.fe.groups import Grouping
from powermean.model.variance import FloatArray
from powermean.solve.support_lp import METHOD, LinprogResult, LPControls
from powermean.solve.support_problem import NONE, SupportProblem
from powermean.solve.support_witness import DualWitness, dual_witness

if TYPE_CHECKING:
    from powermean.solve.support_gamma_lp import Sparse, SparseMatrix


@dataclass(frozen=True)
class OverlapLP:
    """Work on a certified overlap, including the discarded second IR attempt."""

    iterations: int
    set_aside_reason: str
    set_aside_attempts: int
    set_aside_sweeps: int
    backend: str = METHOD


class _Optimize(Protocol):
    def linprog(
        self,
        c: FloatArray,
        a_ub: None,
        b_ub: None,
        a_eq: SparseMatrix,
        b_eq: FloatArray,
        /,
        *,
        bounds: FloatArray,
        method: str,
        options: Mapping[str, float | int | bool],
    ) -> LinprogResult: ...


def _equations(
    problem: SupportProblem, grouping: Grouping, n_equations: int
) -> SparseMatrix:
    """Build sparse ``B.T`` with each original slope and every encoded level."""
    rows, slopes = np.nonzero(problem.design)
    observations, equations = [rows], [slopes]
    values = [problem.design[rows, slopes]]
    n = problem.n_rows
    row_ids = np.arange(n, dtype=np.intp)
    ones = np.ones(n)
    offset = problem.n_slopes
    for block in grouping.blocks:
        observations.append(row_ids)
        equations.append(offset + block.codes)
        values.append(ones)
        offset += block.n_levels
    sparse = cast("Sparse", importlib.import_module("scipy.sparse"))
    return sparse.csr_array(
        (
            np.concatenate(values),
            (np.concatenate(equations), np.concatenate(observations)),
        ),
        shape=(n_equations, n),
    )


def certify_overlap(
    problem: SupportProblem,
    controls: LPControls,
    *,
    workspace: int,
    tolerance: float,
    threshold: float,
) -> tuple[DualWitness, int] | None:
    """Find ``B.T @ h = 0`` with ``h >= 1`` on all zero rows, then check it afresh.

    The workspace is in loading-element units. Charge the dense-equivalent shape
    ``n * (p + sum(levels))`` plus one slot per variable and equality. The charge
    bounds the sparse entry count and both vector dimensions, including the level
    coordinates, before allocating any solver input. The actual program is sparse.
    """
    grouping = problem.grouping
    if grouping is None:
        return None
    n = problem.n_rows
    m = problem.n_slopes + sum(block.n_levels for block in grouping.blocks)
    if n * (m + 1) + m > workspace:
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            zero = problem.z == 0
            if not zero.any():
                return None
            bounds = np.empty((n, 2))
            bounds[:, 0] = -np.inf
            bounds[:, 1] = np.inf
            bounds[zero, 0] = 1.0
            optimize = cast("_Optimize", importlib.import_module("scipy.optimize"))
            result = optimize.linprog(
                np.zeros(n),
                None,
                None,
                _equations(problem, grouping, m),
                np.zeros(m),
                bounds=bounds,
                method=METHOD,
                options=controls.options,
            )
            success: object = getattr(result, "success", None)
            status: object = getattr(result, "status", None)
            iterations: object = getattr(result, "nit", None)
            point: object = getattr(result, "x", None)
        except Exception:
            # This optional backend can fail to load, allocate or solve.
            return None
    if caught:
        return None

    if (
        not isinstance(success, bool | np.bool_)
        or not success
        or not isinstance(status, int | np.integer)
        or isinstance(status, bool)
        or status != 0
        or not isinstance(iterations, int | np.integer)
        or isinstance(iterations, bool)
        or not 0 <= iterations <= controls.max_iterations
        or not isinstance(point, np.ndarray)
        or point.shape != (n,)
        or point.dtype.kind != "f"
        or point.dtype.itemsize > 8
        or not np.isfinite(point).all()
        or (point[zero] < 1.0).any()
    ):
        return None
    dual = dual_witness(problem, point, NONE, tolerance=tolerance, threshold=threshold)
    return (dual, int(iterations)) if dual.passed else None
