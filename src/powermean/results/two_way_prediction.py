"""Fixed-slope PPML recovery on the cells observed by a two-way GMM fit."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from powermean.data.accounting import frozen_index
from powermean.errors import ConvergenceError
from powermean.fe.groups import Grouping, encode_blocks
from powermean.fe.profile import profile
from powermean.fe.profile_arithmetic import prepare_profile
from powermean.fe.reductions import immutable_array
from powermean.formula.frame import Design
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray, PowerVariance
from powermean.results.prediction import boundary_delta, slope_index
from powermean.solve.m1_options import AbsorbedOptions


def _support(
    rows: IndexArray, columns: IndexArray, positive: NDArray[np.bool_], nr: int, nc: int
) -> tuple[NDArray[np.bool_], IndexArray]:
    """Select edges that can carry positive mass with the observed margins.

    Starting from the observed feasible flow, every edge can gain mass and
    every positive edge can also lose mass. A zero edge can gain mass exactly
    when a return path closes a directed cycle. SCCs find all such edges at once.
    """
    destinations = columns + nr
    source = np.concatenate((rows, destinations[positive]))
    target = np.concatenate((destinations, rows[positive]))
    graph = csr_matrix(
        (np.ones(source.size, dtype=np.int8), (source, target)),
        shape=(nr + nc, nr + nc),
    )
    _, component = connected_components(graph, directed=True, connection="strong")
    return component[rows] == component[destinations], component.astype(np.intp)


@dataclass(frozen=True)
class TwoWayRecovery:
    """The index and normalized log-scale effects of the recovered mean."""

    q: FloatArray
    effects: tuple[FloatArray, FloatArray]


@dataclass(frozen=True)
class GMMIndex:
    """Replay the solver's centered, power-of-two-scaled slope coordinates."""

    exponent: NDArray[np.int32]
    origin: FloatArray
    mean: FloatArray
    amplitude: FloatArray
    theta: FloatArray

    def evaluate(self, matrix: FloatArray) -> FloatArray:
        """Evaluate replacement covariates in their fitted coordinates."""
        with np.errstate(over="raise", invalid="raise"):
            scaled = np.ldexp(matrix, -self.exponent) - self.origin - self.mean
            scaled /= self.amplitude
        return slope_index(scaled, self.theta)


@dataclass(frozen=True)
class TwoWayPrediction:
    """Compact retained inputs. Recovery starts only when the caller requests it."""

    y: FloatArray
    index: FloatArray
    rows: IndexArray
    columns: IndexArray
    levels: tuple[pd.Index, pd.Index]
    design: Design
    projection: GMMIndex
    common: float

    def __post_init__(self) -> None:
        """Own the previously mutable estimation arrays."""
        object.__setattr__(self, "y", immutable_array(self.y))
        object.__setattr__(self, "index", immutable_array(self.index))
        object.__setattr__(self, "rows", frozen_index(self.rows))
        object.__setattr__(self, "columns", frozen_index(self.columns))

    def counterfactual_delta(
        self, matrix: FloatArray, zero: NDArray[np.bool_] | None
    ) -> FloatArray:
        """Change the slope index without subtracting an uncentered common level."""
        with np.errstate(over="raise", invalid="raise"):
            if zero is None:
                return self.projection.evaluate(matrix) - self.index
            active = ~zero
            delta = boundary_delta(matrix)
            delta[active] = (
                self.projection.evaluate(matrix[active]) - self.index[active]
            )
            return delta

    @cached_property
    def recovered(self) -> TwoWayRecovery:
        """Balance the margins at the GMM slopes and keep exact boundary zeros."""
        if not np.isfinite(self.index).all():
            raise FloatingPointError("the fitted GMM slope index is not representable")
        nr, nc = map(len, self.levels)
        active, components = _support(self.rows, self.columns, self.y > 0, nr, nc)
        r, c = self.rows[active], self.columns[active]
        grouping = encode_blocks((r, c))
        variance = PowerVariance(1.0)
        controls = AbsorbedOptions(absorbed="profiled")
        shift = float(self.index[active][0])
        with np.errstate(over="raise", invalid="raise"):
            offset = self.index[active] - shift
        problem = prepare_profile(
            grouping,
            self.y[active],
            offset,
            np.ones(len(r)),
            variance=variance,
        )
        solved = profile(
            problem,
            tolerance=controls.profile_tol,
            max_sweeps=controls.profile_max_sweeps,
        )
        if not solved.converged:
            raise ConvergenceError(
                "convergence.prediction_recovery",
                {"reason": solved.reason},
                status=solved.status,
                last_state=None,
            )
        q = np.full(len(self.y), -np.inf)
        q[active] = solved.index
        left, right = _normalize(
            solved.state.coefficients,
            grouping,
            components,
            nr,
            nc,
            shift=shift + self.common,
        )
        return TwoWayRecovery(
            immutable_array(q), (immutable_array(left), immutable_array(right))
        )


def _normalize(
    coefficients: tuple[FloatArray, ...],
    grouping: Grouping,
    components: IndexArray,
    nr: int,
    nc: int,
    *,
    shift: float,
) -> tuple[FloatArray, FloatArray]:
    """Set the first active column effect in each component to zero."""
    r, c = (np.asarray(b.levels, dtype=np.intp) for b in grouping.blocks)
    left, right = np.full(nr, np.nan), np.full(nc, np.nan)
    left[r], right[c] = coefficients
    anchors = np.full(len(components), nc, dtype=np.intp)
    np.minimum.at(anchors, components[nr + c], c)
    constants = right[anchors.clip(max=nc - 1)]
    with np.errstate(over="ignore", invalid="ignore"):
        left[r] += constants[components[r]] - shift
        right[c] -= constants[components[nr + c]]
    return left, right
