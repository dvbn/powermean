"""Aggregate exogenous difference moments and local root certification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from powermean.errors import ConvergenceError, IdentificationError

FloatArray = NDArray[np.float64]
_RANK_TOL = 1e-12
_ROOT_TOL = 1e-10
_RANGE = 150.0
_EPS = np.finfo(np.float64).eps


def _triple(left: FloatArray, middle: FloatArray, right: FloatArray) -> FloatArray:
    """Evaluate L M' R with the smaller square intermediate."""
    if left.shape[0] <= left.shape[1]:
        return (left @ middle.T) @ right
    return left @ (middle.T @ right)


def _within(grid: FloatArray, directed: bool) -> FloatArray:
    """Subtract the least-squares additive row and column effects."""
    rows, cols = grid.sum(axis=1), grid.sum(axis=0)
    nr, nc = grid.shape[:2]
    total = rows.sum(axis=0)
    if directed:
        # The two normal equations omit the diagonal. The column effects sum to zero.
        denominator = nr * (nr - 2)
        left = ((nr - 1) * rows + cols - total / (nr - 1)) / denominator
        right = (rows + (nr - 1) * cols - total) / denominator
        residual = grid - left[:, None] - right[None, :]
        diagonal = np.arange(nr)
        residual[diagonal, diagonal] = 0.0
        return cast("FloatArray", residual)
    return cast(
        "FloatArray", grid - rows[:, None] / nc - cols[None, :] / nr + total / (nr * nc)
    )


def _check_design(grid: FloatArray, directed: bool) -> FloatArray:
    """Check the rank of the identifying contrasts and scale the instruments."""
    within = _within(grid, directed)
    flat = within.reshape(-1, grid.shape[-1])
    scale = np.max(np.abs(flat), axis=0)
    resolved = scale > _RANK_TOL
    standardized = flat[:, resolved] / scale[resolved]
    singular = np.linalg.svd(standardized, compute_uv=False)
    rank = 0
    if singular.size:
        # The input grid has unit-scale columns. Centering and removing the
        # effects can leave much smaller contrasts, magnifying projection
        # roundoff when those contrasts are rescaled to unit size.
        error = 32 * _EPS * max(grid.shape[:2]) * np.sqrt(flat.shape[0])
        error *= np.linalg.norm(1.0 / scale[resolved])
        rank = int(np.count_nonzero(singular > max(_RANK_TOL * singular[0], error)))
    if rank != grid.shape[-1]:
        raise IdentificationError(
            "identification.rank_deficient", {"rank": rank, "columns": grid.shape[-1]}
        )
    return cast("FloatArray", np.moveaxis(within / scale, -1, 0).copy())


@dataclass(frozen=True)
class _Evaluation:
    """Unnormalized focal sums and factors at a coefficient vector."""

    moment: FloatArray
    a: FloatArray
    b: FloatArray
    u: FloatArray
    v: FloatArray
    mass: FloatArray


class DifferenceMoments:
    """Evaluate both published moments on a validated cell grid."""

    def __init__(
        self, y: FloatArray, x: FloatArray, mask: NDArray[np.bool_], *, gmm1: bool
    ) -> None:
        self.mask = mask
        self.ones = mask.astype(np.float64)
        self.gmm1 = gmm1
        directed = not bool(mask.all())
        self.unit_b = gmm1 and not directed
        self.x = x
        self.t = _check_design(x, directed)
        self.roundoff = 32 * _EPS * max(mask.shape)
        active = (y > 0).astype(np.float64)
        # Count ordered nonzero same-diagonal products on admissible rectangles.
        # The product with the mask enforces available cross-links. Inclusion-
        # exclusion removes the repeated rows and columns.
        rows, cols = active.sum(axis=1), active.sum(axis=0)
        count = rows.sum()
        admissible = (
            np.sum(active * _triple(self.ones, active, self.ones))
            if directed
            else count**2
        )
        if admissible - rows @ rows - cols @ cols + count <= 0:
            raise IdentificationError("identification.jochmans_no_products")
        _, exponent = np.frexp(y.max())
        self.y = np.ldexp(y, -exponent)
        if np.any(self.y[active != 0] < np.exp(-_RANGE)):
            raise _failure("numerical_domain", None)

    def evaluate(self, theta: FloatArray) -> _Evaluation:
        """Build factors and the moment using one common outcome normalization."""
        index = self.x @ theta
        if not np.isfinite(index).all() or np.max(np.abs(index)) > _RANGE:
            raise FloatingPointError("Jochmans index outside the numerical domain")
        a = self.y * np.exp(-index) if self.gmm1 else self.y
        b = self.ones if self.gmm1 else self.ones * np.exp(index)
        if self.unit_b:
            u = np.full_like(a, a.sum())
            v = a.sum(axis=1)[:, None] * a.sum(axis=0)[None, :]
        else:
            u, v = _triple(b, a, b), _triple(a, b, a)
        positive, negative = a * u, b * v
        moment = self.t.reshape(len(theta), -1) @ (positive - negative).ravel()
        return _Evaluation(moment, a, b, u, v, positive + negative)

    def derivatives(
        self, state: _Evaluation, *, project: bool = False
    ) -> tuple[FloatArray, FloatArray | None]:
        """Contract the exact Jacobian and optionally form the incident sums."""
        a, b, u, v = state.a, state.b, state.u, state.v
        k = len(self.t)
        jacobian = np.empty((k, k))
        incidence = np.empty((k, self.mask.size)) if project else None
        flat_x = self.x.reshape(-1, k)
        if self.unit_b:
            rows, cols = a.sum(axis=1), a.sum(axis=0)
        for j, t in enumerate(self.t):
            ta, tb = t * a, t * b
            a_gradient = b_gradient = None
            if self.gmm1 or project:
                if self.unit_b:
                    a_gradient = t * u + ta.sum() - (t @ cols)[:, None] - rows @ t
                else:
                    a_gradient = t * u + _triple(b, ta, b)
                    a_gradient -= _triple(tb, a, b) + _triple(b, a, tb)
                a_gradient *= a
            if not self.gmm1 or project:
                b_gradient = _triple(ta, b, a) + _triple(a, b, ta)
                b_gradient -= t * v + _triple(a, tb, a)
                b_gradient *= b
            gradient = a_gradient if self.gmm1 else b_gradient
            assert gradient is not None
            sign = -1 if self.gmm1 else 1
            jacobian[j] = sign * (gradient.ravel() @ flat_x)
            if incidence is not None:
                assert a_gradient is not None
                assert b_gradient is not None
                incidence[j] = (a_gradient + b_gradient).ravel()
        return jacobian, incidence

    def direction(self, state: _Evaluation) -> FloatArray:
        """Solve the Newton equation after checking resolved slope information."""
        jacobian, _ = self.derivatives(state)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        floor = max(_RANK_TOL * singular[0], self.roundoff * state.mass.sum())
        if singular[-1] <= floor:
            raise IdentificationError("identification.jochmans_jacobian")
        return np.asarray(np.linalg.solve(jacobian, state.moment), dtype=np.float64)

    def covariance(self, state: _Evaluation, scale: FloatArray) -> FloatArray:
        """Use all incident terms and the actual Jacobian for the sandwich."""
        if self.mask.shape == (2, 2):
            # Each cell's projection equals the sole rectangle's moment.
            # It vanishes at the root and only measures residual error nearby.
            raise FloatingPointError("One rectangle cannot support covariance")
        jacobian, incidence = self.derivatives(state, project=True)
        assert incidence is not None
        # J = J_raw/M and h = N * incidence/M, so the factors cancel in h/N.
        # A bound for numerical noise in the incident sums uses the absolute
        # product mass (including terms that cancel) and the contraction size.
        mass = state.mass.ravel()
        peak = mass.max()
        rounding = self.roundoff * peak * np.linalg.norm(mass / peak)
        singular = np.linalg.svd(incidence, compute_uv=False)
        if singular[-1] <= max(rounding, _RANK_TOL * singular[0]):
            raise FloatingPointError(
                "Jochmans cell projections lack resolved full rank"
            )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            influence = np.linalg.solve(jacobian, incidence)
            influence = influence / scale[:, None]
            covariance = influence @ influence.T
        sd = np.sqrt(covariance.diagonal())
        if not np.isfinite(covariance).all() or np.any(
            covariance.diagonal() < 2 * np.finfo(np.float64).tiny
        ):
            raise FloatingPointError("Jochmans covariance has an unrepresentable entry")
        # Test positive definiteness on the correlation scale to allow different
        # regressor units and keep the small eigenvalue above rounding noise.
        eigenvalues = np.linalg.eigvalsh((covariance / sd[:, None]) / sd)
        if eigenvalues[0] <= 32 * _EPS * len(sd) * eigenvalues[-1]:
            raise FloatingPointError("Jochmans covariance lacks resolved full rank")
        return covariance


def _failure(
    status: str, last_state: FloatArray | None, *, certificate: LocalRoot | None = None
) -> ConvergenceError:
    """Refuse a fit at its last accepted point, on the reported slope scale."""
    return ConvergenceError(
        "convergence.jochmans",
        status=status,
        last_state=last_state,
        certificate=certificate,
    )


@dataclass(frozen=True)
class LocalRoot:
    """Residual checks at the returned local root, in scaled coordinates."""

    iterations: int
    relative_moment: float
    correction: float
    tolerance: float = _ROOT_TOL
    status: str = "local_solution"


def solve(
    moments: DifferenceMoments, start: FloatArray, scale: FloatArray, max_iter: int
) -> tuple[FloatArray, _Evaluation, LocalRoot]:
    """Damped Newton with both moment and local correction checks at the endpoint."""
    with np.errstate(over="ignore", invalid="ignore"):
        theta = start * scale
        try:
            state = moments.evaluate(theta)
        except FloatingPointError as error:
            raise _failure("numerical_domain", start) from error
    for iteration in range(max_iter + 1):
        step = moments.direction(state)
        norm = float(np.max(np.abs(state.moment)))
        relative_moment = norm / float(state.mass.sum())
        correction = float(np.max(np.abs(step)))
        if relative_moment <= _ROOT_TOL and correction <= _ROOT_TOL:
            return theta, state, LocalRoot(iteration, relative_moment, correction)
        if iteration == max_iter:
            break
        for backtrack in range(30):
            fraction = 0.5**backtrack
            trial = theta - fraction * step
            try:
                proposed = moments.evaluate(trial)
            except FloatingPointError:
                continue
            if np.max(np.abs(proposed.moment)) < (1 - 1e-4 * fraction) * norm:
                theta, state = trial, proposed
                break
        else:
            raise _failure("line_search_failed", theta / scale)
    raise _failure("iteration_limit", theta / scale)
