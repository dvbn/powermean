"""Full-coordinate implicit cancelled derivative actions (design03 5.1-5.2)."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.fe._cancel_storage import IndexArray, finite_product, finite_vector
from powermean.fe.cancel import CancelStructure, segment_sum, to_original, to_sorted
from powermean.fe.retained import CenteredDesign
from powermean.inference.correction._inputs import checked_array
from powermean.inference.correction._row_derivatives import (
    BiasRowPrimitives,
    CellRowPrimitives,
)
from powermean.inference.correction.budget import (
    OperatorWorkspacePlan,
)
from powermean.systems.cancelled import CancelledState

# Row pairs handled at once by the within-group forms: a batch buffer only; it
# selects no tolerance, rank, sample or estimator.
PAIR_CHUNK = 1 << 18

# Largest magnitude of an input for which neither the within-group forms nor
# the column loops they replace can overflow: every quantity of either is a sum
# of fewer than 1e90 products of at most seven inputs.
SAFE_SCALE = 1e30


class _Pairs(NamedTuple):
    """The ordered pairs ``(i, j)`` of sorted rows of some whole groups.

    ``left`` and ``right`` are the rows of each pair, ``left`` nondecreasing.
    By row of the chunk: ``offsets`` is its first pair, ``diagonal`` its pair
    with itself, ``sizes`` the size of its group and ``groups`` that group's
    index in the chunk. ``first`` is the chunk's first sorted row.
    """

    first: int
    left: IndexArray
    right: IndexArray
    offsets: IndexArray
    diagonal: IndexArray
    sizes: IndexArray
    groups: IndexArray


def _pair_chunks(structure: CancelStructure) -> Iterator[_Pairs]:
    """Every ordered pair of rows of one group, whole groups at a time."""
    segments, counts = structure.segments, structure.counts
    cumulative = np.cumsum(counts * counts)
    group = 0
    while group < counts.size:
        done = int(cumulative[group - 1]) if group else 0
        stop = int(np.searchsorted(cumulative, done + PAIR_CHUNK, side="right"))
        stop = max(stop, group + 1)
        first, last = int(segments[group]), int(segments[stop])
        codes = structure.sorted_codes[first:last]
        sizes = counts[codes]
        offsets = np.cumsum(sizes) - sizes
        heads = segments[codes]
        rows = np.arange(first, last)
        left = np.repeat(rows, sizes)
        right = np.repeat(heads - offsets, sizes) + np.arange(left.size)
        diagonal = offsets + rows - heads
        yield _Pairs(first, left, right, offsets, diagonal, sizes, codes - group)
        group = stop


def _centered(
    values: NDArray[np.float64], pairs: _Pairs, *, over_left: bool, over_right: bool
) -> NDArray[np.float64]:
    """Subtract within-group means of pair values over ``i``, over ``j`` or both."""
    rows = pairs.sizes.size
    result = values.copy()
    if over_right:
        local = pairs.left - pairs.first
        row_sums = np.add.reduceat(values, pairs.offsets)
        result -= (row_sums / pairs.sizes)[local]
    if not over_left:
        return result
    other = pairs.right - pairs.first
    sums = np.bincount(other, weights=values, minlength=rows)
    result -= (sums / pairs.sizes)[other]
    if over_right:
        totals = np.bincount(pairs.groups, weights=row_sums)[pairs.groups]
        result += (totals / (pairs.sizes * pairs.sizes))[local]
    return result


def _pair_forms(
    design: CenteredDesign,
    matrix: NDArray[np.float64],
    index: IndexArray | None = None,
) -> Iterator[tuple[_Pairs, NDArray[np.float64]]]:
    """Yield chunks of ``b_i' M b_j`` for rows ``b`` of the centered design.

    A row of ``B`` is its centered slopes and its retained indicators minus
    their group means. The form is therefore gathered from ``M`` at the levels
    of the two rows and centered within the group over ``j`` in its
    slope-by-level part, over ``i`` in its level-by-slope part and over both in
    its level-by-level part; no row-by-column array is formed. With ``index``,
    the entry of ``M`` at full columns ``(a, b)`` is ``matrix[index[a], index[b]]``.
    """
    structure, k = design.structure, design.n_slopes
    columns = [
        start + codes
        for codes, start in zip(
            structure.retained_sorted_codes, design.block_starts, strict=True
        )
    ]
    slope_rows = np.arange(k)
    if index is not None:
        columns = [index[a] for a in columns]
        slope_rows = index[:k]
    slopes = design.centered_slopes
    upper, left = matrix[slope_rows], matrix[:, slope_rows]
    slope_square = upper[:, slope_rows]

    def chunk_forms(pairs: _Pairs) -> NDArray[np.float64]:
        i, j = pairs.left, pairs.right
        levels = np.zeros(i.size)
        for a in columns:
            for b in columns:
                levels += matrix[a[i], b[j]]
        forms = _centered(levels, pairs, over_left=True, over_right=True)
        if k:
            si, sj = slopes[i], slopes[j]
            forms += np.einsum("ik,kl,il->i", si, slope_square, sj)
            by_level, by_slope = np.zeros(i.size), np.zeros(i.size)
            for a in columns:
                by_level += np.einsum("ik,ki->i", si, upper[:, a[j]])
                by_slope += np.einsum("ik,ik->i", left[a[i]], sj)
            forms += _centered(by_level, pairs, over_left=False, over_right=True)
            forms += _centered(by_slope, pairs, over_left=True, over_right=False)
        return forms

    for pairs in _pair_chunks(structure):
        yield pairs, chunk_forms(pairs)


def group_forms(
    design: CenteredDesign,
    matrix: NDArray[np.float64],
    weights: NDArray[np.float64],
    index: IndexArray | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Within-group forms of a full-coordinate matrix ``M``, by sorted row ``i``.

    Returns ``b_i' M b_i``, ``sum_j (b_i' M b_j) w_j`` and
    ``sum_j w_j (b_j' M b_i)``, the sums over the rows ``j`` of the group of
    ``i`` and ``b`` the rows of the centered design. These are the row sums that
    a loop over the columns of ``M`` accumulates from design products, formed
    here from the entries of ``M`` that each group touches. With ``index``,
    ``M[a, b]`` is ``matrix[index[a], index[b]]``
    (:meth:`~powermean.fe.geometry_design.NumericalGeometry.gathered`).
    """
    n = design.structure.n_rows
    diagonal, right, left = np.empty(n), np.empty(n), np.empty(n)
    for pairs, forms in _pair_forms(design, matrix, index):
        rows = pairs.sizes.size
        chunk = slice(pairs.first, pairs.first + rows)
        diagonal[chunk] = forms[pairs.diagonal]
        right[chunk] = np.add.reduceat(forms * weights[pairs.right], pairs.offsets)
        left[chunk] = np.bincount(
            pairs.right - pairs.first,
            weights=forms * weights[pairs.left],
            minlength=rows,
        )
    return diagonal, right, left


def group_right_form(
    design: CenteredDesign,
    matrix: NDArray[np.float64],
    weights: NDArray[np.float64],
    index: IndexArray | None = None,
) -> NDArray[np.float64]:
    """Compute only the right-weighted part of ``group_forms``."""
    result = np.empty(design.structure.n_rows)
    for pairs, forms in _pair_forms(design, matrix, index):
        chunk = slice(pairs.first, pairs.first + pairs.sizes.size)
        result[chunk] = np.add.reduceat(forms * weights[pairs.right], pairs.offsets)
    return result


def _scatter(
    design: CenteredDesign,
    values: NDArray[np.float64],
    pairs: _Pairs,
    out: NDArray[np.float64],
) -> None:
    """Add ``sum_pairs values_ij b_i b_j'`` to ``out``, the transpose of gathering."""
    structure, k = design.structure, design.n_slopes
    i, j = pairs.left, pairs.right
    blocks = list(
        zip(
            design.block_starts,
            structure.block_sizes,
            structure.retained_sorted_codes,
            strict=True,
        )
    )
    both = _centered(values, pairs, over_left=True, over_right=True)
    for a_start, a_size, a_codes in blocks:
        for b_start, b_size, b_codes in blocks:
            cells = a_codes[i] * b_size + b_codes[j]
            block = np.bincount(cells, weights=both, minlength=a_size * b_size)
            out[a_start : a_start + a_size, b_start : b_start + b_size] += (
                block.reshape(a_size, b_size)
            )
    if not k:
        return
    slopes = design.centered_slopes
    si, sj = slopes[i], slopes[j]
    out[:k, :k] += (si * values[:, None]).T @ sj
    over_j = _centered(values, pairs, over_left=False, over_right=True)
    over_i = _centered(values, pairs, over_left=True, over_right=False)
    for start, size, codes in blocks:
        span = slice(start, start + size)
        for column in range(k):
            out[column, span] += np.bincount(
                codes[j], weights=si[:, column] * over_j, minlength=size
            )
            out[span, column] += np.bincount(
                codes[i], weights=sj[:, column] * over_i, minlength=size
            )


def group_congruence(
    design: CenteredDesign,
    diagonal: NDArray[np.float64] | None,
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``B' (diag(a) + sum_g u_g v_g') B`` as a full p by p matrix.

    ``a`` is ``diagonal`` (absent for zero), ``u`` and ``v`` are ``left`` and
    ``right`` in sorted rows, and ``u_g v_g'`` couples the rows of one group.
    The matrix is what a loop over unit columns assembles from design products
    and group sums; here each group adds its entries at the levels it touches.
    """
    p = design.n_columns
    out = np.zeros((p, p))
    for pairs in _pair_chunks(design.structure):
        values = left[pairs.left] * right[pairs.right]
        if diagonal is not None:
            rows = slice(pairs.first, pairs.first + pairs.sizes.size)
            values[pairs.diagonal] += diagonal[rows]
        _scatter(design, values, pairs, out)
    return out


def _second_zeta_sorted(
    state: CancelledState,
    primitives: BiasRowPrimitives,
    du_sorted: NDArray[np.float64],
    dv_sorted: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute uncentered sorted row curvature terms ``zeta_i(du, dv)``."""
    structure = state.design.structure
    du_sorted = checked_array(du_sorted, (structure.n_rows,), "du_sorted")
    dv_sorted = checked_array(dv_sorted, (structure.n_rows,), "dv_sorted")
    codes = structure.sorted_codes
    k_vec = state.k
    with np.errstate(over="raise", invalid="raise"):
        s_ku = segment_sum(structure, finite_product(k_vec * du_sorted))
        s_kv = segment_sum(structure, finite_product(k_vec * dv_sorted))
        coupled = finite_product(dv_sorted * s_ku[codes] + du_sorted * s_kv[codes])
        direct = finite_product(-primitives.k_dot * du_sorted * dv_sorted)
        return finite_product(direct + primitives.eta_weights * coupled)


@dataclass(frozen=True, slots=True)
class CancelledBiasState:
    """Fresh-point second-derivative and cell operator in cancelled coordinates."""

    state: CancelledState
    primitives: BiasRowPrimitives
    _cell: CellRowPrimitives | None
    workspace_plan: OperatorWorkspacePlan

    @property
    def cell_primitives(self) -> CellRowPrimitives:
        """Cell data exist only after an explicit cell-capable evaluation."""
        if self._cell is None:
            raise ValueError(
                "cell primitives were not requested; evaluate with cell=True"
            )
        return self._cell

    @property
    def design(self) -> CenteredDesign:
        """Underlying group-centered design ``B = M_C[X, F_R]``."""
        return self.state.design

    @property
    def group_cancel_scale(self) -> NDArray[np.float64]:
        """Unweighted cancellation scale ``A_hat_g = n_g^-1 sum_{i in g} Z_i/m_i``."""
        return self.cell_primitives.group_cancel_scale

    @property
    def fitted_mean_original(self) -> NDArray[np.float64]:
        """After-cancellation fitted mean ``M_hat_i = A_hat_{g(i)} m_hat_i``."""
        return to_original(self.design.structure, self.cell_primitives.fitted_mean)

    @property
    def cell_residual_original(self) -> NDArray[np.float64]:
        """After-cancellation cell residual ``e_hat_i = Z_i - M_hat_i``."""
        return to_original(self.design.structure, self.cell_primitives.cell_residual)

    def pi_centered_sorted(self, values: ArrayLike) -> NDArray[np.float64]:
        """Compute ``u_i' v = (B_i - Bbar_g)' v`` in sorted group order."""
        structure = self.design.structure
        image = self.design.matvec_sorted(values)
        with np.errstate(over="raise", invalid="raise"):
            means = segment_sum(structure, finite_product(self.state.pi * image))
            return finite_product(image - means[structure.sorted_codes])

    def group_adjoint_linear_weights(self, adjoint: ArrayLike) -> NDArray[np.float64]:
        """Compute sorted linear adjoint weights ``gamma_g(w)_i = -k_i * w_hat_i``."""
        w_hat = self.pi_centered_sorted(adjoint)
        with np.errstate(over="raise", invalid="raise"):
            return finite_product(-self.state.k * w_hat)

    def second_derivative_sorted(
        self, du_sorted: NDArray[np.float64], dv_sorted: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Evaluate ``D^2 rho[du, dv]`` in sorted group order."""
        structure = self.design.structure
        zeta = _second_zeta_sorted(self.state, self.primitives, du_sorted, dv_sorted)
        with np.errstate(over="raise", invalid="raise"):
            zeta_sums = segment_sum(structure, zeta)
            return finite_product(
                zeta - self.state.pi * zeta_sums[structure.sorted_codes]
            )

    def second_derivative_vec(
        self, left: ArrayLike, right: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate ``D^2 S(theta)[left, right]`` in full column coordinates."""
        du = self.design.matvec_sorted(left)
        dv = self.design.matvec_sorted(right)
        return self.design.rmatvec_sorted(self.second_derivative_sorted(du, dv))

    def group_second_derivative_form(
        self, adjoint: ArrayLike, left: ArrayLike, right: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate ``w' D^2 psi_g(theta)[u, v]`` for each cancelled group ``g``."""
        w_hat = self.pi_centered_sorted(adjoint)
        du = self.design.matvec_sorted(left)
        dv = self.design.matvec_sorted(right)
        zeta = _second_zeta_sorted(self.state, self.primitives, du, dv)
        with np.errstate(over="raise", invalid="raise"):
            return segment_sum(self.design.structure, finite_product(w_hat * zeta))

    def cell_score_dot(self, values: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``f_i(theta)' v = lambda_i u_i' v`` in original row order."""
        u_dot_v = self.pi_centered_sorted(values)
        with np.errstate(over="raise", invalid="raise"):
            scores_sorted = finite_product(
                self.cell_primitives.lambda_weights * u_dot_v
            )
        return to_original(self.design.structure, scores_sorted)

    def cell_score_rmatvec(self, row_weights: ArrayLike) -> NDArray[np.float64]:
        """Evaluate ``sum_i row_weights_i f_i(theta)`` without an ``n x p`` matrix."""
        s = self.design.structure
        alpha = to_sorted(s, finite_vector(row_weights, s.n_rows, "row_weights"))
        with np.errstate(over="raise", invalid="raise"):
            mu = finite_product(self.cell_primitives.lambda_weights * alpha)
            mu_sums = segment_sum(s, mu)
            centered = finite_product(mu - self.state.pi * mu_sums[s.sorted_codes])
        return self.design.rmatvec_sorted(centered)

    def cell_derivative_vec(
        self, row_weights: ArrayLike, values: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate ``sum_i row_weights_i (nabla f_i(theta) v)`` in ``O(n)`` work."""
        s = self.design.structure
        p = self.cell_primitives
        alpha = to_sorted(s, finite_vector(row_weights, s.n_rows, "row_weights"))
        bv = self.design.matvec_sorted(values)
        with np.errstate(over="raise", invalid="raise"):
            mu = finite_product(p.lambda_weights * alpha)
            mu_group = segment_sum(s, mu)
            term1 = finite_product(p.u_lambda_weights * alpha)
            term2 = finite_product(
                mu_group[s.sorted_codes] * self.primitives.eta_weights
            )
            xi = finite_product((term1 + term2) * bv)
            xi_sums = segment_sum(s, xi)
            centered = finite_product(xi - self.state.pi * xi_sums[s.sorted_codes])
            return finite_product(-self.design.rmatvec_sorted(centered))

    def cell_derivative_form(
        self, left: ArrayLike, right: ArrayLike
    ) -> NDArray[np.float64]:
        """Evaluate ``left' nabla f_i(theta) right`` at each original row ``i``."""
        s = self.design.structure
        p = self.cell_primitives
        w_hat = self.pi_centered_sorted(left)
        bv = self.design.matvec_sorted(right)
        with np.errstate(over="raise", invalid="raise"):
            direct = finite_product(-p.u_lambda_weights * w_hat * bv)
            coupling = segment_sum(
                s, finite_product(self.primitives.eta_weights * w_hat * bv)
            )
            indirect = finite_product(-p.lambda_weights * coupling[s.sorted_codes])
            return to_original(s, finite_product(direct + indirect))


def gathers_apply(
    bias: CancelledBiasState, n_identified: int, *matrices: NDArray[np.float64]
) -> bool:
    """Whether the within-group forms stand in for the column loops here.

    They do when they gather fewer entries than the row-by-column products of
    the loops hold, and when every input, ``matrices`` included, is at most
    ``SAFE_SCALE`` in magnitude, so that the loops would raise nothing either.
    Otherwise the loops run: they decide, raise and report as before.
    """
    design, state, primitives = bias.design, bias.state, bias.primitives
    structure = design.structure
    counts = structure.counts
    width = design.n_slopes + len(structure.block_sizes)
    gathered = int(np.sum(counts * counts)) * width * width
    if gathered > structure.n_rows * n_identified:
        return False
    inputs = (
        state.k,
        state.rho,
        primitives.k_dot,
        primitives.eta_weights,
        design.centered_slopes,
        *matrices,
    )
    return all(
        values.size == 0 or float(np.abs(values).max()) <= SAFE_SCALE
        for values in inputs
    )
