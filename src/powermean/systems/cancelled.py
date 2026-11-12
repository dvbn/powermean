"""Fresh cancelled/mixed group moments and nonsymmetric Jacobian actions.

The cancelled score of BBP (``eq:fe-growing-cancel-score``).
For already transformed z and centered transformed offset c, q=B theta+c, with
B=M_C[X,F_R] and ordinary unweighted M_C:

    w=a m²/V, t=a mz/V, pi=w/W_g, rho=t-pi T_g, S=B' rho,
    k=(e-1)t+(2-e)(T_g/W_g)w, H=-dS/dtheta'.

H v=B'[k Bv-pi sum_g(k Bv)]; H' v=B'[k(Bv-sum_g(pi Bv))].
These actions are distinct. Inference must solve with the true transpose,
not a working-regression curvature. This module evaluates raw cancelled moments; it
defines no convex criterion or frequency-weight/zero-weight statistical policy.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import NamedTuple, Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.fe._cancel_storage import (
    BuilderRecord,
    FloatStorage,
    IndexArray,
    IndexStorage,
    finite_product,
    finite_vector,
)
from powermean.fe.cancel import CancelStructure, segment_sum, to_original, to_sorted
from powermean.fe.retained import CenteredDesign
from powermean.model.variance import FloatArray, Variance
from powermean.systems._cancel_rows import group_residuals, row_components

BoolArray = NDArray[np.bool_]


class CsrArray(Protocol):
    """Typed view of the CSR array from :meth:`CancelledState.group_scores_sparse`."""

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix dimensions."""
        ...

    @property
    def nnz(self) -> int:
        """Number of stored entries."""
        ...

    @property
    def data(self) -> FloatArray:
        """Stored values."""
        ...

    @property
    def indices(self) -> NDArray[np.int32] | NDArray[np.int64]:
        """Column indices of stored entries."""
        ...

    @property
    def indptr(self) -> NDArray[np.int32] | NDArray[np.int64]:
        """Row pointers of stored entries."""
        ...

    def toarray(self) -> FloatArray:
        """Dense array copy."""
        ...

    def __matmul__(self, other: FloatArray, /) -> FloatArray:
        """Dense matrix or vector product."""
        ...


class CooArray(Protocol):
    """Typed view of a SciPy COO array before CSR conversion."""

    def tocsr(self) -> CsrArray:
        """Convert to CSR format."""
        ...


class Sparse(Protocol):
    """The object of ``scipy.sparse`` used here."""

    def coo_array(
        self,
        arg1: tuple[FloatArray, tuple[ArrayLike, ArrayLike]],
        /,
        *,
        shape: tuple[int, int],
    ) -> CooArray:
        """A COO array of ``(data, (rows, cols))``."""
        ...


sparse = cast("Sparse", importlib.import_module("scipy.sparse"))


class CancelledRows(NamedTuple):
    """Detached q, w, t, pi, rho and k in input positional row order.

    No mean is exponentiated just for reporting: a finite q may have an
    unrepresentable exp(q) even when all requested moment terms are finite.
    """

    q: FloatArray
    w: FloatArray
    t: FloatArray
    pi: FloatArray
    rho: FloatArray
    k: FloatArray


class CancelledAccounting(NamedTuple):
    """Detached group flags in code order; no row or group is deleted."""

    n_rows: int
    n_groups: int
    singleton_groups: BoolArray
    all_zero_groups: BoolArray
    n_singleton_groups: int
    n_all_zero_groups: int
    n_rows_in_singleton_groups: int
    n_rows_in_all_zero_groups: int


@dataclass(frozen=True, eq=False, init=False)
class CancelledState(BuilderRecord):
    """An owned current-point evaluation, built by :func:`cancelled_equations`.

    Row-array properties are in sorted order and have detached view metadata.
    ``rows`` restores input positional order. No input point, weights or variance
    object is retained for later evaluation, and no cross-call cache is used.
    """

    __slots__ = (
        "_group_t",
        "_group_w",
        "_k",
        "_pi",
        "_positive_counts",
        "_q",
        "_rho",
        "_score",
        "_t",
        "_theta",
        "_w",
        "design",
    )

    design: CenteredDesign
    _theta: FloatStorage
    _q: FloatStorage
    _w: FloatStorage
    _t: FloatStorage
    _pi: FloatStorage
    _rho: FloatStorage
    _k: FloatStorage
    _group_w: FloatStorage
    _group_t: FloatStorage
    _positive_counts: IndexStorage
    _score: FloatStorage

    @property
    def theta(self) -> FloatArray:
        """Owned coefficients in full-column order."""
        return self._theta.view()

    @property
    def q(self) -> FloatArray:
        """Sorted transformed index."""
        return self._q.view()

    @property
    def w(self) -> FloatArray:
        """Sorted information weights."""
        return self._w.view()

    @property
    def t(self) -> FloatArray:
        """Sorted weighted relative outcomes, exactly zero where z is zero."""
        return self._t.view()

    @property
    def pi(self) -> FloatArray:
        """Sorted within-group w shares."""
        return self._pi.view()

    @property
    def rho(self) -> FloatArray:
        """Sorted cancelled residuals."""
        return self._rho.view()

    @property
    def k(self) -> FloatArray:
        """Sorted exact negative-Jacobian row weights."""
        return self._k.view()

    @property
    def group_w(self) -> FloatArray:
        """W_g in cancelled code order."""
        return self._group_w.view()

    @property
    def group_t(self) -> FloatArray:
        """T_g in cancelled code order."""
        return self._group_t.view()

    @property
    def positive_counts(self) -> IndexArray:
        """Number of positive transformed outcomes in each group."""
        return self._positive_counts.view()

    @property
    def score(self) -> FloatArray:
        """Unnormalized S=sum_g psi_g, in full-column coordinates."""
        return self._score.view()

    def _image(self, values: ArrayLike) -> FloatArray:
        """Checked Bv in segment order."""
        return self.design.matvec_sorted(values)

    def jac_vec(self, values: ArrayLike) -> FloatArray:
        """H v, where H=-dS/dtheta', including the group outer-product term."""
        structure = self.design.structure
        with np.errstate(over="raise", invalid="raise"):
            ku = self.k * self._image(values)
            sums = segment_sum(structure, ku)
            return self.design.rmatvec_sorted(
                ku - self.pi * sums[structure.sorted_codes]
            )

    def jac_t_vec(self, values: ArrayLike) -> FloatArray:
        """H' v, the actual adjoint, without a symmetry assumption."""
        structure = self.design.structure
        with np.errstate(over="raise", invalid="raise"):
            image = self._image(values)
            means = segment_sum(structure, self.pi * image)
            return self.design.rmatvec_sorted(
                self.k * (image - means[structure.sorted_codes])
            )

    def group_score_dot(self, values: ArrayLike) -> FloatArray:
        """psi_g' v for each cancelled group, in code order."""
        with np.errstate(over="raise", invalid="raise"):
            return segment_sum(self.design.structure, self.rho * self._image(values))

    def group_jacobian_form(self, left: ArrayLike, right: ArrayLike) -> FloatArray:
        """Left' H_g right for each group, retaining the nonsymmetric term."""
        structure = self.design.structure
        with np.errstate(over="raise", invalid="raise"):
            bu, bv = self._image(left), self._image(right)
            sums = segment_sum(structure, self.k * bv)
            means = segment_sum(structure, self.pi * bu)
            return finite_product(
                segment_sum(structure, self.k * bu * bv) - means * sums
            )

    def row_term_dot(self, values: ArrayLike) -> FloatArray:
        """rho_i (B_i-Bbar_g)' v at input rows."""
        structure = self.design.structure
        with np.errstate(over="raise", invalid="raise"):
            image = self._image(values)
            means = segment_sum(structure, self.pi * image)
            terms = finite_product(self.rho * (image - means[structure.sorted_codes]))
        return to_original(structure, terms)

    def group_scores_sparse(self) -> CsrArray:
        """Detached G by d group scores using at most G K + n r entries.

        Retained group/level cells are discovered from integer pairs, avoiding
        mixed-radix integer overflow and any row-by-retained-level allocation.
        """
        structure = self.design.structure
        groups, slopes = structure.n_groups, self.design.n_slopes
        with np.errstate(over="raise", invalid="raise"):
            slope_part = segment_sum(
                structure, self.rho[:, None] * self.design.centered_slopes
            )
            rows = [np.repeat(np.arange(groups), slopes)]
            cols = [np.tile(np.arange(slopes), groups)]
            values = [slope_part.reshape(-1)]
            rho_sums = segment_sum(structure, self.rho)
            for codes, start in zip(
                structure.retained_sorted_codes,
                self.design.block_starts,
                strict=True,
            ):
                pairs = np.column_stack((structure.sorted_codes, codes))
                cells, inverse = np.unique(pairs, axis=0, return_inverse=True)
                group = cells[:, 0]
                counts = np.bincount(inverse)
                rows.append(group)
                cols.append(start + cells[:, 1])
                values.append(
                    np.bincount(inverse, weights=self.rho)
                    - counts / structure.counts[group] * rho_sums[group]
                )
        data = finite_product(np.concatenate(values))
        return sparse.coo_array(
            (data, (np.concatenate(rows), np.concatenate(cols))),
            shape=(groups, self.design.n_columns),
        ).tocsr()

    def rows(self) -> CancelledRows:
        """Fresh detached row quantities in input positional order."""
        structure = self.design.structure
        return CancelledRows(
            *(
                to_original(structure, array)
                for array in (
                    self.q,
                    self.w,
                    self.t,
                    self.pi,
                    self.rho,
                    self.k,
                )
            )
        )

    def accounting(self) -> CancelledAccounting:
        """Count singleton and all-zero groups separately, retaining every row."""
        structure = self.design.structure
        singleton = structure.counts == 1
        all_zero = self.positive_counts == 0
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


def _log_weights(structure: CancelStructure, weights: ArrayLike | None) -> FloatArray:
    """Positive a, sorted; zero-weight centering policy is not supplied."""
    if weights is None:
        return np.zeros(structure.n_rows)
    array = finite_vector(weights, structure.n_rows, "obs_weights")
    if np.any(array <= 0):
        raise ValueError(
            "obs_weights must be positive; zero-weight policy is unresolved"
        )
    return np.log(to_sorted(structure, array))


def cancelled_equations(
    design: CenteredDesign,
    z: ArrayLike,
    theta: ArrayLike,
    *,
    variance: Variance,
    centered_offset: ArrayLike | None = None,
    obs_weights: ArrayLike | None = None,
) -> CancelledState:
    """Evaluate a fresh point without solving, mapping order or selecting rows.

    Parameters
    ----------
    design : CenteredDesign
        Full-column implicit design from ``centered_design``.
    z : array_like
        Already transformed, finite nonnegative outcomes in input row order.
    theta : array_like
        Finite full-column coefficients; identified callers must map into these.
    variance : Variance
        Current package protocol (``log_value``, not ``log_variance``).
    centered_offset : array_like, optional
        Already transformed and unweighted-centered offset, input row order.
        Added exactly once. The caller owns its centering/order precondition.
    obs_weights : array_like, optional
        Positive equation multipliers, default one; no replication semantics.

    Returns
    -------
    CancelledState
        Owned arrays and true forward/transpose negative-Jacobian actions. A new
        call always evaluates new inputs; only the immutable design is shared.
    """
    if type(design) is not CenteredDesign:
        raise TypeError("design must come from centered_design")
    structure = design.structure
    coefficients = design.checked_coefficients(theta).copy()
    outcomes = to_sorted(structure, finite_vector(z, structure.n_rows, "z"))
    if np.any(outcomes < 0):
        raise ValueError("z must be nonnegative")
    log_weights = _log_weights(structure, obs_weights)
    q = design.matvec_sorted(coefficients)
    if centered_offset is not None:
        offset = finite_vector(centered_offset, structure.n_rows, "centered_offset")
        with np.errstate(over="raise", invalid="raise"):
            q = finite_product(q + to_sorted(structure, offset))
    rows = row_components(outcomes, q, log_weights, variance)
    group_w, group_t = segment_sum(structure, rows.w), segment_sum(structure, rows.t)
    pi, rho, k = group_residuals(rows, group_w, group_t, structure.sorted_codes)
    # Exact algebraic zero; logarithmic round trips must not create singleton noise.
    rho[structure.counts[structure.sorted_codes] == 1] = 0.0
    positive_counts = np.add.reduceat((outcomes > 0).astype(np.intp), structure.starts)
    score = design.rmatvec_sorted(rho)
    record = object.__new__(CancelledState)
    object.__setattr__(record, "design", design)
    object.__setattr__(record, "_theta", FloatStorage.store(coefficients))
    object.__setattr__(record, "_q", FloatStorage.store(q))
    object.__setattr__(record, "_w", FloatStorage.store(rows.w))
    object.__setattr__(record, "_t", FloatStorage.store(rows.t))
    object.__setattr__(record, "_pi", FloatStorage.store(pi))
    object.__setattr__(record, "_rho", FloatStorage.store(rho))
    object.__setattr__(record, "_k", FloatStorage.store(k))
    object.__setattr__(record, "_group_w", FloatStorage.store(group_w))
    object.__setattr__(record, "_group_t", FloatStorage.store(group_t))
    object.__setattr__(record, "_positive_counts", IndexStorage.store(positive_counts))
    object.__setattr__(record, "_score", FloatStorage.store(score))
    return record
