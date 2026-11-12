"""The eigenvalue adjustment of an indefinite signed multiway covariance.

The operation is separate from every covariance kernel: it takes the reported, factor
scaled covariance of a signed multiway request with two or more dimensions, the frame
scales of its coordinates and the sequence of the convention in force, and returns the
matrix that the result reports, with the raw matrix, the frame matrix, the eigenvalues
it replaced and the diagnostics of the final matrix.  It is never applied to a
component, to a summand before the signed sum, to a DDG estimate, to HC, HAC or Conley,
or inside an objective, and the ``"none"`` preset and ``"psd": "none"`` leave the
estimate raw.  Neither the scales nor the sequence has a default: the caller declares
both.

The frame.  The native programs form the covariance on standardized
columns and undo the standardization afterwards; the equivalent operation on the
package's factor-scaled reported covariance ``V_q`` is a diagonal congruence,

    V_T = T V_q T,    F = the sequence's operation on V_T,    V_adj = T^-1 F T^-1,
    T = diag(s_1, ..., s_p, 1),    s_j = max(sd_j, 1e-3),
    sd_j = sqrt((sum_i x_ij^2 - (sum_i x_ij)^2 / N) / (N - 1)),

``x_j`` the user's regressor column ``j`` as supplied, the sum over the rows of the
scale sample, ``N`` their count (frequency counts do not expand rows for it), the
reported intercept or the private constant at scale exactly one, the response scale a
common positive factor that commutes with the operation and is neither applied nor
recorded.  :func:`column_scales` is that formula: an unweighted row-sample standard
deviation without centering the column itself, floored at ``1e-3`` for every column,
zero and near-zero columns included, a missing computed scale (a negative rounded
variance) refused as the programs refuse it.  The frame map and its inverse are the
entrywise product and quotient by ``s_j s_k``, the scale product formed once per pair
so that the images are exactly symmetric, range-safe: an entry that is not
representable, or a nonzero entry that would become zero, raises
``FloatingPointError`` and nothing is replaced.  With every scale equal to one both maps
are the identity bit for bit.

The sequences, on the frame matrix:

1. ``"modern"`` (``reghdfe`` 6.13.1; the ``"regression"`` preset, and ``"none"`` with an
   explicit ``"psd": "cgm"`` or a custom record with ``psd="cgm"``): the frame matrix is
   made symmetric and decomposed as ``V = U diag(w) U'``; if its smallest eigenvalue is
   negative, exactly ``min(w) < 0`` as the native test reads, the negative eigenvalues
   are set to zero and the matrix is rebuilt as ``U diag(w_+) U'``; when an intercept is
   reported and the adjustment is active, the RAW frame block of the other
   coefficients, backed up before the clip, is clipped on its own by the same rule (if
   its own smallest eigenvalue is negative, else kept raw) and written over that block.
   No second full repair follows, so the final matrix need not be positive
   semidefinite.  Witness (exact): the raw ``[[-1, 2], [2, -1]]`` with the
   intercept first and ``T = 1`` clips to all entries ``1/2``, its raw slope block
   ``[-1]`` becomes ``0``, and the final ``[[1/2, 1/2], [1/2, 0]]`` has determinant
   ``-1/4``.
2. ``"legacy"`` (``ppmlhdfe`` 2.3.4 through its ``reghdfe5`` callee; the ``"stata"``
   preset): the same clip of the full frame matrix and no block rule.  For a fit whose
   reported coefficients hold no intercept (an absorbed fit) the frame matrix holds,
   beside the reported coefficients, the private reporting constant
   (``constant``: its raw cross-covariances with the reported coefficients and its raw
   variance, in reported coordinates, factor applied once, scale one), because the
   program forces one and its clip acts on the full matrix; the reported result is the
   inverse frame map restricted to the reported coefficients, and the private row is
   retained in the record, never reported.

When the adjustment is inactive the reported matrix is the raw one, bit for bit.  A
diagonal congruence preserves inertia, so the final reported matrix is indefinite
exactly when the final frame matrix is, and no final diagonal is negative in either
sequence (each diagonal comes from one projection).  The caller prints the note
``covariance.psd_adjusted`` whenever the adjustment is active (the clip of the frame
matrix, the private constant included for the legacy sequence) and
``covariance.indefinite`` when the final reported matrix is not positive semidefinite;
neither is issued here.  Diagnostics are recorded truthfully for the FINAL matrices as
well as the raw and frame ones, with the port's indefiniteness tolerance
(``eps * n * max |w|``) beside the exact native trigger.  Nothing is floored, no
threshold is added, no universal positive semidefiniteness is claimed, and a zero
diagonal that the clipping creates stays zero (a zero standard error, degenerate).
Whole-stack treatment of stacked fits is not done here.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from powermean.inference.numeric import finite_array, immutable_array, symmetric_part
from powermean.model.variance import FloatArray

EMPTY: FloatArray = immutable_array(np.zeros(0))
FLOOR = 1e-3
"""The native floor of every frame scale (``compute_stdevs`` and its legacy twin)."""
Sequence = Literal["modern", "legacy"]
SEQUENCES: tuple[Sequence, ...] = ("modern", "legacy")
BoolArray = NDArray[np.bool_]


def _square(matrix: ArrayLike) -> FloatArray:
    """A finite real square matrix, as its symmetric part."""
    values = finite_array(matrix, "covariance")
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not values.size:
        raise ValueError("covariance must be a nonempty square matrix")
    return symmetric_part(values)


def _intercept(intercept: object, size: int) -> int | None:
    """The index of the reported intercept, or ``None``; a bool is not an index."""
    if intercept is None:
        return None
    if isinstance(intercept, (bool, np.bool_)) or not isinstance(intercept, Integral):
        raise ValueError(
            "intercept must be the integer index of a coefficient, or None"
        )
    index = int(intercept)
    if not 0 <= index < size:
        raise ValueError("intercept must index a coefficient of the covariance")
    return index


def _sequence(sequence: object) -> Sequence:
    """One of the two native sequences, by exact string (no default)."""
    if sequence not in SEQUENCES:
        raise ValueError("sequence must be 'modern' or 'legacy'")
    found: Sequence = "modern" if sequence == "modern" else "legacy"
    return found


def _scales(scales: ArrayLike, size: int, intercept: int | None) -> FloatArray:
    """One positive finite scale per reported coordinate; the intercept's is one."""
    values = finite_array(scales, "frame scales")
    if values.ndim != 1 or values.shape != (size,):
        raise ValueError("frame scales must hold one value per coefficient")
    if not np.all(values > 0):
        raise ValueError("frame scales must be positive")
    if intercept is not None and values[intercept] != 1.0:
        raise ValueError("the reported intercept has frame scale one")
    return values


def _constant(constant: ArrayLike, size: int) -> FloatArray:
    """The private constant's raw row: ``p`` cross-covariances and its variance."""
    values = finite_array(constant, "private constant")
    if values.ndim != 1 or values.shape != (size + 1,):
        raise ValueError(
            "the private constant holds one cross-covariance per coefficient and "
            "its variance"
        )
    return values


def _indefinite(eigenvalues: FloatArray, size: int) -> bool:
    """The port's indefiniteness rule: ``min(w) < -eps * n * max |w|``."""
    scale = float(np.max(np.abs(eigenvalues)))
    return bool(np.min(eigenvalues) < -np.finfo(float).eps * size * scale)


def _outer(scales: FloatArray) -> FloatArray:
    """``s_j s_k``, formed once per pair (exactly symmetric), range-safe."""
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        outer: FloatArray = scales[:, None] * scales[None, :]
    if not np.isfinite(outer).all() or np.any(outer == 0):
        raise FloatingPointError("the frame scale products are not representable")
    return outer


def _range_safe(found: FloatArray, matrix: FloatArray, name: str) -> FloatArray:
    """``found`` finite, with no zero made of a nonzero entry."""
    if not np.isfinite(found).all() or np.any((found == 0) & (matrix != 0)):
        raise FloatingPointError(f"{name} is not representable in float64")
    return found


def frame_image(matrix: ArrayLike, scales: ArrayLike) -> FloatArray:
    """``T V T``: the entrywise product ``(s_j s_k) V_jk``.

    Parameters
    ----------
    matrix : array
        A finite real square matrix in reported coordinates.
    scales : array
        One positive finite scale per coordinate.

    Returns
    -------
    array
        The frame image, owned; the identity bit for bit when every scale is one.

    Raises
    ------
    ValueError
        A matrix that is not finite and square, or scales that are not one positive
        finite value per coordinate.
    FloatingPointError
        An image entry that is not representable, or a nonzero entry made zero.
    """
    values = finite_array(matrix, "covariance")
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("covariance must be a square matrix")
    outer = _outer(_scales(scales, values.shape[0], None))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        found: FloatArray = values * outer
    return immutable_array(_range_safe(found, values, "the frame image"))


def frame_inverse(matrix: ArrayLike, scales: ArrayLike) -> FloatArray:
    """``T^-1 F T^-1``: the entrywise quotient ``F_jk / (s_j s_k)``.

    The inverse of :func:`frame_image` with the same scales; the same refusals.
    """
    values = finite_array(matrix, "frame matrix")
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("frame matrix must be a square matrix")
    outer = _outer(_scales(scales, values.shape[0], None))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        found: FloatArray = values / outer
    return immutable_array(_range_safe(found, values, "the inverse frame image"))


@dataclass(frozen=True, eq=False)
class FrameScales:
    """The frame scales of a set of reported regressor columns.

    Attributes
    ----------
    scales : array
        ``s_j = max(sd_j, 1e-3)``, one per column.
    deviations : array
        ``sd_j`` as computed, before the floor.
    floored : array of bool
        Whether the floor replaced ``sd_j`` (``sd_j < 1e-3``).
    count : int
        ``N``, the rows of the scale sample.
    """

    scales: FloatArray
    deviations: FloatArray
    floored: BoolArray
    count: int


def column_scales(columns: ArrayLike) -> FrameScales:
    """The native frame scale of every column of the scale sample (module docstring).

    Parameters
    ----------
    columns : array
        The user's regressor columns as supplied, one row per row of the scale sample
        (compressed rows: a frequency count expands nothing), at least two rows.

    Returns
    -------
    FrameScales
        ``max(sd_j, 1e-3)`` with the deviations, the floor flags and ``N``.

    Raises
    ------
    ValueError
        Columns that are not a finite real matrix of at least two rows, or a column
        whose computed scale is missing (a negative rounded variance), which the
        programs refuse.
    """
    values = finite_array(columns, "columns")
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("the scale sample holds at least two rows of columns")
    count = int(values.shape[0])
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        total = np.sum(values, axis=0)
        squares = np.sum(values * values, axis=0)
        variance: FloatArray = (squares - total * total / count) / (count - 1)
    if np.any(variance < 0) or not np.isfinite(variance).all():
        raise ValueError("a computed frame scale is missing (negative variance)")
    deviations: FloatArray = np.sqrt(variance)
    floored = deviations < FLOOR
    scales = np.where(floored, FLOOR, deviations)
    return FrameScales(
        scales=immutable_array(scales),
        deviations=immutable_array(deviations),
        floored=np.frombuffer(floored.tobytes(), dtype=np.bool_),
        count=count,
    )


def clip_negative_eigenvalues(
    matrix: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Rules 1 and 2 on one symmetric matrix: the clipped matrix, ``w``, the replaced.

    Parameters
    ----------
    matrix : array
        A finite real square matrix.

    Returns
    -------
    tuple of array
        The symmetric part of ``matrix`` when its smallest eigenvalue is not negative,
        else ``U diag(w_+) U'`` (symmetric part); the eigenvalues ``w`` in ascending
        order; the negative eigenvalues that were set to zero (empty when none).
    """
    symmetric = _square(matrix)
    found, vectors = np.linalg.eigh(symmetric)
    eigenvalues: FloatArray = np.asarray(found, dtype=np.float64)
    negative = eigenvalues < 0
    if not np.any(negative):
        return symmetric, eigenvalues, EMPTY
    kept = np.where(negative, 0.0, eigenvalues)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        rebuilt = (vectors * kept) @ vectors.T
    return symmetric_part(rebuilt), eigenvalues, eigenvalues[negative]


@dataclass(frozen=True, eq=False)
class PsdAdjustment:
    """The record of one eigenvalue adjustment of a reported signed multiway covariance.

    Attributes
    ----------
    raw : array
        The matrix handed in, as its symmetric part: the factor-scaled signed estimate
        of the reported coefficients in reported coordinates, untouched.
    adjusted : array
        The final matrix of the reported coefficients that the result reports: ``raw``
        itself when the adjustment is inactive.
    adjusted_full : array
        The inverse frame map of the final frame matrix, every frame coordinate: equal
        to ``adjusted`` without a private constant, else one row and column longer,
        the private constant last.  Retained, never reported.
    intercept : int or None
        The index of the reported intercept, or ``None`` when none is reported.
    sequence : {"modern", "legacy"}
        The native sequence applied.
    scales : array
        The frame scales ``T`` of every frame coordinate, the intercept's or the private
        constant's exactly one.
    constant : array or None
        The private constant's raw row (cross-covariances, then its variance) in
        reported coordinates, or ``None``.
    frame : array
        The frame matrix ``T V T`` handed to the sequence's operation, the private
        constant included where the legacy sequence requires it.
    active : bool
        Whether the smallest eigenvalue of the frame matrix was negative, so that the
        clip rebuilt it.
    eigenvalues : array
        The eigenvalues of the frame matrix, ascending (those of ``raw`` when ``T = 1``
        and no constant enters).
    replaced : array
        The negative eigenvalues of the frame matrix set to zero; empty when inactive.
    block_eigenvalues : array or None
        The eigenvalues of the backed-up RAW frame block of the other coefficients when
        the modern block rule ran (an intercept reported, the adjustment active, at
        least one other coefficient); ``None`` otherwise.
    block_replaced : array
        The negative eigenvalues of that raw block set to zero; empty when the block
        was kept raw or the block rule did not run.
    final_frame : array
        The final frame matrix after the sequence's operation (``frame`` when
        inactive).
    final_frame_eigenvalues : array
        Its eigenvalues, ascending.
    final_eigenvalues : array
        The eigenvalues of ``adjusted``, ascending.
    final_minimum : float
        The smallest of them, as computed.
    final_indefinite : bool
        Whether ``adjusted`` is indefinite by the port's tolerance; true only when the
        modern block rule overwrote the block of a clipped matrix with a block that
        does not complete it to a positive semidefinite matrix.
    """

    raw: FloatArray
    adjusted: FloatArray
    adjusted_full: FloatArray
    intercept: int | None
    sequence: Sequence
    scales: FloatArray
    constant: FloatArray | None
    frame: FloatArray
    active: bool
    eigenvalues: FloatArray
    replaced: FloatArray
    block_eigenvalues: FloatArray | None
    block_replaced: FloatArray
    final_frame: FloatArray
    final_frame_eigenvalues: FloatArray
    final_eigenvalues: FloatArray
    final_minimum: float
    final_indefinite: bool

    @property
    def count(self) -> int:
        """The eigenvalues set to zero, frame matrix and raw block together."""
        return int(self.replaced.size + self.block_replaced.size)

    @property
    def block_active(self) -> bool:
        """Whether the raw block was itself clipped before it overwrote the block."""
        return bool(self.block_replaced.size)

    @property
    def uniform(self) -> bool:
        """Whether every frame scale is one, so that the frame is the identity."""
        return bool(np.all(self.scales == 1.0))


def _full_raw(
    raw: FloatArray, scales: FloatArray, constant: FloatArray | None
) -> tuple[FloatArray, FloatArray]:
    """The frame coordinates: the reported matrix, plus the private constant row."""
    if constant is None:
        return raw, scales
    size = raw.shape[0]
    full = np.zeros((size + 1, size + 1))
    full[:size, :size] = raw
    full[size, :size] = constant[:size]
    full[:size, size] = constant[:size]
    full[size, size] = constant[size]
    return symmetric_part(full), np.append(scales, 1.0)


def _operation(
    frame: FloatArray, sequence: Sequence, index: int | None, size: int
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray | None, FloatArray]:
    """The sequence's operation on the frame matrix (module docstring).

    Returns the final frame matrix, the eigenvalues of the frame matrix, those
    replaced, the raw block's eigenvalues (or ``None``) and those replaced.
    """
    clipped, eigenvalues, replaced = clip_negative_eigenvalues(frame)
    block_eigenvalues: FloatArray | None = None
    block_replaced = EMPTY
    if not replaced.size:
        return frame, eigenvalues, replaced, None, block_replaced
    final = clipped
    if sequence == "modern" and index is not None and size > 1:
        keep = np.array([j != index for j in range(frame.shape[0])])
        block = frame[np.ix_(keep, keep)]  # the RAW frame block, backed up
        block_adjusted, block_eigenvalues, block_replaced = clip_negative_eigenvalues(
            block
        )
        final = np.array(clipped, copy=True)
        final[np.ix_(keep, keep)] = block_adjusted
    return final, eigenvalues, replaced, block_eigenvalues, block_replaced


def adjust_psd(
    covariance: FloatArray,
    *,
    intercept: int | None,
    scales: ArrayLike,
    sequence: Sequence,
    constant: ArrayLike | None = None,
) -> PsdAdjustment:
    """The native single-fit adjustment of one reported covariance, in its frame.

    Parameters
    ----------
    covariance : array
        The factor-scaled signed multiway covariance of the reported coefficients in
        reported coordinates, finite, square, at least one coefficient.
    intercept : int or None
        The index of the reported intercept, or ``None`` when none is reported.
    scales : array
        The frame scales of the reported coefficients, one positive finite double
        each, the intercept's exactly one.
    sequence : {"modern", "legacy"}
        The native sequence of the convention in force.
    constant : array, optional
        For the legacy sequence of a fit without a reported intercept: the private
        constant's raw cross-covariances with the reported coefficients followed by its
        raw variance, in reported coordinates, the factor applied once.  Refused with
        the modern sequence or beside a reported intercept.

    Returns
    -------
    PsdAdjustment
        The raw, frame and final matrices with the eigenvalues replaced and the
        diagnostics of the final matrices (module docstring).

    Raises
    ------
    ValueError
        A matrix that is not finite, real, square and nonempty; an intercept that does
        not index a coefficient; scales that are not one positive finite value per
        coefficient or an intercept scale other than one; a sequence outside the two;
        a private constant of the wrong length or where it does not belong.
    FloatingPointError
        A frame image, rebuilt entry or inverse image that is not representable.
    """
    raw = _square(covariance)
    size = raw.shape[0]
    index = _intercept(intercept, size)
    chosen = _sequence(sequence)
    reported_scales = _scales(scales, size, index)
    private: FloatArray | None = None
    if constant is not None:
        if chosen != "legacy" or index is not None:
            raise ValueError(
                "the private constant belongs to the legacy sequence of a fit "
                "without a reported intercept"
            )
        private = _constant(constant, size)
    full_raw, full_scales = _full_raw(raw, reported_scales, private)
    frame = frame_image(full_raw, full_scales)
    final_frame, eigenvalues, replaced, block_eigenvalues, block_replaced = _operation(
        frame, chosen, index, size
    )
    active = bool(replaced.size)
    adjusted_full = frame_inverse(final_frame, full_scales) if active else full_raw
    adjusted = adjusted_full[:size, :size] if active else raw
    final_eigenvalues: FloatArray = np.asarray(
        np.linalg.eigvalsh(adjusted), dtype=np.float64
    )
    return PsdAdjustment(
        raw=immutable_array(raw),
        adjusted=immutable_array(adjusted),
        adjusted_full=immutable_array(adjusted_full),
        intercept=index,
        sequence=chosen,
        scales=immutable_array(full_scales),
        constant=None if private is None else immutable_array(private),
        frame=immutable_array(frame),
        active=active,
        eigenvalues=immutable_array(eigenvalues),
        replaced=immutable_array(replaced),
        block_eigenvalues=(
            None if block_eigenvalues is None else immutable_array(block_eigenvalues)
        ),
        block_replaced=immutable_array(block_replaced),
        final_frame=immutable_array(final_frame),
        final_frame_eigenvalues=immutable_array(
            np.asarray(np.linalg.eigvalsh(final_frame), dtype=np.float64)
        ),
        final_eigenvalues=immutable_array(final_eigenvalues),
        final_minimum=float(np.min(final_eigenvalues)),
        final_indefinite=_indefinite(final_eigenvalues, size),
    )


def reported_errors(covariance: FloatArray) -> FloatArray:
    """``sqrt(V_jj)`` of a matrix already in reported coordinates (no floor).

    NaN where a variance is negative, zero where it is zero.  The direct map of the
    order scale does not apply to an adjusted matrix: the adjustment acts in the frame
    of the reported coordinates, so its standard errors are read from the final
    reported diagonal.
    """
    variances = np.diag(finite_array(covariance, "covariance"))
    errors = np.full(variances.shape, np.nan)
    valid = variances >= 0
    errors[valid] = np.sqrt(variances[valid])
    return immutable_array(errors)


__all__ = [
    "FLOOR",
    "SEQUENCES",
    "FrameScales",
    "PsdAdjustment",
    "Sequence",
    "adjust_psd",
    "clip_negative_eigenvalues",
    "column_scales",
    "frame_image",
    "frame_inverse",
    "reported_errors",
]
