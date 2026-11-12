"""Local equivalents of the four scikit-learn names that the carried files imported.

The carried Newton files import nothing from scikit-learn.  Two names of
their source served the L-BFGS fallback, which is removed with them.  The four others
are replaced here, each by the part of scikit-learn's object that the carried files
use, with the arithmetic of scikit-learn:

========================  ==========================================  ==============
scikit-learn              here                                        used by
========================  ==========================================  ==============
``HalfSquaredError``      :class:`HalfSquaredError`                   default loss of
                                                                      ``NewtonSolver``
``ConvergenceWarning``    :class:`ConvergenceWarning`                 ``NewtonSolver``
``safe_sparse_dot``       :func:`safe_sparse_dot`                     ``sandwich_dot``
``squared_norm``          :func:`squared_norm`                        ``l2_penalty``
========================  ==========================================  ==============

``ConvergenceWarning`` is the category of the iteration's own diagnostics.  It is not
:class:`powermean.errors.NumericalWarning`, which reports the secondary status of the
certificate, and neither derives from the other.
"""

from __future__ import annotations

from typing import Literal, overload, override

import numpy as np

from powermean._vendor._scipy import sparse
from powermean._vendor._typing import BaseLoss, FloatArray, SparseMatrix


class ConvergenceWarning(UserWarning):
    """A convergence problem of the Newton iteration, as scikit-learn's category."""


def squared_norm(x: FloatArray) -> np.float64:
    """The squared Euclidean or Frobenius norm of ``x``, the dot product of its entries.

    The entries are taken in the order of memory (``np.ravel(x, order="K")``), as by
    scikit-learn.  ``x`` is a floating array; scikit-learn's warning for integer
    arrays, whose dot product may overflow, has no case here.
    """
    x = np.ravel(x, order="K")
    value: np.float64 = np.dot(x, x)
    return value


@overload
def safe_sparse_dot(
    a: SparseMatrix, b: SparseMatrix, *, dense_output: Literal[True]
) -> FloatArray: ...


@overload
def safe_sparse_dot(
    a: FloatArray | SparseMatrix,
    b: FloatArray | SparseMatrix,
    *,
    dense_output: bool = False,
) -> FloatArray | SparseMatrix: ...


def safe_sparse_dot(
    a: FloatArray | SparseMatrix,
    b: FloatArray | SparseMatrix,
    *,
    dense_output: bool = False,
) -> FloatArray | SparseMatrix:
    """The product ``a @ b`` of two matrices, each dense or sparse.

    The product of two sparse matrices is returned as a dense array when
    ``dense_output`` is true, as by scikit-learn; every other product is returned as
    the operator computes it, a dense array unless both factors are sparse.  Operands
    of more than two dimensions, which scikit-learn also accepts, do not occur in the
    carried files.
    """
    ret: FloatArray | SparseMatrix = a @ b
    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        dense: FloatArray = ret.toarray()
        return dense
    return ret


def _double(values: FloatArray) -> FloatArray:
    """``values`` in double precision, as a compiled loss reads them.

    A single-precision value converts to a double exactly, and an array of doubles is
    returned as it is.
    """
    return np.asarray(values, dtype=np.float64)


def _stored(values: FloatArray, like: FloatArray) -> FloatArray:
    """Doubles ``values``, rounded once to the floating dtype of ``like``.

    scikit-learn's compiled loss writes each double into an output array allocated
    like one of its inputs, so a single-precision input gives single-precision
    values; an array of doubles is returned as it is.  An integer ``like``, which
    scikit-learn refuses, counts as double precision, so that nothing is truncated.
    """
    stored: FloatArray = values.astype(np.result_type(like.dtype, 0.0), copy=False)
    return stored


class HalfSquaredError(BaseLoss):
    """Half the squared error, ``(raw_prediction - y_true)^2 / 2``, of each row.

    The default pointwise loss of ``NewtonSolver``, with the identity link.  Its
    gradient is ``raw_prediction - y_true`` and its hessian one; a sample weight
    multiplies the three.  As in scikit-learn's compiled loss, the inputs are read as
    doubles, the operations are those of double precision in the same order, and each
    result is rounded once to the dtype of the array in which scikit-learn stores it:
    that of ``y_true`` for the loss, that of ``raw_prediction`` for the gradient and the
    hessian.  For inputs of one floating dtype, double or single precision, every value
    is therefore scikit-learn's, bit for bit.  Unlike scikit-learn's, the arrays need
    not be contiguous, and ``n_threads`` is accepted for the interface and not used,
    since NumPy evaluates the rows.
    """

    n_classes: int | None = None
    is_multiclass: bool = False

    @override
    def loss(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> FloatArray:
        """``0.5 (raw_prediction - y_true)^2``, times ``sample_weight`` if given."""
        difference = _double(raw_prediction) - _double(y_true)
        loss = 0.5 * difference * difference
        if sample_weight is not None:
            loss = _double(sample_weight) * loss
        return _stored(loss, y_true)

    @override
    def loss_gradient(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> tuple[FloatArray, FloatArray]:
        """The loss and the gradient ``raw_prediction - y_true`` of each row."""
        gradient = _double(raw_prediction) - _double(y_true)
        loss = 0.5 * gradient * gradient
        if sample_weight is not None:
            weights = _double(sample_weight)
            loss, gradient = weights * loss, weights * gradient
        return _stored(loss, y_true), _stored(gradient, raw_prediction)

    @override
    def gradient(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> FloatArray:
        """``raw_prediction - y_true``, times ``sample_weight`` if given."""
        gradient = _double(raw_prediction) - _double(y_true)
        if sample_weight is not None:
            gradient = _double(sample_weight) * gradient
        return _stored(gradient, raw_prediction)

    @override
    def gradient_hessian(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> tuple[FloatArray, FloatArray]:
        """The gradient and the hessian, one in each row before its weight."""
        gradient = _double(raw_prediction) - _double(y_true)
        hessian = np.ones_like(gradient)
        if sample_weight is not None:
            weights = _double(sample_weight)
            gradient, hessian = weights * gradient, weights * hessian
        return _stored(gradient, raw_prediction), _stored(hessian, raw_prediction)
