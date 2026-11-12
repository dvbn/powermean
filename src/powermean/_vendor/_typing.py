"""Types of the carried Newton files: arrays, sparse matrices and pointwise losses.

The two scikit-learn files of :mod:`powermean._vendor` carry no annotations in their
source; the annotations added to them use the names of this module.

- ``FloatArray``: an array of doubles, the precision in which the package calls the
  carried files.  Their arithmetic
  is that of the source for any floating dtype; the annotations state how the package
  uses it.
- :class:`SparseMatrix`: a sparse matrix or sparse array of SciPy, described by the
  operations that the carried files apply to one.  SciPy publishes no type
  information (no ``py.typed`` marker, and no stub package among the dependencies),
  so the protocol lists those operations, and only those: an object without them, a
  string for instance, is not a ``FloatArray | SparseMatrix``.
  :mod:`powermean._vendor._scipy` types ``issparse`` so that the type checker takes
  the sparse member where it is true and the dense member where it is false.
- ``Intercept``: the intercept of a coefficient vector, a double, or one value per
  class.
- :class:`BaseLoss` and :class:`MulticlassLoss`: what ``LinearModelLoss`` needs from a
  pointwise loss.  The losses of scikit-learn (``sklearn._loss``) have this
  interface, and :class:`powermean._vendor._compat.HalfSquaredError` implements it.
"""

from __future__ import annotations

from typing import Protocol, overload

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Intercept = float | FloatArray


class SparseMatrix(Protocol):
    """A sparse matrix or sparse array of SciPy, as the carried files use one.

    Its shape and transpose; its product with a dense array, which is dense, and with
    another sparse matrix, which is sparse; the product of a dense array with it,
    which is dense; its sum along an axis, which is dense (a ``numpy.matrix`` for a
    sparse matrix); and its dense copy.  Every sparse format of SciPy has them.
    """

    shape: tuple[int, int]
    T: SparseMatrix

    @overload
    def __matmul__(self, other: FloatArray, /) -> FloatArray: ...

    @overload
    def __matmul__(self, other: SparseMatrix, /) -> SparseMatrix: ...

    def __rmatmul__(self, other: FloatArray, /) -> FloatArray:
        """The dense product ``other @ self``."""
        ...

    def sum(self, *, axis: int) -> FloatArray:
        """The sums of the entries along ``axis``, a dense array."""
        ...

    def toarray(self) -> FloatArray:
        """The matrix as a dense array."""
        ...


class BaseLoss(Protocol):
    """A pointwise loss of the raw prediction, with its first two derivatives.

    ``LinearModelLoss`` passes every argument by keyword and divides the returned
    arrays in place, so each method returns new arrays.  A sample weight multiplies
    the value of its row.  ``n_threads`` is the number of threads that a compiled loss
    may use.

    Attributes
    ----------
    n_classes : int or None
        The number of classes of a multiclass loss, ``None`` for a loss with one
        output.  ``LinearModelLoss`` uses it on its multiclass paths only.
    is_multiclass : bool
        Whether the raw prediction has one column per class.
    """

    n_classes: int | None
    is_multiclass: bool

    def loss(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> FloatArray:
        """The loss of each row."""
        ...

    def loss_gradient(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> tuple[FloatArray, FloatArray]:
        """The loss of each row and its derivative in the raw prediction."""
        ...

    def gradient(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> FloatArray:
        """The derivative of the loss of each row in the raw prediction."""
        ...

    def gradient_hessian(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> tuple[FloatArray, FloatArray]:
        """The first and second derivatives of the loss of each row."""
        ...


class MulticlassLoss(BaseLoss, Protocol):
    """A multiclass loss, which also returns the predicted probabilities."""

    def gradient_proba(
        self,
        *,
        y_true: FloatArray,
        raw_prediction: FloatArray,
        sample_weight: FloatArray | None,
        n_threads: int,
    ) -> tuple[FloatArray, FloatArray]:
        """The derivatives of the loss of each row and the class probabilities."""
        ...
