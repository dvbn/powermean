# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Newton solver for Generalized Linear Models.

Carried from scikit-learn under the notice ``SKLEARN_COPYING`` of this folder.  The
arithmetic, the
names and the structure are those of the source, without its L-BFGS fallback: a
failed Newton step or line search stops the iteration without convergence, and no
other solver is run.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import cast

import numpy as np

from powermean._vendor import _scipy as scipy
from powermean._vendor._compat import ConvergenceWarning, HalfSquaredError
from powermean._vendor._typing import FloatArray, SparseMatrix
from powermean._vendor.vendor_linear_loss import LinearModelLoss


class NewtonSolver(ABC):
    """Newton solver for GLMs.

    This class implements Newton/2nd-order optimization routines for GLMs. Each Newton
    iteration aims at finding the Newton step which is done by the inner solver. With
    Hessian H, gradient g and coefficients coef, one step solves:

        H @ coef_newton = -g

    For our GLM / LinearModelLoss, we have gradient g and Hessian H:

        g = X.T @ loss.gradient + l2_reg_strength * coef
        H = X.T @ diag(loss.hessian) @ X + l2_reg_strength * identity

    Backtracking line search updates coef = coef_old + t * coef_newton for some t in
    (0, 1].

    This is a base class, actual implementations (child classes) may deviate from the
    above pattern and use structure specific tricks.

    Usage pattern:
        - initialize solver: sol = NewtonSolver(...)
        - solve the problem: sol.solve(X, y, sample_weight)

    References
    ----------
    - Jorge Nocedal, Stephen J. Wright. (2006) "Numerical Optimization"
      2nd edition
      https://doi.org/10.1007/978-0-387-40065-5

    - Stephen P. Boyd, Lieven Vandenberghe. (2004) "Convex Optimization."
      Cambridge University Press, 2004.
      https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf

    Parameters
    ----------
    coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
        Initial coefficients of a linear model.
        If shape (n_classes * n_dof,), the classes of one feature are contiguous,
        i.e. one reconstructs the 2d-array via
        coef.reshape((n_classes, -1), order="F").

    linear_loss : LinearModelLoss
        The loss to be minimized.

    l2_reg_strength : float, default=0.0
        L2 regularization strength.

    tol : float, default=1e-4
        The optimization problem is solved when each of the following conditions is
        fulfilled:
        1. maximum |gradient| <= tol
        2. Newton decrement d: 1/2 * d^2 <= tol

    max_iter : int, default=100
        Maximum number of Newton steps allowed.

    n_threads : int, default=1
        Number of OpenMP threads to use for the computation of the Hessian and gradient
        of the loss function.

    verbose : int, default=0
        Diagnostic output: from 1, each iteration and its convergence check are
        printed; from 2, also each trial of the line search.

    Attributes
    ----------
    coef_old : ndarray of shape coef.shape
        Coefficient of previous iteration.

    coef_newton : ndarray of shape coef.shape
        Newton step.

    gradient : ndarray of shape coef.shape
        Gradient of the loss w.r.t. the coefficients.

    gradient_old : ndarray of shape coef.shape
        Gradient of previous iteration.

    loss_value : float
        Value of objective function = loss + penalty.

    loss_value_old : float
        Value of objective function of previous iteration.

    raw_prediction : ndarray of shape (n_samples,) or (n_samples, n_classes)

    converged : bool
        Indicator for convergence of the solver.

    iteration : int
        Number of Newton steps, i.e. calls to inner_solve

    use_fallback_lbfgs_solve : bool
        Set to True when inner_solve or line_search fails: the Newton step cannot be
        computed or is not a descent direction, or the line search finds no
        acceptable step. The iteration then stops without convergence, and no
        fallback solver is run; the name is that of the source.

    gradient_times_newton : float
        gradient @ coef_newton, set in inner_solve and used by line_search. If the
        Newton step is a descent direction, this is negative.
    """

    coef: FloatArray
    linear_loss: LinearModelLoss
    l2_reg_strength: float
    tol: float
    max_iter: int
    n_threads: int
    verbose: int
    raw_prediction: FloatArray
    loss_value: np.float64
    loss_value_old: np.float64
    coef_old: FloatArray
    coef_newton: FloatArray
    gradient: FloatArray
    gradient_old: FloatArray
    hessian: FloatArray
    gradient_times_newton: np.float64
    converged: bool
    iteration: int
    use_fallback_lbfgs_solve: bool

    def __init__(
        self,
        *,
        coef: FloatArray,
        linear_loss: LinearModelLoss = LinearModelLoss(
            base_loss=HalfSquaredError(), fit_intercept=True
        ),
        l2_reg_strength: float = 0.0,
        tol: float = 1e-4,
        max_iter: int = 100,
        n_threads: int = 1,
        verbose: int = 0,
    ) -> None:
        self.coef = coef
        self.linear_loss = linear_loss
        self.l2_reg_strength = l2_reg_strength
        self.tol = tol
        self.max_iter = max_iter
        self.n_threads = n_threads
        self.verbose = verbose

    def setup(
        self,
        X: FloatArray | SparseMatrix,
        y: FloatArray,
        sample_weight: FloatArray | None,
    ) -> None:
        """Precomputations.

        If None, initializes:
            - self.coef
        Sets:
            - self.raw_prediction
            - self.loss_value
        """
        _, _, self.raw_prediction = self.linear_loss.weight_intercept_raw(self.coef, X)
        self.loss_value = self.linear_loss.loss(
            coef=self.coef,
            X=X,
            y=y,
            sample_weight=sample_weight,
            l2_reg_strength=self.l2_reg_strength,
            n_threads=self.n_threads,
            raw_prediction=self.raw_prediction,
        )

    @abstractmethod
    def update_gradient_hessian(
        self,
        X: FloatArray | SparseMatrix,
        y: FloatArray,
        sample_weight: FloatArray | None,
    ) -> None:
        """Update gradient and Hessian."""

    @abstractmethod
    def inner_solve(
        self,
        X: FloatArray | SparseMatrix,
        y: FloatArray,
        sample_weight: FloatArray | None,
    ) -> None:
        """Compute Newton step.

        Sets:
            - self.coef_newton
            - self.gradient_times_newton
        """

    def line_search(
        self,
        X: FloatArray | SparseMatrix,
        y: FloatArray,
        sample_weight: FloatArray | None,
    ) -> None:
        """Backtracking line search.

        Sets:
            - self.coef_old
            - self.coef
            - self.loss_value_old
            - self.loss_value
            - self.gradient_old
            - self.gradient
            - self.raw_prediction
        """
        # line search parameters
        beta, sigma = 0.5, 0.00048828125  # 1/2, 1/2**11
        eps = 16 * np.finfo(self.loss_value.dtype).eps
        t: float = 1  # step size

        # gradient_times_newton, the product of the gradient and the Newton step, was
        # computed in inner_solve.
        armijo_term = sigma * self.gradient_times_newton
        _, _, raw_prediction_newton = self.linear_loss.weight_intercept_raw(
            self.coef_newton, X
        )

        self.coef_old = self.coef
        self.loss_value_old = self.loss_value
        self.gradient_old = self.gradient

        # The sum of the absolute values of gradient_old, computed when first needed.
        sum_abs_grad_old: float = -1

        is_verbose = self.verbose >= 2
        if is_verbose:
            print("  Backtracking Line Search")
            print(f"    eps=16 * finfo.eps={eps}")

        for i in range(21):  # until and including t = beta**20 ~ 1e-6
            self.coef = self.coef_old + t * self.coef_newton
            raw = self.raw_prediction + t * raw_prediction_newton
            self.loss_value, self.gradient = self.linear_loss.loss_gradient(
                coef=self.coef,
                X=X,
                y=y,
                sample_weight=sample_weight,
                l2_reg_strength=self.l2_reg_strength,
                n_threads=self.n_threads,
                raw_prediction=raw,
            )
            # Note: If coef_newton is too large, loss_gradient may produce inf values,
            # potentially accompanied by a RuntimeWarning.
            # This case will be captured by the Armijo condition.

            # 1. Check Armijo / sufficient decrease condition.
            # The smaller (more negative) the better.
            loss_improvement = self.loss_value - self.loss_value_old
            check = loss_improvement <= t * armijo_term
            if is_verbose:
                print(
                    f"    line search iteration={i + 1}, step size={t}\n"
                    f"      check loss improvement <= armijo term: {loss_improvement} "
                    f"<= {t * armijo_term} {check}"
                )
            if check:
                break
            # 2. Deal with relative loss differences around machine precision.
            tiny_loss = np.abs(self.loss_value_old * eps)
            check = np.abs(loss_improvement) <= tiny_loss
            if is_verbose:
                print(
                    "      check loss |improvement| <= eps * |loss_old|:"
                    f" {np.abs(loss_improvement)} <= {tiny_loss} {check}"
                )
            if check:
                if sum_abs_grad_old < 0:
                    sum_abs_grad_old = scipy.linalg.norm(self.gradient_old, ord=1)
                # 2.1 Check sum of absolute gradients as alternative condition.
                sum_abs_grad = scipy.linalg.norm(self.gradient, ord=1)
                check = sum_abs_grad < sum_abs_grad_old
                if is_verbose:
                    print(
                        "      check sum(|gradient|) < sum(|gradient_old|): "
                        f"{sum_abs_grad} < {sum_abs_grad_old} {check}"
                    )
                if check:
                    break

            t *= beta
        else:
            warnings.warn(
                (
                    f"Line search of Newton solver {self.__class__.__name__} at"
                    f" iteration #{self.iteration} did not converge after 21 line"
                    " search refinement iterations. The Newton iteration stops"
                    " without convergence; no fallback solver is run."
                ),
                ConvergenceWarning,
                stacklevel=1,
            )
            if self.verbose:
                print(
                    "  Line search did not converge; the iteration stops without a "
                    "fallback solver."
                )
            self.use_fallback_lbfgs_solve = True
            return

        self.raw_prediction = raw
        if is_verbose:
            print(
                f"    line search successful after {i + 1} iterations with "
                f"loss={self.loss_value}."
            )

    def check_convergence(
        self,
        X: FloatArray | SparseMatrix,
        y: FloatArray,
        sample_weight: FloatArray | None,
    ) -> None:
        """Check for convergence.

        Sets self.converged.
        """
        if self.verbose:
            print("  Check Convergence")
        # Note: Checking maximum relative change of coefficient <= tol is a bad
        # convergence criterion because even a large step could have brought us close
        # to the true minimum. That criterion divides the change of each coefficient
        # by the larger of one and the absolute value of its previous value.

        # 1. Criterion: maximum |gradient| <= tol
        #    The gradient was already updated in line_search()
        g_max_abs = np.max(np.abs(self.gradient))
        check = g_max_abs <= self.tol
        if self.verbose:
            print(f"    1. max |gradient| {g_max_abs} <= {self.tol} {check}")
        if not check:
            return

        # 2. Criterion: For Newton decrement d, check 1/2 * d^2 <= tol
        #       d = sqrt(grad' @ hessian^-1 @ grad)
        #         = sqrt(coef_newton' @ hessian @ coef_newton)
        #    See Boyd, Vandenberghe (2009) "Convex Optimization" Chapter 9.5.1.
        # NumPy's annotations type this product of vectors as an array; it is a double.
        d2 = cast("np.float64", self.coef_newton @ self.hessian @ self.coef_newton)
        check = 0.5 * d2 <= self.tol
        if self.verbose:
            print(f"    2. Newton decrement {0.5 * d2} <= {self.tol} {check}")
        if not check:
            return

        if self.verbose:
            loss_value = self.linear_loss.loss(
                coef=self.coef,
                X=X,
                y=y,
                sample_weight=sample_weight,
                l2_reg_strength=self.l2_reg_strength,
                n_threads=self.n_threads,
            )
            print(f"  Solver did converge at loss = {loss_value}.")
        self.converged = True

    def finalize(
        self,
        X: FloatArray | SparseMatrix,
        y: FloatArray,
        sample_weight: FloatArray | None,
    ) -> None:
        """Finalize the solver's results.

        Some solvers may need this, others not. This optional hook does nothing in
        the base class: ``solve`` calls it once, after the iteration, so that a
        solver that changes the parametrization of the coefficients during the
        iteration can return them in the usual one, as NewtonCholeskySolver does for
        the multinomial loss.
        """
        pass

    def solve(
        self,
        X: FloatArray | SparseMatrix,
        y: FloatArray,
        sample_weight: FloatArray | None,
    ) -> FloatArray:
        """Solve the optimization problem.

        This is the main routine.

        Order of calls:
            self.setup()
            while iteration:
                self.update_gradient_hessian()
                self.inner_solve()
                self.line_search()
                self.check_convergence()
            self.finalize()

        Returns
        -------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Solution of the optimization problem.
        """
        # setup usually:
        #   - initializes self.coef if needed
        #   - initializes and calculates self.raw_predictions, self.loss_value
        self.setup(X=X, y=y, sample_weight=sample_weight)

        self.iteration = 1
        self.converged = False
        self.use_fallback_lbfgs_solve = False

        while self.iteration <= self.max_iter and not self.converged:
            if self.verbose:
                print(f"Newton iter={self.iteration}")

            self.use_fallback_lbfgs_solve = False  # Set again by a failed step.

            # 1. Update Hessian and gradient
            self.update_gradient_hessian(X=X, y=y, sample_weight=sample_weight)

            # No early stop is made here, even close to the optimum, usually detected
            # by zero gradients at this stage; check_convergence decides after the
            # step.

            # 2. Inner solver
            #    Calculate Newton step/direction
            #    This usually sets self.coef_newton and self.gradient_times_newton.
            self.inner_solve(X=X, y=y, sample_weight=sample_weight)
            if self.use_fallback_lbfgs_solve:
                break

            # 3. Backtracking line search
            #    This usually sets self.coef_old, self.coef, self.loss_value_old
            #    self.loss_value, self.gradient_old, self.gradient,
            #    self.raw_prediction.
            self.line_search(X=X, y=y, sample_weight=sample_weight)
            # line_search may set the flag, but the type checker keeps, across the call,
            # the False of the test above: the cast reads the flag as the bool it is.
            if cast("bool", self.use_fallback_lbfgs_solve):
                break

            # 4. Check convergence
            #    Sets self.converged.
            self.check_convergence(X=X, y=y, sample_weight=sample_weight)

            # 5. Next iteration
            self.iteration += 1

        # A failed step stops the iteration without convergence, and no fallback
        # solver is run: inner_solve or line_search reported the failure (a Newton
        # step that is not a descent direction in verbose output only). Otherwise the
        # iteration stopped at max_iter.
        if not self.converged and not self.use_fallback_lbfgs_solve:
            warnings.warn(
                (
                    f"Newton solver did not converge after {self.iteration - 1} "
                    "iterations."
                ),
                ConvergenceWarning,
                stacklevel=1,
            )

        self.iteration -= 1
        self.finalize(X=X, y=y, sample_weight=sample_weight)
        return self.coef


class NewtonCholeskySolver(NewtonSolver):
    """Cholesky based Newton solver.

    Inner solver for finding the Newton step H w_newton = -g uses Cholesky based linear
    solver.

    As in the source, the step is computed by scipy.linalg.solve with
    assume_a="sym", the solver for symmetric matrices, whatever the name of the class.
    """

    hessian_warning: bool | np.bool_
    is_multinomial_with_intercept: bool
    is_multinomial_no_penalty: bool

    def setup(
        self,
        X: FloatArray | SparseMatrix,
        y: FloatArray,
        sample_weight: FloatArray | None,
    ) -> None:
        """Precomputations, and the gradient and the hessian allocated.

        Sets the attributes of NewtonSolver.setup, then:
            - self.gradient
            - self.hessian
            - self.is_multinomial_with_intercept
            - self.is_multinomial_no_penalty
        and moves multinomial coefficients to the convention of inner_solve.
        """
        super().setup(X=X, y=y, sample_weight=sample_weight)
        if self.linear_loss.base_loss.is_multiclass:
            # Easier with ravelled arrays, e.g., for scipy.linalg.solve.
            # As with LinearModelLoss, we always are contiguous in n_classes.
            self.coef = self.coef.ravel(order="F")
        # Note that the computation of gradient in LinearModelLoss follows the shape of
        # coef.
        self.gradient = np.empty_like(self.coef)
        # But the hessian is always 2d.
        n = self.coef.size
        self.hessian = np.empty_like(self.coef, shape=(n, n))
        # To help case distinctions.
        self.is_multinomial_with_intercept = (
            self.linear_loss.base_loss.is_multiclass and self.linear_loss.fit_intercept
        )
        self.is_multinomial_no_penalty = (
            self.linear_loss.base_loss.is_multiclass and self.l2_reg_strength == 0
        )
        if self.is_multinomial_no_penalty:
            # See inner_solve. The provided coef might not adhere to the convention
            # that the last class is set to zero.
            # This is done by the usual freedom of a (overparametrized) multinomial to
            # add a constant to all classes which doesn't change predictions.
            n_classes = cast("int", self.linear_loss.base_loss.n_classes)
            coef = self.coef.reshape(n_classes, -1, order="F")  # easier as 2d
            coef -= coef[-1, :]  # coef -= coef of last class
        elif self.is_multinomial_with_intercept:
            # See inner_solve. Same as above, but only for the intercept.
            n_classes = cast("int", self.linear_loss.base_loss.n_classes)
            # intercept -= intercept of last class
            self.coef[-n_classes:] -= self.coef[-1]

    def update_gradient_hessian(
        self,
        X: FloatArray | SparseMatrix,
        y: FloatArray,
        sample_weight: FloatArray | None,
    ) -> None:
        """Update gradient and Hessian in place, and the hessian warning."""
        _, _, self.hessian_warning = self.linear_loss.gradient_hessian(
            coef=self.coef,
            X=X,
            y=y,
            sample_weight=sample_weight,
            l2_reg_strength=self.l2_reg_strength,
            n_threads=self.n_threads,
            gradient_out=self.gradient,
            hessian_out=self.hessian,
            raw_prediction=self.raw_prediction,  # this was updated in line_search
        )

    def inner_solve(
        self,
        X: FloatArray | SparseMatrix,
        y: FloatArray,
        sample_weight: FloatArray | None,
    ) -> None:
        """Compute Newton step, or flag a failure.

        Sets:
            - self.coef_newton
            - self.gradient_times_newton
        or sets self.use_fallback_lbfgs_solve when the pointwise hessian has many
        negative values, when the Hessian matrix is singular or ill-conditioned, or
        when the Newton step is not a descent direction. The iteration then stops
        without convergence, and no fallback solver is run.
        """
        if self.hessian_warning:
            warnings.warn(
                (
                    f"The inner solver of {self.__class__.__name__} detected a "
                    "pointwise hessian with many negative values at iteration "
                    f"#{self.iteration}. The Newton iteration stops without "
                    "convergence; no fallback solver is run."
                ),
                ConvergenceWarning,
                stacklevel=1,
            )
            if self.verbose:
                print(
                    "  The inner solver detected a pointwise Hessian with many "
                    "negative values; the iteration stops without a fallback solver."
                )
            self.use_fallback_lbfgs_solve = True
            return

        # Note: The following case distinction could also be shifted to the
        # implementation of HalfMultinomialLoss instead of here within the solver.
        if self.is_multinomial_no_penalty:
            # The multinomial loss is overparametrized for each unpenalized feature, so
            # at least the intercepts. This can be seen by noting that predicted
            # probabilities are invariant under shifting all coefficients of a single
            # feature j for all classes by the same amount c:
            #   coef[k, :] -> coef[k, :] + c    =>    proba stays the same
            # where we have assumed coef.shape = (n_classes, n_features).
            # Therefore, also the loss (-log-likelihood), gradient and hessian stay the
            # same, see
            # Noah Simon and Jerome Friedman and Trevor Hastie. (2013) "A Blockwise
            # Descent Algorithm for Group-penalized Multiresponse and Multinomial
            # Regression". https://doi.org/10.48550/arXiv.1311.6529
            #
            # We choose the standard approach and set all the coefficients of the last
            # class to zero, for all features including the intercept.
            # Note that coef was already dealt with in setup.
            n_classes = cast("int", self.linear_loss.base_loss.n_classes)
            n_dof = self.coef.size // n_classes  # degree of freedom per class
            n = self.coef.size - n_dof  # effective size
            self.gradient[n_classes - 1 :: n_classes] = 0
            self.hessian[n_classes - 1 :: n_classes, :] = 0
            self.hessian[:, n_classes - 1 :: n_classes] = 0
            # We also need the reduced variants of gradient and hessian where the
            # entries set to zero are removed. For 2 features and 3 classes with
            # arbitrary values, "x" means removed:
            #   gradient is  0  1  x  3  4  x
            #
            #   hessian is   0  1  x  3  4  x
            #                1  7  x  9 10  x
            #                x  x  x  x  x  x
            #                3  9  x 21 22  x
            #                4 10  x 22 28  x
            #                x  x  x  x  x  x
            # The following slicing triggers copies of gradient and hessian.
            gradient: FloatArray
            hessian: FloatArray
            gradient = self.gradient.reshape(-1, n_classes)[:, :-1].flatten()
            hessian = self.hessian.reshape(n_dof, n_classes, n_dof, n_classes)[
                :, :-1, :, :-1
            ].reshape(n, n)
        elif self.is_multinomial_with_intercept:
            # Here, only intercepts are unpenalized. We again choose the last class and
            # set its intercept to zero.
            # Note that coef was already dealt with in setup.
            self.gradient[-1] = 0
            self.hessian[-1, :] = 0
            self.hessian[:, -1] = 0
            gradient, hessian = self.gradient[:-1], self.hessian[:-1, :-1]
        else:
            gradient, hessian = self.gradient, self.hessian

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", scipy.linalg.LinAlgWarning)
                self.coef_newton = scipy.linalg.solve(
                    hessian, -gradient, check_finite=False, assume_a="sym"
                )
                if self.is_multinomial_no_penalty:
                    self.coef_newton = np.c_[
                        self.coef_newton.reshape(n_dof, n_classes - 1), np.zeros(n_dof)
                    ].reshape(-1)
                    assert self.coef_newton.flags.f_contiguous
                elif self.is_multinomial_with_intercept:
                    self.coef_newton = np.r_[self.coef_newton, 0]
                # NumPy's annotations type this product of vectors as an array; it is
                # a double.
                self.gradient_times_newton = cast(
                    "np.float64", self.gradient @ self.coef_newton
                )
                if self.gradient_times_newton > 0:
                    if self.verbose:
                        print(
                            "  The inner solver found a Newton step that is not a "
                            "descent direction; the iteration stops without a "
                            "fallback solver."
                        )
                    self.use_fallback_lbfgs_solve = True
                    return
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgWarning) as e:
            warnings.warn(
                f"The inner solver of {self.__class__.__name__} stumbled upon a "
                "singular or very ill-conditioned Hessian matrix at iteration "
                f"{self.iteration}. The Newton iteration stops without convergence; "
                "no fallback solver is run.\n"
                "Further options are to use another solver or to avoid such situation "
                "in the first place. Possible remedies are removing collinear features"
                " of X or increasing the penalization strengths.\n"
                "The original Linear Algebra message was:\n" + str(e),
                scipy.linalg.LinAlgWarning,
                stacklevel=1,
            )
            # Possible causes:
            # 1. hess_pointwise is negative. But this is already taken care in
            #    LinearModelLoss.gradient_hessian.
            # 2. X is singular or ill-conditioned
            #    This might be the most probable cause.
            #
            # There are many possible ways to deal with this situation. Most of them
            # add, explicitly or implicitly, a matrix to the hessian to make it
            # positive definite, confer to Chapter 3.4 of Nocedal & Wright 2nd ed.
            # Instead, the iteration stops here, and no fallback solver is run.
            if self.verbose:
                print(
                    "  The inner solver stumbled upon a singular or ill-conditioned "
                    "Hessian matrix; the iteration stops without a fallback solver."
                )
            self.use_fallback_lbfgs_solve = True
            return

    def finalize(
        self,
        X: FloatArray | SparseMatrix,
        y: FloatArray,
        sample_weight: FloatArray | None,
    ) -> None:
        """Return multinomial coefficients to the symmetric parametrization."""
        if self.is_multinomial_no_penalty:
            # Our convention is usually the symmetric parametrization where
            # sum(coef[classes, features], axis=0) = 0.
            # We convert now to this convention. Note that it does not change
            # the predicted probabilities.
            n_classes = cast("int", self.linear_loss.base_loss.n_classes)
            self.coef = self.coef.reshape(n_classes, -1, order="F")
            self.coef -= np.mean(self.coef, axis=0)
        elif self.is_multinomial_with_intercept:
            # Only the intercept needs an update to the symmetric parametrization.
            n_classes = cast("int", self.linear_loss.base_loss.n_classes)
            self.coef[-n_classes:] -= np.mean(self.coef[-n_classes:])
