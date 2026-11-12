"""The names printed for an estimator, a mean order and the coverage of the theory.

- "PPML" and "GPML" name ``omega = 1`` with ``tau = 1`` and ``tau = 2`` only; the
  other cases are described by their indices.  "Log OLS" names ``omega = 0``.  An NB2
  working variance is named with its ``kappa`` and marked as an element of the package
  that is not in BBP.
- A prediction is the fitted conditional power mean of order ``omega``: the conditional
  mean of ``Y`` at ``omega = 1`` only, the geometric mean at ``omega = 0``, the harmonic
  mean at ``omega = -1``.
- The coverage label of a no-FE joint fit is ``unique_root`` for the power family with
  ``tau`` in ``[1, 2]`` at a positive order (``prop:iols-exact-sample-root``), at a
  negative order on positive outcomes (the same transformed-problem
  sample conditions, a package-design extension and not a new BBP theorem), for log
  OLS and for joint NB2 (a convexity argument, not in BBP);
  it is ``selected_root`` for the power family outside ``[1, 2]``, the root that root
  Newton reaches from the certified nearest convex fit of the same sample (PPML below
  1, Gamma above 2) or from the caller's recorded start, certified as a regular root
  and not as unique.  The label is
  conditional on the sample conditions and on the numerical certificate, which are
  separate: a label is not a certification, and neither is a statement about the
  asymptotic validity of the inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from powermean.data.specs import Estimator, FixedEffectFormulation
from powermean.model.variance import NB2Variance, PowerVariance

CoverageLabel = Literal["unique_root", "selected_root"]
NOT_IN_BBP = "a design element not in BBP"


@dataclass(frozen=True)
class Coverage:
    """A coverage label and its basis."""

    label: CoverageLabel
    basis: str


def order_text(omega: float) -> str:
    """``omega`` as printed: an integer without a fraction, else Python's shortest."""
    return str(int(omega)) if omega == int(omega) else repr(omega)


def mean_label(omega: float) -> str:
    """The name of the power mean of order ``omega``."""
    if omega == 1:
        return "arithmetic mean"
    if omega == 0:
        return "geometric mean"
    if omega == -1:
        return "harmonic mean"
    return f"power mean of order {order_text(omega)}"


def variance_text(estimator: Estimator) -> str:
    """The working variance as printed; none on the linear route."""
    if estimator.route == "linear":
        return "none (log OLS)"
    variance = estimator.variance
    if isinstance(variance, NB2Variance):
        return f"NB2, m + kappa m^2 with kappa = {variance.kappa!r} ({NOT_IN_BBP})"
    return f"m^tau, tau = {order_text(variance.tau)}"


def estimator_name(estimator: Estimator) -> str:
    """The name of the estimator by the naming rules."""
    omega = estimator.mean.omega
    if estimator.route == "linear":
        return "log OLS (omega = 0)"
    variance = estimator.variance
    if isinstance(variance, PowerVariance) and omega == 1:
        if variance.tau == 1:
            return "PPML (omega = 1, tau = 1)"
        if variance.tau == 2:
            return "GPML (omega = 1, tau = 2)"
    indices = f"omega = {order_text(omega)}"
    if isinstance(variance, NB2Variance):
        return f"power-mean regression, {indices}, NB2 working variance"
    return f"power-mean regression, {indices}, tau = {order_text(variance.tau)}"


def prediction_label(omega: float) -> str:
    """What a prediction is: the conditional mean of ``Y`` at ``omega = 1`` only."""
    if omega == 1:
        return "fitted conditional mean of the outcome (arithmetic mean, omega = 1)"
    return (
        f"fitted conditional {mean_label(omega)} of the outcome "
        f"(omega = {order_text(omega)})"
    )


def coverage(estimator: Estimator) -> Coverage:
    """The coverage label of a no-FE fit and its basis."""
    omega = estimator.mean.omega
    formulation = estimator.fixed_effects
    if (
        isinstance(formulation, FixedEffectFormulation)
        and formulation.cancelled is not None
    ):
        variance = estimator.variance
        if isinstance(variance, PowerVariance) and variance.tau in (1.0, 2.0):
            basis = (
                "profiled PPML" if variance.tau == 1.0 else "BBP prop:fe-fixed-design"
            )
            return Coverage("unique_root", basis)
        return Coverage(
            "selected_root",
            "BBP prop:fe-growing-selection; certified regular root, not uniqueness",
        )
    if estimator.route == "linear":
        return Coverage("unique_root", "log OLS, least squares on log Y")
    variance = estimator.variance
    if isinstance(variance, NB2Variance):
        return Coverage(
            "unique_root",
            "joint NB2: a convexity argument, " + NOT_IN_BBP,
        )
    if not 1 <= variance.tau <= 2:
        return Coverage(
            "selected_root",
            "joint estimation with tau outside [1, 2]: the root reached by Newton "
            "from the certified nearest convex fit of the same sample (PPML below 1, "
            "Gamma above 2) or from the caller's recorded start; a certified regular "
            "root, not unique; prop:fe-growing-selection, a large-sample statement, "
            "concerns the Gamma-weighted start",
        )
    if omega > 0:
        return Coverage("unique_root", "BBP prop:iols-exact-sample-root")
    return Coverage(
        "unique_root",
        "negative mean order on positive outcomes: the same transformed-problem "
        "sample conditions as prop:iols-exact-sample-root, a package-design "
        "extension, not a new BBP theorem",
    )
