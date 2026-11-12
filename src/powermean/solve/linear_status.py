"""The status of a linear solve: the fresh certificate of its point, or its reason.

After the
least-squares solve returns, :func:`conclude_linear` certifies the point afresh
(:func:`powermean.solve.linear_certificate.certify_linear`, with the controls of the
resolved options) and composes the status of the solve from that certificate alone:

    verified             the certificate passes at the strict level
    verified_secondary   it passes at the secondary level only
    nonfinite            otherwise, the point is not finite or its evaluation is
                         not representable (the gate ``evaluation``)
    singular_system      otherwise, the curvature is not positive definite or the
                         equilibrated system fails its test of conditioning
    solver_error         otherwise: an available point that misses the score, the
                         index or the coefficient gate.  The least-squares solve has
                         no iteration, line search or budget, so neither
                         ``iteration_limit`` nor ``line_search_failure`` can describe
                         it, and the gates alone decide

Where the solve formed no point, there is nothing to certify, and the status is that
of the candidate's own reason (:data:`NO_POINT_STATUS`), each kept apart:

    rank_deficient         singular_system   the design has not full column rank:
                                             the evidence of a singular system
    factorization_failure  solver_error      the decomposition did not converge,
                                             which is no evidence of singularity or
                                             of a missing root
    numerical_breakdown    nonfinite         a value of the solve is not
                                             representable in float64

The certificate is then ``None``: no point is invented, no evaluation is claimed, no
rank is read as a certificate, and ``no_finite_root`` is never assigned here, since
no existence analysis is formed on this path.  The native reasons of
the Newton solver (:mod:`powermean.solve.status`) are not those of a linear solve;
:class:`LinearConclusion` keeps the candidate with its own reason beside the
certificate and the status.  No status is added to :data:`~powermean.solve.status.
STATUSES`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from powermean.solve.assessment import certificate_controls
from powermean.solve.linear import LinearCandidate, LinearReason
from powermean.solve.linear_certificate import (
    LinearCertificate,
    LinearScaleProvider,
    certify_linear,
)
from powermean.solve.linear_evaluation import LinearProblem
from powermean.solve.options import SolverOptions
from powermean.solve.status import Status

_WITHOUT_POINT: dict[LinearReason, Status] = {
    "rank_deficient": "singular_system",
    "factorization_failure": "solver_error",
    "numerical_breakdown": "nonfinite",
}
NO_POINT_STATUS: Mapping[LinearReason, Status] = MappingProxyType(_WITHOUT_POINT)
# The status of a fresh certificate that fails at a finite and regular point.
AVAILABLE_POINT_FAILURE: Status = "solver_error"


def linear_status(
    reason: LinearReason, certificate: LinearCertificate | None
) -> Status:
    """The status of a linear solve from its reason and the fresh certificate.

    With a certificate, the table of the module: the level, then the gates that
    describe the point, then ``solver_error``.  Without one, the reason of the
    candidate decides; a candidate that formed a point must carry a certificate.
    """
    if certificate is None:
        if reason == "full_column_rank":
            raise ValueError("a linear candidate with a point needs its certificate")
        return NO_POINT_STATUS[reason]
    if reason != "full_column_rank":
        raise ValueError("a linear candidate without a point has no certificate")
    if certificate.level == "strict":
        return "verified"
    if certificate.level == "secondary":
        return "verified_secondary"
    if "evaluation" in certificate.strict:
        return "nonfinite"
    if "curvature" in certificate.strict or "conditioning" in certificate.strict:
        return "singular_system"
    return AVAILABLE_POINT_FAILURE


@dataclass(frozen=True, eq=False)
class LinearConclusion:
    """The status of a linear solve, with the candidate and certificate it rests on.

    ``candidate`` is the solver's record with its own reason, ``certificate`` the
    fresh certificate of its point, or ``None`` where no point was formed, and
    ``status`` one of :data:`~powermean.solve.status.STATUSES`.
    """

    candidate: LinearCandidate
    certificate: LinearCertificate | None
    status: Status


def conclude_linear(
    problem: LinearProblem,
    candidate: LinearCandidate,
    options: SolverOptions,
    *,
    reported: LinearScaleProvider | None,
    frequency: bool,
) -> LinearConclusion:
    """Certify the point of a linear solve afresh and compose its status.

    Parameters
    ----------
    problem : LinearProblem
        The original linear equation that was solved.
    candidate : LinearCandidate
        The solver's candidate, with or without a point.
    options : SolverOptions
        The resolved options, whose certificate controls apply.
    reported : LinearScaleProvider or None
        The reported scale and the standard errors at the point.
    frequency : bool
        Whether the weights of the solve are frequency weights.

    Returns
    -------
    LinearConclusion
        The certificate, where a point exists, and the status.

    Raises
    ------
    TypeError
        An argument is not of its type.
    ValueError
        A candidate that solved another number of rows or columns than the
        problem's, or whose units or weights are not the problem's.
    """
    if type(candidate) is not LinearCandidate:
        raise TypeError("candidate must be a LinearCandidate")
    if type(options) is not SolverOptions:
        raise TypeError("options must be a SolverOptions")
    _check_same_problem(problem, candidate)
    certificate = None
    if candidate.coefficients is not None:
        certificate = certify_linear(
            problem,
            candidate.coefficients,
            certificate_controls(options),
            reported=reported,
            frequency=frequency,
        )
    status = linear_status(candidate.reason, certificate)
    return LinearConclusion(candidate, certificate, status)


def _check_same_problem(problem: LinearProblem, candidate: LinearCandidate) -> None:
    """The candidate solved this problem: its dimension, units and weights."""
    if type(problem) is not LinearProblem:
        raise TypeError("problem must be a LinearProblem")
    units = candidate.units
    same = (
        candidate.dimension == problem.dimension
        and units.nobs == problem.units.nobs
        and units.identifiers.tolist() == problem.units.identifiers.tolist()
        and units.inactive.tolist() == problem.units.inactive.tolist()
        and candidate.weights.tolist() == problem.active_weights.tolist()
    )
    if not same:
        raise ValueError("the candidate did not solve this linear problem")
