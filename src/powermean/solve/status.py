"""The status of a solve: the certificate of its candidate beside its native reason.

After the solver returns, :func:`conclude`
certifies the retained candidate afresh (:func:`powermean.solve.certificate.certify`,
with the controls of the solve's options) and composes the status of the solve from
that certificate and the native reason of the retained attempt, which it keeps beside
the certificate.  The certificate assesses the candidate alone and receives no native
detail; the status needs both, and neither stands for the other.

    verified             the certificate passes at the strict level
    verified_secondary   it passes at the secondary level only
    nonfinite            otherwise, the candidate is not finite or its evaluation is
                         not representable (the gate ``evaluation``)
    singular_system      otherwise, the observed curvature is not positive definite or
                         the equilibrated system fails its test of conditioning (the
                         gates ``curvature`` and ``conditioning``)
    iteration_limit      otherwise, the native reason ``iteration_limit`` or
                         ``evaluation_limit``
    line_search_failure  otherwise, the native reason ``line_search_failure``
    solver_error         otherwise: a native stop, or a failed Newton step of the
                         solver (``factorization_failure``, ``non_descent_direction``,
                         ``numerical_breakdown``), at a finite and regular candidate
                         that misses the score, the index or the coefficient gate

How the native reasons are kept apart from the gates:

- A native stop never verifies a root: whatever the native reason, only the
  certificate gives ``verified`` and ``verified_secondary``, and a native stop at a
  candidate that the certificate rejects is ``solver_error``.  The solver's own test
  stopped where the certificate does not certify.
- ``iteration_limit`` and ``line_search_failure`` come from the native reason only,
  never from failed quantities.
- ``evaluation_limit`` is a native reason and not a status: a spent budget of
  evaluations is reported as ``iteration_limit``, the status of a spent budget, as
  SciPy's L-BFGS-B reports both of its budgets with one flag and names them in its
  message; the reason itself stays in :attr:`Conclusion.reason`.
- ``nonfinite`` and ``singular_system`` describe the candidate, and come from its gates
  before the native reason: a rank-deficient design ends with the status of the
  certificate.  In the convex systems that the solver
  accepts, ``C`` is positive semidefinite, so a curvature that is not positive
  definite is a singular one.
- ``no_finite_root`` needs an existence certificate, which this path does not form:
  it is never assigned here.

A failure of the coefficient gate alone, including missing, negative or nonfinite
standard errors, takes the status of the native reason; the certificate shows
which coefficient failed and why.  No status is added:
``ConvergenceError``, ``ExistenceError`` and ``NumericalWarning`` carry
:data:`STATUSES` as they are.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, get_args

from powermean.solve.assessment import certificate_controls
from powermean.solve.certificate import Certificate, certify
from powermean.solve.iols.candidate import IolsCandidate
from powermean.solve.joint_root_records import JointRootCandidate
from powermean.solve.newton_trace import NativeReason, NewtonCandidate
from powermean.solve.reported import ReportedScaleProvider
from powermean.systems.joint import JointSystem

# The candidates of a conclusion: the convex Newton solver's, the optional no-FE IOLS
# solver's, and the joint root solve's outside ``tau in [1, 2]``, which concludes
# itself (:mod:`powermean.solve.joint_root`) and is never passed to :func:`conclude`.
AnyNonlinearCandidate = NewtonCandidate | IolsCandidate | JointRootCandidate

Status = Literal[
    "verified",
    "verified_secondary",
    "line_search_failure",
    "iteration_limit",
    "singular_system",
    "nonfinite",
    "no_finite_root",
    "solver_error",
]
STATUSES: tuple[Status, ...] = get_args(Status)
# The status of a certificate that fails at a finite and regular candidate, by the
# native reason of the attempt that produced the candidate.
_BY_REASON: dict[NativeReason, Status] = {
    "native_stop": "solver_error",
    "iteration_limit": "iteration_limit",
    "evaluation_limit": "iteration_limit",
    "line_search_failure": "line_search_failure",
    "factorization_failure": "solver_error",
    "non_descent_direction": "solver_error",
    "numerical_breakdown": "solver_error",
}
NATIVE_STATUS: Mapping[NativeReason, Status] = MappingProxyType(_BY_REASON)


def compose_status(certificate: Certificate, reason: NativeReason) -> Status:
    """The status of a candidate's certificate and its attempt's native reason.

    The table of the module: the level of the certificate, then the gates that
    describe the candidate, then the native reason.
    """
    if certificate.level == "strict":
        return "verified"
    if certificate.level == "secondary":
        return "verified_secondary"
    if "evaluation" in certificate.strict:
        return "nonfinite"
    if "curvature" in certificate.strict or "conditioning" in certificate.strict:
        return "singular_system"
    return NATIVE_STATUS[reason]


@dataclass(frozen=True, eq=False)
class Conclusion:
    """The status of a solve, with the candidate, certificate and reason it rests on.

    ``candidate`` is the solver's record, ``certificate`` that of its retained
    coefficients, ``reason`` the native reason of the retained attempt and ``status``
    one of :data:`STATUSES`.
    """

    candidate: AnyNonlinearCandidate
    certificate: Certificate
    reason: NativeReason
    status: Status


def conclude(
    system: JointSystem,
    candidate: NewtonCandidate | IolsCandidate,
    *,
    reported: ReportedScaleProvider | None,
    frequency: bool,
) -> Conclusion:
    """Certify the retained candidate of a solve afresh and compose its status.

    Parameters
    ----------
    system : JointSystem
        The system that was solved.
    candidate : NewtonCandidate or IolsCandidate
        The solver's candidate and trace; its options give the controls.
    reported : ReportedScaleProvider or None
        The reported scale and the standard errors at the candidate.
    frequency : bool
        Whether the weights of the system are frequency weights.

    Returns
    -------
    Conclusion
        The certificate, the native reason and the status.
    """
    controls = certificate_controls(candidate.options)
    certificate = certify(
        system,
        candidate.coefficients,
        controls,
        reported=reported,
        frequency=frequency,
    )
    reason = (
        candidate.reason
        if isinstance(candidate, IolsCandidate)
        else candidate.attempts[candidate.retained].reason
    )
    status = compose_status(certificate, reason)
    return Conclusion(candidate, certificate, reason, status)
