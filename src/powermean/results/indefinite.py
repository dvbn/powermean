"""The note of a certified root whose curvature is not positive definite (ruling 78).

Owner's decision of 14 November 2026 (ruling 78): for a variance power outside
``[1, 2]`` the estimating equation can have several roots; the package keeps
certifying the root that it reaches, and, when the curvature at the certified root is
not positive definite, tells the user in one note that the equation may have other
roots.

Only the certificate's own record decides, with no new threshold or computation: a
certificate reads its scaled score in the current symmetric curvature where that is
positive definite (``"current_symmetric_hessian"``) and otherwise, at a selected root,
in the solve's fixed merit metric (``"fixed_merit"``); a certificate that has neither
scores NaN and certifies nothing.  A certified fit whose final certificate records
``"fixed_merit"`` is therefore exactly a certified root whose curvature is not
positive definite, on the routes whose certificate waives the curvature gate at a
selected root (the joint dense and absorbed certificates outside ``[1, 2]``, the
cancelled certificate away from the convex variances).  The fit issues the note once, as
:class:`~powermean.errors.NumericalWarning` ``numerical.indefinite_curvature`` at the
caller's line (:func:`powermean.interface.dispatch.fit_prepared`), and ``summary()``
prints its text with the other notes.  A fit that is not certified issues nothing.
"""

from __future__ import annotations

from powermean import messages
from powermean.results.retained import RetainedFit
from powermean.solve.certificate import Certificate

NOTE = "numerical.indefinite_curvature"
FIXED_MERIT = "fixed_merit"


def indefinite(certified: bool, score_metric: object) -> bool:
    """Whether a certified fit's final certificate read the fixed merit metric."""
    return bool(certified) and score_metric == FIXED_MERIT


def note(status: object) -> str:
    """The summary line of the note, with the fit's status."""
    return f"Note: {messages.render(NOTE, {'status': status})}"


def retained_indefinite(retained: RetainedFit) -> bool:
    """Whether a fit without fixed effects is a certified root of indefinite curvature.

    The joint certificate (:class:`~powermean.solve.certificate.Certificate`) records
    its metric; the linear certificate of log OLS has none and never reads a merit.
    """
    certificate = retained.certificate
    return isinstance(certificate, Certificate) and indefinite(
        retained.certified, certificate.score_metric
    )
