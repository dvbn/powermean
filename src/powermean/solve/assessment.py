"""The independent assessment that the retry of the convex Newton solver consumes.

A native stop of the Newton iteration is the solver's own test and never a
verification: a solver proposes, the certificate decides.  The one decision of the
solver that needs more is its retry: it starts a second attempt only when an
independent audit says that the native stop missed its gate.  This module states
what the solver receives from such an assessment and nothing else; it computes
none.  The certificate supplies the assessor.

An :class:`Assessment` of a point holds what the retry reads of its audit:

    accepted   the point passes the independent gate, which ends the attempts
    score      compared with 0.99 times the score of the start
    error      the common error, compared between the two attempts

The package's certificate measures the scaled score and its other criteria.  The
assessor receives the system being
solved, the coefficients and the four controls of the certificate
(:class:`CertificateControls`), which the iteration never reads: ``tol``,
``tol_secondary``, ``coef_tol`` and ``rcond_min``.  An exception of the assessor is
not caught.  :class:`AssessmentLog` counts and times the assessments of one solve
apart from the iteration, so that the cost of the audit is reported with the rest
and not hidden in it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

import numpy as np

from powermean.model.variance import FloatArray
from powermean.solve.options import SolverOptions
from powermean.systems.joint import JointSystem


@dataclass(frozen=True)
class CertificateControls:
    """The four options of the certificate, for the assessor.

    ``tol`` and ``tol_secondary`` bound the scaled score and the index correction,
    ``coef_tol`` the coefficient correction in standard errors, and ``rcond_min`` the
    reciprocal condition number; their values are those of :class:`SolverOptions`.
    """

    tol: float
    tol_secondary: float
    coef_tol: float
    rcond_min: float


def certificate_controls(options: SolverOptions) -> CertificateControls:
    """The controls of the certificate among the resolved options."""
    return CertificateControls(
        tol=options.tol,
        tol_secondary=options.tol_secondary,
        coef_tol=options.coef_tol,
        rcond_min=options.rcond_min,
    )


def _flag(value: object, name: str) -> bool:
    """``value`` if it is a ``bool``; a NumPy boolean or a number is refused.

    Typed as ``object`` because it is the run-time check of what the field declares.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _number(value: object, name: str) -> float:
    """A real number other than a boolean, as ``float``; it may be infinite or NaN.

    Typed as ``object`` because it is the run-time check of what the field declares.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    return float(value)


@dataclass(frozen=True)
class Assessment:
    """An independent assessment of one point, as the retry reads it.

    Parameters
    ----------
    accepted : bool
        Whether the point passes the independent gate.
    score : float
        The score of the point, compared with 0.99 times that of the start; an
        infinite or NaN score reports a point that could not be assessed.
    error : float
        The common error, compared between the attempts; it may be infinite or NaN.

    Notes
    -----
    ``accepted`` must be a ``bool``; ``score`` and ``error`` real numbers other than
    booleans, stored as ``float``.  Another value raises ``TypeError``.
    """

    accepted: bool
    score: float
    error: float

    def __post_init__(self) -> None:
        """Check the types of the fields and store the two numbers as ``float``."""
        object.__setattr__(self, "accepted", _flag(self.accepted, "accepted"))
        object.__setattr__(self, "score", _number(self.score, "score"))
        object.__setattr__(self, "error", _number(self.error, "error"))

    @property
    def finite_error(self) -> bool:
        """Whether the common error is a finite number."""
        return math.isfinite(self.error)


def _returned(value: object) -> Assessment:
    """``value`` if it is an :class:`Assessment`, the run-time check of an assessor."""
    if not isinstance(value, Assessment):
        raise TypeError("an assessor must return an Assessment")
    return value


class Assessor(Protocol):
    """An independent assessment of coefficients of a system, by the certificate."""

    def __call__(
        self,
        system: JointSystem,
        gamma: FloatArray,
        controls: CertificateControls,
        /,
    ) -> Assessment:
        """The assessment of ``gamma`` on ``system`` with the controls given."""
        ...


class AssessmentLog:
    """The assessments of one solve: each one made, counted and timed.

    Parameters
    ----------
    assessor : Assessor
        The independent assessment, supplied by the caller.
    system : JointSystem
        The system being solved, handed to every assessment.
    controls : CertificateControls
        The controls of the certificate, handed to every assessment.

    Attributes
    ----------
    count : int
        The assessments made.
    wall, cpu : float
        Their elapsed and processor seconds, apart from those of the iteration.
    """

    def __init__(
        self,
        assessor: Assessor,
        system: JointSystem,
        controls: CertificateControls,
    ) -> None:
        self.assessor = assessor
        self.system = system
        self.controls = controls
        self.count = 0
        self.wall = 0.0
        self.cpu = 0.0

    def __call__(self, gamma: FloatArray) -> Assessment:
        """The assessment of ``gamma``, which must be an :class:`Assessment`."""
        wall, cpu = time.perf_counter(), time.process_time()
        result = self.assessor(self.system, gamma, self.controls)
        self.wall += time.perf_counter() - wall
        self.cpu += time.process_time() - cpu
        self.count += 1
        return _returned(result)
