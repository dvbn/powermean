"""The records of a joint root solve outside ``tau in [1, 2]``: start, stages, result.

The starting values and the continuation path are recorded, with the resolved solver,
options and start, and the trace; the start that produced the reported solution is
stored.  The solve itself is :mod:`powermean.solve.joint_root`.  These records hold
what ran and what it cost; none certifies anything: the conclusion's certificate
decides, and a candidate is never a certified solution.

- :class:`JointRootInitializer`: the start of the root solve, the caller's (accepted
  and recorded as given, never certified) or the nearest convex fit of the same system
  (PPML below ``tau = 1``, Gamma above ``tau = 2``), with the convex solver's own
  candidate and its fresh convex certificate, its status and native reason;
- :class:`StageSummary`: the gates and quantities of a stage point's certificate,
  without its evaluation;
- :class:`JointRootStage`: one variance stage, its attempts (the
  :class:`~powermean.solve.cancelled_records.RootAttemptRecord` of the shared kernel)
  and the summary of its retained point;
- :class:`JointRootCandidate`: the whole solve: start, resolved options and root
  options, initializer, merit metric and its fixed factor, stages, the stage of the
  target, continuation, the candidate point, the native reason of the conclusion, the
  counted work and the time, the initializer's included.

The solve-layer types are read for annotations only, so that
:mod:`powermean.solve.status` can name the candidate without an import cycle; arrays
are stored as owned bytes and handed out as fresh read-only views.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from powermean.fe._cancel_storage import FloatStorage
    from powermean.fe.admission_records import AdmissionWork
    from powermean.model.variance import FloatArray, PowerVariance
    from powermean.solve.cancelled_records import (
        MeritMetric,
        RootAttemptRecord,
        RootOptions,
    )
    from powermean.solve.certificate import Certificate, GateName, Level, ScoreMetric
    from powermean.solve.newton_trace import NativeReason, NewtonCandidate, Timing
    from powermean.solve.options import SolverOptions
    from powermean.solve.starts import Start
    from powermean.solve.status import Status

InitializerKind = Literal["ppml_tau1", "gamma_tau2", "user_start"]


@dataclass(frozen=True, eq=False)
class JointRootInitializer:
    """The start of the root solve and how it was obtained.

    Attributes
    ----------
    kind : {"ppml_tau1", "gamma_tau2", "user_start"}
        The nearest convex fit below ``tau = 1`` or above ``tau = 2``, or the caller's
        start.
    variance : PowerVariance or None
        The convex working variance of the initializer fit; ``None`` for a caller's
        start.
    certified : bool
        Whether the convex fit's fresh certificate accepted its point; a caller's
        start is never certified.
    status, reason : str or None
        The status of the convex fit and the native reason of its retained attempt;
        ``None`` for a caller's start.
    newton : NewtonCandidate or None
        The convex solver's candidate and trace.
    certificate : Certificate or None
        The convex certificate of the initializer's point, on the same system.
    """

    kind: InitializerKind
    variance: PowerVariance | None
    certified: bool
    status: Status | None
    reason: NativeReason | None
    _point: FloatStorage
    newton: NewtonCandidate | None
    certificate: Certificate | None

    @property
    def point(self) -> FloatArray:
        """The starting point of the root solve, a fresh read-only view."""
        return self._point.view()

    @property
    def iterations(self) -> int:
        """The convex solve's accepted Newton steps, charged to ``max_iter``."""
        return 0 if self.newton is None else self.newton.iterations

    @property
    def evaluations(self) -> int:
        """The convex solve's loss calls, charged to ``max_evaluations``."""
        return 0 if self.newton is None else self.newton.evaluations.total


@dataclass(frozen=True, eq=False)
class StageSummary:
    """The certificate of a stage point, read without its evaluation."""

    level: Level | None
    strict: tuple[GateName, ...]
    secondary: tuple[GateName, ...]
    scaled_score: float
    index_correction: float
    rcond: float
    positive_definite: bool
    conditioned: bool
    score_metric: ScoreMetric
    errors: Mapping[GateName, str]


@dataclass(frozen=True, eq=False)
class JointRootStage:
    """One variance stage of the root solve: its attempts and its certified point."""

    variance: PowerVariance
    attempts: tuple[RootAttemptRecord, ...]
    retained: int
    summary: StageSummary
    status: Status
    iterations: int
    evaluations: int


@dataclass(frozen=True, eq=False)
class JointRootCandidate:
    """The candidate of one joint root solve and its trace; not a certified solution.

    Attributes
    ----------
    start : Start
        The start of the fit in the solver's coordinates: the caller's (source
        ``"user"``) or the automatic start of the convex initializer fit.
    options, root_options
        The resolved options and root-Newton options with which the solve ran.
    initializer : JointRootInitializer
        The start of the root solve.
    merit_metric : {"gamma", "gram"}
        The fixed merit metric requested; its lower factor where it was formed.
    stages : tuple of JointRootStage
        The direct stage on the target, then any continuation stages, in order;
        empty where no root stage opened; a stage may accept an exact zero-step root.
    target_stage : int or None
        The stage whose retained attempt gives the native reason; ``None`` without
        stages.
    continuation_used : bool
        Whether a continuation stage ran.
    reason : str
        The native reason of the conclusion.
    work : AdmissionWork
        The counted operations of the solve, begun and completed.
    timing : Timing
        The time of the whole solve, the initializer's included.
    """

    start: Start
    options: SolverOptions
    root_options: RootOptions
    initializer: JointRootInitializer
    merit_metric: MeritMetric
    _merit_cholesky: FloatStorage | None
    stages: tuple[JointRootStage, ...]
    target_stage: int | None
    continuation_used: bool
    _point: FloatStorage
    reason: NativeReason
    work: AdmissionWork
    timing: Timing

    @property
    def coefficients(self) -> FloatArray:
        """The candidate point, a fresh read-only view."""
        return self._point.view()

    @property
    def merit_cholesky(self) -> FloatArray | None:
        """The lower factor ``L_M`` of the fixed merit metric, where it was formed."""
        return None if self._merit_cholesky is None else self._merit_cholesky.view()

    @property
    def root_iterations(self) -> int:
        """The accepted root-Newton steps of every stage."""
        return sum(stage.iterations for stage in self.stages)

    @property
    def root_evaluations(self) -> int:
        """The equation evaluations of every stage's attempts."""
        return sum(stage.evaluations for stage in self.stages)

    @property
    def iterations(self) -> int:
        """Every accepted step, the initializer's included, charged to ``max_iter``."""
        return self.initializer.iterations + self.root_iterations

    @property
    def evaluations(self) -> int:
        """Every evaluation, the initializer's included."""
        return self.initializer.evaluations + self.root_evaluations
