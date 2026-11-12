"""The records of the absorbed joint root solve outside ``tau in [1, 2]``.

A joint fit outside ``[1, 2]`` starts from the nearest convex member (PPML below 1,
Gamma above 2, on the same sample and formulation), and its continuation path is
recorded; the result is a selected local root, certified as a regular root, not as
unique.  The solve itself is
:mod:`powermean.solve.absorbed_root`.  These records hold what ran and what it cost;
none certifies anything: the certificate of the selected point decides, and a
candidate is never a certified solution.

- :class:`RootPoint`: one EXACT point of the full joint system ``[X, E_J]``: the
  slopes, the coordinates of the kept incidence ``E_J``
  (:mod:`powermean.fe.certificate_basis`), the base ``fl(X theta + o)``, the effects
  of every original block (the coordinates placed at their levels, exact zeros at
  references and dependent levels) and the fitted index rebuilt from them by the
  accepted ordered additions.  No absorption, profile or recovery forms any part of
  it, so it holds no inner-accuracy record;
- :class:`MeritFactor`: the fixed positive-definite merit metric ``M`` formed once at
  the start, its scale, the symmetric scaled matrix factorized and the factor that
  whitens ``u = D^-1/2 L^-1 P (S/s)``, ``||u||^2 = S' M^-1 S``;
- :class:`RootDirectionRecord`: one Newton direction: its factor outcome, nonzeros,
  conditioning and the counted solves;
- :class:`RootStageSummary` and :class:`AbsorbedRootStage`: one variance stage, its
  attempts (the :class:`~powermean.solve.cancelled_records.RootAttemptRecord` of the
  shared kernel) and the gates of its retained point's certificate;
- :class:`AbsorbedRootInitializer`: the caller's start, or the nearest convex fit of
  the same sample by the REQUESTED organization's unchanged convex solver, with that
  solver's candidate and its own audit (its history, certificate and inner checks,
  which actually ran);
- :class:`AbsorbedRootCandidate`: the whole solve.

The organization requested at the fit (``"index"`` or ``"profiled"``) is recorded as
requested; the operation of the root stages is :data:`OPERATION` whatever was
requested, and the convex initializer is the requested organization's own solve.
Neither is labelled as the other.  Arrays are owned read-only copies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np

from powermean.data.accounting import frozen_index
from powermean.fe.reductions import immutable_array
from powermean.solve.absorbed_direction import Normalization

if TYPE_CHECKING:
    from powermean.data.sample_fe_state import FESample
    from powermean.fe._cancel_storage import FloatStorage
    from powermean.fe.admission_records import AdmissionWork
    from powermean.model.order import IndexArray
    from powermean.model.variance import FloatArray, PowerVariance
    from powermean.solve.absorbed_certificate import (
        CandidateAudit,
        FEGate,
        FEPointCertificate,
        Level,
    )
    from powermean.solve.absorbed_certificate_factor import FactorRecord
    from powermean.solve.absorbed_certificate_geometry import SparseMatrix
    from powermean.solve.cancelled_records import (
        MeritMetric,
        RootAttemptRecord,
        RootOptions,
    )
    from powermean.solve.absorbed import AbsorbedOptions
    from powermean.solve.newton_absorbed import IndexCandidate
    from powermean.solve.newton_profiled import ProfiledCandidate
    from powermean.solve.newton_trace import NativeReason, Timing
    from powermean.solve.options import SolverOptions

RootInitializerKind = Literal["ppml_tau1", "gamma_tau2", "user_start"]
# The operation of every root stage, whatever organization the fit requested.
OPERATION = "sparse full-joint root Newton"
ScoreMetric = Literal["current_symmetric_hessian", "fixed_merit", "none"]


# ------------------------------------------------------------------- the point


@dataclass(frozen=True, eq=False)
class RootPoint:
    """One exact point of the full joint system (module docstring).

    Attributes
    ----------
    theta : array of float
        The slopes in the coordinates of the sample's scaled design (``K`` values).
    alpha : array of float
        The coordinates of the kept incidence ``E_J``, one per kept column.
    base : array of float
        ``fl(X theta + o)``, the transformed offset entered once.
    effects : tuple of arrays of float
        ``alpha`` placed at the levels of every original block
        (:func:`~powermean.fe.certificate_basis.embed`): exact zeros at references and
        dependent levels.  A deterministic representative with the indicator rank's
        bounds, never a claim of unique three-plus-block effects.
    index : array of float
        ``((base + effects_1[c_1]) + effects_2[c_2]) + ...``, the accepted ordered
        reconstruction: the authoritative fitted index of the point.
    normalization : Normalization or None
        The accepted component normalization of ``effects``
        (:func:`~powermean.solve.absorbed_direction.normalize`), formed for a selected
        point only; ``None`` for the iterates.
    """

    theta: FloatArray
    alpha: FloatArray
    base: FloatArray
    effects: tuple[FloatArray, ...]
    index: FloatArray
    normalization: Normalization | None = None

    def __post_init__(self) -> None:
        """Own the arrays."""
        for name in ("theta", "alpha", "base", "index"):
            object.__setattr__(self, name, immutable_array(getattr(self, name)))
        object.__setattr__(
            self, "effects", tuple(immutable_array(e) for e in self.effects)
        )
        given = self.normalization
        if given is not None:
            if type(given) is not Normalization:
                raise TypeError("normalization must be a Normalization or None")
            object.__setattr__(
                self,
                "normalization",
                Normalization(
                    tuple(immutable_array(e) for e in given.effects),
                    tuple(frozen_index(r) for r in given.references),
                    tuple(immutable_array(s) for s in given.shifts),
                ),
            )

    @property
    def gamma(self) -> FloatArray:
        """``[theta, alpha]``, the point in the coordinates of the root solve."""
        return immutable_array(np.concatenate([self.theta, self.alpha]))


# ------------------------------------------------------------------- the metric


@dataclass(frozen=True, eq=False)
class MeritFactor:
    """The fixed merit metric of one root solve, formed once at its start.

    ``metric`` is the requested metric: ``"gamma"``, the observed bread of the Gamma
    equation of the same rows at the start, ``B' diag(a c_Gamma) B``; or ``"gram"``,
    ``B'B`` on the rows of positive weight.  ``matrix`` is that raw ``M`` (owned
    canonical CSC), ``scale`` its equilibration ``s``, ``symmetric`` the scaled
    symmetric ``Aplus`` factorized, ``record`` the factor attempt (kind ``"merit"``),
    and ``l_indptr``, ``l_indices``, ``l_data`` the unit lower factor ``L`` as SciPy
    returned it, with ``diagonal`` ``d = diag(U) > 0`` and ``perm_r == perm_c``.
    Then ``||D^-1/2 L^-1 Pr (S/s)||^2 = S' M_sym^-1 S``; no other metric replaces it.
    """

    metric: MeritMetric
    matrix: SparseMatrix
    scale: FloatArray
    symmetric: SparseMatrix
    record: FactorRecord
    l_indptr: IndexArray
    l_indices: IndexArray
    l_data: FloatArray
    diagonal: FloatArray
    perm_r: IndexArray

    def __post_init__(self) -> None:
        """Own the factor arrays, the index arrays as ``intp``."""
        object.__setattr__(self, "scale", immutable_array(self.scale))
        for name in ("l_indptr", "l_indices", "perm_r"):
            found = np.asarray(getattr(self, name), dtype=np.intp)
            object.__setattr__(self, name, frozen_index(found))
        object.__setattr__(self, "l_data", immutable_array(self.l_data))
        object.__setattr__(self, "diagonal", immutable_array(self.diagonal))

    @property
    def dimension(self) -> int:
        """``K + |J|``."""
        return int(self.scale.size)


# ------------------------------------------------------------------- the work


@dataclass(frozen=True)
class RootDirectionRecord:
    """One Newton direction of the root solve: factor, conditioning, solves.

    ``status`` is ``"solved"``, or ``"refused"`` where the direction was refused
    before or by its factorization or conditioning (the attempt then stops with
    ``factorization_failure``), or ``"failed"`` where the solve was not representable
    (``numerical_breakdown``); ``l_nnz``/``u_nnz`` the nonzeros of the ordinary
    factor, ``estimate`` the inverse-norm estimate and ``rcond`` the reciprocal
    condition in scaled coordinates (NaN where not formed); ``actions`` the counted
    SuperLU solves of the direction.
    """

    tau: float
    status: Literal["solved", "refused", "failed"]
    reason: str
    l_nnz: int | None
    u_nnz: int | None
    estimate: float
    rcond: float
    actions: Mapping[str, int]

    def __post_init__(self) -> None:
        """Freeze the actual action counts, including on direct construction."""
        object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))


@dataclass(frozen=True, eq=False)
class AbsorbedRootWork:
    """The counted work of a root solve: operations, directions, merit solves.

    ``counts`` the begun and completed operations of the solve's work log,
    ``directions`` every Newton direction in order and ``merit_actions`` the
    triangular solves of the merit whitening.  The initializer's own work is kept by
    its candidate.
    """

    counts: AdmissionWork
    directions: tuple[RootDirectionRecord, ...]
    merit_actions: Mapping[str, int]

    def __post_init__(self) -> None:
        """Freeze the merit counts."""
        object.__setattr__(
            self, "merit_actions", MappingProxyType(dict(self.merit_actions))
        )
        object.__setattr__(self, "directions", tuple(self.directions))


# ------------------------------------------------------------------- the stages


@dataclass(frozen=True, eq=False)
class RootStageSummary:
    """The certificate of a stage point, read without its evaluation."""

    level: Level | None
    strict: tuple[FEGate, ...]
    secondary: tuple[FEGate, ...]
    not_applicable: tuple[FEGate, ...]
    scaled_score: float
    index_correction: float
    rcond: float
    positive_definite: bool
    conditioned: bool
    score_metric: ScoreMetric
    effects: float
    errors: Mapping[str, str]

    def __post_init__(self) -> None:
        """Freeze the stage's diagnostic map before it enters either history."""
        object.__setattr__(self, "errors", MappingProxyType(dict(self.errors)))


@dataclass(frozen=True, eq=False)
class AbsorbedRootStage:
    """One variance stage of the root solve: its attempts and its certified point."""

    variance: PowerVariance
    attempts: tuple[RootAttemptRecord, ...]
    retained: int
    summary: RootStageSummary
    status: str
    iterations: int
    evaluations: int


# ------------------------------------------------------------------- the start


@dataclass(frozen=True, eq=False)
class AbsorbedRootInitializer:
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
        Whether the convex fit's own audit accepted its selected point; a caller's
        start is never certified.
    status, reason : str or None
        The convex fit's status (``"verified"``, ``"verified_secondary"`` or its
        operative reason) and its operative reason; ``None`` for a caller's start.
    candidate : IndexCandidate or ProfiledCandidate or None
        The requested organization's convex candidate on the SAME sample.
    audit : CandidateAudit or None
        Its audit: the admitted history and the certificate of its selected point,
        with the inner checks and work that actually ran.
    point : RootPoint or None
        The start of the root solve in the full joint coordinates: the caller's
        slopes with zero nuisance coordinates, or the convex point's slopes with the
        coordinates of its effects (``alpha = sum_d w_d e_d`` by the exact witnesses);
        ``None`` where the convex fit holds no point with representable effects (no
        root stage then opens).
    gap : float or None
        ``max |q_start - q_convex|`` of that start against the convex certificate's
        authoritative index: recorded, not a gate; ``None`` for a caller's start.
    """

    kind: RootInitializerKind
    variance: PowerVariance | None
    certified: bool
    status: str | None
    reason: str | None
    candidate: IndexCandidate | ProfiledCandidate | None
    audit: CandidateAudit | None
    point: RootPoint | None
    gap: float | None

    @property
    def iterations(self) -> int:
        """The convex solve's accepted Newton steps, charged to ``max_iter``."""
        return 0 if self.candidate is None else self.candidate.iterations

    @property
    def evaluations(self) -> int:
        """The convex solve's pointwise evaluations, charged to ``max_evaluations``."""
        if self.candidate is None:
            return 0
        return sum(
            0 if a.iteration is None else a.iteration.evaluations.total
            for a in self.candidate.attempts
        )


# ------------------------------------------------------------------- the candidate


@dataclass(frozen=True, eq=False)
class AbsorbedRootCandidate:
    """The candidate of one absorbed joint root solve and its trace; not a fit.

    Attributes
    ----------
    sample : FESample
        The authoritative target sample, also the initializer's.
    variance : PowerVariance
        The target working variance, ``tau`` outside ``[1, 2]``.
    start : FloatStorage or None
        The caller's slopes in the solver's coordinates, as given; ``None`` for the
        automatic nearest convex start.
    options, root_options
        The resolved solver and root-Newton options with which the solve ran.
    absorbed : AbsorbedOptions
        The options of the organization REQUESTED at the fit, which its convex
        initializer ran; the root stages are :data:`OPERATION`.
    initializer : AbsorbedRootInitializer
        The start of the root solve.
    merit : MeritFactor or None
        The fixed merit metric, where it was formed; ``merit_reason`` the native
        reason and text of its absence otherwise.
    stages : tuple of AbsorbedRootStage
        The direct stage on the target, then any continuation stages, in order; empty
        where no root stage opened.
    target_stage : int or None
        The stage whose retained attempt gives the native reason.
    continuation_used : bool
        Whether a continuation stage ran.
    point : RootPoint or None
        The selected point, ``None`` where no root stage opened or the point is not
        representable.
    certificate : FEPointCertificate or None
        The fresh certificate of ``point`` on the target equation.
    reason, message : str
        The operative reason of the conclusion and its text.
    native : str or None
        The native reason of the target stage's retained attempt, ``None`` without a
        root stage.
    work : AbsorbedRootWork
        The counted work of the root solve.
    timing : Timing
        The time of the whole solve, the initializer's included.
    """

    sample: FESample
    variance: PowerVariance
    start: FloatStorage | None
    options: SolverOptions
    root_options: RootOptions
    absorbed: AbsorbedOptions
    initializer: AbsorbedRootInitializer
    merit: MeritFactor | None
    merit_reason: str
    stages: tuple[AbsorbedRootStage, ...]
    target_stage: int | None
    continuation_used: bool
    point: RootPoint | None
    certificate: FEPointCertificate | None
    reason: str
    message: str
    native: NativeReason | None
    work: AbsorbedRootWork
    timing: Timing

    @property
    def organization(self) -> str:
        """The organization requested at the fit (its initializer's)."""
        return self.absorbed.absorbed

    @property
    def operation(self) -> str:
        """The operation of the root stages."""
        return OPERATION

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
