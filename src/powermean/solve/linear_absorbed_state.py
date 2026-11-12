"""The candidate of the absorbed linear path: its ending, its admission and its work.

Internal module; nothing here certifies a point or forms inference.  The notation is
that of
:mod:`powermean.solve.linear_absorbed_numeric`.

Endings.  A solve ends at one of :data:`STAGES` with a status that :data:`ENDINGS`
allows there; the records that the ending forms are present and no other.  A sample
without active slopes skips the stages of the slope solve.

Admission.  :class:`AbsorbedLinearCandidate`, at every construction, direct or by
``dataclasses.replace``, re-derives ``t`` and binds, in the order of the solve, the
preparation to the sample's grouping and raw weights; the joint absorption result to
its call (:func:`~powermean.solve.linear_absorbed_point._bound_absorption`: its state,
controls, work and reason, its residual bit for bit and its diagnostics against the
fresh check of that residual, re-run converged or not); the attempts and slopes; the
seed of the joint state and the slopes; the final absorption result to the fixed-slope
call, in the same way; and the point, its normalization and equations to this sample,
these slopes and the final effects, by identity or bit for bit.  It replays no LAPACK
decomposition and no absorber iteration, and states no certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np

from powermean.data.sample_fe_state import FESample, same_bits
from powermean.data.weights import WeightKind
from powermean.fe.absorb import AbsorptionResult, AbsorptionState
from powermean.fe.absorb_numeric import PreparedWeights
from powermean.fe.reductions import immutable_array
from powermean.model.variance import FloatArray
from powermean.solve.linear_absorbed_numeric import (
    FINAL_COLUMNS,
    KINDS,
    Decomposition,
    _own,
    fixed_slope_residual,
    full_rank,
    joint_columns,
    joint_values,
    raw_weights,
    seeds,
    signed_response,
    slopes,
)
from powermean.solve.linear_absorbed_point import (
    BOUND,
    FreshEquations,
    LinearControls,
    LinearPoint,
    PointNormalization,
    _bound_absorption,
    blocks_of,
    linear_sample,
)

Stage = Literal[
    "preparation",
    "joint",
    "weighting",
    "rank",
    "factorization",
    "seed",
    "residual",
    "final",
    "point",
    "normalization",
    "equations",
    "complete",
]
STAGES: tuple[Stage, ...] = get_args(Stage)
Status = Literal[
    "complete",
    "range",
    "absorption",
    "rank_deficient",
    "factorization",
    "final_absorption",
    "normalization",
]
STATUSES: tuple[Status, ...] = get_args(Status)
# The statuses with which a solve can end at each stage.
ENDINGS: dict[Stage, tuple[Status, ...]] = {
    "preparation": ("range",),
    "joint": ("absorption",),
    "weighting": ("range",),
    "rank": ("factorization", "rank_deficient"),
    "factorization": ("factorization", "range"),
    "seed": ("range",),
    "residual": ("range",),
    "final": ("range", "final_absorption"),
    "point": ("range",),
    "normalization": ("range", "normalization"),
    "equations": ("range",),
    "complete": ("complete",),
}
# The stages of the slope solve, which a sample without active slopes skips.
SLOPE_STAGES: tuple[Stage, ...] = ("weighting", "rank", "factorization")


# --------------------------------------------------------------------- the work


@dataclass(frozen=True)
class LinearWork:
    """The work of one solve as its records state it; derived, never supplied.

    ``stages``: the stages begun, the last ending the solve unless all completed.
    ``weight_reductions``: ``2 G`` level-total reductions when the preparation formed
    (``G`` by the factory, ``G`` by its constructor's revalidation), ``None`` when it
    refused (the factory does not report how far it reached).  ``absorptions``: absorber
    calls begun; ``sweeps``, ``visits``, ``operators``, ``diagnostic_passes``: the work
    of the results returned; ``decompositions``: LAPACK attempts begun.
    ``verification``: what the candidate's admission re-derives at every construction,
    the solve's own included, listed apart from the solve's work: for every absorption
    result present, its residual relation and its fresh check, which the admission
    re-runs whether or not the result converged.
    """

    stages: tuple[Stage, ...]
    weight_reductions: int | None
    absorptions: int
    sweeps: int
    visits: int
    operators: int
    diagnostic_passes: int
    decompositions: int
    verification: tuple[str, ...]


# --------------------------------------------------------------- the candidate


@dataclass(frozen=True, eq=False)
class AbsorbedLinearCandidate:
    """The candidate of one absorbed linear solve and its evidence; never a certificate.

    ``stage``: where the solve ended (:data:`STAGES`); ``status``: how
    (:data:`ENDINGS`); ``reason``: why, empty when complete.  The records formed:
    ``prepared``, ``joint`` (of ``[t, Xs]``), ``decompositions`` in order, ``seed`` (the
    new single-column state), ``final`` (of ``t - Xs theta``), ``point`` (the last
    genuinely formed full point), ``normalized`` and ``equations``.  Derived: ``t`` and
    ``theta`` (once solved).  ``ValueError`` unless every record present is the one its
    stage forms, bound to this sample and these controls.  No standard error is held.
    """

    sample: FESample
    controls: LinearControls
    stage: Stage
    status: Status
    reason: str
    prepared: PreparedWeights | None = None
    joint: AbsorptionResult | None = None
    decompositions: tuple[Decomposition, ...] = ()
    seed: AbsorptionState | None = None
    final: AbsorptionResult | None = None
    point: LinearPoint | None = None
    normalized: PointNormalization | None = None
    equations: FreshEquations | None = None
    t: FloatArray = field(init=False)
    theta: FloatArray | None = field(init=False)

    def __post_init__(self) -> None:
        """Admit the candidate: its ending, every record's presence and binding."""
        sample = linear_sample(self.sample)
        if type(self.controls) is not LinearControls:
            raise ValueError("controls must be LinearControls")
        object.__setattr__(self, "decompositions", tuple(self.decompositions))
        _ending(self)
        t = signed_response(sample)
        _own(self, t=t, theta=_admitted(self, t))

    @property
    def n_slopes(self) -> int:
        """The number of active slope columns, ``K``."""
        return int(self.sample.active.size)

    @property
    def complete(self) -> bool:
        """Whether every stage completed; a candidate, not a certificate."""
        return self.status == "complete"

    @property
    def weight_kind(self) -> WeightKind | None:
        """The stated kind of the sample's weights; ``None`` without weights."""
        weights = self.sample.data.weights
        return None if weights is None else weights.kind

    @property
    def work(self) -> LinearWork:
        """The work of the solve as the records state it (:class:`LinearWork`)."""
        return _work(self)


def _reached(stage: Stage, step: Stage) -> bool:
    """Whether ``step`` completed: the solve ended at a later stage."""
    return STAGES.index(stage) > STAGES.index(step)


def _ending(found: AbsorbedLinearCandidate) -> None:
    """``ValueError`` unless the stage, status and reason state one ending."""
    stage, status = found.stage, found.status
    if stage not in STAGES or status not in ENDINGS[stage]:
        raise ValueError("the stage and status are not one ending of the solve")
    reason = found.reason
    if not isinstance(reason, str) or (status == "complete") == bool(reason):
        raise ValueError("a reason is given exactly when the solve did not complete")
    if not found.n_slopes and (stage in SLOPE_STAGES or found.decompositions):
        raise ValueError("a sample without slopes has no slope solve")


def _presence(stage: Stage, status: Status) -> dict[str, bool]:
    """Which records the ending forms: every stage before it, and its own result."""
    return {
        "prepared": _reached(stage, "preparation"),
        "joint": _reached(stage, "preparation"),
        "seed": _reached(stage, "seed"),
        "final": _reached(stage, "final") or status == "final_absorption",
        "point": _reached(stage, "point"),
        "normalized": _reached(stage, "normalization") or status == "normalization",
        "equations": stage == "complete",
    }


def _admitted(found: AbsorbedLinearCandidate, t: FloatArray) -> FloatArray | None:
    """Bind every record present to the sample and controls; the slopes, if solved."""
    expected = _presence(found.stage, found.status)
    if any((getattr(found, name) is not None) != on for name, on in expected.items()):
        raise ValueError(BOUND)
    sample, prepared, joint = found.sample, found.prepared, found.joint
    if prepared is None or joint is None:
        return None
    if type(prepared) is not PreparedWeights:
        raise ValueError(BOUND)
    if prepared.grouping is not blocks_of(sample) or not same_bits(
        prepared.weights, raw_weights(sample)
    ):
        raise ValueError(BOUND)
    call = (joint_values(sample, t), joint_columns(found.n_slopes), prepared)
    _bound_absorption(
        found.controls, joint, call, converged=_reached(found.stage, "joint")
    )
    theta = _decomposed(found)
    if theta is not None:
        _bound_seed(found, t, theta, prepared)
    return theta


def _decomposed(found: AbsorbedLinearCandidate) -> FloatArray | None:
    """The slopes once solved (none without slopes); the attempts as they stand.

    ``direct_svd``, ``pivoted_qr`` and ``triangular`` in that order, each of the shape
    of ``W``: none before the slope solve, then as far as the ending reached.
    """
    stage, status, attempts = found.stage, found.status, found.decompositions
    solved = _reached(stage, "factorization")
    if not found.n_slopes:
        return immutable_array(np.zeros(0)) if solved else None
    shape = (found.sample.count, found.n_slopes)
    if any(type(a) is not Decomposition or a.shape != shape for a in attempts):
        raise ValueError(BOUND)
    if tuple(a.kind for a in attempts) != KINDS[: len(attempts)]:
        raise ValueError(BOUND)
    ok = [a.outcome == "ok" for a in attempts]
    full = bool(attempts) and full_rank(attempts[0], found.n_slopes)
    if solved:
        formed = len(ok) == 3 and all(ok) and full
    elif stage == "factorization":
        tails = ([True],) if status == "range" else ([False], [True, False])
        formed = full and ok[1:] in tails
    elif stage == "rank":
        formed = len(ok) == 1 and ok[0] == (status == "rank_deficient") and not full
    else:
        formed = not attempts
    if not formed:
        raise ValueError(BOUND)
    return slopes(attempts[1], attempts[2]) if solved else None


def _bound_seed(
    found: AbsorbedLinearCandidate,
    t: FloatArray,
    theta: FloatArray,
    prepared: PreparedWeights,
) -> None:
    """The seed of the joint state and the slopes; the final absorption from it."""
    joint, seed, final = found.joint, found.seed, found.final
    if joint is None or seed is None:
        return
    if type(seed) is not AbsorptionState or seed.grouping is not prepared.grouping:
        raise ValueError(BOUND)
    try:
        expected = seeds(joint.state.coefficients, theta)
        u = None if final is None else fixed_slope_residual(found.sample, t, theta)
    except FloatingPointError as error:
        raise ValueError(BOUND) from error
    if seed.columns != FINAL_COLUMNS or not all(
        same_bits(a, b) for a, b in zip(seed.coefficients, expected, strict=True)
    ):
        raise ValueError(BOUND)
    if final is None or u is None:
        return
    call = (immutable_array(u[:, None]), FINAL_COLUMNS, prepared)
    _bound_absorption(
        found.controls, final, call, converged=_reached(found.stage, "final")
    )
    _bound_evaluations(found, theta, final)


def _bound_evaluations(
    found: AbsorbedLinearCandidate, theta: FloatArray, final: AbsorptionResult
) -> None:
    """The point of these slopes and final effects; its normalization and equations."""
    point, normalized, equations = found.point, found.normalized, found.equations
    if point is None:
        return
    effects = tuple(a[:, 0] for a in final.state.coefficients)
    if type(point) is not LinearPoint or point.sample is not found.sample:
        raise ValueError(BOUND)
    if not same_bits(point.theta, theta) or not all(
        same_bits(a, b) for a, b in zip(point.effects, effects, strict=True)
    ):
        raise ValueError(BOUND)
    tolerance = found.controls.absorb_tol_final
    if normalized is not None and (
        type(normalized) is not PointNormalization
        or normalized.point is not point
        or normalized.within(tolerance) == (found.status == "normalization")
    ):
        raise ValueError(BOUND)
    if equations is not None and (
        type(equations) is not FreshEquations or equations.point is not point
    ):
        raise ValueError(BOUND)


def _relation(role: str) -> list[str]:
    """The residual relation of an absorption result and its fresh check, re-run."""
    return [f"{role} relation", f"{role} check"]


def _verification(found: AbsorbedLinearCandidate) -> tuple[str, ...]:
    """What the admission re-derives for this candidate, in its order."""
    names = ["t"]
    if found.joint is not None:
        names += _relation("joint")
    if found.n_slopes and found.theta is not None:
        names.append("slopes")
    if found.seed is not None:
        names.append("seed")
    if found.final is not None:
        names += ["fixed-slope residual", *_relation("final")]
    return tuple(names)


def _work(found: AbsorbedLinearCandidate) -> LinearWork:
    """The work of a candidate, from its ending and its records."""
    end = STAGES.index(found.stage) + 1
    skipped = () if found.n_slopes else SLOPE_STAGES
    stages = tuple(s for s in STAGES[:end] if s not in (*skipped, "complete"))
    results = [r for r in (found.joint, found.final) if r is not None]
    raised = found.stage == "final" and found.status == "range"
    blocks = len(blocks_of(found.sample).blocks)
    return LinearWork(
        stages=stages,
        weight_reductions=None if found.prepared is None else 2 * blocks,
        absorptions=len(results) + int(raised),
        sweeps=sum(r.sweeps for r in results),
        visits=sum(r.visits for r in results),
        operators=sum(r.operators for r in results),
        diagnostic_passes=sum(r.diagnostic_passes for r in results),
        decompositions=len(found.decompositions),
        verification=_verification(found),
    )
