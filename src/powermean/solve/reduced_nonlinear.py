"""Fit-owned retained nonlinear coordinates with one conditional FE elimination.

The human-authorized reduced organization is NOT the all-FE profiled algorithm.
Only a genuine NumericFit operation can create this layout. An eliminated block
must be wholly kept in the existing exact canonical basis. No reference/gauge,
rank, row, family or weight decision is changed. The other canonical FE
coordinates are outer nonlinear unknowns, not an inner profile warm state.

Power's existing checked log-sum update solves the conditional level equations;
NB2 uses the existing safeguarded scalar treatment with the same controls.
There is no claim of a closed form for NB2. A trial profiles this one block only.
Final original equations, support, conditioning and requested inference remain
independent consumers of the full reconstructed point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.data.numeric_fit import NumericFit
from powermean.fe.absorb_numeric import (
    checked_difference,
    checked_product,
    checked_sum,
    group_sums,
    prepare_weights,
)
from powermean.fe.certificate_basis import CertificateBasis, embed, lift
from powermean.fe.certificate_coordinates import (
    _direct_retained_coordinates, coordinates, retained_coordinates,
)
from powermean.fe.groups import Grouping
from powermean.fe.profile import BUDGET, ProfileResult, ProfileState, profile
from powermean.fe.profile_arithmetic import (
    ProfileProblem,
    _build_profile_problem,
    fresh_equations,
)
from powermean.fe.profile_nb2 import ScalarWork, profile_nb2
from powermean.fe.reductions import immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray, PowerVariance
from powermean.solve import profiled_loss
from powermean.solve.newton_convex import InnerLimitError
from powermean.systems import profiled
from powermean.systems.profiled import ProfiledProblem, base_index


@dataclass(frozen=True, eq=False)
class ConditionalSolve:
    """A returned scalar-block solve, kept before any later reconstruction fails."""

    values: FloatArray
    theta: FloatArray
    base: FloatArray
    alpha: FloatArray
    prepared: ProfileProblem
    result: ProfileResult
    scalar: ScalarWork | None

    def require(self) -> None:
        """Keep the existing inner-limit versus representability qualification."""
        if self.result.status == BUDGET:
            raise InnerLimitError(f"conditional profile: {self.result.reason}")
        if not self.result.converged:
            raise FloatingPointError(f"conditional profile: {self.result.reason}")


@dataclass(frozen=True, eq=False)
class ConditionalPoint:
    """A complete conditional root, NOT an all-FE profile.

    Coordinates hold slopes and the non-eliminated canonical coefficients.
    State holds every original block's coefficients in the canonical gauge.
    Only scalar conditional provenance is retained; the temporary scalar-solve
    evidence is released. The full index is reconstructed in declared block order
    and the conditional equation checked again at that exact rounded index.
    """

    theta: FloatArray
    base: FloatArray
    index: FloatArray
    state: ProfileState
    coordinates: FloatArray
    block: int
    scalar: ScalarWork | None

    def __post_init__(self) -> None:
        """Detach all changing numerical coordinates from trial working arrays."""
        for name in ("theta", "base", "index", "coordinates"):
            object.__setattr__(self, name, immutable_array(getattr(self, name)))


class ConditionalSpace:
    """One operation's exact canonical partition and one-block root preparation."""

    @classmethod
    def for_problem(
        cls, problem: ProfiledProblem, basis: CertificateBasis,
    ) -> ConditionalSpace | None:
        """Unknown operation/callback seams keep the unchanged requested route."""
        if (type(problem) is not ProfiledProblem
                or type(problem.__dict__.get("_numeric_owner")) is not NumericFit
                or not problem.n_slopes
                or basis.grouping is not problem.grouping
                or cls._selected(basis) is None):
            return None
        callbacks = (
            (profiled_loss.profile_point, profiled_loss._REDUCED_PROFILE),
            (profiled_loss.absorb_slopes, profiled_loss._REDUCED_ABSORB),
            (profiled_loss._absorb_slopes, profiled_loss._REDUCED_PRIVATE_ABSORB),
            (profiled.curvature_totals, profiled._ACCEPTED_CURVATURE_TOTALS),
            (profiled.prepare_weights, profiled._ACCEPTED_PREPARE_WEIGHTS),
            (profile, profiled.profile),
            (profile_nb2, profiled.profile_nb2),
            # The cold conditional opening bypasses these original hooks. A
            # changed hook keeps the requested all-FE route, not a silent bypass.
            (vars(profiled_loss.ProfiledEvaluator).get("start"), _OPENING_START),
            (vars(profiled_loss.ProfiledEvaluator).get("_completed"),
             _OPENING_COMPLETED),
            (vars(profiled_loss.ProfiledEvaluator).get("_profiled"),
             _OPENING_PROFILED),
        )
        if any(current is not accepted for current, accepted in callbacks):
            return None
        return cls(problem, basis)

    def __init__(self, problem: ProfiledProblem, basis: CertificateBasis) -> None:
        owner = problem.__dict__.get("_numeric_owner")
        if type(problem) is not ProfiledProblem or type(owner) is not NumericFit:
            raise ValueError("reduced nonlinear execution needs the numeric owner")
        if basis.grouping is not problem.grouping:
            raise ValueError("the canonical basis belongs to another operation")
        selected = self._selected(basis)
        if selected is None:
            raise ValueError("no wholly kept categorical block can be eliminated")
        self.problem, self.owner, self.basis = problem, owner, basis
        self.block = selected
        place = basis.positions[selected]
        self.eliminated = np.array(place, dtype=np.intp, copy=True)
        self.eliminated.flags.writeable = False
        size = problem.n_slopes + basis.n_kept
        mask = np.ones(size, dtype=np.bool_)
        mask[problem.n_slopes + place] = False
        self.remaining: IndexArray = np.flatnonzero(mask)
        self.remaining.flags.writeable = False
        # Header on the actual admitted immutable block, not caller-owned state.
        grouping = object.__new__(Grouping)
        object.__setattr__(grouping, "blocks", (problem.grouping.blocks[selected],))
        object.__setattr__(grouping, "positions", problem.grouping.positions)
        initial: ProfileProblem = problem.__dict__["_initial"]
        prepared = prepare_weights(grouping, problem.weights)
        self.initial = _build_profile_problem(
            prepared, initial.z, initial.base, problem.variance,
            initial.log_weights, initial.log_outcomes,
        )
        # Exactly the same admitted z/family: join the existing one-slot row
        # lifetime, not a second per-trial evaluation/cache. Group sums/brackets
        # remain local; the independent final certificate still opens a new slot.
        object.__setattr__(self.initial, "_last_eval", initial._last_eval)

    @staticmethod
    def _selected(basis: CertificateBasis) -> int | None:
        """Largest wholly kept original block; deterministic declared tie order."""
        selected, count = None, 0
        for index, place in enumerate(basis.positions):
            if (place.size > count and (place >= 0).all()
                    and np.array_equal(place, np.arange(
                        int(place[0]), int(place[0]) + place.size, dtype=np.intp,
                    ))):
                selected, count = index, place.size
        return selected

    @property
    def dimension(self) -> int:
        """Outer nonlinear dimension, not an observation-sized unknown."""
        return int(self.remaining.size)

    def opening(self, theta: FloatArray) -> FloatArray:
        """The caller's slopes and cold non-eliminated effects, without a profile."""
        if theta.shape != (self.problem.n_slopes,) or not np.isfinite(theta).all():
            raise ValueError("start must hold one finite value per slope")
        full = np.zeros(self.problem.n_slopes + self.basis.n_kept)
        full[:theta.size] = theta
        return immutable_array(full[self.remaining])

    def unpack(self, values: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Embed retained coordinates; eliminated coefficients start at zero."""
        if values.shape != (self.dimension,) or not np.isfinite(values).all():
            raise FloatingPointError("the reduced coordinates are not finite")
        full = np.zeros(self.problem.n_slopes + self.basis.n_kept)
        full[self.remaining] = values
        return full[:self.problem.n_slopes], full[self.problem.n_slopes:]

    def solve(
        self, values: FloatArray, *, tolerance: float, max_sweeps: int,
        warm: ProfileState | None = None,
    ) -> ConditionalSolve:
        """Return the actual conditional outcome before reconstructing full q."""
        theta, alpha = self.unpack(values)
        base = base_index(self.problem, theta)
        effects = embed(self.basis, alpha)
        conditional_base = checked_sum(
            base, lift(self.basis, effects), "conditional base index"
        )
        prepared = self.initial._with_base(conditional_base)
        start = None
        if warm is not None:
            start = ProfileState(
                prepared.grouping, (warm.coefficients[self.block],)
            )
        if type(self.problem.variance) is PowerVariance:
            result = profile(
                prepared, tolerance=tolerance, max_sweeps=max_sweeps, state=start
            )
            scalar = None
        else:
            found = profile_nb2(
                prepared, tolerance=tolerance, max_sweeps=max_sweeps, state=start
            )
            result, scalar = found.profile, found.scalar
        return ConditionalSolve(
            immutable_array(values), immutable_array(theta), base,
            immutable_array(alpha), prepared, result, scalar,
        )

    def reconstruct(self, solved: ConditionalSolve) -> ConditionalPoint:
        """Rebuild/check full original-order q only after a completed scalar root."""
        if (solved.prepared.grouping is not self.initial.grouping
                or solved.prepared.z is not self.initial.z
                or solved.prepared.prepared is not self.initial.prepared):
            raise ValueError("the conditional solve belongs to another operation")
        solved.require()
        result, prepared = solved.result, solved.prepared
        alpha = np.array(solved.alpha, copy=True)
        alpha[self.eliminated] = result.state.coefficients[0]
        effects = embed(self.basis, alpha)
        index = solved.base
        for block, coefficients in zip(
            self.problem.grouping.blocks, effects, strict=True
        ):
            index = checked_sum(
                index, coefficients[block.codes], "reduced reconstructed index"
            )
        checked = fresh_equations(prepared, index)
        if checked.maximum > result.tolerance:
            raise FloatingPointError(
                "the conditional root fails its fresh canonical-index bound"
            )
        # embed produced these fresh immutable numeric buffers. Do not recopy
        # or authenticate an already owned full coefficient state at every trial.
        state = object.__new__(ProfileState)
        object.__setattr__(state, "grouping", self.problem.grouping)
        object.__setattr__(state, "coefficients", effects)
        return ConditionalPoint(
            solved.theta, solved.base, index, state, solved.values,
            self.block, solved.scalar
        )

    def point(
        self, values: FloatArray, *, tolerance: float, max_sweeps: int,
        warm: ProfileState | None = None,
    ) -> ConditionalPoint:
        """Compose a conditional root and its exact original-order reconstruction."""
        return self.reconstruct(self.solve(
            values, tolerance=tolerance, max_sweeps=max_sweeps, warm=warm
        ))

    def coordinates(self, theta: FloatArray, state: ProfileState) -> FloatArray:
        """Use the accepted exact witness map, not a new gauge/reference rule."""
        try:
            eligible = (type(self) is ConditionalSpace
                and ConditionalSpace.__getattribute__ is _OPENING_SPACE_ACCESS
                and getattr(ConditionalSpace, "__getattr__", None) is _OPENING_SPACE_GETATTR
                and all(vars(ConditionalSpace).get(name) is None
                        for name in ("basis", "block", "remaining", "problem"))
                and ProfileState is _OPENING_STATE_CLASS
                and type(state) is _OPENING_STATE_CLASS
                and ProfileState.__getattribute__ is _OPENING_STATE_ACCESS
                and getattr(ProfileState, "__getattr__", None) is _OPENING_STATE_GETATTR
                and vars(ProfileState).get("coefficients") is _OPENING_STATE_FIELD
                and coordinates is _OPENING_COORDINATES
                and retained_coordinates is _OPENING_RETAINED
                and np.array_equal is _OPENING_ARRAY_EQUAL
                and np.arange is _OPENING_ARANGE
                and np.concatenate is _OPENING_CONCATENATE
                and CertificateBasis is _OPENING_BASIS_CLASS
                and vars(CertificateBasis).get("n_kept") is _OPENING_N_KEPT
                and vars(CertificateBasis).get("positions") is None
                and type(self.basis) is CertificateBasis
                and type(vars(self.basis).get("positions")) is tuple
                and bool(vars(self.basis)["positions"])
                and type(vars(self.basis).get("selected")) is np.ndarray
                and type(vars(self.basis)["positions"][0]) is np.ndarray
                and type(theta) is np.ndarray and theta.ndim == 1
                and theta.dtype == np.float64
                and type(self.remaining) is np.ndarray and self.remaining.ndim == 1
                and self.remaining.dtype == np.intp
                and self.block == 0
                and np.array_equal(self.remaining, np.concatenate((
                    np.arange(theta.size, dtype=np.intp),
                    np.arange(theta.size + self.basis.positions[0].size,
                              theta.size + self.basis.n_kept, dtype=np.intp),
                ))))
        except MemoryError:
            eligible = False
        if not eligible:
            alpha = coordinates(self.basis, state.coefficients)
            return immutable_array(np.concatenate((theta, alpha))[self.remaining])
        effects = state.coefficients
        try:
            # Already normally loaded by the numeric interface; no producer or
            # eligibility is replayed, only its literal weak live reader.
            from powermean.interface import numeric_execution as handoff
            namespace = vars(handoff)
            read = namespace.get("_direct_retained_action")
            problem = vars(self).get("problem")
            proof = (read(self.basis, problem)
                     if type(problem) is ProfiledProblem
                     and read is namespace.get("_DIRECT_RETAINED_ACTION")
                     and _direct_retained_coordinates is _OPENING_DIRECT_RETAINED
                     else None)
        except MemoryError:
            proof = None
        if proof is not None:
            retained = _direct_retained_coordinates(self.basis, effects, proof)
            if retained is not None:
                return immutable_array(np.concatenate((theta, retained)))
        retained = retained_coordinates(self.basis, effects)
        if retained is not None:
            return immutable_array(np.concatenate((theta, retained)))
        alpha = coordinates(self.basis, effects)
        return immutable_array(np.concatenate((theta, alpha))[self.remaining])

    def tangent(self, step: FloatArray, observed: FloatArray) -> FloatArray:
        """Differentiate the conditional root using its TRUE observed Jacobian.

        The outer direction may use expected curvature, but the root being
        eliminated is always the actual equation. Its derivative therefore
        cannot substitute Fisher weights for the observed row curvature.
        """
        k = self.problem.n_slopes
        alpha = np.array(step[k:], copy=True)
        alpha[self.eliminated] = 0.0
        effects = lift(self.basis, embed(self.basis, alpha))
        with np.errstate(all="ignore"):
            slopes = self.problem.design @ step[:k]
        base = checked_sum(slopes, effects, "retained direction")
        block = self.problem.grouping.blocks[self.block]
        totals = group_sums(observed, block.codes, block.n_levels)
        if not (totals > 0).all():
            raise FloatingPointError("the conditional Jacobian total is not positive")
        product = checked_product(observed, base, "conditional tangent product")
        sums = group_sums(product, block.codes, block.n_levels)
        with np.errstate(all="ignore"):
            correction = sums / totals
        return checked_difference(
            base, correction[block.codes], "conditional tangent"
        )


_OPENING_COORDINATES = coordinates
_OPENING_START = vars(profiled_loss.ProfiledEvaluator)["start"]
_OPENING_COMPLETED = vars(profiled_loss.ProfiledEvaluator)["_completed"]
_OPENING_PROFILED = vars(profiled_loss.ProfiledEvaluator)["_profiled"]
_OPENING_SPACE_COORDINATES = vars(ConditionalSpace)["coordinates"]
_OPENING_RETAINED = retained_coordinates
_OPENING_DIRECT_RETAINED = _direct_retained_coordinates
_OPENING_SPACE_ACCESS = ConditionalSpace.__getattribute__
_OPENING_SPACE_GETATTR = getattr(ConditionalSpace, "__getattr__", None)
_OPENING_ARRAY_EQUAL = np.array_equal
_OPENING_ARANGE = np.arange
_OPENING_CONCATENATE = np.concatenate
_OPENING_N_KEPT = vars(CertificateBasis).get("n_kept")
_OPENING_BASIS_CLASS = CertificateBasis
_OPENING_STATE_CLASS = ProfileState
_OPENING_STATE_ACCESS = ProfileState.__getattribute__
_OPENING_STATE_GETATTR = getattr(ProfileState, "__getattr__", None)
_OPENING_STATE_FIELD = vars(ProfileState).get("coefficients")

