"""One-fit reduced linear actions for the existing profiled/index organizations.

The nonlinear profiles, provider K-Schur loop, trial
organization and all stopping rules stay with their existing consumers. Only the
linear projection/RHS backend changes. Every current weighted system must pass
the existing partition/resource predicate BEFORE factorization. A failed selected
factor is an error, never a request to run the iterative backend.

No public problem acquires this owner. The admitted NumericFit supplies immutable
row inputs; the canonical basis and pair incidence belong to this operation.
Only a freshly completed observed FINAL projection can offer its ordinary factor
to the independent fresh certificate. Aplus, support, conditioning and equations
are still evaluated there. Factors never enter retained public evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from powermean.data.numeric_fit import NumericFit, _immutable_bytes
from powermean.data.sample_fe_source import Derived
from powermean.fe.absorb import AbsorptionResult, AbsorptionState
from powermean.fe.absorb_acceleration import BREAKDOWN, Work
from powermean.fe.absorb_numeric import (
    Attempted,
    PreparedWeights,
    checked_difference,
    fresh_check,
    zero_coefficients,
)
from powermean.fe.certificate_basis import (
    CertificateBasis,
    embed,
    lift,
    positive_support,
)
from powermean.fe.groups import Block, Grouping
from powermean.fe.reductions import immutable_array
from powermean.model.variance import FloatArray
from powermean.solve import absorbed_certificate_geometry as geometry
from powermean.solve import absorbed_rhs as rhs
from powermean.solve import compact_assembly as compact
from powermean.solve.absorbed_certificate_block import (
    BlockFactor,
    _factor_parts,
    _parts,
    block_factorized,
)
from powermean.solve.absorbed_certificate_factor import (
    FactorRecord,
    SuperLU,
    index_width,
)
from powermean.solve.absorbed_certificate_geometry import (
    Assembly,
    _mixed_entries,
    _slope_entries,
    equilibration,
    moment,
    scaled_ordinary,
)
from powermean.solve.compact_assembly import CompactEntries, JointLayout
from powermean.solve.compact_matrix import CompactMatrix
from powermean.systems import profiled
from powermean.systems.profiled import (
    SCHUR,
    Absorbed,
    ProfiledProblem,
    _prepare_curvature,
)

if TYPE_CHECKING:
    from powermean.solve.absorbed_certificate_point import AdmittedPoint, FreshRows


def _same(left: FloatArray, right: FloatArray) -> bool:
    """Bit equality of owned ordinary float64 arrays, including signed zeros."""
    return left.shape == right.shape and np.array_equal(
        left.view(np.uint64), right.view(np.uint64)
    )


@dataclass(frozen=True, eq=False)
class FinalFactor:
    """An internal, exact-final observed ordinary operator; not a cached verdict."""

    owner: NumericFit
    variance: object
    basis: CertificateBasis
    q: FloatArray
    weights: FloatArray
    raw: CompactMatrix
    scale: FloatArray
    record: FactorRecord
    factor: BlockFactor
    projection: AbsorptionResult | None = None

    def matches(
        self, point: AdmittedPoint, rows: FreshRows, basis: CertificateBasis
    ) -> bool:
        """Match actual numeric owner, point, family, raw weights and coordinates."""
        numeric = point._numeric_problem
        return (
            numeric is not None
            and numeric.__dict__.get("_numeric_owner") is self.owner
            and point.variance is self.variance
            and basis is self.basis
            and point.sample.grouping is basis.grouping
            and _same(point.q, self.q)
            and _same(rows.curvature, self.weights)
            and _same(rows.weights, self.owner.weights)
        )

    def assembly(self, rows: FreshRows) -> Assembly:
        """Fresh score sums, with the exactly bound current-weight Hessian."""
        design = self.owner.design
        return Assembly(design, moment(design, rows.b, self.basis), self.raw)


@dataclass(frozen=True, eq=False)
class Projection:
    """One current factor and its algebraic D-inverse slope map."""

    prepared: PreparedWeights
    raw: CompactMatrix
    scale: FloatArray
    record: FactorRecord
    factor: BlockFactor
    coefficients: FloatArray
    absorbed: Absorbed


@dataclass(frozen=True, eq=False)
class JointFactor:
    """One fresh full-current operator, without slope projection actions."""

    prepared: PreparedWeights
    raw: CompactMatrix
    scale: FloatArray
    record: FactorRecord
    factor: SuperLU


class ReducedExecution:
    """One actual numeric operation, one invariant layout, at most one live factor."""

    def __init__(self, problem: ProfiledProblem, basis: CertificateBasis) -> None:
        if type(problem) is not ProfiledProblem:
            raise ValueError("reduced execution needs the actual numeric fit owner")
        owner = problem.__dict__.get("_numeric_owner")
        if type(owner) is not NumericFit:
            raise ValueError("reduced execution needs the actual numeric fit owner")
        self.problem, self.owner, self.basis = problem, owner, basis
        self.layout = JointLayout(basis, problem.n_slopes)
        self.current: Projection | None = None
        self.final: FinalFactor | None = None

    @classmethod
    def _from_owned(
        cls, problem: ProfiledProblem, basis: CertificateBasis,
    ) -> ReducedExecution:
        """Only the admitted interface composition offers immutable route reuse."""
        headers = _structure_headers(problem, basis)
        if headers is None:
            return cls(problem, basis)
        result = _OwnedReducedExecution(problem, basis)
        result._headers = headers
        return result

    def _assemble(self, curvature: FloatArray) -> CompactMatrix:
        """The original dynamic emitter/sink path for public or changed inputs."""
        problem, basis, block = self.problem, self.basis, self.layout.block
        assert block is not None
        entries = CompactEntries(problem.n_slopes + basis.n_kept, block)
        _slope_entries(problem.design, curvature, entries)
        _mixed_entries(problem.design, curvature, basis, entries)
        self.layout.add_pairs(curvature, entries)
        return entries.finish()

    def _joint_factor(
        self, prepared: PreparedWeights, *, complete: bool = False,
    ) -> JointFactor | None:
        """Assemble once; select the ordinary producer before a factor.

        The projection-only caller retains its original ineligible result. The
        nonlinear caller uses the existing ordinary sparse factor when a reduced
        factor is structurally/resource ineligible, never after a failed attempt.
        """
        problem, basis, block = self.problem, self.basis, self.layout.block
        if block is None or not positive_support(basis, prepared.weights > 0).full:
            if complete:
                raise FloatingPointError("the current FE curvature lacks full support")
            return None
        size = problem.n_slopes + basis.n_kept
        raw = self._assemble(prepared.weights)
        reason = index_width(size, raw.nnz)
        if reason:
            if complete:
                raise FloatingPointError(reason)
            return None
        try:
            scale = equilibration(raw)
        except ValueError as error:
            if complete:
                raise FloatingPointError(str(error)) from error
            return None
        ordinary = scaled_ordinary(raw, scale)
        parts = _parts(ordinary, block, ordinary=True)
        if parts is None:
            if not complete:
                return None
            record, factor = block_factorized(ordinary, "ordinary", block)
        else:
            record, factor = _factor_parts(
                parts, "ordinary", dense_entry_budget=raw.nnz,
            )
        if factor is None:
            raise FloatingPointError(record.reason)
        return JointFactor(prepared, raw, scale, record, factor)

    def _factor(
        self, prepared: PreparedWeights, tolerance: float, acceleration: bool
    ) -> Projection | None:
        """The unchanged projection actions on the one fresh current factor."""
        current = self._joint_factor(prepared)
        if current is None:
            return None
        factor = current.factor
        if not isinstance(factor, BlockFactor):
            raise FloatingPointError(current.record.reason)
        coefficients = self._coefficients(factor, current.scale)
        absorbed = self._projected(prepared, coefficients, tolerance, acceleration)
        return Projection(
            prepared, current.raw, current.scale, current.record,
            factor, coefficients, absorbed,
        )

    def joint_step(
        self, weights: FloatArray, b: FloatArray,
    ) -> tuple[FloatArray, FloatArray, PreparedWeights]:
        """One full-current RHS action, no K projections or iterative FE solves.

        All nuisance RHS entries survive, including scores at zero-C rows.
        No row is divided by C. This direction never offers an origin factor as
        final: only the existing independent observed final producer can do so.
        """
        self.current, self.final = None, None
        prepared = _prepare_curvature(self.problem, weights)
        current = self._joint_factor(prepared, complete=True)
        assert current is not None
        sums = moment(self.problem.design, b, self.basis)
        with np.errstate(all="ignore"):
            solved = current.factor.solve(sums / current.scale) / current.scale
            effect = lift(self.basis, embed(
                self.basis, solved[self.problem.n_slopes:]
            ))
            change = self.problem.design @ solved[:self.problem.n_slopes] + effect
        if not np.isfinite(solved).all() or not np.isfinite(change).all():
            raise FloatingPointError("the reduced full direction is not representable")
        return immutable_array(solved), immutable_array(change), prepared

    def _coefficients(self, factor: BlockFactor, scale: FloatArray) -> FloatArray:
        """D^-1 B' = -F_nu,X (F_X,X)^-1, using only K full RHS columns."""
        k = self.problem.n_slopes
        right = np.zeros((scale.size, k))
        right[np.arange(k), np.arange(k)] = 1.0 / scale[:k]
        with np.errstate(all="ignore"):
            inverse_columns = factor.solve(right) / scale[:, None]
        if not np.isfinite(inverse_columns).all():
            raise FloatingPointError(
                "the reduced slope actions are not representable"
            )
        try:
            coefficients = np.linalg.solve(
                inverse_columns[:k].T, -inverse_columns[k:].T
            ).T
        except np.linalg.LinAlgError as error:
            raise FloatingPointError(
                "the reduced slope actions are singular"
            ) from error
        if not np.isfinite(coefficients).all():
            raise FloatingPointError(
                "the reduced slope coefficients are not representable"
            )
        return immutable_array(coefficients)

    def _projected(
        self, prepared: PreparedWeights, coefficients: FloatArray,
        tolerance: float, acceleration: bool,
    ) -> Absorbed:
        """Recover the full-row projection and fresh original-level check."""
        embedded = embed(self.basis, coefficients)
        residual = checked_difference(
            self.problem.design, lift(self.basis, embedded), "reduced projection"
        )
        state = AbsorptionState(self.problem.grouping, self.problem.columns, embedded)
        result = _projection_result(
            residual, state, prepared, tolerance, acceleration
        )
        if not result.converged:
            return Absorbed(prepared.weights, result, None)
        with np.errstate(all="ignore"):
            matrix = residual.T @ (prepared.weights[:, None] * residual)
        failed = not np.isfinite(matrix).all()
        return Absorbed(prepared.weights, result,
                        None if failed else immutable_array(matrix),
                        SCHUR if failed else "")

    def project(
        self, weights: FloatArray, q: FloatArray, *,
        tolerance: float, acceleration: bool, final_observed: bool,
    ) -> tuple[Absorbed, PreparedWeights] | None:
        """Fresh selected linear action, or pre-attempt ineligibility only."""
        self.current, self.final = None, None
        if (profiled.curvature_totals is not profiled._ACCEPTED_CURVATURE_TOTALS
                or profiled.prepare_weights is not profiled._ACCEPTED_PREPARE_WEIGHTS):
            return None
        prepared = _prepare_curvature(self.problem, weights)
        current = self._factor(prepared, tolerance, acceleration)
        if current is None:
            return None
        self.current = current
        absorbed = current.absorbed
        if absorbed.result.converged and not absorbed.failure and final_observed:
            self.final = FinalFactor(
                self.owner, self.problem.variance, self.basis, immutable_array(q),
                prepared.weights, current.raw, current.scale,
                replace(current.record, call=MappingProxyType({
                    **(current.record.call or {}), "same_fit_exact_final_reuse": True,
                })), current.factor, absorbed.result,
            )
        return absorbed, prepared

    def offer(
        self, point: AdmittedPoint, rows: FreshRows, basis: CertificateBasis
    ) -> FinalFactor | None:
        """No origin/expected/foreign/stale factor enters the fresh certificate."""
        final = self.final
        if final is not None and final.matches(point, rows, basis):
            return final
        return None

    def nuisance(
        self, b: FloatArray, prepared: PreparedWeights, *,
        target: tuple[float, float, rhs.RHSGuard],
        limits: tuple[int, bool],
    ) -> rhs.RHSResult | None:
        """Full original RHS, including zero-C scores, and unchanged accuracy checks."""
        current = self.current
        if current is None or prepared is not current.prepared:
            return None
        tol, forcing, guard = target
        tol, forcing = rhs._tolerance(tol), rhs._forcing(forcing)
        budget, accelerated = rhs._budget(limits[0]), rhs._flag(limits[1])
        right = rhs._right(b, prepared.n_rows)
        start = rhs._evaluate(right, prepared, zero_coefficients(prepared.grouping, 1))
        work = Work().checked(start.passes)
        initial = None if start.check is None else start.check.maximum
        if initial is None:
            outcome = rhs._Outcome(start, rhs.RANGE, start.failure or rhs.RANGE, work)
        else:
            bound = rhs._Target(tol, forcing, initial, guard)
            outcome, _, work = rhs._settled(prepared, start, work, bound)
            if outcome is None:
                outcome = self._right_outcome(right, current, start, work, bound)
        point = outcome.point
        return rhs.RHSResult(
            point.h[:, 0], point.u[:, 0], tuple(a[:, 0] for a in point.alpha),
            point.check, point.native, outcome.status, outcome.reason,
            tol, forcing, initial, guard, budget, accelerated, outcome.work,
        )

    def _right_outcome(
        self, right: FloatArray, current: Projection, start: rhs._Point,
        work: Work, target: rhs._Target,
    ) -> rhs._Outcome:
        """The direct candidate, with the SAME fresh accuracy predicate and work."""
        try:
            point = self._right_point(right, current)
        except FloatingPointError as error:
            return rhs._Outcome(start, rhs.RANGE, str(error), work)
        outcome, point, work = rhs._settled(
            current.prepared, point, work.checked(point.passes), target
        )
        return outcome if outcome is not None else rhs._Outcome(
            point, BREAKDOWN, "the reduced full RHS fails the requested bounds", work
        )

    def _right_point(self, right: FloatArray, current: Projection) -> rhs._Point:
        """D^-1 E'b from full H inverse, without ever dividing a row by C."""
        k = self.problem.n_slopes
        sums = np.zeros(k + self.basis.n_kept)
        sums[k:] = moment(self.problem.design[:, :0], right[:, 0], self.basis)
        with np.errstate(all="ignore"):
            solved = current.factor.solve(sums / current.scale) / current.scale
            coefficients = solved[k:] + current.coefficients @ solved[:k]
        if not np.isfinite(coefficients).all():
            raise FloatingPointError("the reduced full RHS is not representable")
        alpha = embed(self.basis, coefficients[:, None])
        return rhs._evaluate(right, current.prepared, alpha)


type _Header = tuple[tuple[int, ...], tuple[int, ...], str, bytes]
_OWNED_BYTES = _immutable_bytes


def _structure_headers(
    problem: ProfiledProblem, basis: CertificateBasis,
) -> tuple[_Header, ...] | None:
    """Bind actual complete immutable buffers and headers, without reading rows."""
    if (type(problem) is not ProfiledProblem or type(basis) is not CertificateBasis
            or _immutable_bytes is not _OWNED_BYTES):
        return None
    owner = problem.__dict__.get("_numeric_owner")
    if (type(owner) is not NumericFit
            or ((kind := type(basis.grouping)) is not Grouping and kind is not Derived)
            or problem.grouping is not owner._grouping):
        return None
    design = problem.design
    if (type(design) is not np.ndarray or design.dtype != np.dtype(np.float64)
            or design.dtype.metadata is not None
            or design.shape != owner._design.shape
            or _immutable_bytes(design) is not owner._design.data):
        return None
    arrays = [basis.selected, *basis.positions]
    for block in basis.grouping.blocks:
        if type(block) is not Block:
            return None
        arrays.extend((block.codes, block.counts))
    headers: list[_Header] = [(design.shape, design.strides, design.dtype.str,
                              owner._design.data)]
    for array in arrays:
        if (type(array) is not np.ndarray or array.ndim != 1
                or array.dtype != np.dtype(np.intp)
                or array.dtype.metadata is not None):
            return None
        data = _immutable_bytes(array)
        if data is None:
            return None
        headers.append((array.shape, array.strides, array.dtype.str, data))
    return tuple(headers)


def _same_headers(left: tuple[_Header, ...], right: tuple[_Header, ...]) -> bool:
    """Compare metadata and immutable OWNER identity, never whole byte values."""
    return len(left) == len(right) and all(
        first[:3] == second[:3] and first[3] is second[3]
        for first, second in zip(left, right, strict=True)
    )


class _OwnedReducedExecution(ReducedExecution):
    """One private assembly strategy; factors/actions remain the shared backend."""

    _headers: tuple[_Header, ...]

    def _assemble(self, curvature: FloatArray) -> CompactMatrix:
        """Fill fixed routes only while actual ownership and producers still match."""
        headers = _structure_headers(self.problem, self.basis)
        if (headers is None or not _same_headers(headers, self._headers)
                or not _original_structure(self.layout)):
            return super()._assemble(curvature)
        structure = self.layout._structure
        if structure is None:
            raw, self.layout._structure = geometry._preparing_hessian(
                self.problem.design, curvature, self.layout,
            )
            return raw
        try:
            return geometry._prepared_hessian(
                self.problem.design, curvature, self.layout, structure,
            )
        except MemoryError:
            self.layout._structure = None
        del structure
        return super()._assemble(curvature)


def _original_structure(layout: JointLayout) -> bool:
    """Changed defining or consumed callbacks retain original assembly semantics."""
    return (
        geometry.original_values() and compact.original_entries(layout)
        and all(globals()[name] is value for name, value in _STRUCTURE_PRODUCERS)
        and CompactEntries is compact.CompactEntries
        and JointLayout is compact.JointLayout
        and _slope_entries is geometry._SLOPE_ENTRIES
        and _mixed_entries is geometry._MIXED_ENTRIES
    )


_STRUCTURE_PRODUCERS = tuple((name, globals()[name]) for name in (
    "_structure_headers", "_same_headers", "_original_structure",
))


def _projection_result(
    residual: FloatArray, state: AbsorptionState, prepared: PreparedWeights,
    tolerance: float, acceleration: bool,
) -> AbsorptionResult:
    """Keep the actual candidate and attempted original-level check on refusal."""
    attempted = Attempted()
    try:
        checked = fresh_check(residual, prepared, attempted)
    except FloatingPointError as error:
        return AbsorptionResult(
            residual, state, rhs.RANGE, str(error), tolerance, None, None,
            0, 0, 0, attempted.count, acceleration,
        )
    passed = checked.maximum <= tolerance
    return AbsorptionResult(
        residual, state, "converged" if passed else BREAKDOWN,
        "" if passed else "the reduced projection fails the requested fresh bound",
        tolerance, checked.column_errors, checked.maximum, 0, 0, 0,
        attempted.count, acceleration,
    )


_FINAL_PROJECTION_TYPES = (
    ReducedExecution, _OwnedReducedExecution, FinalFactor, Projection,
    Absorbed, AbsorptionResult, AbsorptionState, PreparedWeights,
    ProfiledProblem, NumericFit, CertificateBasis,
)
_FINAL_PROJECTION_ABSENT = object()
_FINAL_PROJECTION_MEMBERS = tuple(
    (kind, name, vars(kind).get(name, _FINAL_PROJECTION_ABSENT)) for kind, names in (
        (ReducedExecution, ("__getattribute__", "__getattr__", "offer", "project",
                            "_factor", "_projected", "current", "final",
                            "owner", "problem", "basis")),
        (_OwnedReducedExecution, ("__getattribute__", "__getattr__", "offer", "project",
                                  "_factor", "_projected", "current", "final",
                                  "owner", "problem", "basis")),
        (FinalFactor, ("__getattribute__", "__getattr__", "matches", "projection",
                       "owner", "variance", "basis", "q", "weights")),
        (Projection, ("__getattribute__", "__getattr__", "prepared", "absorbed")),
        (Absorbed, ("__getattribute__", "__getattr__", "result")),
        (AbsorptionResult, ("__getattribute__", "__getattr__", "__init__",
                            "__post_init__", "state", "residual", "tolerance",
                            "acceleration", "sweeps", "visits", "operators",
                            "diagnostic_passes")),
        (AbsorptionState, ("__getattribute__", "__getattr__", "__init__",
                           "__post_init__", "grouping", "columns", "coefficients")),
        (PreparedWeights, ("__getattribute__", "__getattr__", "weights")),
        (ProfiledProblem, ("__getattribute__", "__getattr__", "grouping", "columns")),
        (NumericFit, ("__getattribute__", "__getattr__", "design", "weights")),
        (CertificateBasis, ("__getattribute__", "__getattr__", "grouping")),
    ) for name in names
)
_FINAL_PROJECTION_PRODUCERS = (
    _projection_result, _same, _prepare_curvature, checked_difference,
    fresh_check, embed, lift,
)


def _completed_projection(
    execution, point, rows, basis, values, names, weights,
) -> AbsorptionResult | None:
    """Read only the actual completed final action on the existing live fit owner.

    This is eligibility, not numerical acceptance. The certificate reconstructs
    and checks the returned data afresh, and no execution/factor enters evidence.
    Changed hooks keep their original later lookup/cold auxiliary call order.
    """
    try:
        if not all(a is b for a, b in zip((
            ReducedExecution, _OwnedReducedExecution, FinalFactor, Projection,
            Absorbed, AbsorptionResult, AbsorptionState, PreparedWeights,
            ProfiledProblem, NumericFit, CertificateBasis,
        ), _FINAL_PROJECTION_TYPES, strict=True)):
            return None
        kind = type(execution)
        if kind is not ReducedExecution and kind is not _OwnedReducedExecution:
            return None
        if type(basis) is not CertificateBasis:
            return None
        if (any(vars(cls).get(name, _FINAL_PROJECTION_ABSENT) is not original
                for cls, name, original in _FINAL_PROJECTION_MEMBERS)
                or not all(a is b for a, b in zip((
                    _projection_result, _same, _prepare_curvature, checked_difference,
                    fresh_check, embed, lift,
                ), _FINAL_PROJECTION_PRODUCERS, strict=True))
                or any(name in vars(execution) for name in
                       ("offer", "project", "_factor", "_projected"))):
            return None
        current, final = execution.current, execution.final
        if (type(execution.owner) is not NumericFit
                or type(execution.problem) is not ProfiledProblem
                or type(point._numeric_problem) is not ProfiledProblem
                or type(current) is not Projection or type(final) is not FinalFactor
                or type(current.absorbed) is not Absorbed
                or type(current.prepared) is not PreparedWeights):
            return None
        result = final.projection
        if (type(names) is not tuple or any(type(name) is not str for name in names)
                or type(result) is not AbsorptionResult
                or result is not current.absorbed.result
                or type(result.state) is not AbsorptionState
                or type(result.state.columns) is not tuple
                or result.state.grouping is not execution.problem.grouping
                or result.state.columns != names
                or result.tolerance != point.inner.tolerance
                or result.acceleration is not point.inner.acceleration
                or final.basis is not execution.basis
                or final.weights is not current.prepared.weights):
            return None
        if any(type(value) is not np.ndarray or value.dtype != np.float64
               or value.dtype.metadata is not None for value in
               (point.q, rows.curvature, rows.weights, final.q, final.weights, values,
                weights)):
            return None
        if not _same(weights, rows.curvature):
            return None
        # Existing offer authenticates exact owner/family/basis/q/current C/raw a.
        # No old converged/maximum/column-error flag is consumed here.
        if execution.offer(point, rows, basis) is not final:
            return None
        if not _same(values, execution.owner.design):
            return None
        return result
    except MemoryError:
        # Only optional matching/header work; no fresh witness arithmetic ran.
        return None
