"""The local automatic start of the fitted-index route: one block, no slope, PPML.

The convex start is ``log(mean Z)``, zero slopes and effects from one profile sweep.
Internal module: ``powermean`` exports nothing of it.  An adaptation of the analytic
block formula: Blanchard, Higham and Higham (IMA J. Numer. Anal. 41, 2021, Sections 1
and 2) and SciPy v1.16.2 ``scipy/special/_logsumexp.py`` support evaluating it from
shifted components and supply no error bound for it.  No code is copied; every
profiler and reduction of the package is unchanged.

The domain (:func:`applies`).  Only the automatic first attempt (``attempt == 0``) of a
problem with exactly one fixed-effect block, no retained slope column and the exact
``PowerVariance`` with ``tau == 1`` or the exact ``NB2Variance`` with ``kappa == 0``,
by exact comparisons; every other start keeps the actual accepted profiler.  The NB2
alias runs the SAME canonical PPML arithmetic, the checked ``PowerVariance(1)`` problem
on the family's own prepared weights, outcomes, base and logs, as the accepted
``profile_nb2`` builds it at ``kappa = 0``; the requested family stays the solver's
everywhere else.

The mathematics.  With ``b`` the transformed base (the offset once; no slope), raw
weights ``a`` and outcomes ``z``, the exact one-block update of PPML at any current
``q = b + c`` (``c`` constant within each level) ends at

    alpha*_g = log sum_{i in g} a_i z_i - log sum_{i in g} a_i exp(b_i),

the increment being ``alpha* - c``: the one sweep of the default start ends at
``alpha*`` whatever the requested ``c = log(mean z)``.  This is the SAME mathematical
sweep; only its rounding changes.  With the accepted log inputs at the BASE, ``vA = log
a + log z`` (rows with ``a > 0`` and ``z > 0``) and ``vB = log a + b`` (rows with ``a >
0``), the others ``-inf``, and per level the paired max-shifted components

    mA = max vA,  sA = sum exp(vA - mA),      mB = max vB,  sB = sum exp(vB - mB),

the coefficient is the absolute replacement (not ``c`` plus an increment)

    alpha* = (mA - mB) + log(sA / sB),

each step checked.  The complete log sums ``LA = mA + log sA`` and ``LB = mB + log sB``
are formed and required finite, the accepted range checks of the profile, although
their rounded difference is not the answer.  Each admitted level has a row of positive
weight and outcome, so both maxima are finite and both sums hold ``exp(0) = 1``: a
smaller shifted exponential that underflows to zero, or a finite negative difference
that overflows to ``-inf``, is a vanishing tail and permitted.  No exponential of a log
sum, no raw sum of ``a z`` or of ``a exp(b)``, no ``log1p`` branch, constant, clamp,
zero detector, threshold or second arithmetic path.  Rounding of the logs, exponentials,
sums, quotient and final cancellation remains: no correctly rounded, error-bounded
or convergence guarantee is claimed.

The procedure (:func:`local_start`), from the caller's requested start (the plain
unweighted ``log(mean z)`` in every level, formed and guarded before):

1. ``q0``, the base plus the requested coefficient by the accepted ordered
   reconstruction, and the actual accepted fresh equations there at ``profile_tol``.
   A passing check returns ``q0`` and the requested coefficient unchanged, with no block
   arithmetic; a refused check is a ``range`` result at ``q0``.
2. Otherwise the one visit of the block: the components and ``alpha*`` above, then
   ``q1`` by the accepted ordered reconstruction of every row; a refusal there is a
   ``range`` result at the checked ``q0``.
3. The actual accepted fresh equations at ``q1``: a refusal is a ``range`` result at
   ``q1`` without a check; a pass is ``converged``; otherwise ``budget``, a finite
   one-sweep initializer as the profiler's, not a root or a certificate.

Records and work.  :class:`LocalInitializer` is the outcome, whatever its status, and
:class:`LocalCall` its log record; neither is a profile record, and no profiler call is
attributed to this path.  The work is the accepted :class:`~powermean.fe.profile.Work`
with the profile's meanings: ``sweeps_attempted`` and ``blocks_attempted`` before the
visit's first sums, ``blocks_completed`` and ``sweeps_completed`` once ``alpha*`` and
``q1`` are finite, ``checks_attempted`` and ``checks_completed`` around each fresh
check.  A refusal before the first check (the base, the prepared problem or ``q0``)
raises ``FloatingPointError`` with no work: unavailable, not zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from powermean.fe.absorb import _tolerance
from powermean.fe.absorb_numeric import (
    checked_difference,
    checked_quotient,
    checked_sum,
)
from powermean.fe.profile import (
    BUDGET,
    CONVERGED,
    RANGE,
    STATUSES,
    ProfileState,
    Work,
    _bound,
    _Counters,
)
from powermean.fe.profile_arithmetic import (
    Coefficients,
    EquationCheck,
    ProfileProblem,
    fresh_equations,
    log_inputs,
    reconstruct_index,
)
from powermean.fe.reductions import group_sum, immutable_array
from powermean.model.order import IndexArray
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.systems.profiled import ProfiledProblem, _prepared, base_index

Family = PowerVariance | NB2Variance
IMPLEMENTATION = (
    "absorbed_start.local_start: one-block replacement at the base from paired "
    "max-shifted sums"
)
# The one sweep of the default convex start.
INITIAL_SWEEPS = 1
DOMAIN = (
    "the local start covers the first attempt with one block, no slope column and "
    "PowerVariance(1) or NB2Variance(0) only"
)


def applies(problem: ProfiledProblem, attempt: int) -> bool:
    """Whether the local start replaces the profiler for this automatic start.

    The first attempt, exactly one block, no slope column, and the exact PPML family or
    the exact NB2 ``kappa = 0`` alias: exact comparisons, no near equality.
    """
    variance = problem.variance
    ppml = type(variance) is PowerVariance and variance.tau == 1.0
    alias = type(variance) is NB2Variance and variance.kappa == 0.0
    single = len(problem.grouping.blocks) == 1 and problem.n_slopes == 0
    return attempt == 0 and single and (ppml or alias)


# ------------------------------------------------------------ the arithmetic


@dataclass(frozen=True, eq=False)
class Components:
    """The paired max-shifted components of the level log sums of one input vector.

    Attributes
    ----------
    maximum : array of float
        ``m_g``, the largest log input of each level, owned and read-only.
    total : array of float
        ``s_g``, the row-order sum of ``exp(v - m_g)`` of each level, owned and
        read-only; the level log sum is ``m_g + log s_g``.
    """

    maximum: FloatArray
    total: FloatArray

    def __post_init__(self) -> None:
        """Own the arrays."""
        object.__setattr__(self, "maximum", immutable_array(self.maximum))
        object.__setattr__(self, "total", immutable_array(self.total))


def components(values: FloatArray, codes: IndexArray, n_levels: int) -> Components:
    """The maxima and the normalized sums of the log inputs ``values`` by level.

    Absent terms are ``-inf``.  Every level needs a finite maximum; its terms
    ``exp(v - m)`` are summed in row order by the accepted reduction (``group_sum``),
    a smaller term that underflows, or a finite negative difference that overflows to
    ``-inf``, being a permitted vanishing tail; every sum must be finite and positive.
    ``FloatingPointError`` otherwise.
    """
    maxima = np.full(n_levels, -np.inf)
    np.maximum.at(maxima, codes, values)
    if not np.isfinite(maxima).all():
        raise FloatingPointError("a level has no finite log input")
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        terms = np.exp(values - maxima[codes])
    totals = group_sum(terms, codes, n_levels)
    # Required by the correction-B record; with finite maxima every level holds the
    # term exp(0) = 1 and terms in [0, 1], so this holds by construction.
    if not (np.isfinite(totals).all() and (totals > 0).all()):
        raise FloatingPointError("a normalized level sum is not finite and positive")
    return Components(maxima, totals)


def replacement(numerator: Components, denominator: Components) -> FloatArray:
    """``alpha* = (mA - mB) + log(sA / sB)``, one value per level, each step checked.

    The complete log sums ``m + log s`` of both are formed and required finite (the
    accepted range checks; their difference is not the answer); the ratio of the sums
    must be finite and positive, and its log, the difference of the maxima and the
    coefficient finite.  ``FloatingPointError`` otherwise; nothing is clamped.
    """
    for found in (numerator, denominator):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            complete = found.maximum + np.log(found.total)
        if not np.isfinite(complete).all():
            raise FloatingPointError("a level log sum is not finite")
    ratio = checked_quotient(numerator.total, denominator.total, "ratio of the sums")
    # Required by the correction-B record; the finite complete log sums and the checked
    # quotient above imply both checks below, which are kept as stated.
    if not (np.isfinite(ratio).all() and (ratio > 0).all()):
        raise FloatingPointError("the ratio of the normalized sums is not positive")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(ratio)
    if not np.isfinite(log_ratio).all():
        raise FloatingPointError("the log of the ratio of the sums is not finite")
    delta = checked_difference(
        numerator.maximum, denominator.maximum, "difference of the maxima"
    )
    return checked_sum(delta, log_ratio, "local coefficients")


# --------------------------------------------------------------------- records


@dataclass(frozen=True, eq=False)
class LocalInitializer:
    """The outcome of the local automatic start, whatever its status.

    Attributes
    ----------
    theta : array of float
        The slopes of the start (none), owned and read-only.
    base : array of float
        ``X theta + o``, the transformed offset entered once, owned and read-only.
    variance : PowerVariance or NB2Variance
        The REQUESTED family; the arithmetic was the canonical PPML one.
    start : ProfileState
        The requested start: ``log(mean z)`` in every level.
    state : ProfileState
        The returned coefficient witness.
    index : array of float
        The reconstructed index of ``state``, every row, owned and read-only.
    initial : EquationCheck or None
        The fresh equations at the requested start's index; ``None`` when they refused.
    check : EquationCheck or None
        The fresh equations at ``index``: the initial ones when the requested start is
        returned, the new point's otherwise; ``None`` when unavailable.
    status : str
        ``converged``, ``budget`` (a finite one-sweep initializer above the tolerance)
        or ``range``.
    reason : str
        Empty on convergence; otherwise what stopped the start.
    tolerance : float
        ``profile_tol``; equality passes.
    max_sweeps : int
        One, the sweep of the default start.
    work : Work
        The block-level work, attempted and completed apart.
    implementation : str
        :data:`IMPLEMENTATION`, the identity of what ran.
    """

    theta: FloatArray
    base: FloatArray
    variance: Family
    start: ProfileState
    state: ProfileState
    index: FloatArray
    initial: EquationCheck | None
    check: EquationCheck | None
    status: str
    reason: str
    tolerance: float
    max_sweeps: int
    work: Work
    implementation: str

    def __post_init__(self) -> None:
        """Check the status and own the arrays."""
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        for name in ("theta", "base", "index"):
            object.__setattr__(self, name, immutable_array(getattr(self, name)))

    @property
    def converged(self) -> bool:
        """Whether the fresh check passed at the returned index."""
        return self.status == CONVERGED

    @property
    def maximum(self) -> float | None:
        """The largest equation of the returned check, if it completed."""
        return None if self.check is None else self.check.maximum


@dataclass(frozen=True)
class LocalCall:
    """One logged local automatic start: where, its controls, status, work and maxima.

    The fields of :class:`~powermean.solve.profiled_loss.ProfileCall` for this
    implementation, never one of its records: ``variance`` is the requested family and
    ``implementation`` what ran; ``initial`` and ``maximum`` are the largest equations
    of the initial and the returned check.  ``work``, ``initial`` and ``maximum`` are
    ``None`` when unavailable, ``work`` when the call refused before its first check
    (``status`` ``refused``).
    """

    attempt: int
    iteration: int
    role: Literal["initial"]
    tolerance: float
    max_sweeps: int
    variance: Family
    implementation: str
    status: str
    reason: str
    work: Work | None
    initial: float | None
    maximum: float | None


# ------------------------------------------------------------------ the start


@dataclass(frozen=True, eq=False)
class _Start:
    """The fixed parts of one local start, from which its results are made."""

    theta: FloatArray
    variance: Family
    start: ProfileState
    tolerance: float
    problem: ProfileProblem

    def result(
        self,
        coefficients: Coefficients,
        index: FloatArray,
        checks: tuple[EquationCheck | None, EquationCheck | None],
        *,
        stop: tuple[str, str],
        counters: _Counters,
    ) -> LocalInitializer:
        """The record of a returned point, its checks, its status and reason, work."""
        return LocalInitializer(
            theta=self.theta,
            base=self.problem.base,
            variance=self.variance,
            start=self.start,
            state=ProfileState(self.problem.grouping, coefficients),
            index=index,
            initial=checks[0],
            check=checks[1],
            status=stop[0],
            reason=stop[1],
            tolerance=self.tolerance,
            max_sweeps=INITIAL_SWEEPS,
            work=counters.work(),
            implementation=IMPLEMENTATION,
        )


def _canonical(problem: ProfiledProblem, base: FloatArray) -> ProfileProblem:
    """The checked PPML problem on the family's own prepared inputs at ``base``.

    The construction of the accepted ``kappa = 0`` identity of ``profile_nb2``, for PPML
    and its NB2 alias alike, so that both run the same arithmetic.
    """
    prepared = _prepared(problem, base)
    return ProfileProblem(
        prepared.prepared,
        prepared.z,
        prepared.base,
        PowerVariance(1.0),
        prepared.log_weights,
        prepared.log_outcomes,
    )


def _fresh(
    problem: ProfileProblem, q: FloatArray, counters: _Counters
) -> tuple[EquationCheck | None, str]:
    """The actual fresh equations at ``q``, counted; their failure when they refuse."""
    counters.count("checks_attempted")
    try:
        found = fresh_equations(problem, q)
    except FloatingPointError as error:
        return None, f"fresh check: {error}"
    counters.count("checks_completed")
    return found, ""


def _visit(
    problem: ProfileProblem, counters: _Counters
) -> tuple[Coefficients, FloatArray]:
    """The one visit of the block at the base: ``alpha*`` and the index ``q1``.

    Counted as the profile counts a sweep; ``FloatingPointError`` when an input, a
    component, the coefficient or the index is not representable.
    """
    counters.count("sweeps_attempted")
    counters.count("blocks_attempted")
    numerator, denominator = log_inputs(problem, problem.base)
    block = problem.grouping.blocks[0]
    alpha = replacement(
        components(numerator, block.codes, block.n_levels),
        components(denominator, block.codes, block.n_levels),
    )
    coefficients = (alpha,)
    index = reconstruct_index(problem, coefficients)
    counters.count("blocks_completed")
    counters.count("sweeps_completed")
    return coefficients, index


def local_start(
    problem: ProfiledProblem,
    theta: ArrayLike,
    start: ProfileState,
    *,
    tolerance: float,
) -> LocalInitializer:
    """The automatic one-block start of the local domain (module docstring, 1 to 3).

    Parameters
    ----------
    problem : ProfiledProblem
        A problem of the local domain (:func:`applies` at the first attempt).
    theta : array_like
        The slopes of the start, none.
    start : ProfileState
        The requested start, bound to the problem's grouping.
    tolerance : float
        ``profile_tol``, positive and finite; equality passes.

    Returns
    -------
    LocalInitializer
        The returned point, its checks, status, reason and work.

    Raises
    ------
    ValueError
        A problem outside the local domain, a malformed tolerance, or a start bound to
        another grouping.
    FloatingPointError
        The base or ``q0`` is not representable: a refusal before any work.
    """
    if not applies(problem, 0):
        raise ValueError(DOMAIN)
    tol = _tolerance(tolerance)
    base = base_index(problem, theta)
    slopes: FloatArray = np.array(theta, dtype=np.float64, copy=True)
    canonical = _canonical(problem, base)
    requested = _bound(start, canonical)
    q0 = reconstruct_index(canonical, requested.coefficients)
    fixed = _Start(slopes, problem.variance, requested, tol, canonical)
    counters = _Counters()
    kept = requested.coefficients
    initial, failure = _fresh(canonical, q0, counters)
    if initial is None:
        stop = (RANGE, failure)
        return fixed.result(kept, q0, (None, None), stop=stop, counters=counters)
    if initial.passing(tol):
        stop = (CONVERGED, "")
        return fixed.result(kept, q0, (initial, initial), stop=stop, counters=counters)
    try:
        coefficients, q1 = _visit(canonical, counters)
    except FloatingPointError as error:
        stop = (RANGE, f"block {canonical.grouping.blocks[0].name!r}: {error}")
        return fixed.result(kept, q0, (initial, initial), stop=stop, counters=counters)
    final, failure = _fresh(canonical, q1, counters)
    checks = (initial, final)
    if final is None:
        stop = (RANGE, failure)
    elif final.passing(tol):
        stop = (CONVERGED, "")
    else:
        stop = (BUDGET, _exceeded(final.maximum, tol))
    return fixed.result(coefficients, q1, checks, stop=stop, counters=counters)


def _exceeded(maximum: float, tolerance: float) -> str:
    """The reason of a one-sweep result above the tolerance, as the profile words it."""
    return (
        f"maximum equation {maximum:.3e} exceeds the tolerance {tolerance:.3e} after "
        f"{INITIAL_SWEEPS} sweeps, the budget of {INITIAL_SWEEPS}"
    )
