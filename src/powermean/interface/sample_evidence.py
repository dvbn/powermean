"""The owned evidence of a support invocation: the finite graph of its result.

Internal: no public name, option, default, message, tolerance or route.

The records of a support result (:mod:`powermean.solve.support` and its
witness, LP, rectifier and projection modules) are frozen dataclasses that their
constructors check but, unlike the recession and collinearity records, do not own: a
primal witness
formed by its producer holds read-only arrays of its own, one built directly or by
``dataclasses.replace`` holds whatever it was given.  A result's status, separated set
and positions are read afresh from its witnesses, so a caller's array kept inside an
admitted record could change the evidence of a recorded removal afterward.
:func:`owned_result` rebuilds the result of an invocation from owned copies of every
part of that finite graph, before the constructor checks it again, once, on
the invocation's own problem and controls; that owned record is the one kept.

The graph.  The result, its FE step, its LP outcome (the solves, their marginals and
primal witnesses, the options and the terminal dual mapping), its rectifier outcome
(the weights, the attempt summaries, the last attempt with its projection and that
projection's refinement record), every primal witness (its ORIGINAL proposal, raw,
normalized and authoritative rows, strict and unresolved rows) and every dual witness
(its proposal, construction and column sums).  Their constructors own the records kept
as they are: the support problem, the frozen control records, the absorber's results
(the joint call's and the refinement call's) and the attempt summaries.

Rules.  An array is kept in bytes of its own with its dtype, shape and every bit (signed
zeros, NaN payloads, infinities, empty arrays), read-only for good; a sequence becomes
a tuple, the options a read-only copy, each record its type rebuilt from owned
parts; a scalar is kept as given, an array found there owned too.  Nothing is
converted, cast, filtered, normalized or analyzed again.  A value that cannot be owned
so (an array of Python objects, an array subclass, another value where an array, a
sequence, the options or a record belongs) is refused with :data:`UNOWNED`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from powermean.fe.absorb import AbsorptionResult
from powermean.solve.support import FEStep, SupportControls, SupportResult
from powermean.solve.support_lp import LPControls, LPDual, LPOutcome, LPSolve
from powermean.solve.support_problem import SupportProblem
from powermean.solve.support_projection import Projection, Refinement
from powermean.solve.support_rectifier import (
    Attempt,
    Final,
    RectifierControls,
    RectifierOutcome,
)
from powermean.solve.support_witness import ColumnSums, DualWitness, PrimalWitness

UNOWNED = "the evidence holds a part that is not of its record's owned types"


# ------------------------------------------------------------------- the leaves


def _array(value: object) -> Any:
    """A copy of a numerical array in bytes of its own: dtype, shape, bits kept."""
    if type(value) is not np.ndarray:
        raise ValueError(UNOWNED)
    if value.dtype.hasobject:
        raise ValueError(UNOWNED)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _kept(value: object) -> Any:
    """A scalar as given; an array found there owned (:func:`_array`)."""
    return _array(value) if isinstance(value, np.ndarray) else value


def _each(value: object, item: Callable[[object], Any]) -> Any:
    """A tuple of the items of a tuple or list, each owned by ``item``."""
    if not isinstance(value, tuple | list):
        raise ValueError(UNOWNED)
    return tuple(item(found) for found in value)


def _maybe(value: object, item: Callable[[object], Any]) -> Any:
    """``None``, or ``value`` owned by ``item``."""
    return None if value is None else item(value)


def _shared[T](value: object, kind: type[T]) -> T:
    """A record that its constructor owns, kept as the very object."""
    if not isinstance(value, kind):
        raise ValueError(UNOWNED)
    return value


def _options(value: object) -> Mapping[str, Any]:
    """A read-only copy of the options of an LP outcome."""
    if not isinstance(value, Mapping):
        raise ValueError(UNOWNED)
    return MappingProxyType({key: _kept(found) for key, found in value.items()})


def _arrays(value: object) -> Any:
    """A tuple of owned arrays (:func:`_each`)."""
    return _each(value, _array)


def _names(value: object) -> Any:
    """A tuple of kept scalars: failures and warnings (:func:`_each`)."""
    return _each(value, _kept)


# --------------------------------------------------------------- the witnesses


def _primal(value: object) -> PrimalWitness:
    """A primal witness from owned parts, its original proposal included."""
    if not isinstance(value, PrimalWitness):
        raise ValueError(UNOWNED)
    return PrimalWitness(
        problem=_shared(value.problem, SupportProblem),
        method=_kept(value.method),
        tolerance=_kept(value.tolerance),
        threshold=_kept(value.threshold),
        failure=_names(value.failure),
        proposed_slopes=_maybe(value.proposed_slopes, _array),
        proposed_effects=_maybe(value.proposed_effects, _arrays),
        raw=_maybe(value.raw, _array),
        scale=_kept(value.scale),
        slopes=_maybe(value.slopes, _array),
        effects=_maybe(value.effects, _arrays),
        z=_maybe(value.z, _array),
        positive_max=_kept(value.positive_max),
        zero_min=_kept(value.zero_min),
        strict=_array(value.strict),
        unresolved=_array(value.unresolved),
    )


def _sums(value: object) -> ColumnSums:
    """Column sums from owned arrays."""
    if not isinstance(value, ColumnSums):
        raise ValueError(UNOWNED)
    return ColumnSums(
        _array(value.numerators), _array(value.denominators), _array(value.ratios)
    )


def _dual(value: object) -> DualWitness:
    """A dual witness from owned parts: proposal, construction and column sums."""
    if not isinstance(value, DualWitness):
        raise ValueError(UNOWNED)
    return DualWitness(
        problem=_shared(value.problem, SupportProblem),
        tolerance=_kept(value.tolerance),
        threshold=_kept(value.threshold),
        raw=_array(value.raw),
        separated=_array(value.separated),
        constructed=_array(value.constructed),
        failure=_names(value.failure),
        normalizer=_kept(value.normalizer),
        normalized=_maybe(value.normalized, _array),
        margin=_kept(value.margin),
        slopes=_maybe(value.slopes, _sums),
        levels=_each(value.levels, _sums),
    )


# ---------------------------------------------------------- the FE and LP steps


def _fe(value: object) -> FEStep:
    """The FE step from owned parts."""
    if not isinstance(value, FEStep):
        raise ValueError(UNOWNED)
    return FEStep(
        levels=_arrays(value.levels),
        rows=_array(value.rows),
        witness=_maybe(value.witness, _primal),
        exact=_kept(value.exact),
    )


def _solve(value: object) -> LPSolve:
    """One charged LP solve from owned parts."""
    if not isinstance(value, LPSolve):
        raise ValueError(UNOWNED)
    return LPSolve(
        remaining=_array(value.remaining),
        status=_kept(value.status),
        success=_kept(value.success),
        message=_kept(value.message),
        iterations=_kept(value.iterations),
        reported=_kept(value.reported),
        warnings=_names(value.warnings),
        failure=_kept(value.failure),
        coefficients=_maybe(value.coefficients, _array),
        marginals=_maybe(value.marginals, _arrays),
        objective=_kept(value.objective),
        primal=_maybe(value.primal, _primal),
    )


def _mapping(value: object) -> LPDual:
    """The terminal dual mapping of the LP from owned parts."""
    if not isinstance(value, LPDual):
        raise ValueError(UNOWNED)
    return LPDual(
        mapped=_array(value.mapped),
        box=_array(value.box),
        stationarity=_array(value.stationarity),
        box_ratios=_array(value.box_ratios),
        witness=_maybe(value.witness, _dual),
        tolerance=_kept(value.tolerance),
    )


def _lp(value: object) -> LPOutcome:
    """The LP outcome from owned parts: options, scales, solves, witnesses, dual."""
    if not isinstance(value, LPOutcome):
        raise ValueError(UNOWNED)
    return LPOutcome(
        controls=_shared(value.controls, LPControls),
        method=_kept(value.method),
        options=_options(value.options),
        scales=_array(value.scales),
        solves=_each(value.solves, _solve),
        witnesses=_each(value.witnesses, _primal),
        dual=_maybe(value.dual, _mapping),
        completion=_kept(value.completion),
        reason=_kept(value.reason),
    )


# ------------------------------------------------------------ the rectifier


def _projection(value: object) -> Projection:
    """One support projection from owned parts; the absorber's results kept."""
    if not isinstance(value, Projection):
        raise ValueError(UNOWNED)
    return Projection(
        status=_kept(value.status),
        reason=_kept(value.reason),
        scales=_array(value.scales),
        absorption=_maybe(value.absorption, _absorbed),
        singular_values=_maybe(value.singular_values, _array),
        cutoff=_kept(value.cutoff),
        rank=_kept(value.rank),
        pivots=_maybe(value.pivots, _array),
        range_residual=_kept(value.range_residual),
        slopes=_maybe(value.slopes, _array),
        effects=_maybe(value.effects, _arrays),
        fitted=_maybe(value.fitted, _array),
        residual=_maybe(value.residual, _array),
        level_means=_arrays(value.level_means),
        slope_ratios=_maybe(value.slope_ratios, _array),
        refinement=_maybe(value.refinement, _refinement),
    )


def _refinement(value: object) -> Refinement:
    """The requirement on ``u`` from owned parts; its call's absorber result kept."""
    if not isinstance(value, Refinement):
        raise ValueError(UNOWNED)
    return Refinement(
        reason=_kept(value.reason),
        adopted=_kept(value.adopted),
        budget=_kept(value.budget),
        evaluations=_kept(value.evaluations),
        result=_maybe(value.result, _absorbed),
    )


def _absorbed(value: object) -> AbsorptionResult:
    """The absorber's result, which its constructor owns (``fe/absorb.py``)."""
    return _shared(value, AbsorptionResult)


def _attempt(value: object) -> Attempt:
    """An attempt summary, a frozen record of scalars."""
    return _shared(value, Attempt)


def _final(value: object) -> Final:
    """The rectifier's last attempt from owned parts."""
    if not isinstance(value, Final):
        raise ValueError(UNOWNED)
    return Final(
        u=_array(value.u),
        projection=_projection(value.projection),
        z=_maybe(value.z, _array),
        residual=_maybe(value.residual, _array),
        weighted=_maybe(value.weighted, _array),
        primal=_maybe(value.primal, _primal),
        dual=_maybe(value.dual, _dual),
    )


def _ir(value: object) -> RectifierOutcome:
    """The rectifier outcome from owned parts: weights, attempts, last, witnesses."""
    if not isinstance(value, RectifierOutcome):
        raise ValueError(UNOWNED)
    return RectifierOutcome(
        controls=_shared(value.controls, RectifierControls),
        penalty=_kept(value.penalty),
        weights=_maybe(value.weights, _array),
        charged=_kept(value.charged),
        attempts=_each(value.attempts, _attempt),
        last=_maybe(value.last, _final),
        witnesses=_each(value.witnesses, _primal),
        dual=_maybe(value.dual, _dual),
        completion=_kept(value.completion),
        reason=_kept(value.reason),
    )


# --------------------------------------------------------------- the result


def owned_result(
    result: SupportResult, problem: SupportProblem, controls: SupportControls
) -> SupportResult:
    """The support result rebuilt on ``problem`` and ``controls`` from owned parts.

    Parameters
    ----------
    result : SupportResult
        The result of the invocation, already bound to ``problem`` and ``controls``.
    problem : SupportProblem
        The invocation's own problem, which the rebuilt result keeps.
    controls : SupportControls
        The invocation's own controls, which the rebuilt result keeps.

    Returns
    -------
    SupportResult
        The result whose every array, sequence and record of the finite graph is owned
        (module docstring), checked again by the accepted constructor: every witness
        replayed from its original proposal, the dual with the witnessed union.

    Raises
    ------
    ValueError
        A part that cannot be owned (:data:`UNOWNED`), or the accepted refusals.
    """
    return SupportResult(
        problem,
        controls,
        _kept(result.method),
        _maybe(result.fe, _fe),
        _maybe(result.lp, _lp),
        _maybe(result.ir, _ir),
        _each(result.witnesses, _primal),
        _maybe(result.dual, _dual),
        _kept(result.completion),
        _kept(result.reason),
    )
