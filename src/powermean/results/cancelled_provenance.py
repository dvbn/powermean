"""Detach cancelled solver diagnostics without copying the outer captured fit.

Provenance is lossless, but mutable array metadata returned to callers
must not mutate the retained record. Immutable owned storage and current-point
IdentifiedState are shared; only outward diagnostic records are detached.
"""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType

from powermean.data.accounting import MissingAccount
from powermean.data.raw import RawInputs
from powermean.results.owned import _owned_attempt, _owned_coefficients
from powermean.solve.cancelled_records import (
    CancelledCertificate,
    RootInitializerRecord,
    RootStageRecord,
    Tau1SolveResult,
)


def _mapping_memo(value: object, memo: dict[int, object], seen: set[int]) -> None:
    """Copy immutable mapping wrappers in the finite owned diagnostic graph.

    Python's deepcopy deliberately refuses mappingproxy. Its values still require
    detachment when they contain arrays; the wrapper is not a reason to share them.
    No property, callback, numerical operation or external state is consulted.
    """
    if id(value) in seen or id(value) in memo:
        return
    seen.add(id(value))
    if isinstance(value, MappingProxyType):
        for item in value.values():
            _mapping_memo(item, memo, seen)
        memo[id(value)] = MappingProxyType(copy.deepcopy(dict(value), memo))
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _mapping_memo(getattr(value, field.name), memo, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _mapping_memo(item, memo, seen)


def detached_certificate(value: CancelledCertificate) -> CancelledCertificate:
    """Detach the coefficient-check arrays; immutable byte storage stays shared."""
    check = value.measures.coefficients
    return replace(
        value,
        measures=replace(
            value.measures,
            coefficients=None if check is None else _owned_coefficients(check),
        ),
    )


def detached_initializer(value: RootInitializerRecord) -> RootInitializerRecord:
    """Return detached diagnostic arrays, not a copy of the captured fit."""
    attempts = value.attempts
    return replace(
        value,
        certificate=(
            None
            if value.certificate is None
            else detached_certificate(value.certificate)
        ),
        attempts=(
            None
            if attempts is None
            else attempts._replace(
                attempts=tuple(_owned_attempt(item) for item in attempts.attempts)
            )
        ),
    )


def detached_stages(values: tuple[RootStageRecord, ...]) -> tuple[RootStageRecord, ...]:
    """Detach certificates while sharing immutable point storage and equation state."""
    return tuple(
        replace(
            value,
            certified=replace(
                value.certified,
                certificate=detached_certificate(value.certified.certificate),
            ),
        )
        for value in values
    )


def detached_tau1(value: Tau1SolveResult) -> Tau1SolveResult:
    """Detach the optional inner-solver trace only when explicitly requested.

    The outer captured fit/equation state is not copied. The shared engines expose
    ndarray diagnostic metadata, so their optional trace needs its own outward copy.
    """
    candidate = value.candidate
    copied = None
    if candidate is not None:
        raw = candidate.sample.base.inputs
        # RawInputs' intentionally non-pickleable column mapping is copied through
        # its normal owning constructor, not a whole outer-capture deepcopy.
        raw_copy = RawInputs(
            y=raw.y,
            design=raw.design,
            column_names=raw.column_names,
            weights=raw.weights,
            offset=raw.offset,
            exposure=raw.exposure,
            metadata=raw.metadata,
        )
        memo: dict[int, object] = {id(raw): raw_copy}
        for missing in (candidate.sample.base.missing, candidate.sample.ledger.missing):
            memo[id(missing)] = MissingAccount._owned(
                dict(missing.groups), missing.excluded
            )
        _mapping_memo(candidate, memo, set())
        copied = copy.deepcopy(candidate, memo)
    return replace(
        value,
        certified=replace(
            value.certified,
            certificate=detached_certificate(value.certified.certificate),
        ),
        history=tuple(detached_certificate(item) for item in value.history),
        candidate=copied,
    )
