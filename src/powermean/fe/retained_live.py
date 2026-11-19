"""The canonical lower live reader of the retained direct opening (Layer18).

The reduced nonlinear opening consumes a producer-issued retained action only
through this module: it neither imports nor looks up ``powermean.interface``. The
numeric owner issues one weak, passive binding at its two existing boundaries, the
fresh producer basis of its admitted sample (:func:`_issue_live_retained`) and the
explicit rebound pair (:func:`_rebind_live_retained`). Every use
(:func:`_live_retained`) reads afresh:

* this module's canonical helpers and the slots of its lower classes;
* the issuing owner's exact-type weak anchors, their current field links and the
  class slot/field/method originals that owner pinned when it was imported;
* the actual lower producer reader, proof and complete storage headers of the
  source basis for its admitted sample, the complete current headers of the bound
  basis, the numeric owner's shared grouping storage and the exact rebound problem.

A binding holds weak anchors and immutable identities, never a sample, numeric,
owner or provider graph, so a copied owner, basis or reader, or one kept after the
owner's release, grants nothing. Proof and headers are read from the actual
producer, never accepted from a caller. The only code called from a binding is the
literal reader below, after its code object is authenticated. Both callers of the
lower reader (:func:`_direct_retained_action`, :func:`_rebind_live_retained`)
first check its fixed import-time identity, so a substituted reader is refused
before it is called and cannot skip the guard it would contain. Author-approved
contract change: replacing only an upper module's private helper alias after
issuance is no longer a use-time revocation; the owner checks its aliases before
it issues. As for the producer readers, interpreter attacks on closure cells/code
and concurrent writes are outside scope.
"""

from __future__ import annotations

from types import FunctionType
from typing import Any
from weakref import ref

from powermean.data.numeric_fit import NumericFit
from powermean.data.sample_fe_source import Derived
from powermean.data.sample_fe_state import FESample
from powermean.fe.certificate_basis import (
    CertificateBasis,
    _construction_code,
    _matching_retained_headers,
    _operational_data,
    _produced_retained_action,
    _retained_header,
    _retained_headers,
)
from powermean.fe.groups import Block, Grouping

# The issuing owner's passive description: exact anchor classes, the
# (anchor, field, target) links over (*anchors, numeric, sample), and its
# import-time (class, name, original) slot/field/method pins.
_Owner = tuple[
    tuple[type, ...],
    tuple[tuple[int, str, int], ...],
    tuple[tuple[type, str, object], ...],
]

_LIVE_RETAINED = "_live_retained_rows"


def _same_group_storage(left: object, right: object) -> bool:
    """Compare complete immutable owners and headers, not row values or flags."""
    if (type(left) is not Grouping and type(left) is not Derived) or type(
        right
    ) is not Grouping:
        return False
    a, b = vars(left), vars(right)
    aa, bb = a.get("blocks"), b.get("blocks")
    if type(aa) is not tuple or type(bb) is not tuple or len(aa) != len(bb):
        return False
    arrays = [(a.get("positions"), b.get("positions"))]
    for x, y in zip(aa, bb, strict=True):
        if type(x) is not Block or type(y) is not Block:
            return False
        arrays.extend((vars(x).get(k), vars(y).get(k)) for k in ("codes", "counts"))
    for x, y in arrays:
        first, second = _retained_header(x), _retained_header(y)
        if (
            first is None
            or second is None
            or first[1] is not second[1]
            or first[2:] != second[2:]
        ):
            return False
    return True


def _owner_live(objects: tuple[object, ...], owner: _Owner) -> bool:
    """The issuing owner's exact chain, read afresh: no callback or namespace."""
    kinds, links, pins = owner
    count = len(kinds)
    return (
        len(objects) == count + 2
        and all(found is not None for found in objects)
        and all(
            type(found) is kind
            for found, kind in zip(objects[:count], kinds, strict=True)
        )
        and all(vars(kind).get(name) is original for kind, name, original in pins)
        and all(vars(objects[a]).get(name) is objects[b] for a, name, b in links)
    )


def _issue_live_retained(
    basis: CertificateBasis,
    sample: object,
    numeric: object,
    anchors: tuple[object, ...],
    owner: _Owner,
) -> None:
    """Issue one weak live reader on the owner's fresh producer basis, or none.

    The producer proof and both header sets are read here. The owner supplies only
    its own anchors and pins; they must hold now and are read again at every use.
    """
    try:
        if (
            _live_retained_original is not _LIVE_RETAINED_GUARD
            or not _live_retained_original()
            or type(numeric) is not NumericFit
            or not _owner_live((*anchors, numeric, sample), owner)
        ):
            return
        payload = _produced_retained_action(basis, sample)
        if payload is None or not _same_group_storage(
            vars(sample).get("grouping"), vars(numeric).get("_grouping")
        ):
            return
        headers = _retained_headers(basis, vars(basis)["grouping"])
        scope = (
            tuple(ref(found) for found in anchors),
            owner,
            ref(numeric),
            ref(sample),
            ref(basis),
            ref(basis),
            None,
        )

        def construction() -> tuple[Any, Any, Any]:
            return scope, payload, headers

        object.__setattr__(basis, _LIVE_RETAINED, construction)
    except (MemoryError, TypeError):
        return  # Optional live binding only, never a numerical action.


def _rebind_live_retained(
    owner: object, source: object, found: object, problem: object
) -> None:
    """Carry only this owner's genuine live source binding to its rebound pair."""
    try:
        if _live_retained is not _LIVE_RETAINED_READER:
            return  # A substituted lower reader is refused, never called.
        carried = _live_retained(source, None)
        if carried is None or carried[0][0][0]() is not owner:
            return
        scope0, payload, _ = carried
        numeric = scope0[2]()
        if type(found) is not CertificateBasis:
            return
        a, b = vars(source), vars(found)
        if (
            any(
                a.get(name) is not b.get(name)
                for name in (
                    "route",
                    "rank",
                    "rows",
                    "selected",
                    "references",
                    "columns",
                    "positions",
                    "witnesses",
                    "echelon",
                    "_operational",
                )
            )
            or vars(problem).get("_numeric_owner") is not numeric
            or vars(problem).get("grouping") is not b.get("grouping")
            or not _same_group_storage(
                b.get("grouping"), vars(numeric).get("_grouping")
            )
        ):
            return
        headers = _retained_headers(found, b["grouping"])
        if headers is None:
            return
        scope = (*scope0[:4], ref(found), scope0[5], ref(problem))

        def construction() -> tuple[Any, Any, Any]:
            return scope, payload, headers

        object.__setattr__(found, _LIVE_RETAINED, construction)
    except (MemoryError, TypeError):
        return  # Optional live binding only, never a numerical action.


def _live_retained(basis: object, problem: object) -> tuple[Any, Any, Any] | None:
    """Code-authenticated weak live binding; copies and stale owners grant nothing."""
    if (
        type(basis) is not CertificateBasis
        or _live_retained_original is not _LIVE_RETAINED_GUARD
        or not _live_retained_original()
    ):
        return None
    reader = vars(basis).get(_LIVE_RETAINED)
    if type(reader) is not FunctionType or (
        reader.__code__ is not _construction_code(_issue_live_retained)
        and reader.__code__ is not _construction_code(_rebind_live_retained)
    ):
        return None
    scope, payload, headers = reader()
    anchors, owner, numeric_at, sample_at, bound, source_at, problem_at = scope
    numeric, sample = numeric_at(), sample_at()
    if (
        bound() is not basis
        or type(numeric) is not NumericFit
        or not _owner_live((*(weak() for weak in anchors), numeric, sample), owner)
    ):
        return None
    source = source_at()
    actual = _produced_retained_action(source, sample)
    grouping = vars(basis).get("grouping")
    if (
        actual is None
        or actual[0] is not payload[0]
        or (
            _operational_data(source) is not None
            and _operational_data(basis) is not payload[0]
        )
        or not _matching_retained_headers(basis, grouping, headers)
        or not _same_group_storage(grouping, vars(numeric).get("_grouping"))
    ):
        return None
    if problem is not None and (
        problem_at is None
        or problem_at() is not problem
        or vars(problem).get("_numeric_owner") is not numeric
        or vars(problem).get("grouping") is not grouping
    ):
        return None
    return scope, payload, headers


def _direct_retained_action(basis: object, problem: object) -> Any:
    """The producer-issued retained action of a live bound pair, else ``None``."""
    if _live_retained is not _LIVE_RETAINED_READER:
        return None  # A substituted lower reader is refused, never called.
    carried = _live_retained(basis, problem)
    return None if carried is None else carried[1][0]


_LIVE_RETAINED_CLASSES = (
    NumericFit,
    CertificateBasis,
    FESample,
    Grouping,
    Derived,
    Block,
)
_LIVE_RETAINED_SLOTS = tuple(
    (kind, name, vars(kind).get(name))
    for kind in _LIVE_RETAINED_CLASSES
    for name in ("__getattribute__", "__getattr__", "__dict__")
)
_LIVE_RETAINED_CALLS = (
    _produced_retained_action,
    _retained_header,
    _retained_headers,
    _matching_retained_headers,
    _construction_code,
    _operational_data,
    _same_group_storage,
    _owner_live,
    _issue_live_retained,
    _rebind_live_retained,
    _live_retained,
    _direct_retained_action,
)


def _live_retained_original() -> bool:
    """Constant-size check of this module's canonical helpers and lower slots."""
    kinds = (NumericFit, CertificateBasis, FESample, Grouping, Derived, Block)
    if any(a is not b for a, b in zip(kinds, _LIVE_RETAINED_CLASSES, strict=True)):
        return False
    current = (
        _produced_retained_action,
        _retained_header,
        _retained_headers,
        _matching_retained_headers,
        _construction_code,
        _operational_data,
        _same_group_storage,
        _owner_live,
        _issue_live_retained,
        _rebind_live_retained,
        _live_retained,
        _direct_retained_action,
    )
    return all(
        a is b for a, b in zip(current, _LIVE_RETAINED_CALLS, strict=True)
    ) and all(
        vars(kind).get(name) is expected
        for kind, name, expected in _LIVE_RETAINED_SLOTS
    )


_LIVE_RETAINED_GUARD = _live_retained_original
# The reader's fixed import-time identity. Its two callers check it before every
# call, so a substituted lower reader is refused uncalled; the reader itself then
# runs the guard above. One identity test per call, no traversal.
_LIVE_RETAINED_READER = _live_retained
