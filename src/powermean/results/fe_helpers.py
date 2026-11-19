"""Shared result argument and normal-reference helpers; unchanged no-FE arithmetic."""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import MappingProxyType
from typing import Any, Literal, TypedDict, cast

import numpy as np

from powermean import messages
from powermean.data.raw import _OwnedMapping
from powermean.errors import SpecificationError
from powermean.fe._cancel_storage import FloatStorage, IndexStorage
from powermean.model.variance import FloatArray
from powermean.results.owned import owned_array


def _array_view(
    value: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Detach metadata, sharing numeric backing only when it is immutable bytes."""
    base: object = value
    while isinstance(base, np.ndarray):
        base = base.base
    if not value.dtype.hasobject and isinstance(base, bytes):
        return value.view()
    return owned_array(value)


# Exact types whose deep copy is the value itself (``copy`` copies them atomically).
_ATOMS = frozenset({int, float, bool, complex, str, bytes, type(None)})


def _detached(value: object, memo: dict[int, object]) -> object:
    """Clone an owned diagnostic graph, not its enclosing retained fit.

    Dataclass constructors, numerical callbacks and validators are not replayed.
    The memo preserves diagnostic aliases; bytes stay shared and ndarray headers
    detach. A readonly flag alone never authorizes sharing writable backing.
    A value of an exact atomic type is its own deep copy and is returned without a
    memo entry: the memo can hold nothing else under its id, since every original
    entered there is kept alive.
    """
    if type(value) in _ATOMS:
        return value
    if id(value) in memo:
        return memo[id(value)]
    copied = _copied(value, memo)
    memo[id(value)] = copied
    # Lazy fields can return temporary tuples/headers. Their IDs remain in the
    # memo, so keep their originals alive in deepcopy's invocation-local slot.
    alive = cast("list[object] | None", memo.get(id(memo)))
    if alive is None:
        memo[id(memo)] = [value]
    else:
        alive.append(value)
    return copied


_ABSENT = object()


def _plain_tuple(value: tuple[Any, ...], memo: dict[int, Any]) -> tuple[Any, ...]:
    """The detached copy of a plain tuple, item by item.

    An atom is its own copy. A plain tuple item takes the steps of
    :func:`_detached` here, in the same order: the memo's copy if it holds one,
    else its own copy, entered in the memo and kept alive once it is complete.
    Any other item goes through :func:`_detached`. The list that keeps originals
    alive is the memo's own, made here if no value has entered it yet.
    """
    alive = memo.get(id(memo))
    if alive is None:
        alive = memo[id(memo)] = []
    items: list[Any] = []
    append = items.append
    for item in value:
        kind = type(item)
        if kind in _ATOMS:
            append(item)
        elif kind is tuple:
            known = memo.get(id(item), _ABSENT)
            if known is _ABSENT:
                known = _plain_tuple(item, memo)
                memo[id(item)] = known
                alive.append(item)
            append(known)
        else:
            append(_detached(item, memo))
    return tuple(items)


def _copied_dataclass(value: Any, memo: dict[int, object]) -> object:
    """Return a shallow dataclass copy with recursively detached fields."""
    copied = copy.copy(value)
    memo[id(value)] = copied
    for field in fields(value):
        object.__setattr__(
            copied, field.name, _detached(getattr(value, field.name), memo)
        )
    return copied


def _copied_mapping(value: Any, memo: dict[int, object]) -> object:
    """Return a detached mapping of the same wrapper type."""
    items = {key: _detached(item, memo) for key, item in value.items()}
    if isinstance(value, _OwnedMapping):
        return _OwnedMapping(items)
    return MappingProxyType(items) if isinstance(value, MappingProxyType) else items


def _copied(value: object, memo: dict[int, object]) -> object:
    """The detached copy of one value that the memo does not hold yet.

    A plain tuple is tested first; it takes the tuple branch below in any case.
    """
    if type(value) is tuple:
        return _plain_tuple(value, memo)
    if isinstance(value, np.ndarray):
        return _array_view(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _copied_dataclass(value, memo)
    if isinstance(value, (_OwnedMapping, MappingProxyType, dict)):
        return _copied_mapping(value, memo)
    if (
        isinstance(value, tuple)
        and type(value) is not FloatStorage
        and type(value) is not IndexStorage
    ):
        return tuple(_detached(item, memo) for item in value)
    if isinstance(value, list):
        return [_detached(item, memo) for item in value]
    return copy.deepcopy(value, memo)


def detached_record[RecordT](value: RecordT) -> RecordT:
    """Return independent outward record wrappers of the same concrete type."""
    return cast("RecordT", _detached(value, {}))


PredictionKind = Literal["power_mean"]
PREDICTION_KINDS: tuple[PredictionKind, ...] = ("power_mean",)
DEFAULT_PREDICTION: PredictionKind = "power_mean"
PREDICTION_KEYWORDS: tuple[str, ...] = ("type",)


class PredictionOptions(TypedDict, total=False):
    """The keyword options of :meth:`Result.predict`, as ``DESIGN.md``, 4.4, names them.

    ``type`` is the kind of prediction, ``"power_mean"`` by default: the fitted
    conditional power mean of the order of the fit, the only kind of the first
    milestone.  No other key exists; :func:`prediction_kind` refuses one at run time,
    as the type checker refuses it at the call.
    """

    type: PredictionKind


def prediction_kind(options: Mapping[str, object]) -> PredictionKind:
    """The kind of prediction that the keywords of ``predict`` name.

    Parameters
    ----------
    options : mapping
        The keywords given, as the collector received them.

    Returns
    -------
    str
        ``"power_mean"``, the default and the only kind.

    Raises
    ------
    SpecificationError
        A keyword other than ``type`` (``specification.unknown_option``, with the
        valid keyword), or a ``type`` that the first milestone does not have
        (``specification.invalid_value``, with the valid kinds).
    """
    for key in options:
        if key not in PREDICTION_KEYWORDS:
            raise SpecificationError(
                "specification.unknown_option",
                {
                    "kind": "keyword of predict",
                    "key": key,
                    "valid": messages.format_keys(PREDICTION_KEYWORDS),
                },
            )
    kind = options.get("type", DEFAULT_PREDICTION)
    for valid in PREDICTION_KINDS:
        if kind == valid:
            return valid
    raise SpecificationError(
        "specification.invalid_value",
        {
            "argument": "type",
            "value": kind,
            "valid": messages.format_keys(PREDICTION_KINDS),
        },
    )


def normal_quantile(level: float) -> float:
    """``q`` with ``P(|Z| <= q) = level`` under the standard normal distribution."""
    if not isinstance(level, float) or not 0 < level < 1:
        raise ValueError("level must be a float strictly between zero and one")
    return statistics.NormalDist().inv_cdf((1 + level) / 2)


def two_sided_p(statistic: FloatArray) -> FloatArray:
    """``P(|Z| >= |z|)`` under the standard normal, NaN where ``z`` is undefined."""
    values = np.full(statistic.shape, np.nan)
    for position, value in enumerate(statistic.tolist()):
        if not math.isnan(value):
            values[position] = math.erfc(abs(value) / math.sqrt(2.0))
    return values
