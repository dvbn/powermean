"""Shared result argument and normal-reference helpers; unchanged no-FE arithmetic."""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import MappingProxyType
from typing import Literal, TypedDict, cast

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


def _detached(value: object, memo: dict[int, object]) -> object:
    """Clone an owned diagnostic graph, not its enclosing retained fit.

    Dataclass constructors, numerical callbacks and validators are not replayed.
    The memo preserves diagnostic aliases; bytes stay shared and ndarray headers
    detach. A readonly flag alone never authorizes sharing writable backing.
    """
    if id(value) in memo:
        return memo[id(value)]
    copied: object
    if isinstance(value, np.ndarray):
        copied = _array_view(value)
    elif is_dataclass(value) and not isinstance(value, type):
        copied = copy.copy(value)
        memo[id(value)] = copied
        for field in fields(value):
            object.__setattr__(
                copied, field.name, _detached(getattr(value, field.name), memo)
            )
    elif isinstance(value, _OwnedMapping):
        copied = _OwnedMapping(
            {key: _detached(item, memo) for key, item in value.items()}
        )
    elif isinstance(value, (MappingProxyType, dict)):
        items = {key: _detached(item, memo) for key, item in value.items()}
        copied = (
            MappingProxyType(items) if isinstance(value, MappingProxyType) else items
        )
    elif (isinstance(value, tuple) and type(value) is not FloatStorage
          and type(value) is not IndexStorage):
        copied = tuple(_detached(item, memo) for item in value)
    elif isinstance(value, list):
        copied = [_detached(item, memo) for item in value]
    else:
        copied = copy.deepcopy(value, memo)
    memo[id(value)] = copied
    # Lazy fields can return temporary tuples/headers. Their IDs remain in the
    # memo, so keep their originals alive in deepcopy's invocation-local slot.
    alive = cast("list[object] | None", memo.get(id(memo)))
    if alive is None:
        memo[id(memo)] = [value]
    else:
        alive.append(value)
    return copied


def detached_record[RecordT](value: RecordT) -> RecordT:
    """Return independent outward record wrappers of the same concrete type."""
    return cast("RecordT", _detached(value, {}))


PredictionKind = Literal["power_mean"]
PREDICTION_KINDS: tuple[PredictionKind, ...] = ("power_mean",)
DEFAULT_PREDICTION: PredictionKind = "power_mean"
PREDICTION_KEYWORDS: tuple[str, ...] = ("type",)


class PredictionOptions(TypedDict, total=False):
    """The keyword options of :meth:`Result.predict`.

    ``type`` is the kind of prediction, ``"power_mean"`` by default: the fitted
    conditional power mean of the order of the fit, the only kind available.  No other
    key exists; :func:`prediction_kind` refuses one at run time, as the type checker
    refuses it at the call.
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
        valid keyword), or a ``type`` that is not available
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
