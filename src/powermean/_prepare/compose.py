"""From an evaluated model frame to the sample construction, by position.

This module covers steps 1 ("parse the formula; evaluate transformations; apply
``subset``") and 2 of the sample construction.  :func:`prepare_frame` takes a
:class:`~powermean.formula.frame.ModelFrame`, which Formulaic evaluated on every row of
the data, and the data themselves, and forms the arrays and masks of
:func:`powermean.data.sample.build_sample`:

- the raw outcome and the trained design as doubles, missing where the frame says so;
- the missing masks by group: ``outcome`` from the raw outcome; ``regressors`` from the
  raw data columns that the regressors read (a Boolean transform can hide a missing
  value, a missing category is encoded like the baseline) together with the evaluated
  design (a transform can make a missing or infinite value from finite data), the two
  masks kept apart on the frame; ``weights``, ``offset`` and ``clusters`` from their
  raw columns, read by :mod:`powermean.formula.validity`;
- the raw weights as their original objects (:class:`~powermean.data.raw.RawWeights`),
  the offset or the exposure as doubles, and the cluster and other requested columns
  as aligned metadata, each as its original values.

The input extraction helpers live in :mod:`powermean._prepare.inputs`, the dependence
provenance records in :mod:`powermean._prepare.dependence`, and the pandas/frame axis
ownership helpers in :mod:`powermean._prepare.owned`.  All extracted helpers are
re-exported here so existing callers of :mod:`powermean._prepare.compose` are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
from numpy.typing import ArrayLike

from powermean._prepare.dependence import (
    ColumnProvenance,
    DependenceProvenance,
    DependenceRole,
    UsedDependence,
    _prepare_dependence,
    _retained_original,
    prepare_dependence,
)
from powermean._prepare.effects import (
    _native_effects, _native_effects_original, read_effects,
)
from powermean._prepare.inputs import (
    Columns,
    _check_same_data,
    _column,
    _doubles,
    _inputs,
    _nan,
    _object_subset,
    _subset_mask,
    _values,
)
from powermean._prepare.owned import (
    _holds_objects,
    _owned_frame,
    _owned_index,
    _owned_multiindex,
    _owned_names,
    _owned_series,
    _owned_table,
)
from powermean.data.accounting import MissingAccount, frozen_codes
from powermean.data.retained import OrderRows, OrderSample, _derive_order_parts
from powermean.data.sample import SampleBase, build_sample
from powermean.data.sample_fe import EffectColumns
from powermean.formula.frame import ModelFrame
from powermean.model.order import _flags

if TYPE_CHECKING:
    __all__ = [
        "ColumnProvenance",
        "Columns",
        "DependenceProvenance",
        "DependenceRole",
        "PreparedFrame",
        "UsedDependence",
        "_check_same_data",
        "_column",
        "_doubles",
        "_holds_objects",
        "_inputs",
        "_nan",
        "_object_subset",
        "_owned_frame",
        "_owned_index",
        "_owned_multiindex",
        "_owned_names",
        "_owned_series",
        "_owned_table",
        "_subset_mask",
        "_values",
        "prepare_dependence",
        "prepare_frame",
    ]


def _isolated_sample_base(base: SampleBase) -> SampleBase:
    """Return a ``SampleBase`` with fresh array objects for its masks."""
    isolated_missing = MissingAccount._owned(
        {group: _flags(mask) for group, mask in base.missing.groups.items()},
        _flags(base.missing.excluded),
    )
    return SampleBase(
        inputs=base.inputs,
        subset=None if base.subset is None else _flags(base.subset),
        missing=isolated_missing,
        complete=_flags(base.complete),
        negative_weight=_flags(base.negative_weight),
        zero_weight=_flags(base.zero_weight),
        codes=frozen_codes(base.codes),
    )


@dataclass(frozen=True, eq=False, init=False)
class PreparedFrame:
    """The frame, its labels and the order-neutral sample base, in one owned state.

    Parameters
    ----------
    frame : ModelFrame
        The evaluated frame, with the fitted spec of its design; copied.
    labels : pandas.Index
        The user's row labels, in input order, as given; copied, its object labels
        and its names included (:func:`_owned_index`).
    columns : Columns
        The columns that were read.
    base : SampleBase
        The order-neutral state of the sample.
    subset_missing : int
        The number of rows whose subset condition was missing, excluded by the subset.
    effects : EffectColumns, optional
        The owned raw labels of the fixed-effect terms; ``None`` without a
        fixed-effect part.
    dependence : DependenceProvenance, optional
        Fit-used and retained-only column provenance.

    Attributes
    ----------
    columns, base, subset_missing, effects, dependence
        As given.
    frame : ModelFrame
        A fresh copy of the retained frame, its fitted spec, owned labels and owned
        axes included, on every access.
    labels : pandas.Index
        A fresh owned copy of the retained labels on every access.
    """

    columns: Columns
    base: SampleBase
    subset_missing: int
    effects: EffectColumns | None
    dependence: DependenceProvenance | None
    _frame: ModelFrame
    _labels: pd.Index[Any]

    def __init__(
        self,
        *,
        frame: ModelFrame,
        labels: pd.Index[Any],
        columns: Columns,
        base: SampleBase,
        subset_missing: int,
        effects: EffectColumns | None = None,
        dependence: DependenceProvenance | None = None,
    ) -> None:
        """Keep an owned copy of the frame and of the labels, and the rest as given."""
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "subset_missing", subset_missing)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "dependence", dependence)
        object.__setattr__(self, "_frame", _owned_frame(frame))
        object.__setattr__(self, "_labels", _owned_index(labels))

    @property
    def frame(self) -> ModelFrame:
        """A fresh copy of the retained frame, with its fitted spec, learned once."""
        return _owned_frame(self._frame)

    @property
    def labels(self) -> pd.Index[Any]:
        """A fresh owned copy of the user's row labels, in input order, as given."""
        return _owned_index(self._labels)

    def for_order(self, omega: float, *, positive_only: bool = False) -> OrderSample:
        """The sample of the mean order ``omega`` (:meth:`SampleBase.for_order`).

        Uses an isolated copy of :attr:`base` and its :class:`MissingAccount` so that
        no in-place shape or dtype mutation on the returned :class:`OrderSample`'s
        missingness masks can reach this :class:`PreparedFrame`.
        """
        return _isolated_sample_base(self.base).for_order(
            omega, positive_only=positive_only
        )

    def _order_parts(
        self, omega: float, *, positive_only: bool = False
    ) -> tuple[OrderSample, OrderRows]:
        """Fresh owned parts with the SAME authoritative base-mask isolation."""
        return _derive_order_parts(
            _isolated_sample_base(self.base), omega, positive_only=positive_only
        )

    def retained_labels(self, sample: OrderSample) -> pd.Index[Any]:
        """The user's labels of the retained rows of ``sample``, by position.

        Parameters
        ----------
        sample : OrderSample
            A sample derived from this frame's base.

        Returns
        -------
        pandas.Index
            ``labels.take(sample.positions)``: the same Index type, names and dtype,
            duplicates and missing labels included; a fresh owned copy, whose object
            labels and names are copies (:func:`_owned_index`).

        Raises
        ------
        ValueError
            A sample of another number of input rows.
        """
        if sample.nobs != len(self._labels):
            raise ValueError("sample does not belong to this frame")
        return _owned_index(self._labels.take(sample.positions))


def prepare_frame(
    frame: ModelFrame,
    data: pd.DataFrame,
    *,
    columns: Columns | None = None,
    subset: pd.Series[Any] | ArrayLike | None = None,
    dependence: UsedDependence | None = None,
) -> PreparedFrame:
    """Feed an evaluated frame to the sample construction, by position.

    Parameters
    ----------
    frame : ModelFrame
        The frame of :func:`powermean.formula.frame.model_frame` on ``data``.
    data : pandas.DataFrame
        The data of the frame, read by position for the columns of ``columns``.
    columns : Columns, optional
        The weights, offset or exposure, cluster and metadata columns; none by
        default.
    subset : array_like of bool or pandas.Series, optional
        The evaluated subset condition, one value per row (module docstring).
    dependence : UsedDependence, optional
        Additional fit-used dependence columns (whose invalid entries join the
        existing ``clusters`` missing group) and retained-only columns (which
        exclude no rows at the fit).

    Returns
    -------
    PreparedFrame
        An owned copy of the frame, its labels and the order-neutral base of the
        sample.
    """
    return _prepare_frame(
        frame, data, columns=columns, subset=subset, dependence=dependence
    )


def _prepare_frame(
    frame: ModelFrame, data: pd.DataFrame, *, columns: Columns | None = None,
    subset: pd.Series[Any] | ArrayLike | None = None,
    dependence: UsedDependence | None = None,
    _retained: Mapping[str, tuple[str, ...]] | None = None,
) -> PreparedFrame:
    """Shared preparation body; no public foreign-input ownership assertion."""
    if type(frame) is not ModelFrame:
        raise TypeError("frame must be a ModelFrame")
    chosen = Columns() if columns is None else columns
    _check_same_data(frame, data)
    if _retained is None:
        inputs, missing, provenance = prepare_dependence(
            frame, data, chosen, dependence
        )
    else:
        inputs, missing, provenance = _prepare_dependence(
            frame, data, chosen, dependence, _retained=_retained
        )
    effects = None
    if frame.parts.fixed_effects:
        # Both components of an interaction mark step 2, before any grouping.
        native = (
            _native_effects(data, frame.parts.fixed_effects, provenance)
            if _retained is not None and _native_retained_original() else None
        )
        effects, missing["effects"] = (
            read_effects(data, frame.parts.fixed_effects) if native is None else native
        )
    mask, subset_missing = _subset_mask(subset, data)
    base = build_sample(inputs, subset=mask, missing=missing)
    return PreparedFrame(
        frame=frame,
        labels=frame.design.labels,
        columns=chosen,
        base=base,
        subset_missing=subset_missing,
        effects=effects,
        dependence=provenance,
    )

_RETAIN_DEPENDENCE = prepare_dependence
_RETAIN_NUMERIC_DEPENDENCE = _prepare_dependence
_RETAIN_EFFECTS = read_effects
_RETAIN_NATIVE_EFFECTS = _native_effects
_RETAIN_EFFECT_GUARD = _native_effects_original
_RETAIN_QUALIFY = _retained_original
_RETAIN_PREPARED = PreparedFrame
_RETAIN_FRAME_METHODS = tuple((name, vars(PreparedFrame).get(name)) for name in (
    "__init__", "__getattribute__", "__getattr__", "for_order", "_order_parts",
))


def _native_retained_original() -> bool:
    """Qualify the existing normal producer, not an alternate preparation engine."""
    return (prepare_dependence is _RETAIN_DEPENDENCE
            and _prepare_dependence is _RETAIN_NUMERIC_DEPENDENCE
            and read_effects is _RETAIN_EFFECTS
            and _native_effects is _RETAIN_NATIVE_EFFECTS
            and _native_effects_original is _RETAIN_EFFECT_GUARD
            and _native_effects_original()
            and _retained_original is _RETAIN_QUALIFY
            and PreparedFrame is _RETAIN_PREPARED
            and all(vars(PreparedFrame).get(name) is original
                    for name, original in _RETAIN_FRAME_METHODS)
            and _retained_original())
