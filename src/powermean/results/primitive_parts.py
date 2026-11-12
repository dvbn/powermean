"""The parts of a primitive capture that internal consumers read, materialized once.

Internal to the primitive snapshot route of
:class:`~powermean.interface.retained_context.PerFitPreparationContext`; the general
route, the package default, uses nothing here.

What is shared.  A :class:`~powermean.results.primitive_snapshot.PrimitiveCapture` is
bytes, plain tuples and plain scalars.  Its consumers read it in two forms.  The
metadata binding and its check, and the retained input snapshot, read records: the
``SampleBase``, the row labels and the ``DependenceProvenance``, selected at a
sample's rows; :class:`PrimitiveParts` materializes those once per context, over the
capture's immutable bytes (read-only arrays that view those bytes, which NumPy refuses
to make writable, and tuples of plain labels).  The fresh validation of a live sample
reads the capture's compact values, its buffers by shape and bytes and its object
columns by their plain tuples (fresh working arrays only for array predicates)
(:class:`~powermean.results.primitive_sample_check.CompactCapture`), formed once per
context as well: genuinely owned immutable data shared inside the execution context,
never a validation result.

What stays fresh.  Each consumer's check runs on every call against the live object it
is given (:meth:`PrimitiveParts.check_sample` is
:func:`~powermean.results.primitive_sample_check.check_compact_sample` on the live
``FESample``: the three checks of the unchanged generic comparator
:func:`~powermean.results.absorbed_sample_binding.check_parts_match_sample`, which
remains the independent reference); nothing is remembered between calls, and no
identity, hash, flag or prior success stands in for a comparison.  Nothing of the parts
is handed outward: every outward accessor of the context, of the preparation and of the
retained snapshots materializes afresh from the capture or selects fresh objects at the
requested rows (:meth:`labels_at`, :meth:`metadata_at`), so a caller's edit of what it
received, including the shape, dtype or strides of an array object, reaches nothing
retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from powermean._prepare.dependence import DependenceProvenance
from powermean._prepare.inputs import Columns
from powermean.data.raw import AlignedColumns
from powermean.data.sample import SampleBase
from powermean.data.sample_fe import EffectColumns
from powermean.data.sample_fe_state import FESample
from powermean.model.order import IndexArray

if TYPE_CHECKING:
    from powermean.results.primitive_compact import CompactCapture
    from powermean.results.primitive_sample_check import check_compact_sample
else:
    from powermean.results.primitive_sample_check import (
        CompactCapture,
        check_compact_sample,
    )

from powermean.results.primitive_snapshot import MetadataParts, PrimitiveCapture

__all__ = ["PrimitiveParts"]


@dataclass(frozen=True, eq=False, init=False)
class PrimitiveParts:
    """The once-materialized parts of one capture (module docstring).

    Formed by :meth:`of` from a :class:`PrimitiveCapture` and held by the context
    that owns that capture; compares by identity, as the records it holds do.

    Attributes
    ----------
    capture : PrimitiveCapture
        The immutable capture the parts were materialized from.
    base : SampleBase
        The order-neutral state of the sample, with its raw inputs.
    labels : pandas.Index
        The user's row labels, in input order.
    columns : Columns
        The columns record of the preparation.
    effects : EffectColumns or None
        The raw labels of the fixed-effect terms.
    dependence : DependenceProvenance or None
        Fit-used and retained-only column provenance.
    metadata : MetadataParts
        The base, the labels and the dependence, as the metadata binding reads them.
    compact : CompactCapture
        The capture's compact values, as the fresh live sample check reads them.
    """

    capture: PrimitiveCapture
    base: SampleBase
    labels: pd.Index[Any]
    columns: Columns
    effects: EffectColumns | None
    dependence: DependenceProvenance | None
    metadata: MetadataParts
    compact: CompactCapture

    def __init__(self, capture: PrimitiveCapture) -> None:
        """Materialize every part of ``capture`` once, through its part methods."""
        if type(capture) is not PrimitiveCapture:
            raise TypeError("capture must be a PrimitiveCapture")
        base = capture.sample_base()
        labels = capture.labels_index()
        dependence = capture.dependence_provenance()
        object.__setattr__(self, "capture", capture)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "columns", capture.columns_record())
        object.__setattr__(self, "effects", capture.effect_columns())
        object.__setattr__(self, "dependence", dependence)
        object.__setattr__(
            self,
            "metadata",
            MetadataParts(base=base, labels=labels, dependence=dependence),
        )
        object.__setattr__(self, "compact", CompactCapture.of(capture))

    @classmethod
    def of(cls, capture: PrimitiveCapture) -> PrimitiveParts:
        """The parts of ``capture``."""
        return cls(capture)

    @property
    def nobs(self) -> int:
        """The number of input rows."""
        return self.base.nobs

    @property
    def labels_size(self) -> int:
        """The number of row labels."""
        return len(self.labels)

    def check_sample(self, sample: FESample) -> None:
        """Validate the live ``sample`` against the capture's values, afresh each call.

        The three sample-versus-frame checks of
        :func:`~powermean.results.absorbed_sample_binding.check_prepared_matches_sample`
        in their order, on the compact values of the capture
        (:func:`~powermean.results.primitive_sample_check.check_compact_sample`);
        ``sample`` is read afresh and nothing is kept between calls.
        """
        check_compact_sample(self.compact, sample)

    def labels_at(self, positions: IndexArray) -> pd.Index[Any]:
        """A fresh ``Index`` of the row labels at ``positions``, in that order.

        A selection, not a copy of every label: the labels of the primitive route are
        plain immutable values, and ``take`` forms a new array of them.
        """
        return self.labels.take(positions)

    def metadata_at(self, positions: IndexArray) -> AlignedColumns:
        """Fresh owned aligned metadata columns at ``positions``, in that order."""
        return self.base.inputs.metadata.take(positions)
