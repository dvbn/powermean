"""The fresh comparison of a live sample with the compact values of a primitive capture.

Internal to the primitive snapshot route of
:class:`~powermean.interface.retained_context.PerFitPreparationContext`; the general
route, the package default, uses nothing here.

What is compared.  The three sample-versus-frame checks of
:func:`~powermean.results.absorbed_sample_binding.check_prepared_matches_sample`, in
their order and with their predicates (the unchanged generic comparator remains the
independent reference of the general route and of every fallback below): the base and
its masks, the inputs and the metadata, the effects and the two partitions.  Each
predicate reads the live ``FESample`` afresh on every call: its arrays, its account of
step 2, the storage of its metadata and weight columns, the source columns of its
derived blocks, the codes and levels of every block.  Nothing is remembered between
calls; no identity, hash, flag or prior success stands in for a comparison.

What is compact.  The capture side is the
:class:`~powermean.results.primitive_compact.CompactCapture` of the context: buffers
by shape and bytes, object columns by their plain tuples with their recorded kind.
Array-based predicates form working arrays from those tuples, each column once per
check, never kept beyond it.  A live column of the exact
:class:`~powermean.data.raw.Column` type is read from storage: a plain tuple is scanned
for its one exact kind, afresh, once per comparison, and compared as a whole when the
kinds agree, or label by label otherwise (the predicate of
:func:`~powermean.results.retained_input_values._same_column_values`: missing labels
equal, same exact type and ``==``); numeric storage is compared by dtype, shape and
bytes.  Only an entirely plain live tuple may skip the column's copy protocol. A live
tuple containing a custom or other nonplain value, a column of another type, or object
storage held in an array takes the generic predicate on its ``values`` as the reference
does.  Integers of any size, finite floats, Booleans and strings are compared exactly,
as the reference compares them.

Refusals are those of the reference: ``ValueError`` with the unbound message at the same
predicate, ``IndexError`` from the same indexing of codes that are not within their
levels, and the error of the same attribute read of a part that is not what the sample
declares.
"""

from __future__ import annotations

from collections.abc import Set
from typing import Any, cast

import numpy as np

from powermean.data.accounting import MissingAccount
from powermean.data.raw import AlignedColumns, AnyArray, Column, RawWeights
from powermean.data.sample_fe_source import Derived, same_bits
from powermean.data.sample_fe_state import FESample
from powermean.fe.groups import Block
from powermean.fe.primitive_codes import first_rows
from powermean.model.order import IndexArray
from powermean.results.absorbed_sample_binding import (
    UNBOUND,
    _first_labels,
    _label_pairs,
    _same_label_value,
    _same_labels,
    _same_pairs,
)
from powermean.results.primitive_compact import (
    CompactCapture,
    _column_values,
    _CompactColumn,
    _CompactEffects,
    _PlainColumn,
    _TypedColumn,
    _values_of,
)
from powermean.results.primitive_snapshot import _Buffer
from powermean.results.retained_input_values import (
    _is_missing_scalar,
    _label_values,
    _plain_kind,
    _same_column_values,
)

__all__ = ["check_compact_sample"]

# The working arrays and kinds of the capture's effect columns, formed in one check.
type _Working = dict[str, tuple[AnyArray, type | None]]


# ------------------------------------------------------------------ predicates


def _same_buffer(enc: _Buffer, live: AnyArray) -> bool:
    """``same_bits(enc.view(), live)``: one shape and the same bytes."""
    return enc.shape == live.shape and enc.data == live.tobytes()


def _same_plain(column: _PlainColumn, stored: tuple[object, ...], kind: type) -> bool:
    """``_same_column_values`` of a plain capture column and a live plain tuple.

    ``kind`` is the one exact kind of the live tuple, scanned afresh by the caller in
    this comparison; a tuple of plain values cannot change its kind.  When it is the
    capture's kind the two tuples are compared as wholes, else label by label with the
    missing, exact-type and equality predicates of the reference.  The capture label is
    a plain built-in value, so no method of a live label runs: a type that differs
    refuses.
    """
    if len(stored) != len(column.labels):
        return False
    if kind is column.plain_type:
        return bool(column.labels == stored)
    for a, b in zip(column.labels, stored, strict=True):
        if _is_missing_scalar(a) and _is_missing_scalar(b):
            continue
        if type(a) is not type(b) or a != b:
            return False
    return True


def _same_compact_column(column: _CompactColumn, live: object) -> bool:
    """``_same_column_values(values, live.values)`` read from the live storage.

    A live column of the exact ``Column`` type is read from its storage; any other
    object, a nonplain tuple, and object storage held in an array are compared on their
    ``values`` as the reference compares them, before any length/type refusal.
    """
    if type(live) is not Column:
        return _same_column_values(_values_of(column), cast("Any", live).values)
    stored = live._stored
    if type(stored) is tuple:
        kind = _plain_kind(stored)
        if kind is None:
            return _same_column_values(_values_of(column), live.values)
        return type(column) is _PlainColumn and _same_plain(column, stored, kind)
    stored = cast("AnyArray", stored)
    shape, dtype = stored.shape, stored.dtype
    if dtype.kind == "O":
        return _same_column_values(_values_of(column), live.values)
    if type(column) is not _TypedColumn:
        return False
    return (
        shape == column.shape
        and dtype == column.dtype
        and (column.data == stored.tobytes())
    )


def _same_missing(compact: CompactCapture, live: MissingAccount) -> bool:
    """``same_missing`` of the capture's account and a live account."""
    groups = live.groups
    if set(compact.missing_names) != set(groups):
        return False
    found = [
        (dtype, enc, np.asarray(groups[group])) for group, dtype, enc in compact.missing
    ]
    found.append((*compact.excluded, np.asarray(live.excluded)))
    return all(dtype == y.dtype and _same_buffer(enc, y) for dtype, enc, y in found)


# ------------------------------------------------------- the three checks


def _check_base_and_masks(compact: CompactCapture, sample: FESample) -> None:
    """The first check: the base's row count, masks, codes and account of step 2."""
    grouping, src, s_base = sample.grouping, sample.source, sample.base
    enc = compact.base
    sub_ok = (enc.subset is None and s_base.subset is None) or (
        enc.subset is not None
        and s_base.subset is not None
        and _same_buffer(enc.subset, s_base.subset)
    )
    if (
        (compact.nobs, compact.nobs, compact.nobs)
        != (sample.ledger.nobs, s_base.nobs, compact.labels_size)
        or None in (compact.effects, grouping, src)
        or not sub_ok
        or not _same_buffer(enc.codes, s_base.codes)
        or not _same_buffer(enc.complete, s_base.complete)
        or not _same_buffer(enc.negative_weight, s_base.negative_weight)
        or not _same_buffer(enc.zero_weight, s_base.zero_weight)
        or not _same_missing(compact, s_base.missing)
        or not _same_missing(compact, sample.ledger.missing)
        or not bool(compact.complete[sample.positions].all())
        or not np.array_equal(
            sample.ledger.codes[compact.excluded_rows], compact.excluded_codes
        )
    ):
        raise ValueError(UNBOUND)


def _check_weights(compact: CompactCapture, sample: FESample, pos: IndexArray) -> None:
    """The weights of the second check: kinds, every row, the retained rows."""
    w_in, w_s, w_d = compact.weights, sample.base.inputs.weights, sample.data.weights
    if w_in is None:
        if w_d is not None or w_s is not None:
            raise ValueError(UNBOUND)
        return
    column, kind = w_in
    if w_d is None or w_s is None:
        raise ValueError(UNBOUND)
    if not (compact.columns.weight_kind == kind == w_d.kind == w_s.kind):
        raise ValueError(UNBOUND)
    same = (
        _same_compact_column(column, w_s.column)
        if type(w_s) is RawWeights
        else _same_column_values(_values_of(column), w_s.values)
    )
    if not same or not same_bits(
        np.asarray(_values_of(column)[pos], dtype=np.float64), w_d.values
    ):
        raise ValueError(UNBOUND)


def _check_offset_or_exposure(
    compact: CompactCapture, sample: FESample, pos: IndexArray
) -> None:
    """The offset or exposure of the second check, symmetric, every row and retained."""
    enc, s_inp, data = compact.base, sample.base.inputs, sample.data
    off_in, off_s = enc.offset, s_inp.offset
    exp_in, exp_s, exp_d = enc.exposure, s_inp.exposure, data.exposure
    if (off_in is None, exp_in is None, exp_in is None) != (
        off_s is None,
        exp_s is None,
        exp_d is None,
    ):
        raise ValueError(UNBOUND)
    if off_in is not None:
        assert compact.offset is not None
        if off_s is None or not (
            _same_buffer(off_in, off_s) and same_bits(compact.offset[pos], data.offset)
        ):
            raise ValueError(UNBOUND)
    elif exp_in is not None:
        assert compact.exposure is not None
        if (
            exp_s is None
            or exp_d is None
            or not (
                _same_buffer(exp_in, exp_s) and same_bits(compact.exposure[pos], exp_d)
            )
        ):
            raise ValueError(UNBOUND)
    elif not same_bits(np.zeros(pos.size, dtype=np.float64), data.offset):
        raise ValueError(UNBOUND)


def _same_metadata_column(compact: CompactCapture, metadata: object, name: str) -> bool:
    """One metadata column of the capture against the live aligned column ``name``."""
    column = compact.metadata[name]
    if type(metadata) is AlignedColumns:
        return _same_compact_column(column, metadata.columns[name])
    return _same_column_values(_values_of(column), cast("Any", metadata)[name])


def _check_inputs_and_metadata(compact: CompactCapture, sample: FESample) -> None:
    """The second check: outcome, design, weights, offset or exposure, metadata."""
    enc, s_inp, pos, cols = (
        compact.base,
        sample.base.inputs,
        sample.positions,
        compact.columns,
    )
    opt_cols = (cols.weights is None, cols.offset is None, cols.exposure is None)
    opt_inp = (enc.weights is None, enc.offset is None, enc.exposure is None)
    if not (
        _same_buffer(enc.y, s_inp.y)
        and _same_buffer(enc.design, s_inp.design)
        and enc.column_names == s_inp.column_names == sample.data.column_names
        and same_bits(compact.y[pos], sample.data.y)
        and same_bits(compact.design[pos], sample.data.design)
        and opt_cols == opt_inp
    ):
        raise ValueError(UNBOUND)
    _check_weights(compact, sample, pos)
    _check_offset_or_exposure(compact, sample, pos)
    metadata = s_inp.metadata
    if (
        (compact.nobs, metadata.rows, compact.metadata_names)
        != (compact.nobs, sample.base.nobs, tuple(metadata))
        or (cols.cluster is not None and cols.cluster not in compact.metadata_set)
        or any(name not in compact.metadata_set for name in cols.metadata)
        or any(
            not _same_metadata_column(compact, metadata, name)
            for name in compact.metadata_names
        )
    ):
        raise ValueError(UNBOUND)


def _pairs_label_by_label(
    lvls: AnyArray, c0: AnyArray, c1: AnyArray, codes: IndexArray, first: IndexArray
) -> None:
    """The label-by-label route of the interaction check of the reference."""
    c0_list, c1_list = c0.tolist(), c1.tolist()
    first0, first1 = _first_labels(c0), _first_labels(c1)
    for k, r0 in enumerate(first.tolist()):
        lvl = lvls[k]
        rep0 = (
            first0[c0[r0]]
            if first0 is not None
            else next(
                x for x in c0_list if _same_label_value(x, c0[r0], exact_type=False)
            )
        )
        rep1 = (
            first1[c1[r0]]
            if first1 is not None
            else next(
                x for x in c1_list if _same_label_value(x, c1[r0], exact_type=False)
            )
        )
        if (
            not isinstance(lvl, tuple)
            or len(lvl) != 2
            or not _same_label_value(lvl[0], rep0)
            or not _same_label_value(lvl[1], rep1)
        ):
            raise ValueError(UNBOUND)
    for r in range(int(codes.size)):
        lvl_r = lvls[int(codes[r])]
        if not (
            _same_label_value(lvl_r[0], c0[r], exact_type=False)
            and _same_label_value(lvl_r[1], c1[r], exact_type=False)
        ):
            raise ValueError(UNBOUND)


def _working(
    effects: _CompactEffects, name: str, working: _Working
) -> tuple[AnyArray, type | None]:
    """``_column_values`` of the capture's column ``name``, formed once per check.

    ``working`` belongs to one call of :func:`_check_effects_and_partitions`: working
    arrays of the immutable capture tuples, read-only, gathered from and never handed
    on, so that each column is formed once in that call, not once per block.
    """
    found = working.get(name)
    if found is None:
        found = working[name] = _column_values(effects.columns[name])
    return found


def _check_block(
    effects: _CompactEffects,
    term: tuple[str, ...],
    rows: IndexArray,
    b_got: Block,
    working: _Working,
) -> None:
    """One block against the compact effects: the reference's block check.

    The live levels and codes are read afresh (:func:`_label_values`,
    :func:`_label_pairs`); the first row of each code is ``numpy.unique``'s, found
    without a sort for live codes numbered by first occurrence (:func:`first_rows`).
    The capture's plain columns provide working arrays from immutable tuples, formed
    once per check (:func:`_working`).
    """
    codes, levels = b_got.codes, b_got._levels
    pairs = _label_pairs(levels) if len(term) == 2 else None
    lvls, lvl_kind = _label_values(levels) if pairs is None else (pairs[0], None)
    first = first_rows(codes)
    if first is None:
        _, first = np.unique(codes, return_index=True)
    if len(term) == 1:
        col_vals, col_kind = _working(effects, term[0], working)
        col_vals = col_vals[rows]
        if not _same_column_values(col_vals[first], lvls) or not _same_labels(
            col_vals, lvls[codes], plain=col_kind is not None and lvl_kind is not None
        ):
            raise ValueError(UNBOUND)
        return
    c0, kind0 = _working(effects, term[0], working)
    c0 = c0[rows]
    c1, kind1 = _working(effects, term[1], working)
    c1 = c1[rows]
    gathered = (
        _same_pairs(pairs, (c0, kind0), (c1, kind1), codes, first)
        if codes.shape == rows.shape
        else None
    )
    if gathered is not None:
        if not gathered:
            raise ValueError(UNBOUND)
        return
    _pairs_label_by_label(lvls, c0, c1, codes, first)


def _check_effects_and_partitions(compact: CompactCapture, sample: FESample) -> None:
    """The third check: the source and the final partition against the effects."""
    effects, src, grouping = compact.effects, sample.source, sample.grouping
    if (
        effects is None
        or src is None
        or grouping is None
        or (effects.rows, effects.names, effects.names)
        != (compact.nobs, src.names, grouping.names)
    ):
        raise ValueError(UNBOUND)
    if isinstance(src, Derived) or isinstance(grouping, Derived):
        if not (
            isinstance(src, Derived)
            and isinstance(grouping, Derived)
            and src.source is grouping.source
        ):
            raise ValueError(UNBOUND)
        s = src.source
        observed: tuple[tuple[tuple[str, ...], ...], int, Set[str]] = (
            s.terms,
            s.nobs,
            set(s.columns),
        )
        if (
            observed != (effects.terms, effects.rows, effects.column_set)
            or not same_bits(s.rows, src.positions)
            or any(
                not _same_compact_column(effects.columns[name], s.columns[name])
                for name in effects.column_names
            )
        ):
            raise ValueError(UNBOUND)
    working: _Working = {}
    for p_got in (src, grouping):
        for term, b_got in zip(effects.terms, p_got.blocks, strict=True):
            if b_got.codes.shape != p_got.positions.shape:
                raise ValueError(UNBOUND)
            _check_block(effects, term, p_got.positions, b_got, working)


def check_compact_sample(compact: CompactCapture, sample: FESample) -> None:
    """Verify that the live ``sample`` is bound to the capture's compact values.

    The three checks of :func:`~powermean.results.absorbed_sample_binding.
    check_parts_match_sample` in their order; ``sample`` is read afresh on every call
    and nothing is kept between calls (module docstring).
    """
    _check_base_and_masks(compact, sample)
    _check_inputs_and_metadata(compact, sample)
    _check_effects_and_partitions(compact, sample)
