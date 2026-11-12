"""Sample integrity and prepared-frame binding checks before retained detachment."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd

from powermean._prepare.compose import PreparedFrame
from powermean._prepare.inputs import Columns
from powermean.data.accounting import RETAINED, ZERO_OUTCOME, ZERO_WEIGHT
from powermean.data.raw import AnyArray, Column, _OwnedMapping
from powermean.data.sample import SampleBase
from powermean.data.sample_fe import EffectColumns
from powermean.data.sample_fe_source import Derived, same_bits
from powermean.data.sample_fe_state import FESample, same_missing
from powermean.fe.groups import Block, Grouping
from powermean.model.order import IndexArray
from powermean.results.absorbed_partitions import validate_partition_lineage
from powermean.results.retained_input_values import (
    _is_missing_scalar,
    _label_values,
    _plain_kind,
    _same_column_values,
)
from powermean.results.retained_index_values import (
    numeric_index_snapshots,
    same_numeric_index_values,
)
from powermean.results.retained_labels import _one_exact_type, _same_design_spec

UNBOUND = "the absorbed handoff graph is not bound to a single sample and candidate"
# The exact types of the labels whose first equal label a dictionary finds
# (:func:`_first_labels`); a subclass, such as NumPy's ``str_``, is not one of them.
_PLAIN_LABELS = frozenset({bool, int, float, str})
# Interaction levels, their first and second labels, and the kinds of those labels.
type _Pairs = tuple[AnyArray, AnyArray, AnyArray, type, type]

__all__ = [
    "UNBOUND",
    "_check_prepared_frame_and_provenance",
    "_check_prepared_matches_sample",
    "_check_sample_integrity",
    "_check_sample_slopes",
    "_same_array_values",
    "_same_design_spec",
    "_same_index_values",
    "check_parts_match_sample",
    "check_prepared_matches_sample",
]


def _same_label_value(a: object, b: object, *, exact_type: bool = True) -> bool:
    """Return True if two scalar label values match (with optional exact type check)."""
    if _is_missing_scalar(a) and _is_missing_scalar(b):
        return True
    return (not exact_type or type(a) is type(b)) and bool(a == b)


def _first_labels(values: np.ndarray) -> dict[object, object] | None:
    """Map each label of ``values`` to its first equal label, or ``None`` to scan.

    One pass instead of the scan ``next(x for x in values.tolist() if
    _same_label_value(x, values[r], exact_type=False))`` for every row ``r``, and only
    where both agree: an object array of labels of the exact types ``_PLAIN_LABELS``,
    floats finite.  None of these is missing for ``_is_missing_scalar``, so the scan's
    test is ``==``: exact (``2**53 + 1`` is not ``2.0**53``), an equivalence (``1``,
    ``1.0`` and ``True`` are one label) under which equal labels hash equally.  Filled
    in row order, the dictionary keeps the first label of each class, with its type;
    a lookup matches a key only by identity or ``==``, so equal hashes alone never
    match.  Any other label (missing, infinite, a tuple, a subclass, a caller's class),
    a typed array, whose rows are NumPy scalars, or no row gives ``None``: the scan.
    """
    if values.ndim != 1 or values.dtype.kind != "O":
        return None
    labels = values.tolist()
    if type(labels) is not list or not labels:
        return None
    first: dict[object, object] = {}
    for label in labels:
        kind = type(label)
        if not any(kind is t for t in _PLAIN_LABELS) or (
            kind is float and not math.isfinite(label)
        ):
            return None
        first.setdefault(label, label)
    return first


def _same_index_values(left: pd.Index[Any], right: pd.Index[Any]) -> bool:
    """Return True if two pandas Index objects match on values, dtype, and names."""
    numeric = numeric_index_snapshots(left, right)
    l_arr, r_arr = (
        (np.asarray(left, dtype=object), np.asarray(right, dtype=object))
        if numeric is None
        else numeric
    )
    if (
        (type(left), len(left), left.dtype, len(left.names))
        != (type(right), len(right), right.dtype, len(right.names))
        or not _same_label_value(left.name, right.name)
        or not all(
            _same_label_value(a, b)
            for a, b in zip(left.names, right.names, strict=True)
        )
        or not (
            _same_column_values(l_arr, r_arr)
            if numeric is None
            else same_numeric_index_values(l_arr, r_arr)
        )
    ):
        return False
    if isinstance(left, pd.MultiIndex) and isinstance(right, pd.MultiIndex):
        return left.nlevels == right.nlevels and all(
            _same_index_values(left.levels[i], right.levels[i])
            and np.array_equal(left.codes[i], right.codes[i])
            for i in range(left.nlevels)
        )
    return True


def _same_array_values(left: object, right: object) -> bool:
    """Return True if two arrays have identical dtype, shape, and exact bytes."""
    return (
        isinstance(left, np.ndarray)
        and isinstance(right, np.ndarray)
        and (left.dtype, left.shape) == (right.dtype, right.shape)
        and left.tobytes() == right.tobytes()
    )


def _check_sample_slopes(sample: FESample) -> None:
    """Validate slope design, scales, and direct report map on ``sample``."""
    count, k = sample.count, int(sample.active.size)
    if len(sample.active_names) != k:
        raise ValueError(UNBOUND)
    if k == 0:
        if sample.design is not None or sample.order_scale is not None:
            raise ValueError(UNBOUND)
        return
    d, o = sample.design, sample.order_scale
    if (
        d is None
        or o is None
        or (d.matrix.shape, d.scale.shape) != ((count, k), (k,))
        or (d.column_names, o.dimension) != (sample.active_names, k)
        or not same_bits(o.scale, d.scale)
    ):
        raise ValueError(UNBOUND)
    active_mat = sample.data.design[:, sample.active]
    largest = np.max(np.abs(active_mat), axis=0)
    _, exp = np.frexp(largest)
    exp_scale = np.asarray(np.ldexp(1.0, exp - 1), dtype=np.float64)
    exp_scale[largest == 0] = 1.0
    with np.errstate(under="ignore"):
        exp_mat = np.asarray(active_mat / exp_scale, dtype=np.float64)
    if not (same_bits(d.scale, exp_scale) and same_bits(d.matrix, exp_mat)):
        raise ValueError(UNBOUND)


def _check_sample_problem_and_data(sample: FESample) -> None:
    """Verify ``sample.data`` and ``sample.problem`` against ``sample.base.inputs``."""
    data, pos, s_inp = sample.data, sample.positions, sample.base.inputs
    if (
        (data.column_names, s_inp.metadata.rows, sample.metadata.rows)
        != (s_inp.column_names, sample.base.nobs, sample.count)
        or not same_bits(s_inp.y[pos], data.y)
        or not same_bits(s_inp.design[pos], data.design)
        or data.row_labels != tuple(int(p) for p in pos.tolist())
    ):
        raise ValueError(UNBOUND)
    if sample.omega == 0.0:
        exp_resp = np.asarray(np.log(data.y), dtype=np.float64)
        exp_off = data.offset
    else:
        with np.errstate(over="ignore", under="ignore"):
            exp_resp = np.asarray(data.y**sample.omega, dtype=np.float64)
            exp_resp[data.y == 0] = 0.0
            exp_off = np.asarray(sample.omega * data.offset, dtype=np.float64)
    if not (
        same_bits(sample.problem.response, exp_resp)
        and same_bits(sample.problem.offset, exp_off)
    ):
        raise ValueError(UNBOUND)


def _check_sample_integrity(sample: object) -> FESample:
    """Validate internal structural and byte integrity of an :class:`FESample`."""
    if type(sample) is not FESample:
        raise TypeError("an absorbed handoff requires an FESample")
    grouping, src = sample.grouping, sample.source
    if grouping is None or src is None or not grouping.blocks:
        raise ValueError("an absorbed sample must hold fixed-effect blocks")
    nobs, count = sample.ledger.nobs, sample.count
    if (
        count <= 0
        or nobs < count
        or sample.omega != sample.problem.omega
        or not sample._typed()
    ):
        raise ValueError(UNBOUND)
    ledger, data, pos, p = sample.ledger, sample.data, sample.positions, sample.problem
    kept = [j for j, reason in enumerate(sample.reasons) if reason is None]
    history = bool((ledger.codes == ZERO_OUTCOME).any())
    eligible = (ledger.codes == RETAINED) | (ledger.codes > ZERO_WEIGHT)
    shapes = (pos.shape, p.response.shape, p.offset.shape, data.weight_values.shape)
    counts = (len(src.blocks), ledger.nobs, len(sample.reasons), kept)
    exp_counts = (
        len(grouping.blocks),
        sample.base.nobs,
        data.nparams,
        sample.active.tolist(),
    )
    if (
        shapes != ((count,), (count,), (count,), (count,))
        # Derived lineage may reuse ordinals; bind its current final row order
        # to the ledger before any certificate/basis or interface consumption.
        or not _same_array_values(grouping.positions, pos)
        or counts != exp_counts
        or sample.conditioned != history
        or (history and not sample.positive_only)
        or not same_missing(ledger.missing, sample.base.missing)
        or not same_bits(src.positions, np.flatnonzero(eligible))
        or int(ledger.blocks.max()) >= len(src.blocks)
    ):
        raise ValueError(UNBOUND)
    _check_sample_problem_and_data(sample)
    validate_partition_lineage(src, grouping, nobs)
    _check_sample_slopes(sample)
    return sample


def _check_prepared_base_and_masks(prepared: PreparedFrame, sample: FESample) -> None:
    """Verify full-frame row counts, subset/missing/weight masks, and codes."""
    _check_base_and_masks(
        prepared.base, prepared.effects, int(prepared._labels.size), sample
    )


def _check_base_and_masks(
    base: SampleBase, effects: EffectColumns | None, labels_size: int, sample: FESample
) -> None:
    """The check of :func:`_check_prepared_base_and_masks` on the frame's parts."""
    grouping, src = sample.grouping, sample.source
    sub_ok = (base.subset is None and sample.base.subset is None) or (
        base.subset is not None
        and sample.base.subset is not None
        and same_bits(base.subset, sample.base.subset)
    )
    if (
        (base.nobs, base.nobs, base.nobs)
        != (sample.ledger.nobs, sample.base.nobs, labels_size)
        or None in (effects, grouping, src)
        or not sub_ok
        or not same_bits(base.codes, sample.base.codes)
        or not same_bits(base.complete, sample.base.complete)
        or not same_bits(base.negative_weight, sample.base.negative_weight)
        or not same_bits(base.zero_weight, sample.base.zero_weight)
        or not same_missing(base.missing, sample.base.missing)
        or not same_missing(base.missing, sample.ledger.missing)
        or not bool(base.complete[sample.positions].all())
        or not np.array_equal(
            sample.ledger.codes[~base.complete], base.codes[~base.complete]
        )
    ):
        raise ValueError(UNBOUND)


def _check_prepared_inputs_and_metadata(
    prepared: PreparedFrame, sample: FESample
) -> None:
    """Verify outcome, design, weights, symmetric offset/exposure, and metadata."""
    _check_inputs_and_metadata(prepared.base, prepared.columns, sample)


def _check_inputs_and_metadata(
    base: SampleBase, cols: Columns, sample: FESample
) -> None:
    """The check of :func:`_check_prepared_inputs_and_metadata` on the frame's parts."""
    inp, s_inp = base.inputs, sample.base.inputs
    pos = sample.positions
    opt_cols = (cols.weights is None, cols.offset is None, cols.exposure is None)
    opt_inp = (inp.weights is None, inp.offset is None, inp.exposure is None)
    if not (
        same_bits(inp.y, s_inp.y)
        and same_bits(inp.design, s_inp.design)
        and inp.column_names == s_inp.column_names == sample.data.column_names
        and same_bits(inp.y[pos], sample.data.y)
        and same_bits(inp.design[pos], sample.data.design)
        and opt_cols == opt_inp
    ):
        raise ValueError(UNBOUND)
    w_in, w_s, w_d = inp.weights, s_inp.weights, sample.data.weights
    if w_in is None:
        if w_d is not None or w_s is not None:
            raise ValueError(UNBOUND)
    elif (
        w_d is None
        or w_s is None
        or not (cols.weight_kind == w_in.kind == w_d.kind == w_s.kind)
        or not _same_column_values(w_in.values, w_s.values)
        or not same_bits(np.asarray(w_in.values[pos], dtype=np.float64), w_d.values)
    ):
        raise ValueError(UNBOUND)
    off_in, off_s = inp.offset, s_inp.offset
    exp_in, exp_s, exp_d = inp.exposure, s_inp.exposure, sample.data.exposure
    if (off_in is None, exp_in is None, exp_in is None) != (
        off_s is None,
        exp_s is None,
        exp_d is None,
    ):
        raise ValueError(UNBOUND)
    if off_in is not None:
        if off_s is None or not (
            same_bits(off_in, off_s) and same_bits(off_in[pos], sample.data.offset)
        ):
            raise ValueError(UNBOUND)
    elif exp_in is not None:
        if (
            exp_s is None
            or exp_d is None
            or not (same_bits(exp_in, exp_s) and same_bits(exp_in[pos], exp_d))
        ):
            raise ValueError(UNBOUND)
    elif not same_bits(np.zeros(pos.size, dtype=np.float64), sample.data.offset):
        raise ValueError(UNBOUND)
    if (
        (inp.metadata.rows, s_inp.metadata.rows, tuple(inp.metadata))
        != (base.nobs, sample.base.nobs, tuple(s_inp.metadata))
        or (cols.cluster is not None and cols.cluster not in inp.metadata)
        or any(name not in inp.metadata for name in cols.metadata)
        or any(
            not _same_column_values(inp.metadata[k], s_inp.metadata[k])
            for k in inp.metadata
        )
    ):
        raise ValueError(UNBOUND)


def _check_prepared_frame_and_provenance(prepared: PreparedFrame) -> None:
    """Verify internal frame, label, spec, and dependence provenance consistency."""
    f, d, base = prepared._frame, prepared._frame.design, prepared.base
    m_grps = base.missing.groups
    if (
        type(prepared.subset_missing) is not int
        or prepared.subset_missing < 0
        or not _same_index_values(prepared._labels, d.labels)
        or not _same_design_spec(d.spec, d.spec)
        or tuple((str(t), d.columns[sl]) for t, sl in d.spec.term_slices.items())
        != d.terms
        or (len(f.outcome), base.inputs.metadata.rows, d.matrix.shape, d.columns)
        != (base.nobs, base.nobs, base.inputs.design.shape, base.inputs.column_names)
        or not same_bits(f.outcome_invalid, m_grps["outcome"])
        or not same_bits(
            d.referenced_invalid | d.evaluated_invalid, m_grps["regressors"]
        )
        or bool(f.parts.fixed_effects) != (prepared.effects is not None)
    ):
        raise ValueError(UNBOUND)
    dep = prepared.dependence
    if dep is not None and (
        not same_bits(dep.used_missing_union, m_grps["clusters"])
        or any(
            c.name != k
            or k not in base.inputs.metadata
            or c.missing_mask.shape != (base.nobs,)
            or not _same_column_values(c.values, base.inputs.metadata[k])
            for k, c in dep.columns.items()
        )
    ):
        raise ValueError(UNBOUND)


def _label_pairs(levels: Column) -> _Pairs | None:
    """Return interaction levels split into their two labels, or ``None``.

    Levels that are exact pairs, with labels of one plain kind at each place
    (:func:`_plain_kind`), are their own copies, as plain labels are
    (:func:`_label_values`): the levels, their first and second labels, read from
    storage, and the two kinds.  Any other levels give ``None``.
    """
    if type(levels) is not Column:
        return None
    stored = levels._stored
    if type(stored) is not tuple or _one_exact_type(stored, (tuple,)) is None:
        return None
    found = cast("tuple[tuple[Any, ...], ...]", stored)
    if set(map(len, found)) != {2}:
        return None
    firsts, seconds = zip(*found, strict=True)
    kind0, kind1 = _plain_kind(firsts), _plain_kind(seconds)
    if kind0 is None or kind1 is None:
        return None
    n = len(found)
    return (
        np.fromiter(found, dtype=object, count=n),
        np.fromiter(firsts, dtype=object, count=n),
        np.fromiter(seconds, dtype=object, count=n),
        kind0,
        kind1,
    )


def _codes_within(codes: IndexArray, n_levels: int) -> bool:
    """Return True if ``codes`` is one vector of integer codes in ``[0, n_levels)``."""
    if type(codes) is not np.ndarray or codes.ndim != 1 or codes.dtype.kind not in "iu":
        return False
    return codes.size == 0 or (int(codes.min()) >= 0 and int(codes.max()) < n_levels)


def _same_labels(labels: AnyArray, expected: AnyArray, *, plain: bool) -> bool:
    """Return True if each label equals its expected label, missing labels equal.

    Plain labels (:func:`_plain_kind`) are never missing and their ``==`` runs no caller
    code, so one comparison of the two lists decides as the comparisons label by label.
    """
    if (
        plain
        and type(labels) is np.ndarray
        and type(expected) is np.ndarray
        and labels.ndim == 1
        and labels.shape == expected.shape
    ):
        return bool(labels.tolist() == expected.tolist())
    return all(
        _same_label_value(a, b, exact_type=False)
        for a, b in zip(labels.tolist(), expected.tolist(), strict=True)
    )


def _same_pairs(
    pairs: _Pairs | None,
    left: tuple[AnyArray, type | None],
    right: tuple[AnyArray, type | None],
    codes: IndexArray,
    first: IndexArray,
) -> bool | None:
    """Return whether plain interaction labels match their levels, or ``None``.

    With pair levels and row labels of the same plain kind at each place and codes of
    the levels, the scalar comparisons of
    :func:`_check_block_against_effect_columns` reduce to two gathers per place: the
    first label equal to a label of one plain kind has its type and is equal to it, so
    each level must equal the labels of its first row, and each row the labels of its
    level.  ``None`` for any other labels, levels or codes: compare them label by label.
    """
    if pairs is None:
        return None
    lvls, firsts, seconds, kind0, kind1 = pairs
    (c0, got0), (c1, got1) = left, right
    if not (
        got0 is kind0
        and got1 is kind1
        and type(c0) is np.ndarray
        and type(c1) is np.ndarray
        and _codes_within(codes, len(lvls))
        and _codes_within(first, len(c0))
        and len(first) <= len(lvls)
        and c0.ndim == 1
        and c0.shape == c1.shape == codes.shape
    ):
        return None
    heads = len(first)
    return bool(
        firsts[:heads].tolist() == c0[first].tolist()
        and seconds[:heads].tolist() == c1[first].tolist()
        and firsts[codes].tolist() == c0.tolist()
        and seconds[codes].tolist() == c1.tolist()
    )


def _check_block_against_effect_columns(
    prepared: PreparedFrame, term: tuple[str, ...], rows: np.ndarray, b_got: Block
) -> None:
    """Verify encoded block levels and codes against ``prepared.effects``.

    Labels of one plain kind (:func:`_label_values`, :func:`_label_pairs`) are compared
    as whole gathered lists (:func:`_same_labels`, :func:`_same_pairs`), which decide
    as the comparisons label by label do; any other labels take those comparisons.
    """
    effects = prepared.effects
    assert effects is not None
    _check_block_against_effects(effects, term, rows, b_got)


def _check_block_against_effects(
    effects: EffectColumns, term: tuple[str, ...], rows: np.ndarray, b_got: Block
) -> None:
    """The check of :func:`_check_block_against_effect_columns` on ``effects``."""
    codes, levels = b_got.codes, b_got._levels
    pairs = _label_pairs(levels) if len(term) == 2 else None
    lvls, lvl_kind = _label_values(levels) if pairs is None else (pairs[0], None)
    _, first = np.unique(codes, return_index=True)
    if len(term) == 1:
        col_vals, col_kind = _label_values(effects.columns[term[0]])
        col_vals = col_vals[rows]
        if not _same_column_values(col_vals[first], lvls) or not _same_labels(
            col_vals, lvls[codes], plain=col_kind is not None and lvl_kind is not None
        ):
            raise ValueError(UNBOUND)
        return
    c0, kind0 = _label_values(effects.columns[term[0]])
    c0 = c0[rows]
    c1, kind1 = _label_values(effects.columns[term[1]])
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
    for r in range(int(rows.size)):
        lvl_r = lvls[int(codes[r])]
        if not (
            _same_label_value(lvl_r[0], c0[r], exact_type=False)
            and _same_label_value(lvl_r[1], c1[r], exact_type=False)
        ):
            raise ValueError(UNBOUND)


def _plain_effect_blocks(
    effects: EffectColumns, src: Grouping, grouping: Grouping, nobs: int
) -> bool:
    """Recognize a callback-free segment, without admitting any of its relations.

    Inspect only exact carried records, tuples, dictionaries and integer arrays.
    A foreign participant selects the unchanged ordered checks. Plain labels are
    freshly recognized on this call, never inferred from a previous admission.
    Source levels must be distinct so that label equality implies code equality.
    Final levels need not be distinct: the original checker can accept a forged
    split of one label into two codes, and this optimization must not tighten it.
    """
    if (
        type(effects) is not EffectColumns
        or type(nobs) is not int
        or nobs <= 0
        or type(effects.terms) is not tuple
        or not effects.terms
        or type(effects.columns) is not _OwnedMapping
    ):
        return False
    items = cast("_OwnedMapping[Column]", effects.columns)._items
    if type(items) is not dict or any(type(name) is not str for name in items):
        return False
    terms = effects.terms
    if any(
        type(term) is not tuple
        or len(term) not in (1, 2)
        or any(type(name) is not str for name in term)
        for term in terms
    ):
        return False
    names = {name for term in terms for name in term}
    if set(items) != names:
        return False
    kinds: dict[str, type] = {}
    for name, column in items.items():
        if type(column) is not Column or type(column._stored) is not tuple:
            return False
        if len(column._stored) != nobs:
            return False
        kind = _plain_kind(column._stored)
        if kind is None:
            return False
        kinds[name] = kind
    for partition in (src, grouping):
        if (
            (type(partition) is not Grouping and type(partition) is not Derived)
            or type(partition.blocks) is not tuple
            or len(partition.blocks) != len(terms)
            or not _codes_within(partition.positions, nobs)
            or not partition.positions.size
        ):
            return False
        for term, block in zip(terms, partition.blocks, strict=True):
            if (
                type(block) is not Block
                or type(block.name) is not str
                or type(block._levels) is not Column
                or type(block._levels._stored) is not tuple
            ):
                return False
            levels = cast("tuple[object, ...]", block._levels._stored)
            if not levels or not _codes_within(block.codes, len(levels)):
                return False
            if block.codes.shape != partition.positions.shape:
                return False
            if len(term) == 1:
                if _plain_kind(levels) is not kinds[term[0]]:
                    return False
            else:
                if _one_exact_type(levels, (tuple,)) is None:
                    return False
                pairs = cast("tuple[tuple[object, ...], ...]", levels)
                if any(len(pair) != 2 for pair in pairs):
                    return False
                left, right = zip(*pairs, strict=True)
                if (
                    _plain_kind(left) is not kinds[term[0]]
                    or _plain_kind(right) is not kinds[term[1]]
                ):
                    return False
            # Exact plain scalar/pair hashes and comparisons cannot call user code.
            if partition is src and len(set(levels)) != len(levels):
                return False
            codes = block.codes.astype(np.intp, copy=False)
            if not np.all(np.bincount(codes, minlength=len(levels)) > 0):
                return False
    return True


def _check_plain_effect_restriction(
    effects: EffectColumns,
    nobs: int,
    sample: FESample,
    src: Grouping,
    grouping: Grouping,
) -> bool:
    """Check source labels and the final restriction, or leave the original route.

    This replaces only the callback-free source/final segment of one live check.
    Source labels are still checked against current effect columns. Every final row
    is then checked against current source codes, and each final level against the
    corresponding source label. Distinct source labels make these exactly the same
    final-row label relations, including the first-representative checks.

    Stored Derived ordinals are not evidence here: the row and level maps are made
    from current positions/codes. Existing entry and lineage checks remain in place.
    No map or recognized label survives this call. Foreign/malformed inputs retain
    the original checks, callback ordering and errors rather than a new refusal.
    """
    if (
        type(sample) is not FESample
        or sample.source is not src
        or sample.grouping is not grouping
    ):
        return False
    if not _plain_effect_blocks(effects, src, grouping, nobs):
        return False
    place = np.full(nobs, -1, dtype=np.intp)
    place[src.positions] = np.arange(src.positions.size, dtype=np.intp)
    at = place[grouping.positions]
    if (at < 0).any():
        return False
    del place
    # Keep source-before-final, and formula order within each partition. There is
    # no caller callback anywhere in this recognized segment between these checks.
    for term, block in zip(effects.terms, src.blocks, strict=True):
        _check_block_against_effects(effects, term, src.positions, block)
    for block, origin in zip(grouping.blocks, src.blocks, strict=True):
        levels = block._levels._stored
        assert type(levels) is tuple
        codes = block.codes.astype(np.intp, copy=False)
        first = np.full(len(levels), codes.size, dtype=np.intp)
        np.minimum.at(first, codes, np.arange(codes.size, dtype=np.intp))
        source_codes = origin.codes[at]
        ordinal = source_codes[first]
        if not np.array_equal(ordinal[codes], source_codes):
            raise ValueError(UNBOUND)
        source_levels = origin._levels._stored
        assert type(source_levels) is tuple
        expected = tuple(source_levels[int(code)] for code in ordinal)
        if levels != expected:
            raise ValueError(UNBOUND)
    return True


def _check_prepared_effects_and_partitions(
    prepared: PreparedFrame, sample: FESample
) -> None:
    """Verify that sample source and grouping match ``prepared.effects``."""
    _check_effects_and_partitions(prepared.effects, prepared.base.nobs, sample)


def _check_effects_and_partitions(
    effects: EffectColumns | None, nobs: int, sample: FESample
) -> None:
    """The check of :func:`_check_prepared_effects_and_partitions` on the parts."""
    src, grouping = sample.source, sample.grouping
    if (
        effects is None
        or src is None
        or grouping is None
        or (effects.rows, effects.names, effects.names)
        != (nobs, src.names, grouping.names)
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
        if (
            (s.terms, s.nobs, set(s.columns))
            != (effects.terms, effects.rows, set(effects.columns))
            or not same_bits(s.rows, src.positions)
            or any(
                s.columns[k] is not ecol
                and not _same_column_values(s.columns[k].values, ecol.values)
                for k, ecol in effects.columns.items()
            )
        ):
            raise ValueError(UNBOUND)
    if _check_plain_effect_restriction(effects, nobs, sample, src, grouping):
        return
    for p_got in (src, grouping):
        for term, b_got in zip(effects.terms, p_got.blocks, strict=True):
            if b_got.codes.shape != p_got.positions.shape:
                raise ValueError(UNBOUND)
            _check_block_against_effects(effects, term, p_got.positions, b_got)


def check_prepared_matches_sample(prepared: PreparedFrame, sample: FESample) -> None:
    """Verify that ``sample`` is bound to ``prepared`` at ``sample.positions``."""
    if type(prepared) is not PreparedFrame:
        raise TypeError("prepared must be a concrete PreparedFrame")
    _check_prepared_base_and_masks(prepared, sample)
    _check_prepared_frame_and_provenance(prepared)
    _check_prepared_inputs_and_metadata(prepared, sample)
    _check_prepared_effects_and_partitions(prepared, sample)


def check_parts_match_sample(
    *,
    base: SampleBase,
    effects: EffectColumns | None,
    columns: Columns,
    labels_size: int,
    sample: FESample,
) -> None:
    """Verify that a live ``sample`` is bound to the parts of a frame.

    The three sample-versus-frame checks of :func:`check_prepared_matches_sample`, in
    their order, on the parts that the owning context materialized once from an
    immutable primitive capture (:mod:`powermean.results.primitive_parts`); the frame's
    own internal consistency (:func:`_check_prepared_frame_and_provenance`) was verified
    on the capture's exact materialization when the capture was formed and cannot have
    changed since.  ``sample`` is read afresh on every call.
    """
    _check_base_and_masks(base, effects, labels_size, sample)
    _check_inputs_and_metadata(base, columns, sample)
    _check_effects_and_partitions(effects, base.nobs, sample)


_check_prepared_matches_sample = check_prepared_matches_sample
