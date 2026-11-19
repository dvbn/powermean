"""The existing exact canonical witness map, shared by nonlinear organizations."""

from __future__ import annotations

import sys
from fractions import Fraction
from types import FunctionType, ModuleType
from typing import TYPE_CHECKING, cast

import numpy as np

from powermean.fe.certificate_basis import (
    _FIT_PREPARED_ISSUER,
    _GENERATED_RETAINED,
    _OPERATIONAL_PREPARED_ISSUER,
    CertificateBasis,
    _construction_code,
    _fit_generated,
    _operational_data,
    _packed_original,
    _produced_retained_action,
    _reporting_elimination,
    _witness_proof,
    certificate_basis,
    embed,
)
from powermean.fe.certificate_witness import (
    FactoredRows,
    HeadRecipe,
    HeadRows,
    OperationalBasis,
    PackedHeads,
    SparseValues,
    WitnessBlock,
    WitnessContext,
    WitnessProof,
)
from powermean.fe.reductions import immutable_array
from powermean.model.variance import FloatArray

if TYPE_CHECKING:
    from powermean.fe.certificate_witness import RetainedRows, Witness, _IndexBuffer
    from powermean.fe.connectivity import Components
    from powermean.fe.groups import Grouping
    from powermean.model.order import IndexArray

_INDEX_SIZE = np.dtype(np.intp).itemsize


def _immutable_indices(values: object, stored: bytes) -> bool:
    """Bind the complete current array header to the actual proof-owned bytes."""
    # NumPy's base stub omits buffer objects such as these immutable bytes.
    return (
        type(values) is np.ndarray
        and values.dtype == np.intp
        and values.ndim == 1
        and values.flags.c_contiguous
        and not values.flags.writeable
        and cast("object", values.base) is stored
        and len(stored) == values.nbytes
        and values.strides == (values.itemsize,)
    )


def _operational_context(basis: CertificateBasis) -> OperationalBasis | None:
    """Borrow real producer/live state, never admit a data-only public copy."""
    try:
        # Do not introduce optional guards/getters on the original generic path.
        # The captured built-in slot reads the exact record's dictionary without
        # invoking a changed instance getter or a replacement class descriptor.
        if (
            type(basis) is not _OPENING_CLASSES[0]
            or _OPERATIONAL_FIELDS is not _OPENING_GENERATORS[-1]
        ):
            return None
        raw = _OPERATIONAL_FIELDS.__get__(basis, _OPENING_CLASSES[0])
        data = raw.get("_operational")
        if (
            type(data) is not _OPENING_CLASSES[-1]
            or raw.get("witnesses") is not data
            or raw.get("echelon") is not data
        ):
            return None
        if (
            not _opening_original()
            or not _fit_generated()
            or (operational := _operational_data(basis)) is not data
        ):
            return None
        reader = raw.get(_GENERATED_RETAINED)
        if type(reader) is FunctionType and (
            reader.__code__ is _construction_code(_FIT_PREPARED_ISSUER)
            or reader.__code__ is _construction_code(_OPERATIONAL_PREPARED_ISSUER)
        ):
            scope, _, _ = reader()
            carried = _produced_retained_action(basis, scope[2]())
            if carried is not None and carried[0] is data:
                return operational
        # The existing explicit NumericExecution rebind owns the second live
        # header view. A public data copy has neither construction nor live scope.
        module = sys.modules.get("powermean.interface.numeric_execution")
        if type(module) is ModuleType:
            namespace = vars(module)
            consume = namespace.get("_direct_retained_action")
            if (
                consume is namespace.get("_DIRECT_RETAINED_ACTION")
                and callable(consume)
                and consume(basis, None) is data
            ):
                return operational
    except MemoryError:
        pass  # Only optional borrowing, never the required ordered action.
    return None


_OPERATIONAL_CONTEXT = _operational_context


def _opening_context(
    basis: CertificateBasis,
) -> WitnessContext | OperationalBasis | None:
    """Admit only complete exact factored values and their current numeric headers."""
    operational = (
        _operational_context(basis)
        if _operational_context is _OPERATIONAL_CONTEXT
        else None
    )
    if operational is not None:
        return operational
    proof = _witness_proof(basis)
    if (
        proof is None
        or not _fit_generated()
        or type(proof.blocks) is not tuple
        or not proof.blocks
        or any(type(block) is not WitnessBlock for block in proof.blocks)
    ):
        return None
    context = proof.blocks[0].context
    if type(context) is not WitnessContext or type(context.factors) is not FactoredRows:
        return None
    factors = context.factors
    packed = type(factors.heads) is PackedHeads
    if (
        (not packed and type(factors.heads) is not tuple)
        or type(factors.tails) is not tuple
        or type(factors.tail_columns) is not tuple
        or type(factors.tail_positions) is not tuple
        or type(context.dependent_columns) is not tuple
        or type(context.dependent_values) is not tuple
    ):
        return None
    if any(type(c) is not int for c in context.dependent_columns) or any(
        type(c) is not int for c in factors.tail_positions
    ):
        return None
    raw_basis = vars(basis)
    if (
        type(raw_basis.get("positions")) is not tuple
        or type(raw_basis.get("columns")) is not tuple
    ):
        return None
    if (
        type(context.columns) is not tuple
        or type(context.positions) is not tuple
        or type(context.components) is not tuple
        or any(
            type(raw) is not bytes
            for group in (context.columns, context.positions, context.components)
            for raw in group
        )
    ):
        return None
    if not (
        len(proof.blocks)
        == len(basis.positions)
        == len(basis.columns)
        == len(context.positions)
        == len(context.columns)
        == len(context.components)
    ):
        return None
    start, references = 0, 0
    for k, block in enumerate(proof.blocks):
        if (
            type(block) is not WitnessBlock
            or block.context is not context
            or type(block.levels) is not tuple
            or type(block.values) is not tuple
            or type(block.size) is not int
            or type(block.start) is not int
            or type(block.block) is not int
            or any(type(level) is not int for level in block.levels)
            or block.block != k
            or block.start != start
            or not _immutable_indices(basis.positions[k], context.positions[k])
            or not _immutable_indices(basis.columns[k], context.columns[k])
        ):
            return None
        place, columns = basis.positions[k], basis.columns[k]
        if (
            block.size != place.size
            or columns.shape != place.shape
            or len(context.components[k]) != place.nbytes
            or block.levels != tuple(np.flatnonzero(place < 0).tolist())
            or len(block.values) != len(block.levels)
        ):
            return None
        for level, value in zip(block.levels, block.values, strict=True):
            column = int(columns[level])
            if (column >= 0 and (type(value) is not int or value != column)) or (
                column < 0 and value is not None
            ):
                return None
            references += int(column < 0)
        kept = place[place >= 0]
        if not np.array_equal(kept, np.arange(start, start + kept.size, dtype=np.intp)):
            return None
        start += kept.size
    first = proof.blocks[0]
    count = sum(block.size for block in proof.blocks) - references
    if (
        first.levels
        or (factors.heads.count if packed else len(factors.heads)) != first.size
    ):
        return None
    selected = basis.selected
    if (
        type(selected) is not np.ndarray
        or type(cast("object", selected.base)) is not bytes
        or not _immutable_indices(selected, cast("bytes", selected.base))
        or selected.size != start
        or context.dependent_values
        or not (
            len(factors.tails)
            == len(factors.tail_columns)
            == len(factors.tail_positions)
        )
    ):
        return None
    pivots: list[int] = []
    if packed:
        if not _checked_packed_heads(
            cast("PackedHeads", factors.heads), first.size, count, len(proof.blocks)
        ):
            return None
    else:
        for head in factors.heads:
            if (
                type(head) is not HeadRecipe
                or type(head.pivot) is not int
                or type(head.anchor) is not tuple
                or len(head.anchor) >= len(proof.blocks)
                or any(
                    type(c) is not int or not first.size <= c < count
                    for c in head.anchor
                )
                or len(set(head.anchor)) != len(head.anchor)
            ):
                return None
            pivots.append(head.pivot)
        if pivots != list(range(first.size)):
            return None
    if (
        type(factors.tail_columns) is not tuple
        or any(
            type(c) is not int or not first.size <= c < count
            for c in factors.tail_columns
        )
        or factors.tail_columns != tuple(sorted(set(factors.tail_columns)))
    ):
        return None
    if packed:
        if (
            any(
                int(column) != index
                for index, column in enumerate(selected[: first.size])
            )
            or tuple(selected[first.size :].tolist()) != factors.tail_columns
        ):
            return None
        kept_columns = set(factors.tail_columns)
        dependent = tuple(c for c in range(first.size, count) if c not in kept_columns)
    else:
        pivots.extend(factors.tail_columns)
        if tuple(selected.tolist()) != tuple(pivots):
            return None
        kept_columns = set(pivots)
        dependent = tuple(c for c in range(count) if c not in kept_columns)
    if context.dependent_columns != dependent:
        return None
    dependent_columns = set(dependent)
    for index, (position, row) in enumerate(
        zip(factors.tail_positions, factors.tails, strict=True)
    ):
        if (
            position != first.size + index
            or type(row) is not SparseValues
            or type(row.positions) is not tuple
            or type(row.numerators) is not tuple
            or type(row.denominators) is not tuple
            or len(row.positions) != len(row.numerators)
            or len(row.positions) != len(row.denominators)
            or any(
                type(c) is not int or c not in dependent_columns for c in row.positions
            )
            or row.positions != tuple(sorted(set(row.positions)))
            or any(type(n) is not int or not n for n in row.numerators)
            or any(type(d) is not int or d <= 0 for d in row.denominators)
        ):
            return None
    return context


def _checked_packed_heads(
    heads: PackedHeads,
    first: int,
    count: int,
    blocks: int,
    _buffer: _IndexBuffer = np.frombuffer,
    _index: type[np.intp] = np.intp,
    _size: int = _INDEX_SIZE,
) -> bool:
    """The existing checked retained fallback, streaming packed values.

    This is not the live fast route. Optional direct preparation/qualification
    failure still gets the prior retained arithmetic, without reconstructing H
    head objects or an H-sized pivot/set map. Every used anchor and storage
    boundary is checked afresh; invalid/foreign headers retain full-action
    refusals. Birth IDs are checked because decoding a head would index them.
    """
    if (
        type(heads.count) is not int
        or heads.count != first
        or type(heads.offsets) is not bytes
        or type(heads.anchors) is not bytes
        or type(heads.births) is not bytes
        or type(heads.versions) is not tuple
        or len(heads.offsets) != (first + 1) * _size
        or len(heads.anchors) % _size
        or len(heads.births) != len(heads.anchors)
    ):
        return False
    offsets = _buffer(heads.offsets, dtype=_index)
    anchors = _buffer(heads.anchors, dtype=_index)
    births = _buffer(heads.births, dtype=_index)
    if int(offsets[0]) != 0 or int(offsets[-1]) != anchors.size:
        return False
    for head in range(first):
        low, high = int(offsets[head]), int(offsets[head + 1])
        if not 0 <= low <= high <= anchors.size or high - low >= blocks:
            return False
        found = set()
        for slot in range(low, high):
            column, version = int(anchors[slot]), int(births[slot])
            if (
                not first <= column < count
                or column in found
                or not -1 <= version < len(heads.versions)
            ):
                return False
            found.add(column)
    return True


def _opening_range(
    context: WitnessContext | OperationalBasis,
    effects: tuple[FloatArray, ...],
    _norm: Fraction | None = None,
) -> bool:
    """Sufficient ALL-coordinate range bound, never a new numerical tolerance.

    L bounds every final tail's l1 norm; C=1+(G-1)L bounds each nonreference
    coefficient. Every reference sums at most n such coefficients, so B=n*C
    bounds EVERY original-level witness entry. With n<2**51, 16*n*M*B below
    binary64 max bounds all rounded ratios, products and ordered partial sums,
    including absolute subnormal-rounding errors (M>=1). See the source-bound
    argument in the P66 handoff. Signaling underflow always keeps the full action.
    """
    if np.geterr()["under"] != "ignore" or len(effects) != len(context.positions):
        return False
    n, maximum = 0, 1.0
    for raw, values in zip(context.positions, effects, strict=True):
        if (
            type(values) is not np.ndarray
            or values.dtype != np.float64
            or values.shape != (len(raw) // np.dtype(np.intp).itemsize,)
            or not values.flags.c_contiguous
            or values.flags.writeable
            or values.strides != (values.itemsize,)
        ):
            return False
        # The existing immutable_array producer reshapes a frombuffer view.
        # Bind its complete immutable storage, not only its immediate base type.
        root: object = values.base
        while type(root) is np.ndarray:
            root = root.base
        if (
            type(root) is not bytes
            or len(root) != values.nbytes
            or not np.isfinite(values).all()
        ):
            return False
        n += values.size
        if values.size:
            maximum = max(maximum, float(np.max(np.abs(values))))
    if not 0 < n < 2**51:
        return False
    assert context.factors is not None
    norm = (
        _norm
        if _norm is not None
        else max(
            (
                sum(
                    (
                        abs(Fraction(v, d))
                        for v, d in zip(row.numerators, row.denominators, strict=True)
                    ),
                    Fraction(0),
                )
                for row in context.factors.tails
            ),
            default=Fraction(1),
        )
    )
    bound = n * (1 + (len(effects) - 1) * max(Fraction(1), norm))
    return 16 * n * Fraction.from_float(maximum) * bound < (2**53 - 1) * 2**971


def retained_coordinates(
    basis: CertificateBasis,
    effects: tuple[FloatArray, ...],
) -> FloatArray | None:
    """Same ordered action on later J, only after skipped predicates are proved."""
    try:
        if type(effects) is not tuple or not _opening_original():
            return None
        context = _opening_context(basis)
        if context is None or not _opening_range(context, effects):
            return None
    except MemoryError:
        # Only optional qualification is contained. Required action errors escape.
        return None
    if TYPE_CHECKING:
        # A successful _opening_range has checked the factored context.
        assert context.factors is not None
    minimum = (
        context.factors.heads.count
        if type(context.factors.heads) is PackedHeads
        or type(context.factors.heads) is HeadRows
        else len(context.factors.heads)
    )
    return _retained_action(basis, effects, context, minimum)


def _retained_action(
    basis: CertificateBasis,
    effects: tuple[FloatArray, ...],
    context: WitnessContext | OperationalBasis,
    minimum: int,
) -> FloatArray:
    """One unchanged ordered retained action for either actual numeric storage."""
    alpha = np.zeros(basis.n_kept - minimum)
    with np.errstate(over="raise", invalid="raise"):
        for block in range(len(effects)):
            place, found = basis.positions[block], effects[block]
            kept = place >= minimum
            alpha[place[kept] - minimum] += found[kept]
            levels = np.flatnonzero((place < 0) & (found != 0)).tolist()
            references = context.references(
                block,
                [level for level in levels if int(basis.columns[block][level]) < 0],
                minimum,
            )
            for level in levels:
                column = int(basis.columns[block][level])
                witness = (
                    references[level]
                    if column < 0
                    else context.dependent(column, minimum)
                )
                for position, numerator, denominator in witness:
                    ratio = numerator / denominator
                    alpha[position - minimum] += found[level] * ratio
    if not np.isfinite(alpha).all():
        raise FloatingPointError("the start coordinates are not representable")
    return immutable_array(alpha)


def _direct_retained_coordinates(
    basis: CertificateBasis,
    effects: tuple[FloatArray, ...],
    proof: WitnessProof | OperationalBasis,
) -> FloatArray | None:
    """Consume only a producer-issued action; the live handoff authenticates it.

    Fresh all-coordinate numerical/storage/range checks stay here. Only the
    immutable structural facts and exact tail norm come from that producer.
    Optional qualification ends before required vector or scalar arithmetic.
    """
    try:
        if (
            type(effects) is not tuple
            or not _opening_original()
            or not _packed_original()
        ):
            return None
        action = proof.retained
        context = (
            proof
            if type(proof) is OperationalBasis
            else cast("WitnessProof", proof).blocks[0].context
        )
        if context is None or not _opening_range(
            context, effects, None if action is None else action.norm
        ):
            return None
        if action is None:
            if type(proof) is not OperationalBasis:
                return None
            minimum = cast("HeadRows", proof.factors.heads).count
            kept = None
        else:
            minimum = action.minimum
            kept = tuple(
                (
                    np.frombuffer(levels, dtype=np.intp),
                    np.frombuffer(places, dtype=np.intp),
                )
                for levels, places in zip(
                    action.kept_levels, action.kept_positions, strict=True
                )
            )
    except MemoryError:
        return None
    if kept is None:
        return _retained_action(basis, effects, context, minimum)
    action = cast("RetainedRows", action)  # Only a retained action supplies kept.
    alpha = np.zeros(basis.n_kept - action.minimum)
    with np.errstate(over="raise", invalid="raise"):
        for block, found in enumerate(effects):
            levels, positions = kept[block]
            alpha[positions] += found[levels]
            for level, witness in zip(
                action.levels[block], action.terms[block], strict=True
            ):
                if found[level] != 0:
                    for position, numerator, denominator in witness:
                        ratio = numerator / denominator
                        alpha[position] += found[level] * ratio
    if not np.isfinite(alpha).all():
        raise FloatingPointError("the start coordinates are not representable")
    return immutable_array(alpha)


_OPENING_CLASSES = (
    CertificateBasis,
    WitnessBlock,
    WitnessContext,
    FactoredRows,
    HeadRecipe,
    SparseValues,
    PackedHeads,
    HeadRows,
    OperationalBasis,
)
_OPERATIONAL_FIELDS = vars(CertificateBasis)["__dict__"]
_OPENING_GENERATORS = (
    _operational_context,
    _retained_action,
    _operational_data,
    _produced_retained_action,
    _construction_code,
    _FIT_PREPARED_ISSUER,
    _OPERATIONAL_PREPARED_ISSUER,
    _GENERATED_RETAINED,
    _immutable_indices,
    _opening_context,
    _opening_range,
    _fit_generated,
    _witness_proof,
    np.geterr,
    np.asarray,
    np.zeros,
    np.errstate,
    np.flatnonzero,
    np.isfinite,
    np.array_equal,
    np.arange,
    np.abs,
    np.max,
    immutable_array,
    vars(CertificateBasis).get("positions"),
    vars(CertificateBasis).get("columns"),
    vars(CertificateBasis).get("selected"),
    vars(CertificateBasis).get("n_kept"),
    vars(WitnessBlock).get("size"),
    vars(WitnessBlock).get("start"),
    vars(WitnessBlock).get("levels"),
    vars(WitnessBlock).get("values"),
    vars(WitnessBlock).get("context"),
    vars(WitnessBlock).get("block"),
    vars(WitnessContext).get("columns"),
    vars(WitnessContext).get("components"),
    vars(WitnessContext).get("positions"),
    vars(WitnessContext).get("dependent_columns"),
    vars(WitnessContext).get("dependent_values"),
    vars(WitnessContext).get("factors"),
    vars(SparseValues).get("positions"),
    vars(SparseValues).get("numerators"),
    vars(SparseValues).get("denominators"),
    _direct_retained_coordinates,
    _packed_original,
    _checked_packed_heads,
    PackedHeads.head,
    PackedHeads.__getattribute__,
    getattr(PackedHeads, "__getattr__", None),
    vars(PackedHeads).get("count"),
    vars(PackedHeads).get("offsets"),
    vars(PackedHeads).get("anchors"),
    vars(PackedHeads).get("births"),
    vars(PackedHeads).get("versions"),
    _OPERATIONAL_FIELDS,
)


def _opening_original() -> bool:
    """Constant-size callback selection, not a stored eligibility verdict."""
    classes = (
        CertificateBasis,
        WitnessBlock,
        WitnessContext,
        FactoredRows,
        HeadRecipe,
        SparseValues,
        PackedHeads,
        HeadRows,
        OperationalBasis,
    )
    if any(a is not b for a, b in zip(classes, _OPENING_CLASSES, strict=True)):
        return False
    current = (
        _operational_context,
        _retained_action,
        _operational_data,
        _produced_retained_action,
        _construction_code,
        _FIT_PREPARED_ISSUER,
        _OPERATIONAL_PREPARED_ISSUER,
        _GENERATED_RETAINED,
        _immutable_indices,
        _opening_context,
        _opening_range,
        _fit_generated,
        _witness_proof,
        np.geterr,
        np.asarray,
        np.zeros,
        np.errstate,
        np.flatnonzero,
        np.isfinite,
        np.array_equal,
        np.arange,
        np.abs,
        np.max,
        immutable_array,
        vars(CertificateBasis).get("positions"),
        vars(CertificateBasis).get("columns"),
        vars(CertificateBasis).get("selected"),
        vars(CertificateBasis).get("n_kept"),
        vars(WitnessBlock).get("size"),
        vars(WitnessBlock).get("start"),
        vars(WitnessBlock).get("levels"),
        vars(WitnessBlock).get("values"),
        vars(WitnessBlock).get("context"),
        vars(WitnessBlock).get("block"),
        vars(WitnessContext).get("columns"),
        vars(WitnessContext).get("components"),
        vars(WitnessContext).get("positions"),
        vars(WitnessContext).get("dependent_columns"),
        vars(WitnessContext).get("dependent_values"),
        vars(WitnessContext).get("factors"),
        vars(SparseValues).get("positions"),
        vars(SparseValues).get("numerators"),
        vars(SparseValues).get("denominators"),
        _direct_retained_coordinates,
        _packed_original,
        _checked_packed_heads,
        PackedHeads.head,
        PackedHeads.__getattribute__,
        getattr(PackedHeads, "__getattr__", None),
        vars(PackedHeads).get("count"),
        vars(PackedHeads).get("offsets"),
        vars(PackedHeads).get("anchors"),
        vars(PackedHeads).get("births"),
        vars(PackedHeads).get("versions"),
        _OPERATIONAL_FIELDS,
    )
    return all(a is b for a, b in zip(current, _OPENING_GENERATORS, strict=True))


def coordinates(basis: CertificateBasis, effects: tuple[FloatArray, ...]) -> FloatArray:
    """``alpha = sum_d w_d e_d`` over every original level ``d`` of every block.

    ``w_d`` is the exact rational witness ``D_d = E_J w_d`` of the basis, so that
    ``E_J alpha = sum_d D_d e_d`` in exact arithmetic: a kept level adds its value at
    its coordinate, a reference or dependent level with a nonzero value its witness
    combination.  ``FloatingPointError`` where a value is not representable;
    ``ValueError`` for effects of other blocks.
    """
    positions = basis.positions
    if len(effects) != len(positions):
        raise ValueError("the effects belong to another grouping")
    proof = _witness_proof(basis)
    operational = (
        _operational_context(basis)
        if _operational_context is _OPERATIONAL_CONTEXT
        else None
    )
    try:
        generated = (proof is not None or operational is not None) and _fit_generated()
    except MemoryError:
        # Keep the original per-level action if optional batch selection cannot fit.
        generated = False
    alpha = np.zeros(basis.n_kept)
    ordinary: tuple[tuple[Witness, ...], ...] | None = None
    with np.errstate(over="raise", invalid="raise"):
        for block, (place, values) in enumerate(zip(positions, effects, strict=True)):
            found = np.asarray(values, dtype=np.float64)
            if found.shape != place.shape:
                raise ValueError("the effects belong to another grouping")
            kept = place >= 0
            alpha[place[kept]] += found[kept]
            witnesses: tuple[Witness, ...] = ()
            if proof is None and operational is None:
                if ordinary is None and type(basis) is CertificateBasis:
                    # The checked outward tuple contains every block's witnesses.
                    # Read it once for this exact action.
                    ordinary = basis.witnesses
                witnesses = (basis.witnesses if ordinary is None else ordinary)[block]
            levels = np.flatnonzero(~kept & (found != 0)).tolist()
            demanded = (
                None
                if not generated or proof is None
                else proof.blocks[block].iter_selected(levels)
            )
            references = (
                {}
                if operational is None
                else operational.references(
                    block,
                    [level for level in levels if basis.columns[block][level] < 0],
                )
            )
            for level in levels:
                witness = (
                    references[level]
                    if operational is not None and basis.columns[block][level] < 0
                    else operational.dependent(int(basis.columns[block][level]))
                    if operational is not None
                    else next(demanded)
                    if demanded is not None
                    else proof.witness(block, level)
                    if proof is not None
                    else witnesses[level]
                )
                for position, numerator, denominator in witness:
                    try:
                        ratio = numerator / denominator
                    except OverflowError as error:
                        raise FloatingPointError(str(error)) from error
                    alpha[position] += found[level] * ratio
    if not np.isfinite(alpha).all():
        raise FloatingPointError("the start coordinates are not representable")
    return immutable_array(alpha)


def _apply_reporting_rows(
    columns: tuple[IndexArray, ...], rows: FactoredRows, effects: tuple[FloatArray, ...]
) -> tuple[FloatArray, ...]:
    """Transfer the non-kept coefficients through the factored exact row map."""
    heads = cast(HeadRows, rows.heads)
    later = np.asarray(rows.tail_columns, dtype=np.intp)
    coefficients = np.zeros(sum(np.count_nonzero(col >= 0) for col in columns) + 1)
    for column, effect in zip(columns, effects, strict=True):
        valid = column >= 0
        coefficients[column[valid]] = effect[valid]
    head_values = coefficients[: heads.count].copy()
    tail_values = coefficients[later]
    coefficients[: heads.count] = 0
    coefficients[later] = 0
    if np.any(coefficients):
        with np.errstate(over="raise", invalid="raise"):
            try:
                correction = np.asarray(
                    [
                        sum(
                            coefficients[c] * (n / d)
                            for c, n, d in row.canonical()
                            if coefficients[c] != 0
                        )
                        for row in rows.tails
                    ]
                )
            except OverflowError as error:
                raise FloatingPointError(str(error)) from error
            tail_values += correction
            coefficients[later] = -correction
            first = np.frombuffer(heads.first, dtype=np.intp)
            for codes, column in zip(heads.codes[1:], columns[1:], strict=True):
                code = np.frombuffer(codes, dtype=np.intp)
                head_values += coefficients[column[code[first]]]
    coefficients.fill(0)
    coefficients[: heads.count] = head_values
    coefficients[later] = tail_values
    return tuple(coefficients[column] for column in columns)


def _canonical_reporting_effects(
    grouping: Grouping, parts: Components, effects: tuple[FloatArray, ...]
) -> tuple[FloatArray, ...]:
    """Move the component-normalized effects onto the exact reporting basis."""
    factored = _reporting_elimination(grouping, parts)
    if factored is not None:
        return _apply_reporting_rows(*factored, effects)
    basis = certificate_basis(grouping)
    return embed(basis, coordinates(basis, effects))
