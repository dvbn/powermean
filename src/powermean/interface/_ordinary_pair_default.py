"""Select an ordinary omitted covariance from this fit's completed FE sample.

Immutable component runs can prove common inputs without reading labels or rows.
Other producers require agreement of their actual observed component partitions,
using detached within-block coordinates and final integer codes. An opaque pair
also needs its split/merge check. Nothing survives a call; no WZ support is involved.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np

from powermean.data.fe_component_storage import matching_runs
from powermean.data.raw import Column
from powermean.data.sample_fe_source import (
    Derived,
    Source,
    _native_source_parts,
    _owned_source_original,
)
from powermean.data.sample_fe_state import FESample
from powermean.errors import SpecificationError
from powermean.fe.groups import Block


def _endpoint_codes(
    block: Block, coordinate: int, *, levels: tuple[object, ...] | None = None
) -> np.ndarray:
    """The endpoint ordinal of each interaction level, within this block only.

    One detached exposure preserves aliases across the block's levels, including
    identity-equality endpoints. Hash/equality acts only on those detached copies;
    no label is compared with a separately copied label from another block.
    """
    if levels is None:
        levels = block.levels
    lookup: dict[object, int] = {}
    codes = np.empty(block.n_levels, dtype=np.intp)
    for code, level in enumerate(levels):
        if type(level) is not tuple or len(level) != 2:
            raise ValueError(
                "ordinary pair default requires canonical interaction levels"
            )
        endpoint = level[coordinate]
        found = lookup.get(endpoint)
        if found is None:
            found = len(lookup)
            lookup[endpoint] = found
        codes[code] = found
    return codes


def _stable_components(grouping: Derived) -> bool:
    """Bind the original producer to its actual immutable component storage.

    Read native owners directly: accessing their logical columns would materialize
    labels. Existing run/header checks inspect no rows and invoke no label hooks.
    """
    origin = grouping.source
    if type(grouping) is not Derived or type(origin) is not Source:
        return False
    state = vars(origin)
    if "_native_columns" in state:
        return _native_source_parts(origin) is not None
    return _owned_source_original() and matching_runs(
        state.get("_component_runs"), cast("Mapping[str, Column]", state.get("columns"))
    )


def _same_partition(
    first: Block, left: np.ndarray, second: Block, right: np.ndarray
) -> bool:
    """Compare two observed coordinate partitions without comparing their labels."""
    forward: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for first_code, second_code in zip(first.codes, second.codes, strict=True):
        a, b = int(left[int(first_code)]), int(right[int(second_code)])
        known = forward.get(a)
        if known is None:
            if b in reverse:
                return False
            forward[a], reverse[b] = b, a
        elif known != b:
            return False
    return True


def _observed_components(
    blocks: tuple[Block, ...], terms: tuple[tuple[str, ...], ...]
) -> dict[tuple[int, str], np.ndarray]:
    """Recover coordinates from the actual final interaction blocks.

    One detached exposure per block retains within-block aliases. Names locate
    candidate coordinates; their integer partitions still require comparison.
    """
    coordinates: dict[tuple[int, str], np.ndarray] = {}
    for i, (block, term) in enumerate(zip(blocks, terms, strict=True)):
        if len(term) != 2:
            continue
        levels = block.levels
        for side, name in enumerate(term):
            coordinates[i, name] = _endpoint_codes(block, side, levels=levels)
    return coordinates


def _pair_partition(
    blocks: tuple[Block, ...],
    terms: tuple[tuple[str, ...], ...],
    pair: int,
    first: int,
    second: int,
    date: str,
    coordinates: dict[tuple[int, str], np.ndarray] | None = None,
) -> bool:
    """Check the opaque pair's actual final partition once, with integer maps."""
    left_side = 1 - terms[first].index(date)
    right_side = 1 - terms[second].index(date)
    if coordinates is None:
        left = _endpoint_codes(blocks[first], left_side)
        right = _endpoint_codes(blocks[second], right_side)
    else:
        left = coordinates[first, terms[first][left_side]]
        right = coordinates[second, terms[second][right_side]]
    forward: dict[int, tuple[int, int]] = {}
    reverse: dict[tuple[int, int], int] = {}
    for pair_code, left_code, right_code in zip(
        blocks[pair].codes, blocks[first].codes, blocks[second].codes, strict=True
    ):
        key = (int(left[int(left_code)]), int(right[int(right_code)]))
        code = int(pair_code)
        known = forward.get(code)
        if known is None:
            if key in reverse:
                return False  # The same endpoint pair has several pair codes.
            forward[code], reverse[key] = key, code
        elif known != key:
            return False  # One pair code combines different endpoint pairs.
    return True


def _ordinary_pair(
    sample: FESample, terms: tuple[tuple[str, ...], ...]
) -> Block | None:
    """Return the unique actual pair block, or require a cluster if ambiguous.

    The admitted completed sample keeps canonical Derived blocks of its source
    on its final rows. Require that lineage as well as the source's actual terms;
    matching names alone never establish common components. Immutable runs or
    actual final coordinate partitions must additionally establish common inputs.
    Construction and outcome admission bind the rows, blocks and source columns.
    The caller excludes explicit requests, cancellation and WZ before entry.
    """
    grouping = sample.grouping
    if grouping is None or len(terms) != 3 or len(grouping.blocks) != 3:
        return None
    blocks = grouping.blocks
    roles: list[tuple[int, int, int, str]] = []
    for pair in range(3):
        first, second = (i for i in range(3) if i != pair)
        left, right = terms[first], terms[second]
        if (
            len(left) != 2
            or len(right) != 2
            or len(set(left)) != 2
            or len(set(right)) != 2
        ):
            continue
        common = set(left) & set(right)
        if len(common) != 1:
            continue
        date = next(iter(common))
        endpoints = {left[1 - left.index(date)], right[1 - right.index(date)]}
        if len(terms[pair]) == 1 or (
            len(terms[pair]) == 2 and set(terms[pair]) == endpoints
        ):
            roles.append((pair, first, second, date))
    if not roles:
        return None
    source = sample.source
    if (
        not isinstance(grouping, Derived)
        or not isinstance(source, Derived)
        or grouping.source is not source.source
        or grouping.source.terms != terms
        or grouping.source.nobs != sample.ledger.nobs
        or tuple(block.name for block in blocks)
        != tuple("^".join(term) for term in terms)
    ):
        raise ValueError(
            "ordinary pair default requires canonical common-source FE provenance"
        )
    positions = sample.positions
    if grouping.positions.shape != positions.shape or any(
        block.codes.shape != positions.shape for block in blocks
    ):
        raise ValueError("ordinary pair default requires final sample FE row alignment")
    coordinates = None
    if not _stable_components(grouping):
        if not np.array_equal(grouping.positions, positions):
            raise ValueError(
                "ordinary pair default requires final sample FE row alignment"
            )
        coordinates = _observed_components(blocks, terms)
        observed: list[tuple[int, int, int, str]] = []
        for pair, first, second, date in roles:
            if not _same_partition(
                blocks[first],
                coordinates[first, date],
                blocks[second],
                coordinates[second, date],
            ):
                continue
            if _pair_partition(blocks, terms, pair, first, second, date, coordinates):
                observed.append((pair, first, second, date))
        roles = observed
        if not roles:
            return None
    if len(roles) > 1:
        # Immutable inputs prove every declared role. Other producers have just
        # proved each role's shared coordinate and pair partition independently.
        # Only the immutable route avoids level exposures and row comparisons.
        raise SpecificationError(
            "specification.unsupported_inference",
            {
                "method": "Default pair-clustered covariance",
                "case": (
                    "ambiguous ordinary FE roles; "
                    "pass cluster='your_pair_term' explicitly"
                ),
            },
        )
    if coordinates is None and not np.array_equal(grouping.positions, positions):
        raise ValueError("ordinary pair default requires final sample FE row alignment")
    pair, first, second, date = roles[0]
    if coordinates is not None or _pair_partition(
        blocks, terms, pair, first, second, date
    ):
        return blocks[pair]
    return None
