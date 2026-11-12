"""Logical fixed-effect block selection and controls for one-block cancellation.

One named
fixed-effect block is selected for exact multiplicative cancellation, or ``cancel=True``
selects the unique block with the largest number of distinct levels on the estimation
positions.  All remaining blocks are retained in their recorded model order without any
cap on the number of retained dimensions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from powermean.data.collinearity_records import DiagnosticControls
from powermean.errors import SpecificationError
from powermean.fe.admission import AdmissionInputs
from powermean.fe.admission_records import WorkLog
from powermean.fe.cancel import cancel_structure
from powermean.fe.geometry import numerical_geometry
from powermean.fe.geometry_checks import GeometryInputs
from powermean.fe.geometry_design import NumericalGeometry
from powermean.fe.groups import Labels, _positions, encode_block
from powermean.fe.retained import centered_design
from powermean.inference.wz_covariance import WZControls
from powermean.messages import format_keys
from powermean.model.variance import (
    CancelledFamily,
    FloatArray,
    Variance,
    resolve_cancelled_variance,
)
from powermean.results.cancelled_retained import CancelledReporting
from powermean.results.retained_inference import InferenceRequest
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.cancelled_records import RootOptions
from powermean.solve.absorbed import AbsorbedOptions


@dataclass(frozen=True)
class NamedFEBlock:
    """One named fixed-effect block in recorded model order."""

    name: str
    labels: Labels


BlockSpec = Sequence[NamedFEBlock | tuple[str, Labels]] | Mapping[str, Labels]


@dataclass(frozen=True)
class CancelledBlockInputs:
    """Deferred one-block cancellation specification resolved inside complete fit."""

    slopes: GeometryInputs
    blocks: BlockSpec
    cancel: str | Literal[True] = True


@dataclass(frozen=True)
class CancelledSelection:
    """Bound one-block cancellation selection and ordered retained blocks."""

    inputs: AdmissionInputs
    cancelled_name: str
    retained_names: tuple[str, ...]
    block_order: tuple[str, ...]
    cancelled_index: int
    level_counts: tuple[int, ...]
    selection_mode: Literal["named", "most_levels"]
    blocks: tuple[NamedFEBlock, ...] = ()
    cancel: str | Literal[True] = True

    def build_geometry(
        self,
        *,
        gram_rtol: float = 1e-12,
        gap_min: float = 1e-8,
        max_bytes: int = 64_000_000,
        nuisance_basis: str = "gram_eigh",
        custom_nuisance_basis: object = None,
    ) -> NumericalGeometry:
        """Build ``NumericalGeometry`` for the selected block and retained order."""
        structure = cancel_structure(
            self.inputs.cancelled,
            self.inputs.retained,
            positions=self.inputs.slopes.positions,
        )
        design = centered_design(structure, self.inputs.slopes.raw_slopes)
        return numerical_geometry(
            design,
            gram_rtol=gram_rtol,
            gap_min=gap_min,
            max_bytes=max_bytes,
            nuisance_basis=nuisance_basis,
            custom_nuisance_basis=custom_nuisance_basis,
        )


@dataclass(frozen=True)
class CancelledFitControls:
    """Explicit numerical/admission/resource choices; no package defaults selected."""

    gram_rtol: float
    gap_min: float
    geometry_bytes: int
    admission_bytes: int
    identity_atol: float
    attempt: AttemptBounds
    certificate: AttemptBounds
    inference: AttemptBounds
    diagnostic: DiagnosticControls
    collinear: str


@dataclass(frozen=True)
class CancelledFitPolicy:
    """Explicit lifecycle choices, without default public behavior."""

    request: InferenceRequest | None
    start: FloatArray | None
    keep_influence: bool
    on_failure: str
    variance: Variance | None = None
    tau: float | None = None
    root_options: RootOptions | Mapping[str, object] | None = None
    absorbed_options: AbsorbedOptions | Mapping[str, object] | None = None
    wz: WZControls | None = None
    wz_coefficient: WZControls | None = None


def _invalid_blocks(value: object, valid: str) -> SpecificationError:
    """Build a ``SpecificationError`` for an invalid ``blocks`` argument."""
    return SpecificationError(
        "specification.invalid_value",
        {"argument": "blocks", "value": value, "valid": valid},
    )


def _validate_block_names(items: list[NamedFEBlock]) -> tuple[NamedFEBlock, ...]:
    """Validate that block names are non-empty, unique strings."""
    seen: set[str] = set()
    for item in items:
        if not isinstance(item.name, str) or not item.name.strip():
            raise _invalid_blocks(item.name, "non-empty string block names")
        if item.name in seen:
            raise _invalid_blocks(item.name, "unique fixed-effect block names")
        seen.add(item.name)
    return tuple(items)


def _normalize_blocks(blocks: object) -> tuple[NamedFEBlock, ...]:
    """Validate and normalize a non-empty ordered collection of named FE blocks."""
    if isinstance(blocks, Mapping):
        items = [NamedFEBlock(name=key, labels=val) for key, val in blocks.items()]
    elif isinstance(blocks, Sequence) and not isinstance(blocks, (str, bytes)):
        items = []
        for entry in blocks:
            if isinstance(entry, NamedFEBlock):
                items.append(entry)
            elif isinstance(entry, tuple) and len(entry) == 2:
                items.append(NamedFEBlock(name=entry[0], labels=entry[1]))
            else:
                raise _invalid_blocks(entry, "NamedFEBlock or (name, labels) pairs")
    else:
        raise _invalid_blocks(
            blocks, "a non-empty sequence or mapping of named FE blocks"
        )
    if not items:
        raise _invalid_blocks(blocks, "at least one named fixed-effect block")
    return _validate_block_names(items)


def _resolve_cancel_index(
    normalized: tuple[NamedFEBlock, ...],
    level_counts: tuple[int, ...],
    cancel: object,
) -> tuple[int, Literal["named", "most_levels"]]:
    """Resolve the index of the cancelled block from a name or ``cancel=True``."""
    names = tuple(block.name for block in normalized)
    if cancel is True:
        max_levels = max(level_counts)
        winners = [idx for idx, count in enumerate(level_counts) if count == max_levels]
        if len(winners) > 1:
            tied = tuple(names[idx] for idx in winners)
            valid = (
                f"an explicit block name ({format_keys(tied)} tie at "
                f"{max_levels} levels)"
            )
            raise SpecificationError(
                "specification.invalid_value",
                {"argument": "cancel", "value": True, "valid": valid},
            )
        return winners[0], "most_levels"
    if isinstance(cancel, str):
        for idx, block in enumerate(normalized):
            if block.name == cancel:
                return idx, "named"
        raise SpecificationError(
            "specification.invalid_value",
            {"argument": "cancel", "value": cancel, "valid": format_keys(names)},
        )
    raise SpecificationError(
        "specification.invalid_value",
        {
            "argument": "cancel",
            "value": cancel,
            "valid": f"True or one of {format_keys(names)}",
        },
    )


def select_cancelled_block(
    slopes: GeometryInputs,
    blocks: BlockSpec,
    *,
    cancel: str | Literal[True] = True,
) -> CancelledSelection:
    """Bind one logical FE block for cancellation and retain the rest in model order."""
    if type(slopes) is not GeometryInputs:
        raise TypeError("use GeometryInputs for slopes")
    normalized = _normalize_blocks(blocks)
    blocks_encoded = tuple(
        encode_block(block.labels, name=block.name) for block in normalized
    )
    n_rows = int(np.asarray(slopes.raw_slopes).shape[0])
    _ = _positions(slopes.positions, n_rows)
    if any(encoded.n_rows != n_rows for encoded in blocks_encoded):
        raise ValueError("all FE arrays must have the same number of rows")
    level_counts = tuple(encoded.n_levels for encoded in blocks_encoded)
    cancelled_index, selection_mode = _resolve_cancel_index(
        normalized, level_counts, cancel
    )
    cancelled_block = normalized[cancelled_index]
    retained_blocks = tuple(
        block for idx, block in enumerate(normalized) if idx != cancelled_index
    )
    inputs = AdmissionInputs(
        slopes=slopes,
        cancelled=cancelled_block.labels,
        retained=tuple(block.labels for block in retained_blocks),
    )
    return CancelledSelection(
        inputs=inputs,
        cancelled_name=cancelled_block.name,
        retained_names=tuple(block.name for block in retained_blocks),
        block_order=tuple(block.name for block in normalized),
        cancelled_index=cancelled_index,
        level_counts=level_counts,
        selection_mode=selection_mode,
        blocks=normalized,
        cancel=cancel,
    )


def align_cancelled_reporting(
    selection: CancelledSelection,
    reporting: CancelledReporting,
) -> CancelledReporting:
    """Bind ``CancelledReporting`` names to the resolved ``CancelledSelection``."""
    if reporting.cancelled_name not in ("", "cancelled", selection.cancelled_name):
        raise SpecificationError(
            "specification.conflicting",
            {"first": "reporting.cancelled_name", "second": "selection.cancelled_name"},
        )
    if reporting.retained_names not in ((), selection.retained_names):
        raise SpecificationError(
            "specification.conflicting",
            {"first": "reporting.retained_names", "second": "selection.retained_names"},
        )
    return replace(
        reporting,
        cancelled_name=selection.cancelled_name,
        retained_names=selection.retained_names,
    )


def _run_selection(
    slopes: GeometryInputs,
    blocks: BlockSpec,
    cancel: str | Literal[True],
    work: WorkLog | None,
) -> CancelledSelection:
    """Execute ``select_cancelled_block`` optionally inside ``work``."""
    if work is None:
        return select_cancelled_block(slopes, blocks, cancel=cancel)
    return work.run(
        "select_cancelled_block",
        lambda: select_cancelled_block(slopes, blocks, cancel=cancel),
    )


def resolve_selection(
    inputs: (
        AdmissionInputs | CancelledSelection | CancelledBlockInputs | GeometryInputs
    ),
    reporting: CancelledReporting,
    *,
    work: WorkLog | None = None,
    blocks: BlockSpec | None = None,
    cancel: str | Literal[True] | None = None,
) -> tuple[
    AdmissionInputs,
    CancelledReporting,
    Literal["direct_inputs", "named", "most_levels"],
    tuple[str, ...],
    int,
]:
    """Resolve and align block selection inside the complete-fit work log."""
    if type(reporting) is not CancelledReporting:
        raise TypeError("use a CancelledReporting record")
    inherited_cancel = (
        inputs.cancel
        if isinstance(inputs, (CancelledBlockInputs, CancelledSelection))
        else True
    )
    eff_c: str | Literal[True] = inherited_cancel if cancel is None else cancel
    if isinstance(inputs, GeometryInputs):
        if blocks is None:
            raise TypeError("blocks is required when inputs is GeometryInputs")
        inputs = CancelledBlockInputs(slopes=inputs, blocks=blocks, cancel=eff_c)
    if isinstance(inputs, CancelledBlockInputs):
        sel = _run_selection(inputs.slopes, inputs.blocks, eff_c, work)
    elif isinstance(inputs, CancelledSelection):
        reselect = bool(inputs.blocks) and (work is not None or eff_c != inputs.cancel)
        sel = (
            _run_selection(inputs.inputs.slopes, inputs.blocks, eff_c, work)
            if reselect
            else inputs
        )
    else:
        if type(inputs) is not AdmissionInputs:
            raise TypeError(
                "use AdmissionInputs, CancelledSelection, or CancelledBlockInputs"
            )
        order = (reporting.cancelled_name, *reporting.retained_names)
        return inputs, reporting, "direct_inputs", order, 0
    aligned = align_cancelled_reporting(sel, reporting)
    return sel.inputs, aligned, sel.selection_mode, sel.block_order, sel.cancelled_index


def _check_conflict(
    first_val: object, second_val: object, f_name: str, s_name: str
) -> None:
    """Raise ``SpecificationError`` when two supplied values conflict."""
    if first_val is not None and second_val is not None and first_val != second_val:
        raise SpecificationError(
            "specification.conflicting", {"first": f_name, "second": s_name}
        )


def resolve_policy_variance(
    policy: CancelledFitPolicy,
    variance: Variance | float | None,
    tau: float | None = None,
) -> CancelledFamily:
    """Resolve ``policy``, ``variance``, and ``tau`` into a ``CancelledFamily``."""
    if isinstance(variance, (int, float)) and not isinstance(variance, bool):
        _check_conflict(float(variance), tau, "variance", "tau")
        tau, variance = float(variance), None
    _check_conflict(policy.variance, variance, "policy.variance", "variance")
    _check_conflict(policy.tau, tau, "policy.tau", "tau")
    eff_var = variance if variance is not None else policy.variance
    eff_tau = tau if tau is not None else policy.tau
    return resolve_cancelled_variance(variance=eff_var, tau=eff_tau)
