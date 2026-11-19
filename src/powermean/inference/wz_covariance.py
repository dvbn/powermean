"""Current-candidate WZ inference composition, separate from coefficient bias.

Design 03.5.4 / 04.7: the published left inverse and P/(P-1) own the reported
covariance. SSC retains its reference distribution and request provenance only.
No certificate, optimizer, symmetric CR2 transform or PSD repair is invoked here.
Owner D2: a pair-cancelled fit's own admission (geometry, exact quotient and
slope evidence) is its one geometry check; WZ binds its labels and reuses it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from powermean.data.inference_conventions import ConventionValues
from powermean.data.request_values import Omitted
from powermean.fe.cancel import CancelStructure
from powermean.inference.convention import ClusterCodes, Covariance
from powermean.inference.ssc import (
    AppliedConvention,
    ConventionCounts,
    ConventionCovariance,
    ResolvedConvention,
    ScalarFactor,
    direct_errors,
    reference_of,
    resolve_convention,
)
from powermean.inference.vcov import _result
from powermean.inference.wz_binding import (
    Coordinates,
    Workspace,
    WZLayout,
    WZSource,
    capture_rows,
    identify,
    layout,
    quotient_coordinates,
    unavailable,
)
from powermean.inference.wz_operator import (
    Information,
    PairCovariance,
    Rows,
    evaluate_information,
)
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray, PowerVariance
from powermean.solve.cancelled_operator import AttemptBounds
from powermean.solve.m1_options import AbsorbedOptions
from powermean.solve.options import SolverOptions
from powermean.systems.admitted import CapturedFit
from powermean.systems.identified import IdentifiedState


def convention(
    requested: str | ConventionValues | Omitted,
    counts: ConventionCounts,
    *,
    inherited: ResolvedConvention | None = None,
) -> AppliedConvention:
    """Own P/(P-1) once, retaining the existing convention's reference rules."""
    resolved = resolve_convention(requested, inherited)
    if resolved.origin != "preset" or len(counts.clusters) != 1:
        raise unavailable("named SSC presets and one pair cluster are supported")
    pairs = counts.clusters[0]
    factor = ScalarFactor(pairs, pairs - 1, "WZ P/(P-1)", "method")
    return AppliedConvention(
        resolved,
        counts,
        "cluster",
        None,
        factor,
        (factor,),
        "min",
        reference_of(resolved.values, counts, clustered=True),
        "none",
    )


def bind_request(source: WZSource, cluster: ClusterCodes) -> None:
    """Require the named source's exact codes, not another equal partition."""
    blocks = [b for b in source.grouping.blocks if b.name == cluster.name]
    if len(blocks) != 1 or not np.array_equal(blocks[0].codes, cluster.codes):
        raise unavailable(
            "cluster must retain the exact declared pair source and codes"
        )


@dataclass(frozen=True)
class WZControls:
    """Existing explicit fit controls; no new bounds, tolerance or optimizer policy.

    ``identified`` optionally carries this fit's exact joint coordinates as
    plain values (Coordinates.held: structure digest and level positions), so
    the within-fit assessment capture replicates them by value. They are read
    only for the identical structure digest; anything else is recomputed.
    """

    source: WZSource
    absorbed: AbsorbedOptions
    options: SolverOptions
    bounds: AttemptBounds
    identified: tuple[str, bytes] | None = None

    def joint_coordinates(self, panel: WZLayout) -> Coordinates:
        """The exact joint coordinates of this fit for ``panel``."""
        return identify(panel, self.bounds, self.identified)

    def cancelled(
        self,
        state: IdentifiedState,
        capture: CapturedFit,
        cluster: ClusterCodes,
        *,
        ssc: str,
    ) -> PairCovariance:
        """The covariance at the current cancelled point (see cancelled_information)."""
        return self.cancelled_information(state, capture, cluster, ssc=ssc)[0]

    def cancelled_information(
        self,
        state: IdentifiedState,
        capture: CapturedFit,
        cluster: ClusterCodes,
        *,
        ssc: str,
        previous: tuple[PairCovariance, Information] | None = None,
    ) -> tuple[PairCovariance, Information]:
        """Covariance and same-point information at the current cancelled point.

        ``previous`` is this fit's last result. It is returned unchanged only for
        freshly captured rows with the identical binding digest, request ssc and
        rcond_min (the same point certified again), never across fits.
        """
        current, coordinates = self.cancelled_rows(
            state, capture, cluster, route="pair_cancelled_current"
        )
        if (
            previous is not None
            and previous[1].binding_sha256 == current.binding_sha256
            and previous[1].rcond_min == self.options.rcond_min
            and previous[0].provenance.requested_ssc == ssc
        ):
            return previous
        return evaluate_information(
            current, self.options, self.bounds, ssc=ssc, plan=coordinates.plan
        )

    def cancelled_rows(
        self,
        state: IdentifiedState,
        capture: CapturedFit,
        cluster: ClusterCodes,
        *,
        route: str,
        workspace: Workspace | None = None,
        panel: WZLayout | None = None,
    ) -> tuple[Rows, Coordinates]:
        """Recover the joint fitted mean by the exact PPML pair-scale algebra.

        At tau=1, t=y and mu_it=pi_it sum_t y_it. These are the fresh current
        cancelled rows, evaluated by the existing capture under unit multipliers.
        No companion fit, effect recovery, or final accepted flag is needed.
        ``panel`` is the caller's layout of this same call, reserved first.
        """
        geometry = capture.admission.geometry
        structure = geometry.structure
        n, k = structure.n_rows, geometry.n_slopes_full
        if panel is None:
            panel = layout(
                self.source, cluster.name, (n, k), self.bounds, workspace=workspace
            )
        bind_request(self.source, cluster)
        if state.design is not geometry or geometry.n_slopes_identified != k:
            raise unavailable(
                "all original slopes must remain identified in this current state"
            )
        if capture.obs_weights is not None and not np.all(capture.obs_weights == 1):
            raise unavailable(
                "only unit equation multipliers and no statistical weights"
            )
        pair = next(b for b in self.source.grouping.blocks if b.name == cluster.name)
        if (
            not np.array_equal(pair.codes, structure.cancel_codes)
            or pair.levels != structure.cancel_levels
        ):
            raise unavailable(
                "current cancellation must own the exact pair source labels"
            )
        _cancelled_sources(self.source, structure)
        coordinates = quotient_coordinates(
            panel, structure, capture.admission.quotient.kept
        )
        rows = state.rows()
        with np.errstate(over="raise", invalid="raise", under="raise"):
            mu = rows.pi * state.full_state.group_t[structure.cancel_codes]
        current = capture_rows(
            panel,
            capture.admission.raw_slopes,
            capture.z,
            mu,
            capture.admission.check.source_columns,
            route=route,
            coordinates=coordinates,
        )
        return current, coordinates


type _Encoded = tuple[Sequence[object], NDArray[np.integer]]


def _encoded(block: _Encoded) -> bool:
    """Whether the codes are a vector of signed integers that index the levels."""
    levels, codes = block
    return bool(
        codes.ndim == 1
        and codes.dtype.kind == "i"
        and (codes.size == 0 or (codes.min() >= 0 and codes.max() < len(levels)))
    )


def _same_row_labels(left: _Encoded, right: _Encoded) -> bool:
    """Whether two encoded blocks carry equal labels at every row.

    This is the equality of their tuples of row labels, ``left == right``,
    decided without building them. Blocks with the same codes agree exactly
    when their levels agree at the codes that occur. Any other pair of blocks
    is compared label by label in row order, as the tuples are, and stops at
    the first difference.
    """
    (levels, codes), (others, other_codes) = left, right
    if codes.size != other_codes.size:
        return False
    if np.array_equal(codes, other_codes):
        present = np.flatnonzero(np.bincount(codes, minlength=len(levels)))
        if present.size == len(levels) == len(others):
            return bool(levels == others)
        return all(
            levels[c] is others[c] or bool(levels[c] == others[c])
            for c in present.tolist()
        )
    return all(
        levels[int(a)] is others[int(b)] or bool(levels[int(a)] == others[int(b)])
        for a, b in zip(codes, other_codes, strict=True)
    )


def _cancelled_sources(source: WZSource, structure: CancelStructure) -> None:
    """Bind the current cancellation to the original declared labels, row by row.

    Positions and every row's original pair and retained labels are compared in
    declared order. Blocks whose codes index their levels are compared through
    those codes and levels; any other block is handed with the rest to the
    row-by-row comparison. The fit's admitted geometry and exact quotient are
    reused, not admitted again (D2); a changed source refuses rather than
    re-admits.
    """
    if not np.array_equal(source.grouping.positions, structure.positions):
        raise unavailable("original cancellation row positions changed")
    original: list[_Encoded] = [
        (tuple(block.levels), np.asarray(block.codes))
        for block in source.grouping.blocks
    ]
    cancelled: _Encoded = (structure.cancel_levels, structure.cancel_codes)
    inverse = structure.inverse
    retained: list[_Encoded] = [
        (levels, codes[inverse])
        for levels, codes in zip(
            structure.retained_levels, structure.retained_sorted_codes, strict=True
        )
    ]
    if not all(map(_encoded, (*original, cancelled, *retained))):
        _cancelled_sources_by_rows(source, structure)
        return
    others = [block for block in original if not _same_row_labels(block, cancelled)]
    if (
        not any(_same_row_labels(cancelled, block) for block in original)
        or len(others) != len(retained)
        or not all(map(_same_row_labels, others, retained))
    ):
        raise unavailable(
            "current cancellation no longer owns the original declared labels"
        )


def _cancelled_sources_by_rows(source: WZSource, structure: CancelStructure) -> None:
    """The same binding from each row's labels, for codes of any other form."""
    original = []
    for block in source.grouping.blocks:
        levels = block.levels
        original.append(tuple(levels[int(code)] for code in block.codes))
    # cancel_levels detaches a fresh tuple on each access: read it once.
    cancel_levels = structure.cancel_levels
    cancelled = tuple(cancel_levels[int(c)] for c in structure.cancel_codes)
    retained = tuple(
        tuple(levels[int(c)] for c in codes[structure.inverse])
        for levels, codes in zip(
            structure.retained_levels, structure.retained_sorted_codes, strict=True
        )
    )
    if (
        cancelled not in original
        or tuple(v for v in original if v != cancelled) != retained
    ):
        raise unavailable(
            "current cancellation no longer owns the original declared labels"
        )


def covariance_record(result: PairCovariance, applied: AppliedConvention) -> Covariance:
    """Keep the published signed matrix and the existing diagnostic convention."""
    estimate = _result(
        result.unscaled_adjusted,
        method="WZ",
        nobs=result.provenance.observations,
        df_resid=None,
        correction="WZ P/(P-1)",
        factor=1.0,
    )
    assert applied.factor is not None
    return Covariance(
        result.adjusted,
        "cluster",
        applied.factor.value,
        applied.counts.count,
        result.provenance.pairs,
        estimate,
        applied.reference.kind,
    )


def retained_covariance_record(
    result: PairCovariance, applied: AppliedConvention
) -> ConventionCovariance:
    """Retain the actual convention around the already-scaled WZ matrix.

    The lower joint-scale record and its signed, unscaled estimate are preserved.
    This boundary adds reference/count/factor provenance, not another factor.
    """
    covariance = covariance_record(result, applied)
    return ConventionCovariance(
        covariance.covariance, covariance.estimate, applied, "cluster"
    )


def reported(
    result: PairCovariance,
    order: OrderScale,
) -> tuple[FloatArray, FloatArray | None, str | None]:
    """Apply the actual report map once; keep D49 negative/zero/range semantics."""
    errors = direct_errors(result.adjusted, order)
    try:
        matrix = order.covariance(result.adjusted)
    except FloatingPointError as exc:
        return errors, None, str(exc)
    return errors, matrix, None


def ppml(omega: float, variance: object, *, weights: bool = False) -> None:
    """Check the exact built-in PPML and no-statistical-weight domain."""
    if (
        omega != 1
        or type(variance) is not PowerVariance
        or variance.tau != 1
        or weights
    ):
        raise unavailable(
            "builtin PowerVariance(1), omega=1 and no statistical weights required"
        )
