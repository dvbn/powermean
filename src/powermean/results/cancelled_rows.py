"""Observation-level terms of a cancelled fit: multiway covariance and influence rows.

``design/03_fixed_effects.md``, Section 3.4; ``design/04_inference_and_effects.md``,
Sections 1, 2, 2.1, 3 and 4 (row "Cancelled, mixed"); BBP, Supplement
``app:applications-inference`` (``eq:app-application-twoway-covariance``) and
``eq:fe-power-cancel-score``.  Private and loaded on request only: nothing imports
this module when the package is imported, and a fit or a post-fit request that asks
for no multiway covariance and no row influence of a cancelled fit never loads it.

What is composed.  A cancelled system solves ``S = sum_g psi_g = 0`` with the group
contribution ``psi_g = sum_{i in g} (B_i - Bbar_g) rho_i``; its row terms
``s_i = (B_i - Bbar_g) rho_i`` and its map ``A = L H^-1`` (the adjoint solves
``H' W = L'`` of the actual, generally nonsymmetric ``H = -dS/dtheta'``) are those
of :mod:`powermean.systems.cancelled` and :mod:`powermean.inference.cancelled`,
which already answer a one-way request on a dimension that cuts across the groups.
This module forms nothing of them anew.  It hands the retained row influences
``u_i = A s_i`` of every original row to the convention engine of the joint route
(:func:`powermean.results.retained_inference._computed_inference`): the raw port
with every factor off, the convention's factors once by their owner, the reporting
map once and decision D4's frame and sequence.  No cluster total, sign, factor or
eigenvalue is computed here.

Counts.  The applied convention arrives from
:func:`powermean.results.cancelled_inference.request_convention`: ``N`` is the number
of cancelled groups, ``K`` the identified estimated coordinates, ``G`` the clusters
of each requested dimension on the original rows, with no nesting adjustment.

Frame of decision D4.  A cancelled fit reports no intercept, and its cancelled
effects are not estimated, so no private reporting constant exists: the frame matrix
is that of the reported slopes alone and the two native sequences coincide on it
(file 04, Section 2.1.1; the legacy sequence of an absorbed fit adds a constant row
that has no counterpart here).  The scales are those of the kept regressor columns
as supplied on every original row of the fit, singleton and all-zero groups
included, since cancellation deletes no row.  The frame record states that no
constant entered; the envelope retains no sample fingerprint, and the record says so.

Coverage.  When no requested dimension nests the cancelled groups, the dependence
that the centering creates within a group is not accounted for, and the
``coverage.outside_bbp`` warning of the one-way kernel is issued with the same case.

Notes at a fit.  A fit assesses several candidates, on the root route including a
Gamma initializer that solves another equation, and reads the requested standard
errors at each.  The notes of a candidate (the port's diagnostics of a raw estimate,
decision D4's notes, the coverage warning) describe that candidate's matrix, so a fit
hands a list (``held``) that receives them in place of their issue, each call
replacing the last; the fit issues the list of its reported point once.  A warning of
another class is never held.  After the fit there is one call and the notes are issued
as they arise.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from powermean.data.request_values import given
from powermean.errors import CovarianceWarning, CoverageWarning
from powermean.fe._cancel_storage import finite_product
from powermean.fe.admission_records import WorkLog
from powermean.inference.absorbed_rank import nested_in
from powermean.inference.cancelled import contracted_influence
from powermean.inference.ssc import AppliedConvention
from powermean.inference.vcov import CovarianceWarning as PortWarning
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.results.owned import owned_array
from powermean.results.retained_inference import (
    InferenceRequest,
    Reporting,
    RetainedInference,
    _computed_inference,
)
from powermean.systems.identified import IdentifiedState

if TYPE_CHECKING:
    from powermean.results.cancelled_retained import RetainedCancelledEnvelope
    from powermean.systems.admitted import CapturedFit

# The frame record's sample entry: the cancelled envelope retains no fingerprint.
FRAME_SAMPLE = "cancelled fit: every original row; no sample fingerprint retained"
CROSSING = "clustering with no dimension nesting the cancelled groups"
# The classes of the notes that describe the covariance of one candidate.
NOTES = (CovarianceWarning, PortWarning, CoverageWarning)


def frame_columns(capture: CapturedFit, order: OrderScale) -> FloatArray:
    """The kept regressor columns as supplied, one row per original row of the fit.

    The retained scaled columns times their column scale, the scale sample of
    decision D4's frame (file 04, Section 2.1.2); formed only for a request whose
    resolved treatment is ``"cgm"``.
    """
    admission = capture.admission
    kept = admission.geometry.kept_slope_indices
    with np.errstate(over="raise", invalid="raise"):
        return owned_array(finite_product(admission.raw_slopes[:, kept] * order.scale))


def _reporting(
    request: InferenceRequest,
    applied: AppliedConvention,
    order: OrderScale,
    rows: int,
    capture: CapturedFit | None,
) -> Reporting:
    """The reporting map of a cancelled fit: its scale, no intercept, no constant."""
    psd = request.psd if given(request.psd) else applied.psd
    if request.signed and psd == "cgm":
        if capture is None:
            raise ValueError("the retained regressor columns are unavailable")
        columns = frame_columns(capture, order)
    else:
        columns = np.empty((rows, 0))  # never read: no frame is formed
    return Reporting(order, None, columns, FRAME_SAMPLE, np.arange(rows, dtype=np.intp))


def multiway_inference(
    state: IdentifiedState,
    rhs: FloatArray,
    order: OrderScale,
    request: InferenceRequest,
    applied: AppliedConvention,
    *,
    work: WorkLog,
    capture: CapturedFit | None,
    held: list[warnings.WarningMessage] | None = None,
) -> RetainedInference:
    """A multiway covariance of a cancelled fit from its original-row terms.

    Parameters
    ----------
    state : IdentifiedState
        The cancelled equations at the point, with their row and group actions.
    rhs : array
        ``W = H^-T L'`` of the same point, one column per kept slope.
    order : OrderScale
        The order and the kept column scales of the reported coefficients.
    request : InferenceRequest
        A multiway request with the observed bread, its dimensions bound to every
        original row.
    applied : AppliedConvention
        The convention applied to the cancelled counts (module docstring).
    work : WorkLog
        The ledger of the inference calls, snapshotted into the record.
    capture : CapturedFit or None
        The owned fit problem, read for the frame columns of an adjusted request.
    held : list, optional
        Given by a fit: it receives the notes of this candidate, replacing those of
        the previous one, and none is issued here (module docstring).

    Returns
    -------
    RetainedInference
        The engine's record: the covariance of the slopes with the factors applied
        once, the reported matrix (decision D4's final matrix where it ran), the
        standard errors, the multiway components and the frame record.

    Raises
    ------
    ValueError
        Another request than a multiway one with the observed bread; codes that do
        not align with the original rows or a dimension with fewer than two
        clusters (the engine's refusals); an adjusted request without the retained
        columns.
    FloatingPointError
        A value that is not representable.

    Warns
    -----
    CoverageWarning
        ``coverage.outside_bbp`` when no requested dimension nests the cancelled
        groups.
    CovarianceWarning
        The port's diagnostics of a raw estimate, and decision D4's notes.
    """
    if not request.multiway_path or request.bread != "observed":
        raise ValueError("the row engine answers observed-bread multiway requests")
    if held is None:
        return _answer(state, rhs, order, request, applied, work=work, capture=capture)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        answered = _answer(
            state, rhs, order, request, applied, work=work, capture=capture
        )
    held[:] = [item for item in caught if isinstance(item.message, NOTES)]
    for item in caught:
        if not isinstance(item.message, NOTES):
            warnings.warn(item.message, stacklevel=2)
    return answered


def _answer(
    state: IdentifiedState,
    rhs: FloatArray,
    order: OrderScale,
    request: InferenceRequest,
    applied: AppliedConvention,
    *,
    work: WorkLog,
    capture: CapturedFit | None,
) -> RetainedInference:
    """The engine's answer at one point, its notes issued as they arise."""
    structure = state.design.structure
    count = structure.n_rows
    if not any(nested_in(structure.cancel_codes, c.codes) for c in request.clusters):
        warnings.warn(
            CoverageWarning("coverage.outside_bbp", {"case": CROSSING}), stacklevel=3
        )
    influence = contracted_influence(state, rhs, rows=True, work=work)
    reporting = _reporting(request, applied, order, count, capture)
    answered = work.run(
        "multiway_engine",
        lambda: _computed_inference(
            (request, applied, None, "observed"),
            (structure.n_groups, count, count),
            state.theta,
            influence,
            np.ones(count),
            frequency=False,
            codes=tuple((c.name, c.codes) for c in request.clusters),
            reporting=reporting,
        ),
    )
    return replace(answered, work=work.snapshot())


def influence_frame(
    envelope: RetainedCancelledEnvelope, names: pd.Index[str]
) -> pd.DataFrame:
    """The row influences ``A s_i`` of a certified cancelled fit, by position.

    Parameters
    ----------
    envelope : RetainedCancelledEnvelope
        The retained cancelled fit, certified, with its influence state kept.
    names : pandas.Index
        The names of the reported slopes, in their order.

    Returns
    -------
    pandas.DataFrame
        One row per original row of the fit, indexed by its original position
        (``position``), one column per reported slope, on the reported scale with
        sum normalization, at the certified uncorrected root.  The rows of a
        cancelled group sum to that group's row of the group influences.

    Raises
    ------
    SpecificationError
        ``specification.influence_discarded`` after ``keep_influence=False``.
    ValueError
        The fit is not certified.
    """
    values = envelope.influence(rows=True)
    positions = envelope.geometry.structure.positions
    frame = pd.DataFrame(
        np.array(values, dtype=np.float64, copy=True),
        index=pd.Index(positions.tolist(), name="position"),
        columns=names,
    )
    frame.attrs.update(
        unit="row",
        sampling_unit=envelope.cancelled_name,
        point="certified uncorrected root",
    )
    return frame


__all__ = ["frame_columns", "influence_frame", "multiway_inference"]
