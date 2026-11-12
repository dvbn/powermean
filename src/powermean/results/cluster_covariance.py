"""Cluster and multiway covariances of a fit without fixed effects, at and after it.

The typed cluster requests of a fit without fixed effects (``{"cluster": "a + b"}`` with
the keys ``"multiway"`` and ``"psd"``, and every cluster request with an explicit
``ssc``) are answered by one engine, the convention composition of
:mod:`powermean.results.retained_inference`: the raw port of
:mod:`powermean.inference.multiway` with every factor off, the convention's factors once
by their owner (:mod:`powermean.inference.ssc`), the reporting map once and the frame
and sequence of the ``psd`` adjustment (:mod:`powermean.inference.psd`).  Nothing here
forms a covariance, a factor or an adjustment of its own.  The robust covariance with an
explicit ``ssc`` is answered by the same engine, at the fit and after it, as its request
without dimensions: the primitive ``HC0`` estimate of the active rows, the convention's
robust factor once, its reference and the reporting map once, with no ``psd`` step
(:func:`engine_request`).  The robust covariance without an explicit ``ssc`` keeps its
default provider and record.

Binding (:func:`bound_request`).  The dimensions of the typed request are bound to the
sample's final rows by the retained classifications
(:func:`powermean.results.covariance.cluster_codes`: the exact labels numbered by first
appearance, never converted to doubles; a dimension that was not retained, or that has
a missing value on the estimation sample, is refused and no row is dropped), in the
request's order, repetitions kept, with the request's explicit convention and the
observed bread.

At the fit.  :func:`cluster_preflight` binds the request of the fit and applies its
convention to the counts of the sample's active rows before anything is solved, so that
a dimension that was not retained, a dimension with fewer than two clusters among the
active rows and a convention whose degrees of freedom are not positive are refused at no
cost of a solve.  The certificate's coefficient gate reads the standard errors of the
requested covariance itself at every candidate that it assesses (:class:`ClusterScale`
on the joint system, :class:`LinearClusterScale` on the least-squares equation): the
influence record of the fresh evaluation (the observed bread), the engine at that
candidate, its standard errors, which for a signed request whose ``psd`` treatment is
``"cgm"`` are those of the final reported matrix; the adjustment's notes are withheld
there.  No other covariance, the default robust one included, stands in for the request,
and a request whose standard error is missing at the candidate fails the gate.  The
handoff forms the record of the fit once from the certificate's evaluation
(:func:`fit_record`), with its notes, and binds the certificate's standard errors to
that record's, bit for bit.

The record (:func:`answer_record`).  The answers at the fit and after it share one
record: the covariance of ``theta`` with the factor applied once, the reported matrix
(the adjustment's final matrix where it ran), its standard errors, the convention in
force by its preset name or its custom record (which a later request inherits), its
counts with ``G_min``, the rank and the diagnostics of the raw estimate (the
adjustment's notes where it was active, the raw diagnostics staying in the typed
answer), and the typed answer itself: the bound request, the components with their
signs, cluster counts and factors, the raw estimate, the ``psd`` treatment and, where it
ran, its adjustment and frame records.  No row array is kept, so
``keep_influence=False`` discards the rows while the record answers its own request.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.data.inference_conventions import ConventionValues
from powermean.data.inference_requests import ClusterRequest
from powermean.data.request_values import OMITTED, Omitted, given
from powermean.data.retained import OrderSample
from powermean.data.specs import Inference
from powermean.inference.contributions import Influence, joint_influence
from powermean.inference.linear_provider import fresh_linear_influence
from powermean.inference.multiway import MultiwayCovariance
from powermean.results.covariance import cluster_codes, frequency_of
from powermean.results.retained import Convention, CovarianceRecord
from powermean.results.retained_inference import (
    InferenceRequest,
    RetainedInference,
    candidate_nofe_inference,
    nofe_convention,
)
from powermean.solve.linear_evaluation import LinearEvaluation
from powermean.solve.reported import ReportedScale
from powermean.systems.joint import JointEvaluation, SamplingUnits

# The notes that describe a reported matrix that the ``psd`` adjustment replaced.
ADJUSTED = "covariance.psd_adjusted"
INDEFINITE = "covariance.indefinite"


def cluster_label(dimensions: tuple[str, ...]) -> str:
    """The cluster of a record: the dimensions as the public spelling joins them."""
    return " + ".join(dimensions)


def bound_request(
    sample: OrderSample,
    clusters: ClusterRequest,
    *,
    ssc: str | ConventionValues | Omitted = OMITTED,
) -> InferenceRequest:
    """The engine's request of a typed cluster request, bound to the sample's rows.

    Parameters
    ----------
    sample : OrderSample
        The sample of the fit, whose retained classifications name the dimensions.
    clusters : ClusterRequest
        The typed request: its dimensions in order, repetitions kept, its form and
        its ``"psd"`` key as given.
    ssc : str, ConventionValues or OMITTED, optional
        The explicit convention of the request; omitted, the convention in force
        decides (the default at the fit, the fit's after it).

    Returns
    -------
    InferenceRequest
        The request with one bound dimension per requested one and the observed bread.

    Raises
    ------
    SpecificationError
        ``specification.cluster_not_retained`` or
        ``specification.cluster_missing_values`` for a dimension.
    """
    codes = tuple(cluster_codes(sample, name) for name in clusters.dimensions)
    return InferenceRequest.from_cluster_request(clusters, codes, ssc=ssc)


def engine_request(inference: Inference) -> bool:
    """Whether the request of a fit is answered by the engine (module docstring).

    A typed cluster request, or the robust covariance with an explicit ``ssc``; the
    robust covariance without one, the single cluster variable and a named HC method
    keep their own providers and records.
    """
    if inference.clusters is not None:
        return True
    return inference.covariance == "robust" and given(inference.ssc)


def fit_request(sample: OrderSample, inference: Inference) -> InferenceRequest:
    """The engine's request of a fit (:func:`engine_request`), bound to its rows."""
    clusters = inference.clusters
    if inference.covariance == "cluster" and clusters is not None:
        return bound_request(sample, clusters, ssc=inference.ssc)
    if not engine_request(inference):
        raise ValueError("the request of the fit is not answered by the engine")
    return InferenceRequest("robust", ssc=inference.ssc, bread=inference.bread)


def cluster_preflight(sample: OrderSample, inference: Inference) -> InferenceRequest:
    """The refusals of an engine request that need no point, before the solve.

    The binding of every dimension (:func:`bound_request`), then the convention applied
    to the counts of the sample's active rows, the rows of positive weight that every
    evaluation of the fit holds (:func:`~powermean.results.retained_inference.
    nofe_convention`): fewer than two clusters among them (``ValueError``), fewer than
    two observations under ``N/(N-1)`` (``ValueError``, as the convention
    refuses them) or a convention whose degrees of freedom are not positive
    (``specification.no_residual_dof``) are refused at no cost of a solve.

    Returns
    -------
    InferenceRequest
        The bound request; nothing else is kept.
    """
    request = fit_request(sample, inference)
    weights = sample.data.weight_values
    active = weights > 0
    units = SamplingUnits(
        np.flatnonzero(active), np.flatnonzero(~active), int(active.size)
    )
    nofe_convention(sample, units, weights[active], request)
    return request


def _scale(
    sample: OrderSample, request: InferenceRequest, influence: Influence
) -> ReportedScale:
    """The reported scale of the request at the candidate of ``influence``."""
    answered = candidate_nofe_inference(sample, influence, request, notes=False)
    return ReportedScale(
        gamma=influence.gamma,
        standard_errors=answered.standard_errors,
        order=sample.order_scale,
    )


@dataclass(frozen=True, eq=False)
class ClusterScale:
    """The reported scale at a joint evaluation, by an engine request.

    Parameters
    ----------
    sample : OrderSample
        The sample of the fit, whose rows the request is bound to.
    request : InferenceRequest
        The bound request of the fit (:func:`cluster_preflight`).

    Notes
    -----
    An instance is a :class:`~powermean.solve.reported.ReportedScaleProvider`: each
    call forms the influence record of the evaluation that it receives (the observed
    bread) and answers the request at that candidate, keeping nothing between calls;
    the direct map of the sample's order scale reports its corrections.
    """

    sample: OrderSample
    request: InferenceRequest

    def __call__(self, evaluation: JointEvaluation, /) -> ReportedScale:
        """The record at ``evaluation.gamma``, from that evaluation alone."""
        influence = joint_influence(evaluation, frequency=frequency_of(self.sample))
        return _scale(self.sample, self.request, influence)


@dataclass(frozen=True, eq=False)
class LinearClusterScale:
    """The reported scale at a linear evaluation, by an engine request.

    Parameters
    ----------
    sample : OrderSample
        The sample of the log OLS fit, whose rows the request is bound to.
    request : InferenceRequest
        The bound request of the fit (:func:`cluster_preflight`).

    Notes
    -----
    An instance is a :class:`~powermean.solve.linear_certificate.
    LinearScaleProvider`: each call forms the influence record of the fresh linear
    evaluation that it receives (:func:`~powermean.inference.linear_provider.
    fresh_linear_influence`) and answers the request at that point, keeping nothing
    between calls.
    """

    sample: OrderSample
    request: InferenceRequest

    def __call__(self, evaluation: LinearEvaluation, /) -> ReportedScale:
        """The record at ``evaluation.gamma``, from that evaluation alone."""
        influence = fresh_linear_influence(
            evaluation, frequency=frequency_of(self.sample)
        )
        return _scale(self.sample, self.request, influence)


def _diagnostics(
    answered: RetainedInference, multiway: MultiwayCovariance
) -> tuple[str, ...]:
    """The diagnostics of a multiway record: the adjustment's notes where it ran, else
    raw.

    Where the adjustment was active, the adjustment's notes describe the reported matrix
    (the port's raw diagnostics describe the raw estimate and stay in the answer).
    """
    adjustment = answered.adjustment
    if adjustment is not None and adjustment.active:
        return (ADJUSTED, INDEFINITE) if adjustment.final_indefinite else (ADJUSTED,)
    return tuple(multiway.estimate.diagnostics)


def _factor(answered: RetainedInference) -> float | tuple[float, ...]:
    """The convention's one factor, or that of each multiway component."""
    factor = answered.applied.factor
    if factor is not None:
        return factor.value
    multiway = answered.multiway
    if multiway is None:
        raise ValueError("a one-way or robust covariance has one convention factor")
    return tuple(component.factor.value for component in multiway.components)


def answer_record(
    answered: RetainedInference,
    cluster: str | None,
    default: CovarianceRecord | None = None,
) -> CovarianceRecord:
    """The record of one answer of the convention engine (module docstring).

    Parameters
    ----------
    answered : RetainedInference
        The engine's answer, computed or served from the initial cache.
    cluster : str or None
        The cluster of the record: the one name, or the dimensions joined by
        ``" + "`` (:func:`cluster_label`); ``None`` for a robust request.
    default : CovarianceRecord, optional
        The record of the fit, whose rank and diagnostics an answer of the initial
        cache keeps.

    Returns
    -------
    CovarianceRecord
        The covariance in both coordinate systems with its factor, counts, rank,
        diagnostics, convention, reference and applied convention, and the typed
        answer.

    Raises
    ------
    FloatingPointError
        The reported covariance is not representable (its range error).
    """
    applied, multiway = answered.applied, answered.multiway
    if multiway is not None:
        rank, diagnostics = multiway.estimate.rank, _diagnostics(answered, multiway)
    elif answered.covariance is not None:
        estimate = answered.covariance.estimate
        rank, diagnostics = estimate.rank, tuple(estimate.diagnostics)
    elif default is not None:
        rank, diagnostics = default.rank, default.diagnostics
    else:
        raise ValueError("an initial-cache answer requires the initial record")
    resolved = applied.convention
    name = resolved.preset
    convention: Convention = (
        resolved.values if resolved.origin == "custom" or name is None else name
    )
    return CovarianceRecord(
        kind=answered.request.kind,
        cluster=cluster,
        theta=answered.theta_covariance,
        reported=answered.require_reported_covariance(),
        standard_errors=answered.standard_errors,
        factor=_factor(answered),
        count=answered.count,
        clusters=applied.counts.g_min,
        rank=rank,
        diagnostics=diagnostics,
        bread=answered.bread,
        convention=convention,
        reference=applied.reference.kind,
        applied=applied,
        answer=answered,
    )


def fit_record(
    sample: OrderSample, inference: Inference, influence: Influence
) -> CovarianceRecord:
    """The record of the engine request of a fit at its certified point.

    Parameters
    ----------
    sample : OrderSample
        The sample of the fit.
    inference : Inference
        The typed cluster request of the fit, or its robust request with an explicit
        ``ssc`` (:func:`engine_request`).
    influence : Influence
        The influence record of the certificate's fresh evaluation at the point.

    Returns
    -------
    CovarianceRecord
        The record of :func:`answer_record`, formed once, with the adjustment's notes;
        a robust record names no cluster.
    """
    request = fit_request(sample, inference)
    answered = candidate_nofe_inference(sample, influence, request)
    dimensions = inference.dimensions
    return answer_record(answered, cluster_label(dimensions) if dimensions else None)


__all__ = [
    "ClusterScale",
    "LinearClusterScale",
    "answer_record",
    "bound_request",
    "cluster_label",
    "cluster_preflight",
    "engine_request",
    "fit_record",
    "fit_request",
]
