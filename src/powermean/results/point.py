"""The point of a fit with its influence record and default covariance, formed once.

``design/04_inference_and_effects.md``, Sections 1 and 2.3; decisions D49 and D55.
The handoff of a fit (:mod:`powermean.results.handoff`) forms, after its checks, the
point in the coordinates of the solver, the influence record of that point by the
accepted primitives (:mod:`powermean.inference.contributions`: the observed bread,
never the expected information, from the certificate's fresh evaluation; or the
linear candidate with the retained scaled design), the covariance of the point by the
convention for the inference requested at the fit, in the coordinates of the solver
(:class:`~powermean.inference.convention.Covariance`, against which the handoff binds
the standard errors that the certificate's coefficient gate read), and that
covariance reported once (:mod:`powermean.results.covariance`).  Where the certificate
formed no correction, because its equilibrated factorization failed its test of
conditioning, the map ``C^-1`` does not exist and no influence record or covariance is
formed: the point keeps its diagnostics.  Where the linear path formed no
coefficients, there is no point.  On the linear route certified afresh (task M0.22),
the influence record is that of the certificate's fresh evaluation of the original
linear equation (:func:`linear_certified_point`), never the cached residuals of the
candidate.  Nothing here solves, certifies or reselects a row.

A named HC method requested at the fit (``plan/evidence/M1_contract.txt``, rules B1,
B2, B10, B12(a) and Sections 3.1 and 3.2; decisions D57 and D60).  Its covariance at
the point is formed by the accepted entry points of :mod:`powermean.inference.hc`:
:func:`~powermean.inference.hc.joint_hc` of the system that the fit solved at the
certificate's evaluation, which it binds to that system bit for bit, or
:func:`~powermean.inference.hc.linear_hc` of the solve captured before it ran, under
the convention in force at the fit (:func:`fit_convention`: the request's explicit
convention, else the default ``"stata"``, whose factor a named method never takes).
Its record (:func:`powermean.results.covariance.hc_default`) holds the covariance, the
reported one, the direct standard errors and the applied convention, without a row
array, so that ``keep_influence=False`` discards nothing that it keeps.
:func:`fit_preflight` runs the refusals that need no point before the fit solves, and
:func:`joint_provider` and :func:`linear_provider` form the reported-scale provider of
the inference requested at the fit for its certificate: the unchanged robust and
one-way providers, those of :mod:`powermean.inference.hc_provider`, or those of a
typed cluster request.

A typed cluster request at the fit (``{"cluster": "a + b"}`` with ``"multiway"`` and
``"psd"``, or any cluster request with an explicit ``ssc``; contract, rules B1, B3,
B10, B12(a) and Sections 3.1, 3.3 and 6(d), (e); decisions D4, D60 and D65) is
answered by the accepted convention engine (:mod:`powermean.results.cluster_covariance`
over :mod:`powermean.results.retained_inference`): its preflight binds its dimensions
and applies its convention to the counts before the solve, its providers give the
certificate's gate the standard errors of the requested covariance at every candidate
(the final reported matrix's of an adjusted signed request), and its record is formed
once at the certificate's evaluation, with those standard errors (:attr:`Point.errors`)
for the handoff's binding.  The robust covariance with an explicit ``ssc`` (contract,
Sections 3.1 and 6(e)) takes the same engine route
(:func:`~powermean.results.cluster_covariance.engine_request`): its preflight applies
the requested convention to the counts, its providers give every candidate's gate the
standard errors of the robust covariance under that convention, and its record is
formed once and bound in the same way; without an explicit ``ssc`` the robust
covariance keeps the provider and the record of the first milestone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.data.retained import OrderSample
from powermean.data.specs import Inference
from powermean.errors import SpecificationError
from powermean.inference.contributions import (
    Influence,
    joint_influence,
    linear_influence,
)
from powermean.inference.convention import Covariance
from powermean.inference.hc import (
    HCPreflight,
    LinearSolve,
    hc_preflight,
    joint_hc,
    linear_hc,
)
from powermean.inference.hc_provider import HCScale, LinearHCScale
from powermean.inference.linear_provider import (
    LinearCovarianceScale,
    fresh_linear_influence,
)
from powermean.inference.provider import CovarianceScale
from powermean.inference.ssc import ResolvedConvention, resolve_convention
from powermean.model.variance import FloatArray
from powermean.results import covariance
from powermean.results.binding import refuse
from powermean.results.cluster_covariance import (
    ClusterScale,
    LinearClusterScale,
    cluster_preflight,
    engine_request,
    fit_record,
)
from powermean.results.retained import CovarianceRecord
from powermean.results.retained_inference import InferenceRequest
from powermean.results.saturated import SaturatedCovariance, saturated_covariance
from powermean.solve.certificate import Certificate
from powermean.solve.linear import LinearCandidate
from powermean.solve.linear_certificate import LinearCertificate
from powermean.solve.reported import saturated_rows
from powermean.systems.joint import JointSystem


@dataclass(frozen=True, eq=False)
class Point:
    """The point, its influence record and its default covariance, as formed once.

    ``convention`` is the covariance of ``theta`` by the convention, for the inference
    requested at the fit, from which ``default`` is reported; both are ``None`` where
    no influence record could be formed.  ``errors`` are the standard errors of an
    engine request's record (a typed cluster request, or the robust covariance with
    an explicit ``ssc``), against which the handoff binds the certificate's
    (``convention`` is then ``None``: no covariance of the first milestone stands for
    the request); ``None`` otherwise, where the direct map of ``convention`` gives
    them.
    """

    theta: FloatArray | None
    influence: Influence | None
    convention: Covariance | None
    default: CovarianceRecord | SaturatedCovariance | None
    errors: FloatArray | None = None


def frequency_of(sample: OrderSample) -> bool:
    """The stated kind of the weights: frequency, or not (``False`` without weights).

    The accessor of :func:`powermean.results.covariance.frequency_of`, which the
    convention engine reads below this module, kept here for its importers.
    """
    return covariance.frequency_of(sample)


def _saturated_sample(sample: OrderSample) -> bool:
    """Whether the sample is saturated: observations equal to its columns.

    Read only where a request was refused for zero residual degrees of freedom, which
    on a saturated fit is no refusal: its covariance is missing
    (:mod:`powermean.results.saturated`), so no preflight or provider is formed.
    """
    weights = sample.data.weight_values
    return saturated_rows(
        weights[weights > 0], sample.data.nparams, frequency=frequency_of(sample)
    )


def _saturated_refusal(error: SpecificationError, sample: OrderSample) -> bool:
    """Whether ``error`` is the zero-residual-dof refusal of a saturated sample."""
    return error.identifier == "specification.no_residual_dof" and _saturated_sample(
        sample
    )


def _saturated_point(
    certificate: Certificate | LinearCertificate,
    inference: Inference,
    influence: Influence | None,
) -> Point:
    """The point of a saturated certificate: no covariance is formed.

    An accepted certificate gets the missing covariance of its request, with the note
    issued once, at the fit (:mod:`powermean.results.saturated`); ``N`` equals the
    coefficients by the certificate's decision.  A certificate that is not accepted
    gets none and issues nothing: its inference is withheld.
    """
    theta = certificate.gamma
    if not certificate.accepted:
        return Point(theta, influence, None, None)
    clusters = inference.clusters
    cluster = clusters.dimensions if clusters is not None else inference.cluster
    method = inference.method or ("robust" if cluster is None else "cluster")
    default = saturated_covariance(
        method, cluster, inference.bread, theta.size, theta.size, stacklevel=4
    )
    return Point(theta, influence, None, default)


def _typed_point(
    theta: FloatArray, sample: OrderSample, inference: Inference, influence: Influence
) -> Point:
    """The point of an engine request, its record formed once at the point."""
    default = fit_record(sample, inference, influence)
    return Point(theta, influence, None, default, default.standard_errors)


# ------------------------------------------------- the inference requested at the fit


def fit_convention(inference: Inference) -> ResolvedConvention:
    """The convention in force of a request at its fit: explicit, else ``"stata"``.

    At the fit nothing is inherited (contract, Section 3.1): the request's explicit
    ``ssc``, a preset put in force by its name or a custom record, else the default.
    """
    return resolve_convention(inference.ssc)


def fit_preflight(sample: OrderSample, inference: Inference) -> HCPreflight | None:
    """The refusals of a named HC request that need no point, before the fit solves.

    The accepted preflight (:func:`powermean.inference.hc.hc_preflight`) on the
    sample's scaled design and raw weights, every retained row in its order (a row of
    zero weight is no replica), under the convention in force at the fit:
    ``specification.no_residual_dof`` for HC1 at ``N = K`` and
    ``specification.conflicting`` for an explicit second correction (rule B12(a)); the
    kernel's ``ValueError`` for a design that is not of full column rank on its
    active rows, whose leverage is undefined.  ``None`` for the robust and cluster
    covariances; an engine request (a typed cluster request, or the robust covariance
    with an explicit ``ssc``) runs its own refusals here, before the solve
    (:func:`~powermean.results.cluster_covariance.cluster_preflight`: a dimension
    that was not retained or has a missing value, fewer than two clusters among the
    active rows, a convention without positive degrees of freedom), and keeps nothing.
    A refusal for zero residual degrees of freedom of a saturated sample is no
    refusal: ``None``, the fit's covariance is missing (:mod:`powermean.results.
    saturated`).
    """
    try:
        return _fit_preflight(sample, inference)
    except SpecificationError as error:
        if _saturated_refusal(error, sample):
            return None  # a saturated fit forms no covariance (results.saturated)
        raise


def _fit_preflight(sample: OrderSample, inference: Inference) -> HCPreflight | None:
    """:func:`fit_preflight` without the saturated exception."""
    method = inference.method
    if method is None:
        if engine_request(inference):
            cluster_preflight(sample, inference)
        elif inference.cluster is not None:
            codes = covariance.cluster_codes(sample, inference.cluster)
            active = sample.data.weight_values > 0
            if int(np.unique(codes.codes[active]).size) < 2:
                raise SpecificationError(
                    "specification.invalid_value",
                    {
                        "argument": "cluster",
                        "value": inference.cluster,
                        "valid": "at least two clusters on the estimation sample",
                    },
                )
        return None
    return hc_preflight(
        sample.design.matrix,
        sample.data.weight_values,
        method,
        fit_convention(inference),
        frequency=frequency_of(sample),
    )


def _engine_preflight(
    sample: OrderSample, inference: Inference
) -> InferenceRequest | None:
    """The engine request's preflight; ``None`` for a saturated fit's dof refusal."""
    try:
        return cluster_preflight(sample, inference)
    except SpecificationError as error:
        if _saturated_refusal(error, sample):
            return None
        raise


def joint_provider(
    sample: OrderSample,
    inference: Inference,
    system: JointSystem,
    *,
    preflight: HCPreflight | None = None,
) -> CovarianceScale | HCScale | ClusterScale | None:
    """The reported-scale provider of the inference requested at a joint fit.

    The robust or one-way provider of the first milestone, unchanged, the named HC
    provider on the active rows of ``system`` with the request's preflight (formed
    here where none is given), or the provider of an engine request (a typed cluster
    request, its dimensions bound to the sample's rows, or the robust covariance with
    an explicit ``ssc``); the direct map of the sample's order scale every way.
    ``None`` where a saturated fit's request has no residual degrees of freedom: its
    certificate calls no provider (:func:`powermean.solve.reported.saturated`).
    """
    frequency = frequency_of(sample)
    method = inference.method
    if method is None:
        if engine_request(inference):
            engine = _engine_preflight(sample, inference)
            return None if engine is None else ClusterScale(sample, engine)
        cluster = None
        if inference.cluster is not None:
            cluster = covariance.cluster_codes(sample, inference.cluster)
        return CovarianceScale(
            frequency=frequency, cluster=cluster, order=sample.order_scale
        )
    found = fit_preflight(sample, inference) if preflight is None else preflight
    if found is None and _saturated_sample(sample):
        return None  # a saturated fit calls no provider (solve.reported.saturated)
    if found is None or found.method != method:
        raise ValueError("the preflight is not that of the request")
    return HCScale(
        found.applied,
        system.active_design,
        system.active_weights,
        frequency,
        sample.order_scale,
        inference.bread,
    )


def linear_provider(
    sample: OrderSample, inference: Inference, solved: LinearSolve
) -> LinearCovarianceScale | LinearHCScale | LinearClusterScale | None:
    """The reported-scale provider of the inference requested at a log OLS fit.

    The robust or one-way provider of the first milestone, unchanged, the named HC
    provider of the solve captured before it ran, or the provider of an engine request
    (a typed cluster request, its dimensions bound to the sample's rows, or the robust
    covariance with an explicit ``ssc``); the direct map of the sample's order scale
    every way.
    """
    method = inference.method
    if method is None:
        if engine_request(inference):
            engine = _engine_preflight(sample, inference)
            return None if engine is None else LinearClusterScale(sample, engine)
        cluster = None
        if inference.cluster is not None:
            cluster = covariance.cluster_codes(sample, inference.cluster)
        return LinearCovarianceScale(
            frequency=frequency_of(sample), cluster=cluster, order=sample.order_scale
        )
    return LinearHCScale(solved, method, fit_convention(inference), sample.order_scale)


# ------------------------------------------------------------------- the point


def default_covariance(
    sample: OrderSample, inference: Inference, influence: Influence | None
) -> tuple[Covariance | None, CovarianceRecord | None]:
    """The covariance requested at the fit, by the convention and reported once.

    Parameters
    ----------
    sample : OrderSample
        The sample of the fit, for its retained classifications and order scale.
    inference : Inference
        The inference requested at the fit, robust or one cluster dimension.
    influence : Influence or None
        The influence record of the point, or ``None`` where none could be formed.

    Returns
    -------
    tuple
        The convention covariance in the coordinates of the solver and its record on
        the reported scale; ``(None, None)`` without an influence record, for a named
        HC request, whose covariance needs its owner of the rows, and for an engine
        request (a typed cluster request, or the robust covariance with an explicit
        ``ssc``), whose record the convention engine forms at a certified point only
        (:func:`power_point`, :func:`linear_certified_point`).
    """
    if influence is None or inference.method is not None or engine_request(inference):
        return None, None
    cluster = None
    if inference.cluster is not None:
        cluster = covariance.cluster_codes(sample, inference.cluster)
    convention = covariance.convention_covariance(influence, cluster)
    name = None if cluster is None else cluster.name
    return convention, covariance.record(convention, sample.order_scale, name)


def _saturated_influence(
    certificate: Certificate, sample: OrderSample
) -> Influence | None:
    """The influence record of a saturated certificate, where it is representable.

    ``moment_influence_map`` refuses a map with a value beyond the doubles by its
    finiteness check (``ValueError``); on a saturated fit no covariance reads the
    map, so that refusal leaves the record absent instead of ending the fit.
    """
    evaluation = certificate.evaluation
    if not certificate.conditioned or evaluation is None:
        return None
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        try:
            return joint_influence(evaluation, frequency=frequency_of(sample))
        except ValueError:
            return None


def power_point(
    sample: OrderSample,
    inference: Inference,
    certificate: Certificate,
    system: JointSystem,
) -> Point:
    """The point of a certificate, with its influence record where one can be formed.

    For a named HC request, its covariance at the certificate's evaluation by
    :func:`~powermean.inference.hc.joint_hc` of ``system``, the retained equation that
    the fit solved, which binds the evaluation to it bit for bit (module docstring);
    for an engine request, the record of the convention engine at the influence record
    of that evaluation.
    """
    evaluation = certificate.evaluation
    if certificate.saturated:
        # A saturated fit stands with every covariance missing (4d02929), whatever its
        # influence map; a map that float64 cannot represent (a subnormal curvature,
        # 1/J beyond the doubles) is then not served (owner's ruling 103).
        return _saturated_point(
            certificate, inference, _saturated_influence(certificate, sample)
        )
    influence = None
    if certificate.conditioned and evaluation is not None:
        influence = joint_influence(evaluation, frequency=frequency_of(sample))
    if engine_request(inference) and influence is not None:
        return _typed_point(certificate.gamma, sample, inference, influence)
    method = inference.method
    if method is None or influence is None or evaluation is None:
        convention, default = default_covariance(sample, inference, influence)
        return Point(certificate.gamma, influence, convention, default)
    found = joint_hc(
        system,
        evaluation,
        method,
        fit_convention(inference),
        frequency=frequency_of(sample),
        bread=inference.bread,
    )
    convention, default = covariance.hc_default(
        found, sample.order_scale, inference.bread
    )
    return Point(certificate.gamma, influence, convention, default)


def linear_point(
    sample: OrderSample, inference: Inference, linear: LinearCandidate
) -> Point:
    """The point of a bare linear candidate, where one was formed.

    The components of a candidate handed over without its fresh certificate (task
    M0.21): never served as inference, which the result refuses on this route
    without a certificate; a named HC request forms no covariance for it.
    """
    if linear.coefficients is None:
        return Point(None, None, None, None)
    influence = linear_influence(
        linear, sample.design.matrix, frequency=frequency_of(sample)
    )
    convention, default = default_covariance(sample, inference, influence)
    return Point(linear.coefficients, influence, convention, default)


def linear_certified_point(
    sample: OrderSample,
    inference: Inference,
    certificate: LinearCertificate,
    solved: LinearSolve | None,
) -> Point:
    """The point of a fresh linear certificate, with its influence record.

    The influence record comes from the certificate's fresh evaluation of the
    original linear equation (:func:`~powermean.inference.linear_provider.
    fresh_linear_influence`), never from the residuals that the least-squares
    candidate cached; where the certificate formed no correction, because its
    equilibrated factorization failed its test of conditioning, no influence record
    or covariance is formed and the point keeps its diagnostics.  For a named HC
    request, its covariance by :func:`~powermean.inference.hc.linear_hc` of
    ``solved``, the solve captured with the retained equation, whose candidate the
    handoff has bound to the certificate's point; without it the handoff is refused.
    For an engine request, the record of the convention engine at the influence record
    of the fresh evaluation.
    """
    evaluation = certificate.evaluation
    influence: Influence | None = None
    if evaluation is not None and certificate.conditioned:
        influence = fresh_linear_influence(evaluation, frequency=frequency_of(sample))
    if certificate.saturated:
        return _saturated_point(certificate, inference, influence)
    if engine_request(inference) and influence is not None:
        return _typed_point(certificate.gamma, sample, inference, influence)
    method = inference.method
    if method is None or influence is None:
        convention, default = default_covariance(sample, inference, influence)
        return Point(certificate.gamma, influence, convention, default)
    if solved is None:
        raise refuse("a named HC covariance of the linear path needs its solve")
    found = linear_hc(solved, method, fit_convention(inference))
    convention, default = covariance.hc_default(
        found, sample.order_scale, inference.bread
    )
    return Point(certificate.gamma, influence, convention, default)
