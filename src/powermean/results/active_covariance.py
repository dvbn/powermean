"""The active covariance of a result without fixed effects: requests, cache and views.

``design/04_inference_and_effects.md``, Sections 1, 2.3, 3 and 5 (``res.vcov(...)``
"return a new covariance from the stored contributions; ``res.with_vcov(...)`` returns
a result object with it.  No refit."); ``plan/evidence/M1_contract.txt``, rules B1,
B3 and B9 to B12 and Sections 3.1, 3.2 and 6(a), (e) and (j); decisions D41, D49, D54,
D57, D60, D65 and D78.  A request on a certified fit without fixed effects
(:func:`answer`) is served in this order, and nothing of it solves, reselects a row,
replays a formula or changes the retained state:

1. a reserved HAC or Conley request is refused by the explicit later-release refusal
   of decision D78 (:func:`powermean.results.retained_hc.deferred_vcov`), before
   anything is read;
2. the request is parsed (:func:`powermean.results.covariance.request`): ``"robust"``,
   one retained cluster, several retained dimensions (``{"cluster": "a + b"}`` with
   ``"multiway"`` and ``"psd"``) or the exact string of a named HC method, with the
   explicit ``ssc`` and ``bread`` of its call;
3. the covariance requested at the fit answers its own request from its record
   (rule B10), before and after ``keep_influence=False``: the same kind, the same
   typed cluster request and the record's convention (omitted, or the one the fit
   requested);
4. any other named HC method is formed by the accepted retained adapter
   (:func:`powermean.results.retained_hc.retained_hc`) at the certified point, from
   the retained equation (the solve captured before it ran, or the joint system at the
   certificate's evaluation, each bound again bit for bit), under the convention in
   force: the request's explicit ``ssc``, else the fit's, inherited, which never
   multiplies a named method nor conflicts with it and whose reference survives
   (contract, Section 3.1); the request needs the retained influence
   (``specification.influence_discarded`` after ``keep_influence=False``);
5. a robust or one-way cluster request after a named HC fit, or after a fit of a
   typed cluster request, takes the convention in force, including its factor and
   reference (D65, Section 6(e)), through the existing
   :func:`powermean.results.retained_inference.retained_nofe_inference` adapter.
   An explicit convention on another post-fit robust or one-way request, and every
   typed cluster request (several dimensions, a multiway form or a ``"psd"`` key),
   take that same route: the dimensions bound to the retained classifications, the
   port's raw estimate, the factor once, decision D4's frame and sequence where the
   request is signed and its treatment ``"cgm"``, and the record of
   :func:`powermean.results.cluster_covariance.answer_record`, which keeps the typed
   answer.  Such a request needs the retained influence
   (``specification.influence_discarded`` after ``keep_influence=False``), unless it
   is the fit's own (step 3).  The omitted-convention first-milestone bridge keeps its
   unchanged :func:`powermean.results.covariance.compute` arithmetic.

Views (:func:`view_provenance`).  ``with_vcov`` binds an answer as the active
covariance of a new result over the same retained fit: its standard errors, tables,
summary, effects, contrasts and tests read that covariance, with its reference, while
the coefficients, the point, the status and the certificate are the fit's.  The
certificate still describes the covariance requested at the fit: nothing is certified
again, and no root is refused or replaced because a later request is unavailable.  A
view's provenance records its covariance and the degeneracy of its standard errors
(decision D49).

Reference (:func:`t_columns`).  The normal reference of the first milestone keeps its
own arithmetic in the result; a ``t`` reference, which a named method takes from a
convention in force such as ``"regression"`` (``N - K_reference`` degrees of freedom),
gives its critical value and two-sided p-values by the accepted
:func:`powermean.effects.targets.reference_values`, coefficient by coefficient.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np

from powermean.data.request_values import OMITTED, given
from powermean.effects.targets import reference_values
from powermean.errors import SpecificationError
from powermean.inference.convention import ClusterCodes
from powermean.inference.hc import hc_preflight
from powermean.inference.ssc import ReferenceDistribution, resolve_convention
from powermean.model.variance import FloatArray
from powermean.results import covariance
from powermean.results.cluster_covariance import (
    answer_record,
    bound_request,
    cluster_label,
)
from powermean.results.provenance import Provenance, _frozen
from powermean.results.retained import CovarianceRecord, RetainedFit
from powermean.results.retained_hc import deferred_vcov, retained_hc
from powermean.results.retained_hc_records import HCInferenceRequest
from powermean.results.retained_inference import (
    InferenceRequest,
    retained_nofe_inference,
)
from powermean.results.saturated import SaturatedCovariance, saturated_covariance

OTHER = "a covariance other than the fit's"


def _named(retained: RetainedFit, found: covariance.Request) -> CovarianceRecord:
    """A named HC covariance of the retained state, by the accepted adapter."""
    method = found.method
    if method is None:
        raise ValueError("a named request names its method")
    answered = retained_hc(retained, HCInferenceRequest(method, found.ssc, found.bread))
    return covariance.hc_record(
        answered.hc,
        answered.require_reported_covariance(),
        answered.standard_errors,
        found.bread,
    )


def _conventional(retained: RetainedFit, found: covariance.Request) -> CovarianceRecord:
    """The existing convention engine, with its factor and reference once.

    A robust request, one retained cluster, or a typed cluster request whose
    dimensions are bound to the retained classifications in its order, repetitions
    kept, with its form and its ``"psd"`` key; the record keeps the typed answer.
    """
    if found.kind not in ("robust", "cluster"):
        raise ValueError("a conventional request is robust or clustered")
    label = found.cluster
    if found.clusters is not None:
        request = bound_request(retained.sample, found.clusters, ssc=found.ssc)
        label = cluster_label(found.clusters.dimensions)
    else:
        codes: tuple[ClusterCodes, ...] = ()
        if found.cluster is not None:
            codes = (covariance.cluster_codes(retained.sample, found.cluster),)
        request = InferenceRequest(found.kind, codes, found.ssc, found.bread)
    answered = retained_nofe_inference(retained, request)
    # Only the ordinary covariance path in answer calls this adapter.
    return answer_record(
        answered, label, cast("CovarianceRecord | None", retained.default)
    )


def _saturated(
    retained: RetainedFit, default: SaturatedCovariance, found: covariance.Request
) -> SaturatedCovariance:
    """A request on a saturated fit: validated as on any fit, answered as missing.

    The classifications are bound as their routes bind them, and a named method's
    convention is applied by its preflight, so that a classification that was not
    retained or a second correction beside the method is refused as on any fit; the
    refusal for zero residual degrees of freedom is the saturated answer itself
    (:mod:`powermean.results.saturated`).  No covariance is formed.
    """
    sample = retained.sample
    cluster: str | tuple[str, ...] | None = found.cluster
    if found.method is not None:
        try:
            hc_preflight(
                sample.design.matrix,
                sample.data.weight_values,
                found.method,
                resolve_convention(found.ssc),
                frequency=covariance.frequency_of(sample),
            )
        except SpecificationError as error:
            if error.identifier != "specification.no_residual_dof":
                raise
    elif found.clusters is not None:
        bound_request(sample, found.clusters, ssc=found.ssc)
        cluster = found.clusters.dimensions
    elif found.cluster is not None:
        covariance.cluster_codes(sample, found.cluster)
    return saturated_covariance(
        found.method or found.kind,
        cluster,
        found.bread,
        default.count,
        default.standard_errors.size,
        stacklevel=5,
    )


def _recomputed(retained: RetainedFit, found: covariance.Request) -> CovarianceRecord:
    """A robust or cluster covariance formed again from the retained influence."""
    influence = retained.influence
    if influence is None:
        raise SpecificationError(
            "specification.influence_discarded", {"request": OTHER}
        )
    codes = None
    if found.cluster is not None:
        codes = covariance.cluster_codes(retained.sample, found.cluster)
    return covariance.compute(influence, retained.sample.order_scale, codes)


def answer(
    retained: RetainedFit,
    default: CovarianceRecord | SaturatedCovariance,
    spec: object,
    *,
    ssc: object = OMITTED,
    bread: object = OMITTED,
) -> CovarianceRecord | SaturatedCovariance:
    """The covariance of a request on a certified fit without fixed effects.

    Parameters
    ----------
    retained : RetainedFit
        The retained state of the fit, whose point is certified.
    default : CovarianceRecord
        The covariance requested at the fit, its record.
    spec : object
        ``None`` for the fit's own request, ``"robust"``, ``{"cluster": name}``,
        ``{"cluster": "a + b"}`` with the optional keys ``"multiway"`` and ``"psd"``,
        the exact string of a named HC method, or an inference record.
    ssc, bread : object, optional
        The explicit convention and bread of the request (module docstring).

    Returns
    -------
    CovarianceRecord or SaturatedCovariance
        The fit's own record, or the covariance formed from the retained state, each
        map applied once; on a saturated fit, a missing covariance with its note for
        every request that names one (:func:`_saturated`), the fit's own without.

    Raises
    ------
    SpecificationError
        The refusals of the parser and of D78 (``specification.unimplemented``,
        ``specification.invalid_value``, ``specification.unknown_option``); of the
        retained adapter (``specification.influence_discarded``,
        ``specification.no_residual_dof``, ``specification.conflicting``);
        a cluster that was not retained or has a missing value.
    ValueError, FloatingPointError
        The refusals of the retained adapter: records that are not bound, a design
        that is not of full column rank, a value that is not representable.
    """
    deferred_vcov(spec)
    found = covariance.request(spec, retained.inference, ssc=ssc, bread=bread)
    if type(default) is SaturatedCovariance:
        if spec is None and not given(ssc) and not given(bread):
            return default
        return _saturated(retained, default, found)
    default = cast("CovarianceRecord", default)
    if found.matches(default):
        return default
    if found.method is not None:
        return _named(retained, found)
    if default.applied is not None or given(found.ssc) or found.clusters is not None:
        return _conventional(retained, found)
    return _recomputed(retained, found)


def fe_keywords(ssc: object, bread: object) -> None:
    """Refuse the keywords ``ssc`` and ``bread`` on a result with fixed effects.

    The private fixed-effects preview carries a request's convention and bread in its
    ``vcov`` mapping, which these keywords do not reach; omitted, they change nothing.
    """
    if given(ssc) or given(bread):
        raise SpecificationError(
            "specification.unsupported_inference",
            {
                "method": "Passing ssc= or bread= as a keyword",
                "case": "results with fixed effects, whose vcov mapping carries them",
            },
        )


def view_provenance(
    base: Provenance, record: CovarianceRecord | SaturatedCovariance
) -> Provenance:
    """The provenance of a view: the fit's, with its covariance and degeneracy."""
    return replace(
        base, covariance=_frozen(record.to_dict()), degenerate=record.degenerate
    )


def t_columns(
    reference: ReferenceDistribution, level: float, statistic: FloatArray
) -> tuple[float, FloatArray]:
    """The critical value and two-sided p-values of a ``t`` reference.

    Parameters
    ----------
    reference : ReferenceDistribution
        A ``t`` reference with its positive degrees of freedom.
    level : float
        The confidence level, strictly between zero and one.
    statistic : array
        The ratios of the coefficients, NaN where undefined.

    Returns
    -------
    tuple
        The critical value, and one p-value per ratio, NaN where it is undefined.
    """
    if reference.kind != "t":
        raise ValueError("the normal reference keeps the result's own arithmetic")
    quantile = reference_values(reference, level, 0.0)[0]
    p_values = np.array(
        [reference_values(reference, level, value)[1] for value in statistic.tolist()],
        dtype=np.float64,
    )
    return quantile, p_values


__all__ = ["answer", "fe_keywords", "t_columns", "view_provenance"]
