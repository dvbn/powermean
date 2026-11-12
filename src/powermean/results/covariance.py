"""Covariances of the fitted point from its retained primitives, on the reported scale.

Covariances are recomputed from the stored contributions, with no refit, and mapped to
the reported scale.

A covariance is formed again from the retained :class:`~powermean.inference.
contributions.Influence` of the point, the primitive scores ``s_i``, the actual sum
Jacobian ``-C`` and its map ``A = C^-1``, by the ``"stata"`` convention
(:mod:`powermean.inference.convention`): the robust estimator with ``N/(N - 1)``, or one
cluster dimension with ``G/(G - 1)``, the observed bread, the normal reference,
frequency weights as replicated rows.  Nothing is solved, the point is unchanged, no row
is reselected and no factor is applied twice.

Reported scale (:func:`report`).  ``V`` of the solver's coordinates ``theta`` is mapped
by :class:`~powermean.model.order_scale.OrderScale`, entry by entry with exact
exponents: ``V_jk / (d_j d_k) / omega^2``, and the standard errors ``sqrt(V_jj)`` by ``1
/ (d_j |omega|)``.  This is the direct map, not the dense product ``M V M'`` of a
materialized transform: where the column scales differ by hundreds of octaves the
product forms an intermediate ``M V`` whose entry underflows while the result is
representable, and the symmetric part then reports half of an off-diagonal entry (the
witness of ``omega = 1``, ``d = (2^600, 2^-600)``, ``V = [[2^900, 2^-600], [2^-600,
2^-1000]]``, whose reported covariance is ``[[2^-300, 2^-600], [2^-600, 2^200]]``).  A
reported entry that is not representable raises ``FloatingPointError``: nothing is
floored, and a genuine zero covariance keeps its diagnostics.  Negative order reverses
no covariance and no standard error; at ``omega = 0`` the order multiplier is absent and
the inverse column scaling remains.

Cluster dimensions (:func:`cluster_codes`).  A later request names a classification that
the preparation retained with the fit: the requested cluster column, or a column of
``Columns.metadata``, as raw typed values aligned with the retained rows, which are the
input rows of the system.  The codes are the integers of first appearance of the exact
values, without conversion to doubles (``2^53`` and ``2^53 + 1`` stay two labels); a
missing value on the estimation sample is refused, since no row is dropped after the
fit, and so is a floating-point value that is not finite, which the preparation reads as
missing, and a name that was not retained.  No new data, array or mapping supplies a
classification here: that transport belongs to the inference interface.

Requests (:func:`request`).  ``"robust"``; a mapping ``{"cluster": name}``; a mapping of
several dimensions, ``{"cluster": "a + b"}``, with the optional keys ``"multiway"`` and
``"psd"``, checked by the canonical
:class:`~powermean.data.inference_requests.ClusterRequest`; the exact strings of the
seven named HC methods, ``"HC0"`` to ``"HC5"`` and ``"HC4m"``; the
:class:`~powermean.data.specs.Inference` record itself; ``None`` for the covariance
requested at the fit.  The name of a cluster dimension is a retained classification: a
column the preparation read, or the interaction ``a^b`` of two columns that it formed
and retained under that requested name, whose codes number the observed pairs of exact
values by first appearance; a dimension that was not retained is refused as any other
name that was not, with no new sample, data, array or refit.  The key ``ssc`` of a
mapping is unknown, the key ``adjustment`` and the expected bread of a robust or cluster
covariance are refused as not available.  The keywords ``ssc`` and ``bread`` of a
request carry the explicit convention and the bread of a named HC method; a post-fit
robust or cluster request also accepts the explicit convention, taking the fit's when
omitted; its bread remains observed.  An omitted bread is the observed one.  The
convention of an omitted ``ssc`` is resolved by the inference layer (the fit's, after
the fit).  A request matches the record of the fit, which then answers it, when it names
the same covariance: the same kind, the same typed cluster request (dimensions, form and
``"psd"`` key, as given), and a convention that is the record's own (omitted, or the one
the fit requested).

A named HC record (:func:`hc_record`, :func:`hc_default`) holds the kernel's covariance
of ``theta`` with the method's factor applied once, the reported covariance and the
direct standard errors, each map applied once, and the applied convention; the rows,
hats and multipliers of the kernel's record stay with it and are not retained.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import pandas as pd

from powermean import messages
from powermean.data.inference_conventions import ConventionValues
from powermean.data.inference_requests import HC_METHODS, ClusterRequest, HCMethod
from powermean.data.request_values import OMITTED, Omitted, given
from powermean.data.retained import OrderSample
from powermean.data.specs import Bread, Inference, explicit_bread, explicit_ssc
from powermean.data.specs import Covariance as CovarianceKind
from powermean.errors import SpecificationError
from powermean.inference.contributions import Influence
from powermean.inference.convention import (
    ClusterCodes,
    Covariance,
    one_way_covariance,
    robust_covariance,
)
from powermean.inference.hc import HCCovariance
from powermean.inference.influence import influence_values
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.results.retained import Convention, CovarianceRecord

# The keys of a vcov mapping that are reserved and not yet available.
LATER_KEYS: frozenset[str] = frozenset({"adjustment"})
# The keys of a cluster mapping of vcov; "ssc" is not one.
CLUSTER_KEYS: tuple[str, ...] = ("cluster", "multiway", "psd")
VALID_SPECS = (
    "'robust', 'HC0', 'HC1', 'HC2', 'HC3', 'HC4', 'HC4m', 'HC5', or a mapping with "
    "the key 'cluster'"
)
# Default values, not a test of keyword omission: an explicit ssc carries its own
# requested-convention provenance.
INACTIVE: Mapping[str, str] = {"ssc": "stata", "bread": "observed"}
# A cluster request by its typed settings: the dimensions, the form and "psd".
ClusterKey = tuple[tuple[str, ...], object, object]


@dataclass(frozen=True)
class Request:
    """A resolved covariance request: the kind, a cluster's name, a named method.

    ``ssc`` is the request's explicit convention, :data:`OMITTED` where it gives none,
    ``bread`` its bread, and ``clusters`` the typed request of several dimensions or of
    the keys ``"multiway"`` and ``"psd"`` (``cluster`` is then ``None``).
    """

    kind: CovarianceKind
    cluster: str | None
    method: HCMethod | None = None
    ssc: str | ConventionValues | Omitted = OMITTED
    bread: Bread = "observed"
    clusters: ClusterRequest | None = None

    def matches(self, record: CovarianceRecord) -> bool:
        """Whether ``record`` is the covariance of this request, its own cached record.

        A robust or cluster request matches the record of its kind and typed cluster
        request: a default record under the ``"stata"`` convention, omitted or explicit;
        a record of the convention engine under its requested convention, or any where
        the request gives none (it takes the fit's convention in force, the record's).
        A named HC request matches the record of its method and bread whose requested
        convention is its own, or any where the request gives none (it takes the fit's
        convention in force, under which the method's factor and reference are the
        record's).
        """
        if self.method is None:
            if record.method is not None or record.kind != self.kind:
                return False
            if _request_key(self) != _record_key(record):
                return False
            if record.answer is None:
                return isinstance(self.ssc, Omitted)
            requested = record.answer.applied.convention.requested
            return isinstance(self.ssc, Omitted) or self.ssc == requested
        applied = record.applied
        if applied is None or record.method != self.method:
            return False
        own = isinstance(self.ssc, Omitted) or self.ssc == applied.convention.requested
        return own and record.bread == self.bread


def _request_key(found: Request) -> ClusterKey | None:
    """The typed cluster request of a request, or ``None`` for a robust one."""
    if found.clusters is not None:
        clusters = found.clusters
        return clusters.dimensions, clusters.multiway, clusters.psd
    if found.cluster is not None:
        return (found.cluster,), OMITTED, OMITTED
    return None


def _record_key(record: CovarianceRecord) -> ClusterKey | None:
    """The typed cluster request of a record, or ``None`` for a robust one.

    A record of the convention engine holds its bound request; a default record its one
    cluster variable.
    """
    answer = record.answer
    if answer is not None:
        bound = answer.request
        if not bound.clusters:
            return None
        return tuple(c.name for c in bound.clusters), bound.multiway, bound.psd
    if record.cluster is not None:
        return (record.cluster,), OMITTED, OMITTED
    return None


# ------------------------------------------------------------------ the reported scale


def report(covariance: FloatArray, scale: OrderScale) -> tuple[FloatArray, FloatArray]:
    """``V`` of ``theta`` on the reported scale, and its standard errors.

    Parameters
    ----------
    covariance : array
        ``V``, square, one row per coefficient of ``theta``, finite.
    scale : OrderScale
        The accepted order scale of the fit.

    Returns
    -------
    tuple of array
        The reported covariance, entry by entry, and the standard errors ``sqrt`` of
        its variances mapped by ``1 / (d_j |omega|)``: NaN where a variance of ``V``
        is negative, zero where it is zero.

    Raises
    ------
    FloatingPointError
        A reported entry or standard error is not representable in float64.
    """
    reported = scale.covariance(covariance)
    variances = np.diag(np.asarray(covariance, dtype=np.float64))
    errors = np.full(variances.shape, np.nan)
    valid = variances >= 0
    errors[valid] = np.sqrt(variances[valid])
    return reported, scale.standard_errors(errors)


def convention_covariance(
    influence: Influence, cluster: ClusterCodes | None = None
) -> Covariance:
    """The covariance of the point by the convention, in the solver's coordinates.

    Parameters
    ----------
    influence : Influence
        The retained influence record of the point.
    cluster : ClusterCodes, optional
        One cluster dimension; the robust estimator without it.

    Returns
    -------
    Covariance
        The robust covariance with ``N/(N - 1)``, or the one-way covariance with
        ``G/(G - 1)``, of :mod:`powermean.inference.convention`: the record that
        the reported-scale provider of the certificate forms as well.
    """
    if cluster is None:
        return robust_covariance(influence)
    return one_way_covariance(influence, cluster)


def record(
    covariance: Covariance, scale: OrderScale, cluster: str | None
) -> CovarianceRecord:
    """The record of a convention covariance, mapped once to the reported scale.

    Parameters
    ----------
    covariance : Covariance
        The covariance of ``theta`` by the convention.
    scale : OrderScale
        The accepted order scale of the fit.
    cluster : str or None
        The name of the classification clustered on, or ``None``.

    Returns
    -------
    CovarianceRecord
        The covariance in both coordinate systems with its factor, counts, rank and
        diagnostics.
    """
    reported, errors = report(covariance.covariance, scale)
    return CovarianceRecord(
        kind=covariance.kind,
        cluster=cluster,
        theta=covariance.covariance,
        reported=reported,
        standard_errors=errors,
        factor=covariance.factor,
        count=covariance.count,
        clusters=covariance.clusters,
        rank=covariance.estimate.rank,
        diagnostics=covariance.estimate.diagnostics,
    )


def hc_record(
    found: HCCovariance, reported: FloatArray, errors: FloatArray, bread: Bread
) -> CovarianceRecord:
    """The record of a named HC covariance, with its applied convention.

    Parameters
    ----------
    found : HCCovariance
        The kernel's record (:mod:`powermean.inference.hc`): the covariance of
        ``theta`` with the method's factor applied once, the primitive's estimate of
        the adjusted rows and the convention applied to the counts.
    reported, errors : array
        That covariance and its direct standard errors on the reported scale, each
        map applied once.
    bread : {"observed", "expected"}
        The bread of the request; the least-squares equation's observed bread is its
        expected information, so the two name one matrix on the linear path.

    Returns
    -------
    CovarianceRecord
        The covariance in both coordinate systems with the method's factor, ``N``,
        the rank and diagnostics of the primitive's estimate, the convention in force
        (a preset by its name, or the custom record), the reference, the method, the
        applied convention and the leverage geometry; no row array of the kernel's
        record is kept.
    """
    applied = found.applied
    resolved = applied.convention
    name = resolved.preset
    convention: Convention = (
        resolved.values if resolved.origin == "custom" or name is None else name
    )
    return CovarianceRecord(
        kind="hc",
        cluster=None,
        theta=found.covariance,
        reported=reported,
        standard_errors=errors,
        factor=found.factor,
        count=applied.counts.count,
        clusters=None,
        rank=found.estimate.rank,
        diagnostics=found.estimate.diagnostics,
        bread=bread,
        convention=convention,
        reference=applied.reference.kind,
        method=found.method,
        applied=applied,
        hat_geometry=found.hat_geometry,
    )


def hc_default(
    found: HCCovariance, scale: OrderScale, bread: Bread
) -> tuple[Covariance, CovarianceRecord]:
    """The named HC covariance requested at the fit, reported once.

    Parameters
    ----------
    found : HCCovariance
        The kernel's record at the certified point.
    scale : OrderScale
        The order scale of the fit.
    bread : {"observed", "expected"}
        The bread of the request.

    Returns
    -------
    tuple
        The covariance of ``theta`` as the default convention record
        (kind ``"hc"``, the method's factor, ``N``, the primitive's estimate and the
        reference), against which the handoff binds the standard errors that the
        certificate's coefficient gate read, and its record on the reported scale.

    Raises
    ------
    FloatingPointError
        A reported entry or standard error is not representable in float64.
    """
    reported, errors = report(found.covariance, scale)
    convention = Covariance(
        covariance=found.covariance,
        kind="hc",
        factor=found.factor,
        count=found.applied.counts.count,
        clusters=None,
        estimate=found.estimate,
        reference=found.reference.kind,
    )
    return convention, hc_record(found, reported, errors, bread)


def compute(
    influence: Influence, scale: OrderScale, cluster: ClusterCodes | None = None
) -> CovarianceRecord:
    """The covariance of the retained point by the convention, reported once.

    Parameters
    ----------
    influence : Influence
        The retained influence record of the point.
    scale : OrderScale
        The accepted order scale of the fit.
    cluster : ClusterCodes, optional
        One cluster dimension; the robust estimator without it.

    Returns
    -------
    CovarianceRecord
        The covariance in both coordinate systems with its factor, counts, rank and
        diagnostics.
    """
    name = None if cluster is None else cluster.name
    return record(convention_covariance(influence, cluster), scale, name)


def influences(influence: Influence, scale: OrderScale) -> FloatArray:
    """The coefficient influences ``A s_i`` on the reported scale, one row per unit.

    The bread map is applied once by :func:`~powermean.inference.influence.
    influence_values` and the order scale once by its ``influences``: raw scores are
    never reported as influences, and a negative order reverses the rows' sign.
    """
    mapped = influence_values(influence.contributions, influence.influence_map)
    return scale.influences(mapped)


# ------------------------------------------------------------------ cluster dimensions


def _missing(value: object) -> bool:
    """Whether a value of a retained classification is missing.

    A missing marker (``None``, ``pd.NA``, ``pd.NaT``, a NumPy ``NaT``), or a
    floating-point value that is not finite: NaN, and the two infinities, which the
    preparation reads as an invalid value of a variable that a row uses
    (:func:`powermean.formula.validity.values_invalid`), so that a classification
    whose value is infinite on a retained row is refused as a missing one is, not
    numbered as a label.  An integer, however large, is never missing.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, float | np.floating):
        return not math.isfinite(float(value))
    return isinstance(value, np.datetime64 | np.timedelta64) and bool(np.isnat(value))


def retained_names(sample: OrderSample) -> tuple[str, ...]:
    """The names of the classifications retained with the fit, in their order."""
    return tuple(sample.metadata)


def frequency_of(sample: OrderSample) -> bool:
    """The stated kind of the weights: frequency, or not (``False`` without weights)."""
    return sample.data.weights is not None and sample.data.weights.frequency


def _available(names: tuple[str, ...]) -> str:
    """The retained names for the message of a refusal, or ``none``."""
    return messages.format_keys(names) if names else "none"


def cluster_codes(sample: OrderSample, name: str) -> ClusterCodes:
    """The integer codes of the retained classification ``name``, aligned with the rows.

    Parameters
    ----------
    sample : OrderSample
        The sample of the fit, whose retained rows are the input rows of the system.
    name : str
        The name of a classification retained with the fit.

    Returns
    -------
    ClusterCodes
        One code per retained row, numbered by first appearance of the exact values.

    Raises
    ------
    SpecificationError
        ``specification.cluster_not_retained`` for a name that the preparation did not
        retain; ``specification.cluster_missing_values`` for a missing value on the
        estimation sample.
    ValueError
        A retained column that is not aligned with the rows of the sample.
    TypeError
        A value that cannot be a key of a classification.
    """
    names = retained_names(sample)
    if name not in names:
        raise SpecificationError(
            "specification.cluster_not_retained",
            {"name": name, "retained": _available(names)},
        )
    values = sample.metadata[name]
    if values.shape != (sample.count,):
        raise ValueError(f"cluster dimension {name!r} is not aligned with the rows")
    labels: dict[Any, int] = {}
    codes = np.empty(sample.count, dtype=np.intp)
    missing = 0
    for position, value in enumerate(values.tolist()):
        if _missing(value):
            missing += 1
            continue
        codes[position] = labels.setdefault(value, len(labels))
    if missing:
        raise SpecificationError(
            "specification.cluster_missing_values", {"name": name, "count": missing}
        )
    return ClusterCodes(name, codes)


# -------------------------------------------------------------------------- requests


def _refuse(value: object) -> SpecificationError:
    """The refusal of a value that names no covariance."""
    return SpecificationError(
        "specification.invalid_value",
        {"argument": "vcov", "value": value, "valid": VALID_SPECS},
    )


def _cluster_dimensions(value: object) -> tuple[str, ...]:
    """The dimensions of a cluster spelling: one name, or several joined by ``+``.

    One column or a retained interaction ``a^b`` as given; several dimensions
    (``"a + b"``) stripped and checked by the canonical
    :class:`~powermean.data.inference_requests.ClusterRequest`, in their order,
    repetitions kept.  Whether a name was retained is :func:`cluster_codes`'s to
    decide, on the sample of the fit: no name is accepted or formed here.
    """
    if not isinstance(value, str) or not value:
        raise _refuse({"cluster": value})
    if "+" not in value:
        return (value,)
    return ClusterRequest(tuple(part.strip() for part in value.split("+"))).dimensions


def _mapping_request(spec: Mapping[str, object]) -> Request:
    """A request from a ``vcov`` mapping: a cluster request and its keys.

    One dimension without ``"multiway"`` or ``"psd"`` is the single cluster variable;
    every other request is the typed
    :class:`~powermean.data.inference_requests.ClusterRequest`, which checks the form,
    the ``"psd"`` treatment and where it is valid.
    """
    for key in spec:
        if key in LATER_KEYS:
            raise SpecificationError(
                "specification.unimplemented",
                {"feature": f"the vcov key {key!r}"},
            )
        if key not in CLUSTER_KEYS:
            raise SpecificationError(
                "specification.unknown_option",
                {
                    "kind": "vcov key",
                    "key": key,
                    "valid": messages.format_keys(CLUSTER_KEYS),
                },
            )
    if "cluster" not in spec:
        raise _refuse(dict(spec))
    dimensions = _cluster_dimensions(spec["cluster"])
    settings = {key: spec[key] for key in ("multiway", "psd") if key in spec}
    if len(dimensions) == 1 and not settings:
        return Request("cluster", dimensions[0])
    clusters = ClusterRequest.from_dict(
        {"type": "ClusterRequest", "dimensions": dimensions, **settings}
    )
    return Request("cluster", None, clusters=clusters)


def _named(spec: object) -> Request:
    """The request of a value: a mapping, ``"robust"`` or a method's exact string."""
    if isinstance(spec, Inference):
        return Request(
            spec.covariance,
            spec.cluster,
            spec.method,
            spec.ssc,
            spec.bread,
            spec.clusters,
        )
    if isinstance(spec, str):
        if spec == "robust":
            return Request("robust", None)
        if spec in HC_METHODS:
            return Request("hc", None, cast(HCMethod, spec))
        raise _refuse(spec)
    if isinstance(spec, Mapping):
        return _mapping_request(spec)
    raise _refuse(spec)


def _with_keywords(found: Request, ssc: object, bread: object) -> Request:
    """The request with the explicit ``ssc`` and ``bread`` of its call, if given.

    A named HC method takes any value of their domains; a robust or cluster request also
    takes any convention, with observed bread.
    """
    if found.method is None:
        if given(bread) and explicit_bread(bread) != "observed":
            raise SpecificationError(
                "specification.unimplemented",
                {"feature": f"bread={bread!r}"},
            )
        return replace(found, ssc=explicit_ssc(ssc) if given(ssc) else found.ssc)
    if given(ssc):
        found = replace(found, ssc=explicit_ssc(ssc))
    if given(bread):
        found = replace(found, bread=explicit_bread(bread))
    return found


def request(
    spec: object,
    inference: Inference,
    *,
    ssc: object = OMITTED,
    bread: object = OMITTED,
) -> Request:
    """The request that ``spec`` names, the fit's own for ``None`` (module docstring).

    Parameters
    ----------
    spec : None, str, mapping or Inference
        ``None``, ``"robust"``, ``{"cluster": name}``, ``{"cluster": "a + b"}`` with
        the optional keys ``"multiway"`` and ``"psd"``, the exact string of a named HC
        method or an inference record.
    inference : Inference
        The inference requested at the fit.
    ssc, bread : object, optional
        The explicit convention and bread of the request, :data:`OMITTED` where the
        call gives none: for ``None``, the fit's own request takes them in place of
        its own; an omitted bread of another request is the observed bread.

    Returns
    -------
    Request
        The kind, the cluster name or typed cluster request, the method, the explicit
        convention and the bread.

    Raises
    ------
    SpecificationError
        A value that names no covariance (``specification.invalid_value``), a key that
        is not one of a cluster request (``specification.unknown_option``), a form or
        a ``"psd"`` treatment outside its domain (``specification.invalid_value``),
        ``"psd"`` outside a signed multiway request (``specification.conflicting``),
        the key ``"adjustment"`` or the expected bread of a robust or cluster
        covariance, which are not yet available
        (``specification.unimplemented``), or an explicit convention or bread outside
        its domain (``specification.invalid_value``).
    """
    if spec is None:
        found = Request(
            inference.covariance,
            inference.cluster,
            inference.method,
            inference.ssc,
            inference.bread,
            inference.clusters,
        )
    else:
        found = _named(spec)
    return _with_keywords(found, ssc, bread)


def frame(record: CovarianceRecord, names: tuple[str, ...]) -> pd.DataFrame:
    """The reported covariance as a labeled table with its method in ``attrs``.

    The table is a fresh writable copy: editing it changes no retained state.
    """
    table = pd.DataFrame(
        np.array(record.reported, dtype=np.float64),
        index=pd.Index(names, name="term"),
        columns=pd.Index(names, name="term"),
    )
    table.attrs.update(record.to_dict())
    return table
