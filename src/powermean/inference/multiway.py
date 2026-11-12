"""Multiway cluster covariances of retained influence rows: two forms, one factor owner.

Internal module: nothing here parses a public keyword or touches the raw port
:mod:`powermean.inference.vcov`, whose ``cluster_covariance`` forms both multiway
forms with its own conventions (its internal spelling ``"cgm2"`` is the ``"ddg"`` of
this package).  This module hands it the influence rows of the active rows
and the integer codes of the requested dimensions, with every factor off
(``correction="none"``), and then applies the convention's factors once, by their one
owner (:mod:`powermean.inference.ssc`).

Forms.  ``"cgm"`` is the signed inclusion-exclusion of Cameron, Gelbach and Miller,
one component per nonempty subset ``T`` of the requested dimensions, each clustered on
the intersection of its dimensions, with the sign ``(-1)^(|T|+1)``; it is the default
of a request with two or more dimensions.  ``"ddg"`` is the sum of the one-way
components of Davezies, D'Haultfoeuille and Guyonvarch, with no intersection term: a
different
covariance convention, positive semidefinite by construction, not a repair.  With one
dimension the two forms are the same one-way estimate bit for bit (the port's own
identity).

Dimensions.  A request names its dimensions in order, repetitions kept, each bound to
the final rows as :class:`~powermean.inference.convention.ClusterCodes`; the codes of
the active rows reach the port as one column per requested position, so that a repeated
dimension stays a dimension of the port's sum (its intersection with itself is itself:
the signed sum collapses to the one-way estimate in exact arithmetic, the DDG sum counts
it twice) and a dimension nested in another keeps its own component and count.  The
port numbers the columns ``cluster_0``, ``cluster_1``, ... ; the records here carry the
requested names by position.  ``G_min`` is the smallest count over the requested
dimensions, repetitions kept, never an intersection count.

Factors.  Under ``cluster_count="min"`` (the ``"stata"`` and
``"regression"`` presets) one factor, formed with ``G_min``, multiplies the whole signed
or DDG sum once: ``q * estimate.covariance``, the composition of
:func:`powermean.inference.ssc.scaled_covariance`.  Under ``cluster_count="each"`` of a
custom record every component takes the convention's cluster factor with its own
cluster count, intersections included (``G_T``, the count of the intersection
clusters of ``T``), and the signed sum of the scaled components is formed here with its
symmetric part rounded once.  No factor is applied twice, no component is
clipped, and the raw port estimate with its components, eigenvalues and diagnostics is
kept untouched.  Weights enter the rows once, before this module: analytic and
probability weights multiply the scores, and frequency rows are literal replicas whose
cluster total is ``sum_{i in g} f_i r_i x_i`` (the rows are not
divided by ``sqrt(f)`` for a cluster total).

The eigenvalue adjustment is a separate operation on the reported
covariance (:mod:`powermean.inference.psd`), never applied here, to a component, or to a
DDG estimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from powermean.data.inference_requests import MULTIWAY_FORMS, Multiway
from powermean.inference.convention import _source_average
from powermean.inference.numeric import symmetric_part
from powermean.inference.ssc import (
    AppliedConvention,
    ReferenceDistribution,
    ScalarFactor,
    cluster_factor_of,
)
from powermean.inference.vcov import CovarianceEstimate, cluster_covariance
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_result

IntegerArray = NDArray[np.integer[Any]]
Dimension = tuple[str, IntegerArray]

# The port's spelling of each form of the design (vcov.py: ``"cgm2"`` is DDG).
PORT_METHODS: dict[str, str] = {"cgm": "cgm", "ddg": "cgm2"}
PORT_RESULT_METHODS: dict[str, str] = {"cgm": "cluster", "ddg": "cluster_cgm2"}


def _form(form: object) -> Multiway:
    """One of the two forms of the design, by exact string."""
    if not isinstance(form, str) or form not in MULTIWAY_FORMS:
        raise ValueError("form must be 'cgm' or 'ddg'")
    found: Multiway = "cgm" if form == "cgm" else "ddg"
    return found


def _dimensions(dimensions: Sequence[Dimension], rows: int) -> tuple[str, ...]:
    """The requested names, in order, repetitions kept; the codes checked per row."""
    names: list[str] = []
    for item in dimensions:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
        ):
            raise ValueError("each dimension is a (name, codes) pair")
        name, codes = item
        if (
            not isinstance(codes, np.ndarray)
            or codes.dtype.kind not in ("i", "u")
            or codes.shape != (rows,)
        ):
            raise ValueError(
                f"cluster dimension {name!r} must have one integer code per active row"
            )
        if np.any(codes < 0):
            raise ValueError(f"cluster dimension {name!r} contains missing labels")
        if np.unique(codes).size < 2:
            raise ValueError(
                f"cluster dimension {name!r} must contain at least two clusters"
            )
        names.append(name)
    if not names:
        raise ValueError("a multiway request names at least one dimension")
    return tuple(names)


def multiway_estimate(
    influence: FloatArray, dimensions: Sequence[Dimension], *, form: Multiway
) -> CovarianceEstimate:
    """The port's raw multiway estimate of the active rows, every factor off.

    Parameters
    ----------
    influence : array
        The influence rows ``u_i`` of the active rows, in their order, two or more.
    dimensions : sequence of (str, array)
        The requested dimensions in order, repetitions kept, each with one integer
        code per active row.
    form : {"cgm", "ddg"}
        The signed form, or the sum of the one-way components.

    Returns
    -------
    CovarianceEstimate
        The port's estimate of the source-average rows under ``correction="none"``:
        the signed or DDG sum, every component (one per nonempty subset for the
        signed form, named ``cluster_j`` by position), its eigenvalues, rank and
        diagnostics.  A singular, zero or indefinite estimate is returned as it is,
        with the port's :class:`~powermean.inference.vcov.CovarianceWarning`.

    Raises
    ------
    ValueError
        Rows that are not a matrix of at least two rows; a dimension that is not one
        nonnegative integer code per row, or with fewer than two clusters; a form
        outside the two.
    FloatingPointError
        A source-average row that is not representable.
    """
    chosen = _form(form)
    if not isinstance(influence, np.ndarray) or influence.ndim != 2:
        raise ValueError("influence rows must be a matrix")
    if influence.shape[0] < 2:
        raise ValueError("a cluster request needs at least two active rows")
    _dimensions(dimensions, influence.shape[0])
    codes = np.column_stack([np.asarray(c, dtype=np.intp) for _, c in dimensions])
    return cluster_covariance(
        _source_average(influence), codes, multiway_method=PORT_METHODS[chosen]
    )


@dataclass(frozen=True, eq=False)
class MultiwayComponent:
    """One component of a multiway estimate, with the factor applied to it.

    Attributes
    ----------
    dimensions : tuple of str
        The requested names of the component's dimensions, by position.
    positions : tuple of int
        The positions of those dimensions in the request.
    sign : int
        ``+1`` or ``-1``: the sign of the component in the signed sum, ``+1`` in DDG.
    n_clusters : int
        The number of intersection clusters of the component.
    factor : ScalarFactor
        The factor applied to this component: the common factor under
        ``cluster_count="min"``, its own under ``"each"``.
    covariance : array
        The port's unsigned, unscaled component, owned.
    """

    dimensions: tuple[str, ...]
    positions: tuple[int, ...]
    sign: int
    n_clusters: int
    factor: ScalarFactor
    covariance: FloatArray


@dataclass(frozen=True, eq=False)
class MultiwayCovariance:
    """A multiway covariance of ``theta`` under a convention: the factors applied once.

    Attributes
    ----------
    covariance : array
        The factor-scaled covariance, exactly symmetric, owned: ``q`` times the port's
        sum under ``"min"``; the signed sum of the scaled components under ``"each"``.
    estimate : CovarianceEstimate
        The port's raw estimate, every factor off: its covariance, components,
        eigenvalues, rank and diagnostics, untouched.
    applied : AppliedConvention
        The convention in force, its counts, factors, reference and ``psd`` field.
    form : {"cgm", "ddg"}
        The form.
    dimensions : tuple of str
        The requested names, in order, repetitions kept.
    components : tuple of MultiwayComponent
        The components with their names, signs, counts and applied factors.
    """

    covariance: FloatArray
    estimate: CovarianceEstimate
    applied: AppliedConvention
    form: Multiway
    dimensions: tuple[str, ...]
    components: tuple[MultiwayComponent, ...]

    @property
    def signed(self) -> bool:
        """Whether this is a signed multiway estimate: two or more dimensions, CGM."""
        return self.form == "cgm" and len(self.dimensions) >= 2

    @property
    def factor(self) -> ScalarFactor | None:
        """The one common factor; ``None`` under ``"each"`` with several dimensions."""
        return self.applied.factor

    @property
    def g_min(self) -> int:
        """The smallest cluster count over the requested dimensions."""
        found = self.applied.counts.g_min
        assert found is not None
        return found

    @property
    def reference(self) -> ReferenceDistribution:
        """The reference distribution of the convention in force."""
        return self.applied.reference


def _component_positions(name: str) -> int:
    """The position of a port column name ``cluster_j``."""
    prefix = "cluster_"
    if not name.startswith(prefix) or not name[len(prefix) :].isdigit():
        raise ValueError("the port names its columns cluster_j")
    return int(name[len(prefix) :])


def _checked_estimate(
    estimate: CovarianceEstimate,
    applied: AppliedConvention,
    form: Multiway,
    names: tuple[str, ...],
) -> None:
    """The estimate is the raw port estimate of this request, every factor off."""
    counts = applied.counts
    if applied.kind != "cluster" or counts.g_min is None:
        raise ValueError("a multiway covariance takes a cluster request")
    if len(counts.clusters) != len(names) or not names:
        raise ValueError("the request names one dimension per cluster count")
    if estimate.correction != "none" or estimate.factor != 1.0:
        raise ValueError("the raw estimate is formed with every factor off")
    if estimate.method != PORT_RESULT_METHODS[form]:
        raise ValueError("the estimate is of another form than requested")
    expected = len(names) if form == "ddg" else 2 ** len(names) - 1
    if len(estimate.components) != expected:
        raise ValueError("the estimate does not hold the request's components")


def _components(
    estimate: CovarianceEstimate,
    applied: AppliedConvention,
    names: tuple[str, ...],
) -> tuple[MultiwayComponent, ...]:
    """The port's components with the requested names and their applied factors."""
    counts = applied.counts
    components: list[MultiwayComponent] = []
    for part in estimate.components:
        positions = tuple(_component_positions(n) for n in part.dimensions)
        if len(positions) == 1 and part.n_clusters != counts.clusters[positions[0]]:
            raise ValueError("the estimate does not hold the request's clusters")
        factor = (
            applied.factor
            if applied.factor is not None
            else cluster_factor_of(applied.convention.values, counts, part.n_clusters)
        )
        components.append(
            MultiwayComponent(
                tuple(names[j] for j in positions),
                positions,
                part.sign,
                part.n_clusters,
                factor,
                part.covariance,
            )
        )
    return tuple(components)


def multiway_covariance(
    estimate: CovarianceEstimate,
    applied: AppliedConvention,
    *,
    form: Multiway,
    dimensions: Sequence[str],
) -> MultiwayCovariance:
    """The convention's factors applied once to the port's raw multiway estimate.

    Parameters
    ----------
    estimate : CovarianceEstimate
        The estimate of :func:`multiway_estimate` under ``correction="none"``.
    applied : AppliedConvention
        The cluster convention applied to the request's counts, whose ``clusters``
        hold the cluster count of each requested dimension in order.
    form : {"cgm", "ddg"}
        The form of the estimate.
    dimensions : sequence of str
        The requested names, in order, repetitions kept.

    Raises
    ------
    TypeError
        ``estimate`` is not a ``CovarianceEstimate``.
    ValueError
        A request that is not a cluster request, an estimate of another form or
        with other counts than the request's, or an estimate formed with a factor.
    FloatingPointError
        A scaled covariance that is not representable.
    """
    if type(estimate) is not CovarianceEstimate:
        raise TypeError("estimate must be a CovarianceEstimate")
    chosen = _form(form)
    names = tuple(dimensions)
    _checked_estimate(estimate, applied, chosen, names)
    components = _components(estimate, applied, names)
    with np.errstate(over="ignore", invalid="ignore"):
        if applied.factor is not None:
            covariance: FloatArray = applied.factor.value * estimate.covariance
        else:
            total = np.zeros_like(estimate.covariance)
            for scaled in components:
                total = total + scaled.sign * (scaled.factor.value * scaled.covariance)
            covariance = symmetric_part(_finite_result(total, "covariance"))
    return MultiwayCovariance(
        _immutable(_finite_result(covariance, "covariance")),
        estimate,
        applied,
        chosen,
        names,
        components,
    )


__all__ = [
    "PORT_METHODS",
    "PORT_RESULT_METHODS",
    "MultiwayComponent",
    "MultiwayCovariance",
    "multiway_covariance",
    "multiway_estimate",
]
