"""The reported scale of the certificate, from the covariance of a fresh evaluation.

The certificate compares the Newton correction of a candidate
with the standard errors at that candidate, which a later stage supplies through a
:class:`~powermean.solve.reported.ReportedScaleProvider`; the solve stage does not
import this one, and this module imports its contract.

:class:`CovarianceScale` is such a provider.  Called with the evaluation that the
certificate formed afresh at its candidate, it forms the influence record of that
evaluation (:func:`~powermean.inference.contributions.joint_influence`: the observed
bread, never the expected information), the robust covariance or the covariance of
one cluster dimension with the factors of the ``"stata"`` preset
(:mod:`powermean.inference.convention`), and the standard errors of the reported
coefficients for its map: the dense ``M`` (``sqrt(diag(M V M'))`` of
:func:`~powermean.inference.convention.reported_covariance`, the identity by default)
or the direct map of the
:class:`~powermean.model.order_scale.OrderScale` (:func:`direct_standard_errors`:
the standard errors ``s_j / d_j / |omega|``, each with exact exponents, so that no
intermediate of a mixed-scale candidate overflows or underflows on the way to a
representable result; the full covariance ``V_jk / (d_j d_k) / omega^2`` is the
result's, by the same direct map, so that no off-diagonal entry is halved by a dense
product and no certificate is refused for a variance that is not representable
where its standard error is).  It returns a
:class:`~powermean.solve.reported.ReportedScale` of that very candidate, the
evaluation's ``gamma`` itself, with its own standard errors and the very map it holds,
so that the retry assessments and the final certificate of one solve report every
correction by the same map: the provider keeps no state between calls, so no record
of another candidate is ever returned.

Zero standard errors at an exact root, as for a saturated model whose rows have zero
scores, are returned as computed, with the zero-covariance diagnostics of the
primitive; the certificate passes a zero correction over them and flags inference as
degenerate.  A negative variance on the reported scale gives a missing standard
error (NaN), which the certificate never passes; nothing is floored or replaced, and an
error of the covariance (an unrepresentable value, invalid frequency counts or cluster
codes, fewer than two observations or clusters) is raised, not caught.  The caller
states whether the weights are frequency weights, as it states it to the certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powermean.inference.contributions import joint_influence
from powermean.inference.convention import (
    ClusterCodes,
    Covariance,
    one_way_covariance,
    reported_covariance,
    robust_covariance,
)
from powermean.inference.numeric import finite_array, immutable_array
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.solve.reported import ReportedScale
from powermean.systems.joint import JointEvaluation


def direct_standard_errors(covariance: Covariance, order: OrderScale) -> FloatArray:
    """The standard errors on the reported scale by the direct map of the order scale.

    Parameters
    ----------
    covariance : Covariance
        The covariance ``V`` of the solver's coordinates.
    order : OrderScale
        The order scale of those coordinates.

    Returns
    -------
    array
        ``sqrt(V_jj)`` mapped by ``1 / (d_j |omega|)`` with exact exponents
        (``OrderScale.standard_errors``), NaN where a variance of ``V`` is negative
        and zero where it is zero.  No dense map is materialized, and the full
        reported covariance, which the result forms by the same direct map
        (``powermean.results.covariance.report``), is not formed here: a standard
        error is representable where the variance that squares it may not be.

    Raises
    ------
    TypeError
        ``covariance`` is not a ``Covariance`` or ``order`` not an ``OrderScale``.
    FloatingPointError
        A reported standard error is not representable in float64.
    """
    if type(covariance) is not Covariance:
        raise TypeError("covariance must be a Covariance")
    if type(order) is not OrderScale:
        raise TypeError("order must be an OrderScale")
    variances = np.diag(covariance.covariance)
    errors = np.full(variances.shape, np.nan)
    valid = variances >= 0
    errors[valid] = np.sqrt(variances[valid])
    return order.standard_errors(errors)


@dataclass(frozen=True, eq=False)
class CovarianceScale:
    """The reported scale at the candidate of an evaluation, by its covariance.

    Parameters
    ----------
    frequency : bool
        Whether the weights of the system are frequency weights.
    cluster : ClusterCodes, optional
        One cluster dimension; the robust covariance without it.
    transform : array, optional
        ``M``, one row per reported coefficient and one column per coefficient of
        ``gamma``, the dense form of the map; the identity without it and without
        ``order``.  Not together with ``order``.
    order : OrderScale, optional
        The accepted order scale of the candidate's coordinates, the direct form of
        the map (:func:`direct_reported`).  Not together with ``transform``.

    Notes
    -----
    An instance is a :class:`~powermean.solve.reported.ReportedScaleProvider`.  The
    dense map is stored as an owned read-only copy; another type of argument raises
    ``TypeError``, a map that is not a finite real matrix or both forms at once
    ``ValueError``.
    """

    frequency: bool
    cluster: ClusterCodes | None = None
    transform: FloatArray | None = None
    order: OrderScale | None = None

    def __post_init__(self) -> None:
        """Check the arguments and store the dense map read-only.

        The values are read as ``object``: this is the run-time check of what the
        fields declare.
        """
        frequency: object = self.frequency
        if not isinstance(frequency, bool):
            raise TypeError("frequency must be a bool")
        cluster: object = self.cluster
        if cluster is not None and type(cluster) is not ClusterCodes:
            raise TypeError("cluster must be ClusterCodes or None")
        order: object = self.order
        if order is not None and type(order) is not OrderScale:
            raise TypeError("order must be an OrderScale or None")
        if self.transform is not None:
            if order is not None:
                raise ValueError("transform and order cannot both be given")
            matrix = finite_array(self.transform, "transform")
            if matrix.ndim != 2 or not matrix.size:
                raise ValueError("transform must be a nonempty matrix")
            object.__setattr__(self, "transform", immutable_array(matrix))

    def _covariance(self, evaluation: JointEvaluation) -> Covariance:
        """The convention covariance of the fresh evaluation, robust or one-way."""
        influence = joint_influence(evaluation, frequency=self.frequency)
        if self.cluster is None:
            return robust_covariance(influence)
        return one_way_covariance(influence, self.cluster)

    def __call__(self, evaluation: JointEvaluation, /) -> ReportedScale:
        """The record at ``evaluation.gamma``, from that evaluation alone."""
        covariance = self._covariance(evaluation)
        if self.order is not None:
            return ReportedScale(
                gamma=evaluation.gamma,
                standard_errors=direct_standard_errors(covariance, self.order),
                order=self.order,
            )
        reported = reported_covariance(covariance, self.transform)
        return ReportedScale(
            gamma=evaluation.gamma,
            transform=reported.transform,
            standard_errors=reported.standard_errors,
        )
