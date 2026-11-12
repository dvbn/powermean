"""The reported scale of the linear certificate, from the fresh linear evaluation.

The linear path is the joint system without effects at a unit curvature:
contribution ``x_i e_i``, map ``(X'CX)^-1`` with ``C = diag(a)``.  The
fresh certificate of the linear path (:mod:`powermean.solve.linear_certificate`)
compares its Newton correction with the standard errors at the point, which this
module supplies through the callback :class:`~powermean.solve.linear_certificate.
LinearScaleProvider`; the solve stage imports nothing of this one.

:func:`fresh_linear_influence` forms the influence record of a
:class:`~powermean.solve.linear_evaluation.LinearEvaluation`, the record that the
certificate evaluated afresh at its point: ``-C`` is its Jacobian, ``C = X' diag(a) X``
its bread, the map ``A = C^-1`` the
:func:`~powermean.inference.influence.moment_influence_map` of ``-C``, and the
contributions ``a_i e_i x_i`` its rows, on its rows of positive weight, with the
statistical count ``N`` of the stated kind of the weights.  It reads nothing of a
least-squares candidate: not its cached residuals, which
:func:`~powermean.inference.contributions.linear_influence` reads and which are no
fresh audit of a point, nor its rank.  :class:`LinearCovarianceScale` is the provider:
the robust covariance or that of one cluster dimension with the factors of the
``"stata"`` preset and the same frequency convention as the certificate
(:mod:`powermean.inference.convention`), reported by the dense map ``M`` or by the
direct map of the :class:`~powermean.model.order_scale.OrderScale`
(:func:`~powermean.inference.provider.direct_standard_errors`), as the joint
provider does.  It keeps no state between calls, and returns a record of the very
point it received.
"""

from __future__ import annotations

from dataclasses import dataclass

from powermean.inference.contributions import (
    Influence,
    _check_frequency,
    statistical_count,
)
from powermean.inference.convention import (
    ClusterCodes,
    Covariance,
    one_way_covariance,
    reported_covariance,
    robust_covariance,
)
from powermean.inference.influence import moment_influence_map
from powermean.inference.numeric import finite_array, immutable_array
from powermean.inference.provider import direct_standard_errors
from powermean.model.order_scale import OrderScale
from powermean.model.variance import FloatArray
from powermean.solve.linear_evaluation import LinearEvaluation
from powermean.solve.reported import ReportedScale


def fresh_linear_influence(
    evaluation: LinearEvaluation, *, frequency: bool
) -> Influence:
    """The influence record of a fresh evaluation of the linear equation.

    Parameters
    ----------
    evaluation : LinearEvaluation
        The evaluation at the point, as the certificate formed it afresh.
    frequency : bool
        Whether the weights of the solve are frequency weights.

    Returns
    -------
    Influence
        ``-C`` from the fresh bread, its map, the contributions ``a_i e_i x_i``, the
        raw weights, the units and ``N``.

    Raises
    ------
    TypeError
        ``evaluation`` is not a ``LinearEvaluation`` or ``frequency`` not a ``bool``.
    ValueError
        Invalid frequency counts, or a map that cannot be formed or represented.
    numpy.linalg.LinAlgError
        ``C`` is not of full rank.
    """
    if type(evaluation) is not LinearEvaluation:
        raise TypeError("evaluation must be a LinearEvaluation")
    _check_frequency(frequency)
    count = statistical_count(evaluation.weights, frequency=frequency)
    jacobian = evaluation.jacobian
    return Influence(
        gamma=evaluation.gamma,
        jacobian=jacobian,
        influence_map=moment_influence_map(jacobian),
        contributions=evaluation.contributions,
        weights=evaluation.weights,
        units=evaluation.units,
        frequency=frequency,
        count=count,
    )


@dataclass(frozen=True, eq=False)
class LinearCovarianceScale:
    """The reported scale at the point of a linear evaluation, by its covariance.

    Parameters
    ----------
    frequency : bool
        Whether the weights of the solve are frequency weights.
    cluster : ClusterCodes, optional
        One cluster dimension; the robust covariance without it.
    transform : array, optional
        ``M``, the dense form of the map; the identity without it and without
        ``order``.  Not together with ``order``.
    order : OrderScale, optional
        The accepted order scale of the point's coordinates, the direct form of the
        map.  Not together with ``transform``.

    Notes
    -----
    An instance is a :class:`~powermean.solve.linear_certificate.LinearScaleProvider`.
    The dense map is stored as an owned read-only copy; another type of argument
    raises ``TypeError``, a map that is not a finite real matrix or both forms at
    once ``ValueError``.
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

    def covariance(self, evaluation: LinearEvaluation) -> Covariance:
        """The convention covariance of the fresh evaluation, robust or one-way."""
        influence = fresh_linear_influence(evaluation, frequency=self.frequency)
        if self.cluster is None:
            return robust_covariance(influence)
        return one_way_covariance(influence, self.cluster)

    def __call__(self, evaluation: LinearEvaluation, /) -> ReportedScale:
        """The record at ``evaluation.gamma``, from that evaluation alone."""
        covariance = self.covariance(evaluation)
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
