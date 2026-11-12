"""The canonical records of a custom convention and of the retained declarations.

The factory ``pm.ssc(...)`` builds the canonical custom convention for the ``ssc``
keyword beside ``vcov``.  This module validates declarations only: it resolves no
preset, computes no factor and puts no convention in force.  The value rules and the
serialization are those of :mod:`powermean.data.request_values`; the request records are
in :mod:`powermean.data.inference_requests`.

==========================  ======================================================
:class:`ConventionValues`   the five required fields of a custom convention
:class:`RetainedColumns`    the names declared by ``retain=[...]``
==========================  ======================================================

The treatment of an indefinite signed multiway estimate, ``"cgm"`` or ``"none"``, is the
field ``psd`` of a convention and the key ``"psd"`` of a signed multiway request; its
values are declared here once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, get_args

from powermean.data.request_values import (
    RETAINED,
    choice,
    column_names,
    field_names,
    serialized,
    settings,
)

PsdTreatment = Literal["cgm", "none"]
RobustFactor = Literal["none", "N/(N-1)", "N/(N-K*)"]
ClusterFactor = Literal["none", "G/(G-1)", "(N-1)/(N-K*)*G/(G-1)"]
ClusterCount = Literal["min", "each"]
Reference = Literal["normal", "t"]

PSD_TREATMENTS: tuple[str, ...] = get_args(PsdTreatment)
# The closed values of each field of a custom convention.
CONVENTION_VALUES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "robust": get_args(RobustFactor),
        "cluster": get_args(ClusterFactor),
        "cluster_count": get_args(ClusterCount),
        "reference": get_args(Reference),
        "psd": PSD_TREATMENTS,
    }
)


@dataclass(frozen=True)
class ConventionValues:
    """The five fields of a custom convention, each required.

    The values of the record that ``pm.ssc(robust=, cluster=, cluster_count=,
    reference=, psd=)`` constructs, with their closed values.  ``K*`` is ``K_factor``.
    No preset is resolved, no factor computed and no convention put in force here.

    Parameters
    ----------
    robust : {"none", "N/(N-1)", "N/(N-K*)"}
        The scalar factor of robust, HAC and Conley requests.
    cluster : {"none", "G/(G-1)", "(N-1)/(N-K*)*G/(G-1)"}
        The scalar factor of cluster requests.
    cluster_count : {"min", "each"}
        One factor with ``G_min`` for the whole multiway sum, or each component with
        its own ``G``.
    reference : {"normal", "t"}
        The reference distribution.
    psd : {"cgm", "none"}
        The treatment of an indefinite signed multiway estimate.

    Raises
    ------
    SpecificationError
        A value outside its closed values (``specification.invalid_value``).
    """

    robust: RobustFactor
    cluster: ClusterFactor
    cluster_count: ClusterCount
    reference: Reference
    psd: PsdTreatment

    def __post_init__(self) -> None:
        """Check each field against its closed values, in the order of the fields."""
        for field, choices in CONVENTION_VALUES.items():
            value = choice(field, getattr(self, field), choices)
            object.__setattr__(self, field, value)

    def to_dict(self) -> dict[str, object]:
        """The type and the five fields."""
        return serialized(
            "ConventionValues",
            {field: getattr(self, field) for field in CONVENTION_VALUES},
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> ConventionValues:
        """The record of :meth:`to_dict` (:mod:`powermean.data.request_values`)."""
        return cls(**settings(record, "ConventionValues", field_names(cls)))


def ssc(
    *,
    robust: RobustFactor,
    cluster: ClusterFactor,
    cluster_count: ClusterCount,
    reference: Reference,
    psd: PsdTreatment,
) -> ConventionValues:
    """Declare a custom convention with all five fields required.

    Parameters
    ----------
    robust : {"none", "N/(N-1)", "N/(N-K*)"}
        The factor of robust, HAC and Conley requests; ``K*`` is ``K_factor``.
    cluster : {"none", "G/(G-1)", "(N-1)/(N-K*)*G/(G-1)"}
        The factor of cluster requests.
    cluster_count : {"min", "each"}
        One factor with ``G_min``, or each multiway component's own ``G``.
    reference : {"normal", "t"}
        The reference distribution, with the convention's reference count.
    psd : {"cgm", "none"}
        The treatment of an indefinite signed multiway covariance.

    Returns
    -------
    ConventionValues
        The immutable declaration. No factor or covariance is computed here.

    Raises
    ------
    SpecificationError
        A field outside its closed values (``specification.invalid_value``).
    """
    return ConventionValues(robust, cluster, cluster_count, reference, psd)


@dataclass(frozen=True)
class RetainedColumns:
    """The names declared by ``retain=[...]`` for later requests.

    A declaration: the columns are retained aligned with the input rows, excluding no
    row, and nothing here reads them.

    Parameters
    ----------
    names : tuple of str
        Nonempty column names, in their order, none required.

    Raises
    ------
    SpecificationError
        Another value (``specification.invalid_value``, argument ``retain``).
    """

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Check the names."""
        names = column_names("retain", self.names, RETAINED, 0)
        object.__setattr__(self, "names", names)

    def to_dict(self) -> dict[str, object]:
        """The type and the names as a list."""
        return serialized("RetainedColumns", {"names": self.names})

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> RetainedColumns:
        """The record of :meth:`to_dict` (:mod:`powermean.data.request_values`)."""
        return cls(**settings(record, "RetainedColumns", field_names(cls)))
