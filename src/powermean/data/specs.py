"""The specification records: the estimator and its inference.

The typed specification objects ``Estimator``, ``Computation`` and ``Inference`` remain
available for programmatic use.  ``Computation`` is :mod:`powermean.data.computation`.
The records are frozen; each converts to a mapping that JSON holds (``to_dict``) and
back (``from_dict``), losslessly, and a record of another type is refused.

:class:`Estimator` is the estimator: a mean order (:class:`PowerMean`, any finite real
``omega``, 1 by default), a working variance
(:class:`~powermean.model.variance.PowerVariance` or
:class:`~powermean.model.variance.NB2Variance`, by exact type; ``PowerVariance(tau=1)``
by default, the interface default), the exogenous moment (:class:`ExogenousMean`) and
the formulation without fixed effects (:class:`NoFixedEffects`).  Instruments and fixed
effects are refused as not yet available (``specification.unimplemented``).  At ``omega
= 0`` the estimator is log OLS: the record keeps its variance as exact provenance of the
specification (``route`` is ``"linear"``), and the linear equation receives no working
variance (``powermean.model.order``).

:class:`Inference` names the covariance requested at a fit without fixed effects:
``"robust"``, or ``"cluster"`` on one cluster variable, with the factors of the
``"stata"`` convention and the normal reference (:mod:`powermean.inference.convention`),
constants of the record, where the expected bread stays refused as not yet available;
``"robust"`` with the explicit convention of its request, a preset name or a custom
record, an explicit ``"stata"`` included; or ``"cluster"`` with the typed request of the
expanded menu: the canonical :class:`~powermean.data.inference_requests.ClusterRequest`
(its dimensions in order, an interaction ``a^b`` among them, its multiway form and its
``"psd"`` key, each as given) with the explicit convention of the request, if any; or
``"hc"``, a named HC method with the explicit convention of its request, if any, and its
bread.  A named method owns its scalar factor and takes the reference distribution of
the convention in force, which the inference layer resolves with the counts of the fit;
the robust and cluster factors, their reference and the ``psd`` treatment are the
convention's.  The record holds the request only.  Nothing here computes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, get_args

from powermean import messages
from powermean.data.inference_conventions import ConventionValues
from powermean.data.inference_requests import ClusterRequest, HCMethod, HCRequest
from powermean.data.request_values import OMITTED, Omitted, given, invalid
from powermean.errors import SpecificationError
from powermean.model.order import Route
from powermean.model.order_scale import _mean_order
from powermean.model.variance import NB2Variance, PowerVariance

Variance = PowerVariance | NB2Variance
Covariance = Literal["robust", "cluster", "hc"]
Bread = Literal["observed", "expected"]
# The one convention and reference of the robust covariance and of the single cluster
# variable.
CONVENTION = "stata"
REFERENCE_DISTRIBUTION = "normal"
# The breads of a named HC request, and the preset names of its explicit convention; a
# custom convention is a ConventionValues record.
BREADS: tuple[Bread, ...] = ("observed", "expected")
PRESETS: tuple[str, ...] = ("stata", "none", "regression")
VALID_SSC = "'stata', 'none', 'regression' or a ConventionValues record"
# The two accepted working variances, by the names of their records.
VARIANCES: tuple[str, ...] = ("PowerVariance", "NB2Variance")


# -------------------------------------------------------------------- the estimator


@dataclass(frozen=True)
class PowerMean:
    """The mean order: ``E(Y^omega | X) = exp(omega X'beta)``; log OLS at zero.

    Any finite real order: the order is validated by
    :func:`powermean.model.order_scale._mean_order` (a boolean, a non-real, NaN or an
    infinity is refused with ``ValueError``; ``-0.0`` is stored as ``0.0``).

    Parameters
    ----------
    omega : float, default 1.0
        The mean order.
    """

    omega: float = 1.0

    def __post_init__(self) -> None:
        """Validate ``omega`` and store it as a ``float``."""
        object.__setattr__(self, "omega", _mean_order(self.omega))

    @property
    def route(self) -> Route:
        """``"linear"`` at the order zero, log OLS; ``"power"`` otherwise."""
        return "linear" if self.omega == 0 else "power"

    def to_dict(self) -> dict[str, str | float]:
        """The type and the order."""
        return {"type": "PowerMean", "omega": self.omega}


@dataclass(frozen=True)
class ExogenousMean:
    """The exogenous conditional-mean moment: the regressors are their instruments."""

    def to_dict(self) -> dict[str, str]:
        """The type."""
        return {"type": "ExogenousMean"}


@dataclass(frozen=True)
class NoFixedEffects:
    """The formulation without fixed effects: every coefficient is a regressor's."""

    def to_dict(self) -> dict[str, str]:
        """The type."""
        return {"type": "NoFixedEffects"}


@dataclass(frozen=True)
class FixedEffectFormulation:
    """Internal provenance of ordered joint or selected-cancelled blocks."""

    blocks: tuple[str, ...]
    cancelled: str | None = None

    def __post_init__(self) -> None:
        """Own the ordered logical names and validate the selected dimension."""
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if not self.blocks or any(not isinstance(x, str) or not x for x in self.blocks):
            raise ValueError("fixed-effect names must be nonempty strings")
        if len(set(self.blocks)) != len(self.blocks):
            raise ValueError("fixed-effect names must be distinct")
        if self.cancelled is not None and self.cancelled not in self.blocks:
            raise ValueError("the cancelled dimension must be one of the blocks")

    def to_dict(self) -> dict[str, object]:
        """The actual formulation and logical block order, detached."""
        return {
            "type": "FixedEffectFormulation",
            "blocks": self.blocks,
            "cancelled": self.cancelled,
        }


def _default_variance() -> PowerVariance:
    """``PowerVariance(tau=1)``, the interface default."""
    return PowerVariance(tau=1.0)


# The components that a serialized estimator may name, by their ``type`` entry: the
# records of this module and the two accepted variances, and nothing else.
REGISTRY: Mapping[str, type] = MappingProxyType(
    {
        component.__name__: component
        for component in (
            PowerMean,
            PowerVariance,
            NB2Variance,
            ExogenousMean,
            NoFixedEffects,
            FixedEffectFormulation,
        )
    }
)


@dataclass(frozen=True)
class Estimator:
    """What is estimated: mean order, working variance, moment and formulation.

    Each component is checked by exact type.

    Parameters
    ----------
    mean : PowerMean, default PowerMean()
        The mean order, 1 by default.
    variance : PowerVariance or NB2Variance, default PowerVariance(tau=1.0)
        The working variance.
    moment : ExogenousMean, default ExogenousMean()
        The moment; instruments are refused as not yet available.
    fixed_effects : NoFixedEffects, default NoFixedEffects()
        The formulation; fixed effects are refused as not yet available.

    Raises
    ------
    TypeError
        ``mean`` is not a ``PowerMean``.
    SpecificationError
        ``variance`` is not one of the two accepted variances
        (``specification.invalid_value``); ``moment`` or ``fixed_effects`` asks for
        what is not yet available (``specification.unimplemented``).
    """

    mean: PowerMean = field(default_factory=PowerMean)
    variance: Variance = field(default_factory=_default_variance)
    moment: ExogenousMean = field(default_factory=ExogenousMean)
    fixed_effects: NoFixedEffects | FixedEffectFormulation = field(
        default_factory=NoFixedEffects
    )

    def __post_init__(self) -> None:
        """Check each component by exact type."""
        if type(self.mean) is not PowerMean:
            raise TypeError("mean must be PowerMean")
        if type(self.variance) not in (PowerVariance, NB2Variance):
            raise SpecificationError(
                "specification.invalid_value",
                {
                    "argument": "variance",
                    "value": self.variance,
                    "valid": messages.format_keys(VARIANCES),
                },
            )
        if type(self.moment) is not ExogenousMean:
            raise SpecificationError(
                "specification.unimplemented",
                {"feature": "instrumental-variable moments"},
            )
        if type(self.fixed_effects) not in (NoFixedEffects, FixedEffectFormulation):
            raise SpecificationError(
                "specification.unimplemented",
                {"feature": "fixed effects"},
            )

    @property
    def route(self) -> Route:
        """The route of the mean order: ``"linear"`` at zero, else ``"power"``."""
        return self.mean.route

    def to_dict(self) -> dict[str, object]:
        """The type and the four components, each by its own ``to_dict``."""
        return {
            "type": "Estimator",
            "mean": self.mean.to_dict(),
            "variance": self.variance.to_dict(),
            "moment": self.moment.to_dict(),
            "fixed_effects": self.fixed_effects.to_dict(),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> Estimator:
        """The record of :meth:`to_dict`, each component rebuilt from its type.

        Raises ``ValueError`` for a record of another type, ``KeyError`` for a missing
        component and ``SpecificationError`` (``specification.invalid_value``) for a
        component whose type is not one of :data:`REGISTRY`.
        """
        data: dict[str, Any] = dict(record)
        if data.pop("type", None) != "Estimator":
            raise ValueError("not an Estimator record")
        for key in ("mean", "variance", "moment", "fixed_effects"):
            nested: dict[str, Any] = dict(data[key])
            name = nested.pop("type", None)
            if name not in REGISTRY:
                raise SpecificationError(
                    "specification.invalid_value",
                    {
                        "argument": key,
                        "value": name,
                        "valid": messages.format_keys(REGISTRY),
                    },
                )
            data[key] = REGISTRY[name](**nested)
        return cls(**data)


# -------------------------------------------------------------------- the inference


def explicit_ssc(value: object) -> str | ConventionValues:
    """An explicit convention of a named HC request: a preset name or a custom record.

    The domain of the parser of the conventions
    (:func:`powermean.inference.ssc.resolve_convention`), with its refusal: a preset
    name is put in force by its name, a ``ConventionValues`` record is a custom record
    whatever its values, owned as a copy; anything else, ``None`` included, is refused
    (``specification.invalid_value``, argument ``ssc``).
    """
    if isinstance(value, str) and value in PRESETS:
        return value
    if type(value) is ConventionValues:
        return replace(value)
    raise invalid("ssc", value, VALID_SSC)


def explicit_bread(value: object) -> Bread:
    """An explicit bread: ``"observed"`` or ``"expected"``, an exact string.

    Anything else, ``None`` included, is refused (``specification.invalid_value``,
    argument ``bread``).
    """
    for bread in BREADS:
        if isinstance(value, str) and value == bread:
            return bread
    raise invalid("bread", value, messages.format_keys(BREADS))


@dataclass(frozen=True)
class Inference:
    """The covariance requested at a fit without fixed effects.

    Parameters
    ----------
    covariance : {"robust", "cluster", "hc"}, default "robust"
        The robust covariance, a cluster covariance, or a named HC method.
    cluster : str, optional
        The single cluster variable, ``a^b`` the interaction of
        two; with ``"cluster"``, required unless ``clusters`` is given, refused
        otherwise.
    method : {"HC0", "HC1", "HC2", "HC3", "HC4", "HC4m", "HC5"}, optional
        The named HC method, an exact string; required with ``"hc"``.
    ssc : str, ConventionValues or OMITTED, optional
        The explicit convention of a named HC request, of a typed cluster request or
        of the robust covariance: a preset name or a
        custom record; ``OMITTED`` where the request gives none, so that the default or
        inherited convention is not taken for an explicit one.
    bread : {"observed", "expected"}, default "observed"
        The bread of a named HC request.
    clusters : ClusterRequest, optional
        The typed cluster request of the expanded menu: its dimensions in order,
        repetitions kept, its multiway form
        and its ``"psd"`` key, each as given; with ``"cluster"`` in place of
        ``cluster``, refused otherwise.

    Raises
    ------
    SpecificationError
        Another covariance, method, convention or bread
        (``specification.invalid_value``); a cluster variable beside a covariance
        that is not ``"cluster"`` (``specification.conflicting``).
    ValueError
        No cluster variable, or an empty one, with ``"cluster"``, or both a variable
        and a typed request; a typed request beside another covariance, of another
        type or with the pair adjustment of a fit with fixed effects; no method with
        ``"hc"``; a method, an explicit convention or the expected bread beside the
        single cluster variable; a method or the expected bread beside
        the robust covariance or a typed cluster request.

    Notes
    -----
    The robust covariance without an explicit convention and the single cluster
    variable keep the ``"stata"`` convention and the normal reference,
    constants written by :meth:`to_dict` and required by :meth:`from_dict`.  A named HC
    request records its method, its bread and its explicit convention only, a typed
    cluster request its typed record and its explicit convention only, and a robust
    request with an explicit convention that convention only; the convention in force
    and its reference are resolved with the counts of the fit.
    """

    covariance: Covariance = "robust"
    cluster: str | None = None
    method: HCMethod | None = None
    ssc: str | ConventionValues | Omitted = OMITTED
    bread: Bread = "observed"
    clusters: ClusterRequest | None = None

    def __post_init__(self) -> None:
        """Check the covariance, its cluster request and the settings of a method.

        The covariance is read as ``object``: this is the run-time check of what the
        field declares.
        """
        covariance: object = self.covariance
        if covariance not in get_args(Covariance):
            raise SpecificationError(
                "specification.invalid_value",
                {
                    "argument": "covariance",
                    "value": covariance,
                    "valid": messages.format_keys(get_args(Covariance)),
                },
            )
        if covariance == "cluster":
            self._check_cluster()
        elif self.cluster is not None:
            raise SpecificationError(
                "specification.conflicting",
                {"first": "cluster", "second": f'covariance="{covariance}"'},
            )
        elif self.clusters is not None:
            raise ValueError("a typed cluster request belongs to a cluster covariance")
        if covariance == "hc":
            self._check_method()
        elif self.clusters is not None or (covariance == "robust" and given(self.ssc)):
            self._check_convention()
        elif self.method is not None or given(self.ssc) or self.bread != "observed":
            raise ValueError(
                "the robust and cluster covariances take no method, explicit "
                "convention or expected bread"
            )

    def _check_cluster(self) -> None:
        """The single cluster variable, or one typed cluster request."""
        if self.clusters is None:
            if not isinstance(self.cluster, str) or not self.cluster:
                raise ValueError("a cluster covariance names one cluster variable")
            return
        if self.cluster is not None:
            raise ValueError("a typed cluster request names its dimensions itself")
        if type(self.clusters) is not ClusterRequest:
            raise ValueError("a typed cluster request is a ClusterRequest")
        if given(self.clusters.adjustment):
            raise ValueError("the pair adjustment belongs to a fit with fixed effects")

    def _check_method(self) -> None:
        """The method, an explicit convention and the bread of a named HC request."""
        if self.method is None:
            raise ValueError("a named HC covariance names its method")
        object.__setattr__(self, "method", HCRequest(self.method).method)
        if given(self.ssc):
            object.__setattr__(self, "ssc", explicit_ssc(self.ssc))
        object.__setattr__(self, "bread", explicit_bread(self.bread))

    def _check_convention(self) -> None:
        """The explicit convention of a typed cluster or robust request; no method."""
        if self.method is not None or self.bread != "observed":
            raise ValueError(
                f"a {self.covariance} covariance takes no method and the observed "
                "bread only"
            )
        if given(self.ssc):
            object.__setattr__(self, "ssc", explicit_ssc(self.ssc))

    @property
    def dimensions(self) -> tuple[str, ...]:
        """The requested cluster dimensions in order: none, one, or several."""
        if self.clusters is not None:
            return self.clusters.dimensions
        return () if self.cluster is None else (self.cluster,)

    def _with_ssc(self, record: dict[str, object]) -> dict[str, object]:
        """``record`` with the explicit convention, where one was given."""
        if isinstance(self.ssc, ConventionValues):
            record["ssc"] = self.ssc.to_dict()
        elif isinstance(self.ssc, str):
            record["ssc"] = self.ssc
        return record

    def to_dict(self) -> dict[str, object]:
        """The type, the covariance and its cluster, convention and reference.

        A named HC request writes its method, its bread and its explicit convention
        where one was given (a preset name, or the mapping of a custom record), and no
        convention in force: the round trip keeps the presence of the convention.  A
        typed cluster request writes its record under ``"clusters"`` and its explicit
        convention in the same way, and a robust request with an explicit convention
        that convention alone, without the ``"stata"`` and normal-reference constants.
        """
        if self.clusters is not None:
            return self._with_ssc(
                {
                    "type": "Inference",
                    "covariance": self.covariance,
                    "cluster": None,
                    "clusters": self.clusters.to_dict(),
                }
            )
        if self.covariance == "robust" and given(self.ssc):
            return self._with_ssc(
                {"type": "Inference", "covariance": self.covariance, "cluster": None}
            )
        if self.covariance != "hc":
            return {
                "type": "Inference",
                "covariance": self.covariance,
                "cluster": self.cluster,
                "convention": CONVENTION,
                "reference_distribution": REFERENCE_DISTRIBUTION,
            }
        return self._with_ssc(
            {
                "type": "Inference",
                "covariance": self.covariance,
                "cluster": None,
                "method": self.method,
                "bread": self.bread,
            }
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> Inference:
        """The record of :meth:`to_dict`.

        Raises ``ValueError`` for a record of another type and ``SpecificationError``
        (``specification.unimplemented``) for a robust covariance or a single cluster
        variable whose convention is not ``"stata"`` or whose reference distribution is
        not normal, which such a record does not hold.  A named HC request, a typed
        cluster request and a robust request with an explicit ``"ssc"`` hold no such
        constants.
        """
        data: dict[str, Any] = dict(record)
        if data.pop("type", None) != "Inference":
            raise ValueError("not an Inference record")
        explicit_robust = data.get("covariance") == "robust" and "ssc" in data
        if data.get("covariance") == "hc" or "clusters" in data or explicit_robust:
            ssc = data.pop("ssc", OMITTED)
            if isinstance(ssc, Mapping):
                ssc = ConventionValues.from_dict(ssc)
            if "clusters" in data:
                data["clusters"] = ClusterRequest.from_dict(data["clusters"])
            return cls(ssc=ssc, **data)
        convention = data.pop("convention", None)
        if convention != CONVENTION:
            raise SpecificationError(
                "specification.unimplemented",
                {"feature": f"the convention {convention!r}"},
            )
        reference = data.pop("reference_distribution", None)
        if reference != REFERENCE_DISTRIBUTION:
            raise SpecificationError(
                "specification.unimplemented",
                {
                    "feature": f"the reference distribution {reference!r}",
                },
            )
        return cls(**data)
