"""Small-sample conventions: resolution, counts, factors, references and ownership.

Internal module: the convention is one
explicit object (:class:`~powermean.data.inference_conventions.ConventionValues`) that
never changes the estimator; it supplies scalar factors, a reference distribution and
the PSD rule.  Nothing here parses a public keyword, exposes ``pm.ssc`` or computes a
score: the kernels receive counts and arrays.

Presets, each equal to one canonical custom record:

    stata        N/(N-1)       G_min/(G_min-1)                       normal   cgm
    none         1             1                                     normal   none
    regression   N/(N-K_f)     (N-1)/(N-K_f) * G_min/(G_min-1)       t        cgm

Resolution: the request's explicit ``ssc`` (a preset name or a
custom record; an explicit ``None`` is refused), else the convention inherited from the
fit, else the default ``"stata"``.  The record keeps the requested value with its
presence, the convention in force, its source and, privately, its origin: a preset put
in force by its name (an explicit name, the default, or a fit's convention in force
that was one) or a custom record, whatever its values.  An inherited custom record
stays inherited and custom.

Counts.  ``N`` is the statistical count; ``K`` the exact full-design
rank, slopes plus kept nuisance coordinates, which nesting never changes;
``K_reference = n_slopes + df_a`` with ``df_a`` the request's convention count of
:mod:`powermean.inference.absorbed_rank`, and
``K_factor = K_reference + nested_adjustment``.
``K*`` of the closed option strings is ``K_factor`` in a factor and ``K_reference`` in a
reference.  ``G_min`` is the smallest count over the REQUESTED dimensions, repetitions
kept, never an intersection count.  The ``t`` reference has ``N - K_reference`` degrees
of freedom, or ``min(G_min - 1, N - K_reference)`` for a cluster request.

Ownership.  A robust, HAC or Conley request takes the convention's
robust factor once; a cluster request its cluster factor, one common factor with
``G_min`` under ``cluster_count="min"`` or each component's own ``G`` under ``"each"``,
the reference still using ``G_min``.  A named HC method owns its scalar: HC1
``N/(N-K)`` with the exact ``K`` and ``N > K``, the others 1.  An explicit convention
beside a named method adds nothing when its robust field is ``"none"``, is the same
correction when identical by formula and exact integer counts (``"N/(N-1)"`` with HC1
exactly when ``K = 1``; ``"N/(N-K*)"`` with HC1 exactly when ``K_factor = K``), and is
otherwise a second correction, refused; floating-point values are never compared.  An
inherited or default convention neither applies its factor to a named method nor
conflicts with it, and its reference distribution survives.

Undefined degrees of freedom refuse (``specification.no_residual_dof``) and are never
the exact-zero case: ``N/(N-K*)`` and HC1 need ``N > K``; a ``t`` reference needs a
positive count.  ``N/(N-1)`` needs ``N >= 2`` and ``G/(G-1)`` two clusters, refused as
:mod:`powermean.inference.convention` refuses them.

Sequence of the eigenvalue adjustment.  The resolved convention carries privately the
native sequence in which an indefinite signed multiway estimate is adjusted, selected
by the origin of the convention in force and never by its values: ``"legacy"`` for the
``"stata"`` preset put in force by its name (``ppmlhdfe`` 2.3.4 through its
``reghdfe5`` callee), ``"modern"`` for every other convention in force
(``"regression"``, ``"none"`` with an explicit ``"psd": "cgm"``, and every custom
record, which has no field to select the legacy sequence).  A custom record whose
five values are a preset's keeps that preset's factors, reference and PSD treatment, and
:func:`preset_name` still names the preset its values equal; it is not the preset, and
it selects the modern sequence.  No new public spelling, field or default: the sequence
is read from the convention in force, never requested on its own.

Composition.  The primitive estimate is the one of
:mod:`powermean.inference.absorbed` (``fe_covariance``): source-average rows, frequency
rows divided by ``sqrt(f)`` for the robust meat and not for a cluster total, one stored
row padded with a zero row; the factor multiplies it ONCE
(:func:`scaled_covariance`), and the direct standard errors are the map of
:meth:`powermean.model.order_scale.OrderScale.standard_errors`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from powermean.data.inference_conventions import (
    ClusterCount,
    ConventionValues,
    PsdTreatment,
    Reference,
)
from powermean.data.inference_requests import HC_METHODS, HCMethod
from powermean.data.request_values import OMITTED, Omitted, given, invalid
from powermean.errors import SpecificationError
from powermean.inference.absorbed_rank import AbsorbedCount
from powermean.inference.convention import (
    _replicated_rows,
    _source_average,
    robust_factor,
)
from powermean.inference.psd import Sequence as PsdSequence
from powermean.inference.vcov import (
    CovarianceEstimate,
    cluster_covariance,
    hc_covariance,
)
from powermean.model.order_scale import OrderScale
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, _finite_result

IntegerArray = NDArray[np.integer[Any]]
ConventionSource = Literal["explicit", "inherited", "default"]
ConventionOrigin = Literal["preset", "custom"]
FactorOwner = Literal["convention", "method"]
RequestKind = Literal["robust", "cluster", "hac", "conley"]
MeatKind = Literal["robust", "cluster"]

STATA = ConventionValues("N/(N-1)", "G/(G-1)", "min", "normal", "cgm")
NONE = ConventionValues("none", "none", "min", "normal", "none")
REGRESSION = ConventionValues("N/(N-K*)", "(N-1)/(N-K*)*G/(G-1)", "min", "t", "cgm")
PRESETS: Mapping[str, ConventionValues] = MappingProxyType(
    {"stata": STATA, "none": NONE, "regression": REGRESSION}
)
PRESET_NAMES: tuple[str, ...] = tuple(PRESETS)
VALID_SSC = "'stata', 'none', 'regression' or a ConventionValues record"
ROBUST_KINDS: tuple[RequestKind, ...] = ("robust", "hac", "conley")


# ------------------------------------------------------------------ resolution


def preset_name(values: ConventionValues) -> str | None:
    """The name of the preset equal to ``values``, or ``None`` for a custom record."""
    return next((name for name, preset in PRESETS.items() if preset == values), None)


def sequence_of(values: ConventionValues, origin: ConventionOrigin) -> PsdSequence:
    """Native adjustment sequence of a convention in force (module docstring).

    ``"legacy"`` for the ``"stata"`` preset put in force by its name (``origin`` is
    ``"preset"``), ``"modern"`` for every other convention, every custom record
    included whatever its values.
    """
    if origin not in ("preset", "custom"):
        raise ValueError("origin must be 'preset' or 'custom'")
    found: PsdSequence = (
        "legacy" if origin == "preset" and values == STATA else "modern"
    )
    return found


@dataclass(frozen=True)
class ResolvedConvention:
    """The convention in force for one request, with what was requested.

    Attributes
    ----------
    requested : str, ConventionValues or OMITTED
        The explicit ``ssc`` as given, or :data:`OMITTED` when absent.
    values : ConventionValues
        The convention in force.
    source : {"explicit", "inherited", "default"}
        Where it came from.
    origin : {"preset", "custom"}
        How it was put in force, kept privately: by a preset's name (an explicit name,
        the default, or a fit's convention in force that was one) or by a custom
        record, whatever its values.  It selects the sequence of the eigenvalue
        adjustment.
    """

    requested: str | ConventionValues | Omitted
    values: ConventionValues
    source: ConventionSource
    origin: ConventionOrigin

    def __post_init__(self) -> None:
        """The origin agrees with the values, an explicit request and the default."""
        if self.origin not in ("preset", "custom"):
            raise ValueError("origin must be 'preset' or 'custom'")
        if self.origin == "preset" and preset_name(self.values) is None:
            raise ValueError("a preset origin holds the values of a preset")
        named = isinstance(self.requested, str)
        if self.source == "explicit" and named != (self.origin == "preset"):
            raise ValueError(
                "an explicit preset is requested by name, a custom record as a record"
            )
        if self.source == "default" and (self.origin, self.values) != ("preset", STATA):
            raise ValueError("the default is the 'stata' preset")

    @property
    def preset(self) -> str | None:
        """The preset name equal to the convention in force, or ``None``."""
        return preset_name(self.values)

    @property
    def sequence(self) -> PsdSequence:
        """Native PSD sequence of the convention in force (internal)."""
        return sequence_of(self.values, self.origin)

    @property
    def label(self) -> str:
        """``ssc=...`` as a caller would write it, for a refusal and the frame record.

        A preset by its name; a custom record by its five fields, whatever its values.
        """
        if self.origin == "preset":
            return f"ssc={self.preset!r}"
        fields = ", ".join(
            f"{k}={v!r}" for k, v in self.values.to_dict().items() if k != "type"
        )
        return f"ssc=pm.ssc({fields})"


def _explicit(requested: object) -> ConventionValues:
    """The explicit convention of a request: a preset name or a custom record."""
    if isinstance(requested, str):
        found = PRESETS.get(requested)
        if found is None:
            raise invalid("ssc", requested, VALID_SSC)
        return found
    if type(requested) is ConventionValues:
        return requested
    raise invalid("ssc", requested, VALID_SSC)


def resolve_convention(
    requested: object = OMITTED,
    inherited: ConventionValues | ResolvedConvention | None = None,
) -> ResolvedConvention:
    """The convention in force: explicit, else inherited, else ``"stata"``.

    Parameters
    ----------
    requested : str, ConventionValues or OMITTED
        The explicit ``ssc`` of the request; :data:`OMITTED` when absent.  ``None`` is
        a value, refused.  A name puts its preset in force; a record is a custom
        record, whatever its values.
    inherited : ConventionValues or ResolvedConvention, optional
        The convention of the fit, after the fit: a custom record, whatever its
        values, or the fit's convention in force with its origin (as the retained
        initial request's ``"stata"``, resolved from its recorded name); ``None`` at
        the fit or where no covariance was formed with it.

    Returns
    -------
    ResolvedConvention
        The requested value with its presence, the convention in force, its source and
        its origin.

    Raises
    ------
    SpecificationError
        ``specification.invalid_value`` for an explicit value outside the domain.
    TypeError
        ``inherited`` is neither ``None``, a ``ConventionValues`` nor a
        ``ResolvedConvention``.
    """
    if inherited is not None and type(inherited) not in (
        ConventionValues,
        ResolvedConvention,
    ):
        raise TypeError(
            "inherited must be a ConventionValues, a ResolvedConvention or None"
        )
    if given(requested):
        values = _explicit(requested)
        if isinstance(requested, str):
            return ResolvedConvention(requested, values, "explicit", "preset")
        return ResolvedConvention(values, values, "explicit", "custom")
    if isinstance(inherited, ResolvedConvention):
        return ResolvedConvention(
            OMITTED, inherited.values, "inherited", inherited.origin
        )
    if inherited is not None:
        return ResolvedConvention(OMITTED, inherited, "inherited", "custom")
    return ResolvedConvention(OMITTED, STATA, "default", "preset")


# ---------------------------------------------------------------------- counts


@dataclass(frozen=True)
class ConventionCounts:
    """The counts of one request, checked as a record.

    Attributes
    ----------
    count : int
        ``N``, the statistical count: the exact frequency total of the active rows,
        else their number.
    n_slopes : int
        The identified slope columns, the intercept included where one is reported.
    rank : int
        ``K``, the exact full-design rank: slopes plus kept nuisance coordinates.
    k_reference : int
        ``n_slopes + df_a`` with the request's convention count ``df_a``.
    k_factor : int
        ``k_reference + nested_adjustment``.
    nested_adjustment : int
        One when a block is nested in a requested dimension, zero otherwise.
    clusters : tuple of int
        ``G`` of each requested dimension, in the order of the request, repetitions
        kept; empty for a request without clusters.
    """

    count: int
    n_slopes: int
    rank: int
    k_reference: int
    k_factor: int
    nested_adjustment: int
    clusters: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Check the counts: integers in their domains, consistent with each other."""
        values = (
            self.count,
            self.n_slopes,
            self.rank,
            self.k_reference,
            self.k_factor,
            self.nested_adjustment,
            *self.clusters,
        )
        if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise ValueError("convention counts must be integers")
        object.__setattr__(self, "clusters", tuple(int(g) for g in self.clusters))
        if self.count < 1:
            raise ValueError("the statistical count N must be positive")
        if (
            self.n_slopes < 0
            or self.rank < self.n_slopes
            or self.k_reference < self.n_slopes
        ):
            raise ValueError("the rank and the convention count include every slope")
        if self.nested_adjustment not in (0, 1):
            raise ValueError("the nested adjustment is zero or one")
        if self.k_factor != self.k_reference + self.nested_adjustment:
            raise ValueError("K_factor is K_reference plus the nested adjustment")
        if self.nested_adjustment and not self.clusters:
            raise ValueError("a noncluster request has no nested adjustment")
        if any(g < 1 for g in self.clusters):
            raise ValueError("every requested dimension has at least one cluster")

    @property
    def g_min(self) -> int | None:
        """The smallest cluster count over the requested dimensions, or ``None``."""
        return min(self.clusters) if self.clusters else None


def convention_counts(
    count: int,
    n_slopes: int,
    rank: int,
    absorbed: AbsorbedCount | None,
    clusters: Sequence[int] = (),
) -> ConventionCounts:
    """The counts of a request from its absorbed convention count.

    Parameters
    ----------
    count : int
        ``N``.
    n_slopes : int
        The identified slope columns (with the intercept of a no-FE design, once).
    rank : int
        ``K``, the exact full-design rank.
    absorbed : AbsorbedCount or None
        The request's count of :func:`powermean.inference.absorbed_rank.absorbed_count`
        on its own cluster dimensions; ``None`` without absorbed effects.
    clusters : sequence of int, optional
        ``G`` of each requested dimension; empty for a robust request.

    Raises
    ------
    ValueError
        A count formed for a cluster request handed to a request without clusters
        (its nesting is request-specific and must be recomputed), or invalid counts.
    """
    groups = tuple(clusters)
    df_a = 0 if absorbed is None else absorbed.absorbed
    nested = 0 if absorbed is None else absorbed.nested_adjustment
    if nested and not groups:
        raise ValueError(
            "a noncluster request recomputes its un-nested convention count"
        )
    reference = n_slopes + df_a
    return ConventionCounts(
        count, n_slopes, rank, reference, reference + nested, nested, groups
    )


# --------------------------------------------------------------------- factors


@dataclass(frozen=True)
class ScalarFactor:
    """One scalar factor as an exact ratio of integer counts.

    ``value`` is ``numerator / denominator`` in one division; ``formula`` the closed
    string of the convention, ``"1"``, or the method's own (``"HC1 N/(N-K)"``);
    ``owner`` who applies it.
    """

    numerator: int
    denominator: int
    formula: str
    owner: FactorOwner

    def __post_init__(self) -> None:
        """A positive denominator."""
        if self.denominator <= 0 or self.numerator <= 0:
            raise ValueError("a factor is a ratio of positive counts")

    @property
    def value(self) -> float:
        """The factor as a double, one division."""
        return self.numerator / self.denominator


@dataclass(frozen=True)
class ReferenceDistribution:
    """The reference distribution of the convention: normal, or ``t`` with ``df``."""

    kind: Reference
    df: int | None

    def __post_init__(self) -> None:
        """``t`` carries positive degrees of freedom; the normal carries none."""
        if (self.kind == "t") != (self.df is not None) or (
            self.df is not None and self.df <= 0
        ):
            raise ValueError("a t reference has positive degrees of freedom")


UNIT = ScalarFactor(1, 1, "1", "convention")


def _no_residual_dof(method: str, count: int, rank: int) -> SpecificationError:
    """The refusal of a correction whose degrees of freedom are not positive."""
    return SpecificationError(
        "specification.no_residual_dof",
        {"method": method, "count": count, "rank": rank},
    )


def robust_factor_of(
    values: ConventionValues, counts: ConventionCounts
) -> ScalarFactor:
    """The convention's robust factor: ``1``, ``N/(N-1)`` or ``N/(N-K_factor)``."""
    n, k = counts.count, counts.k_factor
    if values.robust == "none":
        return UNIT
    if values.robust == "N/(N-1)":
        robust_factor(n)  # the accepted refusal of N < 2
        return ScalarFactor(n, n - 1, values.robust, "convention")
    if n <= k:
        raise _no_residual_dof(f"The robust factor {values.robust!r}", n, k)
    return ScalarFactor(n, n - k, values.robust, "convention")


def cluster_factor_of(
    values: ConventionValues, counts: ConventionCounts, groups: int
) -> ScalarFactor:
    """The convention's cluster factor for ``G = groups``, one exact ratio.

    ``1``, ``G/(G-1)`` or ``(N-1)/(N-K_factor) * G/(G-1)``.
    """
    n, k = counts.count, counts.k_factor
    if values.cluster == "none":
        return UNIT
    if groups < 2:
        raise ValueError("the cluster factor G/(G - 1) needs at least two clusters")
    if values.cluster == "G/(G-1)":
        return ScalarFactor(groups, groups - 1, values.cluster, "convention")
    if n <= k:
        raise _no_residual_dof(f"The cluster factor {values.cluster!r}", n, k)
    return ScalarFactor(
        (n - 1) * groups, (n - k) * (groups - 1), values.cluster, "convention"
    )


def reference_of(
    values: ConventionValues, counts: ConventionCounts, *, clustered: bool
) -> ReferenceDistribution:
    """The reference: normal, or ``t`` with its degrees of freedom.

    ``N - K_reference``, or ``min(G_min - 1, N - K_reference)`` for a cluster request.
    """
    if values.reference == "normal":
        return ReferenceDistribution("normal", None)
    df = counts.count - counts.k_reference
    g_min = counts.g_min
    if clustered:
        if g_min is None:
            raise ValueError("a cluster request names its dimensions")
        df = min(g_min - 1, df)
    if df <= 0:
        raise _no_residual_dof("The t reference", counts.count, counts.k_reference)
    return ReferenceDistribution("t", df)


def method_factor(method: HCMethod, counts: ConventionCounts) -> ScalarFactor:
    """A named HC method's own scalar: HC1 ``N/(N-K)`` with the exact ``K``, else 1."""
    if method not in HC_METHODS:
        raise ValueError(f"unknown HC method {method!r}")
    if method != "HC1":
        return ScalarFactor(1, 1, "1", "method")
    n, k = counts.count, counts.rank
    if n <= k:
        raise _no_residual_dof("HC1", n, k)
    return ScalarFactor(n, n - k, "HC1 N/(N-K)", "method")


def _same_correction(method: HCMethod, robust: str, counts: ConventionCounts) -> bool:
    """Whether an explicit robust factor is HC's own by formula and exact counts."""
    if method != "HC1":
        return False
    if robust == "N/(N-1)":
        return counts.rank == 1
    return counts.k_factor == counts.rank


def owned_factor(
    resolved: ResolvedConvention,
    counts: ConventionCounts,
    *,
    kind: RequestKind,
    method: HCMethod | None = None,
) -> ScalarFactor:
    """The one scalar of a robust, HAC, Conley or named-HC request, and its owner.

    Raises
    ------
    SpecificationError
        ``specification.conflicting``: a named method with a cluster, HAC or Conley
        request, or an explicit convention whose factor is a second correction beside
        the method's.
    """
    if method is None:
        if kind == "cluster":
            raise ValueError("a cluster request takes cluster_factors_of")
        return robust_factor_of(resolved.values, counts)
    if kind != "robust":
        raise SpecificationError(
            "specification.conflicting",
            {"first": f"vcov={method!r}", "second": f"a {kind} request"},
        )
    own = method_factor(method, counts)
    robust = resolved.values.robust
    if resolved.source != "explicit" or robust == "none":
        return own
    if _same_correction(method, robust, counts):
        return own
    raise SpecificationError(
        "specification.conflicting",
        {"first": resolved.label, "second": f"vcov={method!r}"},
    )


@dataclass(frozen=True)
class AppliedConvention:
    """The convention of one request, applied to its counts: factors and reference.

    Attributes
    ----------
    convention : ResolvedConvention
        The convention in force and its provenance.
    counts : ConventionCounts
        The counts.
    kind : {"robust", "cluster", "hac", "conley"}
        The kind of the request.
    method : HC method or None
        A named HC method, which owns the scalar, or ``None``.
    factor : ScalarFactor or None
        The one scalar applied to the primitive covariance; ``None`` only for a
        multiway request under ``cluster_count="each"``, where each component takes
        its own (``components``).
    components : tuple of ScalarFactor
        For a cluster request, the factor of each requested dimension: its own ``G``
        under ``"each"``, the common ``G_min`` factor under ``"min"``; empty otherwise.
    cluster_count : {"min", "each"} or None
        The rule of the components for a cluster request.
    reference : ReferenceDistribution
        The reference distribution of the convention in force.
    psd : {"cgm", "none"}
        The treatment of an indefinite signed multiway estimate, carried for its
        owner.
    """

    convention: ResolvedConvention
    counts: ConventionCounts
    kind: RequestKind
    method: HCMethod | None
    factor: ScalarFactor | None
    components: tuple[ScalarFactor, ...]
    cluster_count: ClusterCount | None
    reference: ReferenceDistribution
    psd: PsdTreatment

    @property
    def sequence(self) -> PsdSequence:
        """Native PSD sequence of the convention in force (internal)."""
        return self.convention.sequence


def apply_convention(
    resolved: ResolvedConvention,
    counts: ConventionCounts,
    *,
    kind: RequestKind,
    method: HCMethod | None = None,
) -> AppliedConvention:
    """The factors and the reference of ``resolved`` on ``counts`` for one request.

    Parameters
    ----------
    resolved : ResolvedConvention
        The convention in force.
    counts : ConventionCounts
        The counts of the request; a cluster request names at least one dimension.
    kind : {"robust", "cluster", "hac", "conley"}
        The kind of the request.
    method : HC method, optional
        A named HC method; ``None`` for the robust estimator and the others.

    Raises
    ------
    SpecificationError
        ``specification.conflicting`` (ownership) or ``specification.no_residual_dof``
        (undefined degrees of freedom).
    ValueError
        A cluster request without dimensions, or a dimension of one cluster under a
        ``G/(G-1)`` factor.
    """
    values = resolved.values
    if kind != "cluster":
        own = owned_factor(resolved, counts, kind=kind, method=method)
        reference = reference_of(values, counts, clustered=False)
        return AppliedConvention(
            resolved, counts, kind, method, own, (), None, reference, values.psd
        )
    if method is not None:
        owned_factor(resolved, counts, kind=kind, method=method)
    g_min = counts.g_min
    if g_min is None:
        raise ValueError("a cluster request names its dimensions")
    common = cluster_factor_of(values, counts, g_min)
    scalar: ScalarFactor | None = common
    if values.cluster_count == "min":
        components = tuple(common for _ in counts.clusters)
    else:
        components = tuple(
            cluster_factor_of(values, counts, g) for g in counts.clusters
        )
        scalar = components[0] if len(components) == 1 else None
    reference = reference_of(values, counts, clustered=True)
    return AppliedConvention(
        resolved,
        counts,
        kind,
        method,
        scalar,
        components,
        values.cluster_count,
        reference,
        values.psd,
    )


# ----------------------------------------------------------------- composition


def primitive_estimate(
    influence: FloatArray,
    weights: FloatArray,
    *,
    frequency: bool,
    cluster: tuple[str, IntegerArray] | None = None,
) -> CovarianceEstimate:
    """The primitive estimate of the active rows, as in ``fe_covariance``.

    Parameters
    ----------
    influence, weights : array
        The influence rows and the raw weights of the active rows, in their order.
    frequency : bool
        Whether the weights are frequency counts.
    cluster : (str, array), optional
        The name and the codes of one cluster dimension on the same rows; ``None``
        for the robust estimator.

    Returns
    -------
    CovarianceEstimate
        ``HC0`` of the source-average rows (``u/sqrt(f)`` for frequency counts, one
        stored row padded with a zero row), or the uncorrected one-way estimate of the
        cluster totals; no factor.
    """
    if not isinstance(frequency, bool):
        raise TypeError("frequency must be a bool")
    if influence.ndim != 2 or weights.shape != (influence.shape[0],):
        raise ValueError("influence rows and weights must be aligned")
    if cluster is None:
        rows = _replicated_rows(influence, weights) if frequency else influence
        if rows.shape[0] == 1:
            rows = np.vstack([rows, np.zeros_like(rows)])
        return hc_covariance(_source_average(rows))
    name, codes = cluster
    if codes.shape != (influence.shape[0],) or np.any(codes < 0):
        raise ValueError(f"cluster dimension {name!r} must have a code per active row")
    if np.unique(codes).size < 2:
        raise ValueError(
            f"cluster dimension {name!r} must contain at least two clusters"
        )
    return cluster_covariance(_source_average(influence), {name: codes})


@dataclass(frozen=True, eq=False)
class ConventionCovariance:
    """A covariance of ``theta`` under a convention: the factor applied once.

    Attributes
    ----------
    covariance : array
        The factor times the primitive's covariance, exactly symmetric, owned.
    estimate : CovarianceEstimate
        The primitive's estimate without the factor, with its diagnostics.
    applied : AppliedConvention
        The convention, counts, factor, reference and provenance.
    kind : {"robust", "cluster"}
        The primitive's kind.
    """

    covariance: FloatArray
    estimate: CovarianceEstimate
    applied: AppliedConvention
    kind: MeatKind

    @property
    def factor(self) -> float:
        """The scalar applied."""
        found = self.applied.factor
        assert found is not None
        return found.value

    @property
    def reference(self) -> ReferenceDistribution:
        """The reference distribution."""
        return self.applied.reference


def scaled_covariance(
    estimate: CovarianceEstimate, applied: AppliedConvention
) -> ConventionCovariance:
    """``factor * estimate.covariance``, once; ``FloatingPointError`` on range.

    A cluster estimate must hold the cluster count of the applied counts.
    """
    if type(estimate) is not CovarianceEstimate:
        raise TypeError("estimate must be a CovarianceEstimate")
    if applied.kind not in ("robust", "cluster"):
        raise ValueError("the primitive covariance is robust or one cluster")
    kind: MeatKind = "robust" if applied.kind == "robust" else "cluster"
    if applied.factor is None:
        raise ValueError("no single factor applies to every component of this request")
    if kind == "cluster":
        components = estimate.components
        if len(components) != 1 or components[0].n_clusters != applied.counts.g_min:
            raise ValueError(
                "the cluster estimate does not hold the request's clusters"
            )
    elif estimate.components:
        raise ValueError("a robust request takes the robust primitive")
    with np.errstate(over="ignore", invalid="ignore"):
        covariance: FloatArray = applied.factor.value * estimate.covariance
    return ConventionCovariance(
        _immutable(_finite_result(covariance, "covariance")), estimate, applied, kind
    )


def direct_errors(covariance: FloatArray, order: OrderScale) -> FloatArray:
    """``sqrt(V_jj)`` by the direct map of the order scale, once.

    NaN where a variance is negative, zero where it is zero; nothing floored.
    The same map as ``powermean.inference.provider.direct_standard_errors``, which
    reads the ``Covariance`` record.
    """
    if type(order) is not OrderScale:
        raise TypeError("order must be an OrderScale")
    variances = np.diag(covariance)
    errors = np.full(variances.shape, np.nan)
    valid = variances >= 0
    errors[valid] = np.sqrt(variances[valid])
    return order.standard_errors(errors)


__all__ = [
    "NONE",
    "PRESETS",
    "PRESET_NAMES",
    "REGRESSION",
    "STATA",
    "UNIT",
    "AppliedConvention",
    "ConventionCounts",
    "ConventionCovariance",
    "ReferenceDistribution",
    "ResolvedConvention",
    "ScalarFactor",
    "apply_convention",
    "cluster_factor_of",
    "convention_counts",
    "direct_errors",
    "method_factor",
    "owned_factor",
    "preset_name",
    "primitive_estimate",
    "reference_of",
    "resolve_convention",
    "robust_factor_of",
    "scaled_covariance",
    "sequence_of",
]
