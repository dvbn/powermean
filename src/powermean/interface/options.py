"""The keywords of the public calls: their collectors, validation and normalization.

No ``**kwargs`` passes between layers, and options are data.

The collectors.  A public call takes its formula and its data as its two named
parameters, positionally or by keyword (``data=df``), and every other argument as a
named keyword of a closed
``TypedDict`` (:class:`FitOptions`,
:class:`PrepareOptions`, :class:`PreparedFitOptions`), so that the formal signature
stays short while each keyword is
declared with its type and documented here.  The collector is a normalization
boundary at the entry point, not an options bag: :func:`collect` reads the keywords
given, refuses a key that no collector declares (``specification.unknown_option``,
listing the valid keywords), refuses the explicit request of a feature that is not
available (``specification.unimplemented``) and
hands explicit typed records to the preparation and the dispatch.  Nothing is
forwarded loosely between layers, no keyword has an ``Any`` type, and the run-time
``inspect.signature`` of a public function shows the collector, not its keys.

Omitted and explicit values.  A keyword that is omitted is omitted: it is told apart
from one given with the same value.  So the default working variance, ``tau = 1``,
does not conflict with an explicit ``variance=pm.NB2(kappa)``; the fixed values of a
convenience wrapper (``ppml``: ``omega = 1``, ``tau = 1``; ``gpml``: ``tau = 2``;
``logols``: ``omega = 0``) refuse a contradictory explicit value and accept an equal
one (``specification.conflicting``); the aliases ``cluster`` and ``vcov`` must agree;
and an unavailable feature that is omitted is not activated and does not invalidate
an ordinary call, while its explicit inactive value (``cancel=None``,
``difference=False``, ``bias_correction=False``, ``moment="multiplicative"``,
``bread="observed"``, ``solver="newton"``) is accepted as a request of nothing.  No
advisor is instantiated.

Named HC methods.  Without fixed effects, ``vcov`` also takes the exact
strings ``"HC0"`` to ``"HC5"`` and ``"HC4m"``, never beside a cluster; ``ssc`` and
``bread`` are then the per-request keywords of that method, validated by their domains
(a preset name or a custom record; ``"observed"`` or ``"expected"``; an explicit
``None`` refused) and recorded with the request
(:class:`~powermean.data.specs.Inference`).  Beside the robust covariance ``bread``
stays an unavailable feature, refused or accepted by the collector as
above, and so does ``bread`` beside a cluster covariance.  A ``vcov`` or ``cluster``
given to a fit replaces the prepared request whole; ``ssc`` or ``bread`` given alone
apply to a prepared named method, and ``ssc`` alone to a prepared robust or cluster
request.

The robust covariance.  The
keyword ``ssc`` beside ``vcov="robust"``, beside neither ``vcov`` nor ``cluster`` (the
robust default), or given alone to a fit of a prepared robust request, is that
request's explicit convention, validated by its domain (a preset name or a custom
record; an explicit ``None`` refused) and recorded with the typed request, whose robust
factor applies once and whose reference is the convention's.  An explicit ``"stata"``
is recorded as explicit too, its factor unchanged.  Omitted, the robust request keeps
the record, the provider and the covariance of the plain request.

Cluster and multiway requests.  The
key ``"cluster"`` of a ``vcov`` mapping, and its shorthand ``cluster``, name one
variable, ``a^b`` the interaction of two, or several dimensions joined by ``+``, as
``{"cluster": "a + b"}``, in their order, repetitions kept and no number of dimensions
fixed; beside it, ``"multiway"`` (``"cgm"``, the signed form and the default of two or
more dimensions, or ``"ddg"``) and ``"psd"`` (``"cgm"`` or ``"none"``,
valid on a signed multiway request only) are the other keys of the mapping, each
validated by the canonical :class:`~powermean.data.inference_requests.ClusterRequest`;
``"ssc"`` is no key of it (an unknown key, whose valid keys are listed)
and ``"adjustment"`` is refused as not available.  The keyword ``ssc`` beside
a cluster request, or given alone to a fit of a prepared cluster request, is that
request's explicit convention, validated by its domain (an explicit ``None`` refused)
and recorded with the typed request.  A single cluster variable with neither
``"multiway"`` nor ``"psd"`` and with ``ssc`` omitted keeps the record, the preparation
and the covariance of the plain one-way request; every other cluster request is typed,
including an explicit ``"stata"`` whose requested-convention provenance is kept.

The tolerance.  ``tol`` is a shorthand for the option ``tol`` of the certificate,
and the one canonical table of defaults and domains is
:class:`~powermean.solve.options.SolverOptions`.  :func:`resolve_solver_options`
applies it: neither given, the canonical default; only one given, that value, validated
by the canonical domain; both given, each validated by the domain first (a Boolean does
not pass as a number) and then compared exactly, never approximately: equal values are
one value, unequal values are refused with ``specification.conflicting`` and no
precedence.  The presence of ``tol`` in a mapping is read before the defaults are
filled, so that an absent nested ``tol`` never conflicts with a top-level one, and a
:class:`~powermean.solve.options.SolverOptions` given as ``solver_options`` is a
complete resolved snapshot whose ``tol`` is a supplied value, equal to the default or
not: nothing infers an omission from a value.  Unknown keys of the mapping keep the
diagnostic of :func:`~powermean.solve.options.resolve_options`.  Presence is read on
the keywords as given: a ``tol`` keyword that is present is a supplied value whatever
it holds, so ``tol=None`` is validated like any other value and refused by the
canonical domain (``specification.invalid_value``), before any comparison; only the
absence of the keyword is the omission that takes the canonical default.  This rule
is the tolerance's; it is not a rule for every
nullable keyword.  A snapshot record is validated again against the canonical table
at this boundary (:func:`resolve_solver_options`): every value it holds is resolved
by the domain of its option, as a mapping's values are, so that a record whose
stored state was formed without the record's own validation (a reconstruction
that bypassed the constructor, as a deserialization can) reaches no solve with a
value outside its domain; a record that lacks a field is refused as not a snapshot.
A valid record resolves to an equal record.

What is normalized.  :class:`Specification` holds the typed records of one fit: the
:class:`~powermean.data.specs.Estimator` (any finite real order; a power variance
with ``tau`` in ``[1, 2]``, or any finite ``tau`` for the no-FE Newton fit at a
nonzero order, whose root outside ``[1, 2]`` is the selected root of
:mod:`powermean.solve.joint_root`, and for the joint FE fit at a nonzero
order, the selected root of :mod:`powermean.solve.absorbed_root`, whose root-Newton
options are the existing nested ``root_options`` record of its ``solver_options``;
or the NB2 variance), ``positive_only``, the
:class:`~powermean.data.specs.Inference` (``"robust"``, one cluster variable, ``a^b``
the interaction of two, a typed cluster request of one or more dimensions, or a named
HC method), the resolved :class:`~powermean.solve.options.
SolverOptions` (with, outside ``[1, 2]``, the resolved
:class:`~powermean.solve.cancelled_records.RootOptions` of the six root-Newton keys,
read flat from the same ``solver_options``), the start
on the reported scale as an owned array (one vector at any
order, the order zero included, where the forward map is ``theta_j = d_j
beta_j`` and the least-squares solve records the start without reading it; the list
of starts of the instrumental systems is not available), the behavior on failure and
whether the influence record
is kept.  :class:`Preparation`
holds what the sample construction reads: the weights with their stated kind, the
offset or the exposure (a column or an expression), the requested cluster dimensions,
the subset condition and the outcome domain.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, TypedDict, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from powermean import messages
from powermean.data.inference_conventions import ConventionValues, RetainedColumns
from powermean.data.inference_requests import HC_METHODS, ClusterRequest, HCMethod
from powermean.data.request_values import OMITTED, Omitted, given
from powermean.data.specs import (
    Estimator,
    Inference,
    PowerMean,
    explicit_bread,
    explicit_ssc,
)
from powermean.data.weights import WEIGHT_KINDS, WeightKind
from powermean.errors import SpecificationError
from powermean.inference.bread import Bread as FEBread
from powermean.interface.fe_bias_options import BiasRequest, resolve_bias_correction
from powermean.interface.fe_options import (
    FEVcovRequest,
    ResolvedFESolverOptions,
    check_deferred_vcov,
    resolve_cancel,
    resolve_fe_solver,
    resolve_fe_solver_options,
    resolve_fe_variance,
    resolve_fe_vcov,
    resolve_retain,
)
from powermean.model.order_scale import _mean_order
from powermean.model.rows import _immutable
from powermean.model.variance import FloatArray, NB2Variance, PowerVariance
from powermean.solve.cancelled_records import ROOT_DOMAINS, RootOptions
from powermean.solve.certificate import outside_convex
from powermean.solve.iols.candidate import refuse_iols
from powermean.solve.iols.options import (
    IolsOptions,
    options_to_kernel_control,
    resolve_iols_options,
)
from powermean.solve.options import (
    DOMAINS,
    OPTION_NAMES,
    SolverOptions,
    resolve_options,
)

OnFailure = Literal["raise", "return"]
ON_FAILURE: tuple[OnFailure, ...] = ("raise", "return")
Bread = Literal["observed"]
Convention = Literal["stata"]
Moment = Literal["multiplicative"]
Solver = Literal["newton", "iols"]
SOLVERS: tuple[Solver, ...] = ("newton", "iols")
VcovSpec = str | Mapping[str, object]
DEFAULT_OMEGA = 1.0
DEFAULT_TAU = 1.0
# The values of the unavailable features that request nothing.
INACTIVE_CANCEL = None
INACTIVE_DIFFERENCE = False
INACTIVE_BIAS_CORRECTION = False
DEFAULT_MOMENT: Moment = "multiplicative"
DEFAULT_SSC: Convention = "stata"
DEFAULT_BREAD: Bread = "observed"
DEFAULT_SOLVER: Solver = "newton"
_CERT_KEYS: frozenset[str] = frozenset(
    {"tol", "tol_secondary", "coef_tol", "rcond_min"}
)


class LaterOptions(TypedDict, total=False):
    """The keywords of ``fit`` whose features are not all available.

    Each is declared so that the explicit request of an unavailable feature is refused
    as such rather than as an unknown keyword; omitted, none is active.

    Parameters
    ----------
    cancel : str or bool, optional
        The fixed-effect block to cancel.  The
        inactive ``None`` is accepted.
    difference : bool, str or mapping, optional
        Two-way difference moments; not available in this version.  The inactive
        ``False`` is accepted.
    bias_correction : bool or str, optional
        ``False`` leaves the fit unchanged. In a cancelled-FE fit,
        ``True`` or ``"short_group"`` requires actual cancelled-group sampling;
        ``"cell"`` requires declared one-way dependence made of complete cancelled
        groups. The certified root and first-order covariance stay uncorrected;
        reported coefficients use the BBP shift. Joint-cell, crossing/multiway
        cell pairing and WZ integration remain unavailable, not zero corrections.
    moment : str, optional
        The instrument moment type; ``"multiplicative"`` is accepted; ``"additive"``
        is not available in this version.
    gmm : str, optional
        The weighting of an overidentified system; not available in this version.
    advise : str or mapping, optional
        The advisory output; not available in this version.  No advisor runs.
    threads : int, optional
        The thread setting of BLAS and kernels; not available in this version.
    ssc : str or ConventionValues, optional
        The small-sample convention.  Beside a named HC method (``vcov="HC0"`` to
        ``"HC5"`` or ``"HC4m"``), the explicit convention of that request:
        ``"stata"``, ``"none"``, ``"regression"`` or a custom record, whose factor
        the method's own must equal by formula and exact counts, else it is refused
        as a second correction.  Beside a cluster request of one or more dimensions,
        the explicit convention of that request, whose cluster factor applies once,
        whose reference is the request's and whose ``"psd"`` treatment decides an
        indefinite signed multiway estimate unless the request's ``"psd"`` key does;
        never a key of the ``vcov`` mapping.  Beside the robust covariance, given or
        by default, the explicit convention of that request, whose robust factor
        applies once and whose reference is the convention's; omitted, the
        ``"stata"`` convention.
    bread : str, optional
        The bread of the covariance.  Beside a named HC method, ``"observed"`` (the
        default) or ``"expected"``; otherwise ``"observed"`` is accepted and
        ``"expected"`` is not available in this version.
    """

    cancel: str | bool | None
    difference: bool | str | Mapping[str, object]
    bias_correction: bool | str
    moment: str
    gmm: str
    advise: str | Mapping[str, object]
    threads: int
    ssc: str | ConventionValues
    bread: str


class PrepareOptions(LaterOptions, total=False):
    """The keywords of ``prepare``: the sample, its domain and the default inference.

    Parameters
    ----------
    positive_only : bool, default False
        Whether the sample keeps the rows of positive outcome only: the outcome
        domain of every fit of the prepared problem.
    weights : str, optional
        The column of the weights.
    weight_type : {"analytic", "frequency", "probability"}, default "analytic"
        The kind of the weights, as the caller states it; with ``weights`` only.
    offset : str, optional
        The offset on the log power-mean scale: a column of the data, or a
        Formulaic expression of its columns that evaluates to one numeric column.
        Not together with ``exposure``.
    exposure : str, optional
        The exposure, whose logarithm is the offset: a column or an expression,
        strictly positive.  Not together with ``offset``.
    subset : array_like of bool or pandas.Series, optional
        The evaluated subset condition, one Boolean per row of the data, as
        ``df["x"] > 0``; a missing condition excludes its row.
    vcov : str or mapping, optional
        The default covariance of the fits: ``"robust"`` (the default),
        ``{"cluster": "a"}`` for one cluster variable, ``"a^b"`` the interaction of
        two, or, without fixed effects, several dimensions as ``{"cluster": "a + b"}``
        with the optional keys ``"multiway"`` (``"cgm"``, the default signed form,
        or ``"ddg"``) and ``"psd"`` (``"cgm"`` or ``"none"``, on a signed multiway
        request only), with its ``ssc``; or a named HC method by its exact string,
        ``"HC0"``, ``"HC1"``, ``"HC2"``, ``"HC3"``, ``"HC4"``, ``"HC4m"`` or
        ``"HC5"``, with its ``ssc`` and ``bread``.  Every requested dimension is
        retained with the sample.  HAC and Conley are reserved and refused.
        ``"robust"`` takes its ``ssc`` too.
    cluster : str, optional
        A shorthand for ``vcov={"cluster": cluster}``, one or several dimensions;
        consistent with ``vcov``.
    """

    positive_only: bool
    weights: str
    weight_type: WeightKind
    offset: str
    exposure: str
    subset: Sequence[object] | ArrayLike | pd.Series[Any]
    vcov: VcovSpec
    cluster: str | Sequence[str]
    retain: str | Sequence[str] | None


class PreparedFitOptions(LaterOptions, total=False):
    """The keywords of ``Prepared.fit`` and the estimation keywords of ``fit``.

    Parameters
    ----------
    omega : float, default 1.0
        The mean order, any finite real; ``0`` is log OLS.
    tau : float, optional
        The index of the power working variance; ``1`` when neither ``tau`` nor
        ``variance`` is given.  Not together with ``variance``.  In ``[1, 2]`` the
        criterion is convex and the root unique.  Outside ``[1, 2]``, for the Newton
        solver at a nonzero ``omega`` only, without fixed effects or with joint
        absorbed fixed effects, the estimate is the
        selected root of the equation, reached by root Newton from the certified
        nearest convex fit of the same sample (PPML below 1, Gamma above 2) or from
        ``start``; cancellation, ``solver="iols"`` and ``omega = 0`` keep their
        existing domains.
    variance : NB2Variance or PowerVariance, optional
        A working-variance object, for example ``pm.NB2(kappa=0.5)`` with any
        ``kappa >= 0``; replaces ``tau``.
    positive_only : bool, optional
        The outcome domain, which the prepared problem fixed: an explicit value must
        equal it.
    vcov : str or mapping, optional
        The covariance of this fit, as in ``prepare``, whose certificate's
        coefficient gate reads its standard errors; every cluster dimension must have
        been retained by the preparation.  Given, it replaces the prepared request
        with its ``ssc`` and ``bread``.
    cluster : str, optional
        A shorthand for ``vcov={"cluster": cluster}``, one or several dimensions;
        consistent with ``vcov``.
    solver : {"newton"}, default "newton"
        The solver.
    solver_options : mapping or SolverOptions, optional
        The options of the certificate and of the convex Newton solver,
        as a mapping of names to values or as the
        typed record of ``solver_defaults``; unknown keys are refused with the valid
        ones.  A fit with ``tau`` outside ``[1, 2]`` also reads the root-Newton keys
        from the mapping: ``root_armijo``, ``root_max_backtracks``,
        ``root_max_iter``, ``merit_metric`` (``"gamma"`` or ``"gram"``),
        ``continuation`` (``"fallback"`` or ``"never"``) and ``continuation_step``,
        with their defaults where omitted; another fit refuses them as unknown.
    tol : float, optional
        The certificate tolerance: a shorthand for the option ``tol`` of
        ``solver_options`` (the module docstring).
    start : array_like, optional
        One starting coefficient vector on the reported scale, one finite value per
        coefficient of the design; mapped to the coordinates of the solver by the
        range-safe forward map and recorded in both coordinates.  At
        ``omega = 0`` the map is ``theta_j = d_j beta_j``, and the least-squares
        solve records the start without reading it.  A list of starts belongs to
        the instrumental systems, which are not available.
    on_failure : {"raise", "return"}, default "raise"
        What happens when no certified solution is obtained: ``ConvergenceError``
        with the last state, or a result flagged as not certified whose inference
        is withheld.
    keep_influence : bool, default True
        Whether the influence contributions are kept after the default covariance
        is formed.
    """

    omega: float
    tau: float
    variance: NB2Variance | PowerVariance
    positive_only: bool
    vcov: VcovSpec
    cluster: str | Sequence[str]
    solver: str
    solver_options: Mapping[str, object] | SolverOptions | IolsOptions
    tol: float
    start: ArrayLike
    on_failure: OnFailure
    keep_influence: bool


class FitOptions(PrepareOptions, PreparedFitOptions, total=False):
    """The keywords of ``fit``, ``ppml``, ``gpml`` and ``logols``.

    The union of :class:`PrepareOptions` and :class:`PreparedFitOptions`: ``fit``
    prepares the sample once and fits it, so it takes every keyword of both.
    """


# The features whose explicit request is refused as not available.
LATER_FEATURES: frozenset[str] = frozenset(
    {
        "cancel",
        "difference",
        "bias_correction",
        "moment",
        "gmm",
        "advise",
        "threads",
        "ssc",
        "bread",
        "retain",
    }
)
FE_ACTIVE_FEATURES: frozenset[str] = frozenset(
    {"cancel", "ssc", "bread", "retain", "bias_correction"}
)
# The per-request keywords of a named HC method without fixed effects.
NAMED_HC_FEATURES: frozenset[str] = frozenset({"ssc", "bread"})
# The per-request keyword of a cluster request without fixed effects.
CLUSTER_FEATURES: frozenset[str] = frozenset({"ssc"})
# The per-request keyword of the robust covariance without fixed effects.
ROBUST_FEATURES: frozenset[str] = frozenset({"ssc"})
# The keys of a cluster mapping of ``vcov``; ``"ssc"`` is not one.
CLUSTER_KEYS: tuple[str, ...] = ("cluster", "multiway", "psd")
VALID_VCOV = (
    "'robust', 'HC0', 'HC1', 'HC2', 'HC3', 'HC4', 'HC4m', 'HC5', or a mapping with "
    "'cluster'"
)
# The keywords of each collector, in the order of their declaration.
LATER_KEYWORDS: tuple[str, ...] = tuple(LaterOptions.__annotations__)
PREPARE_KEYWORDS: tuple[str, ...] = tuple(PrepareOptions.__annotations__)
PREPARED_FIT_KEYWORDS: tuple[str, ...] = tuple(PreparedFitOptions.__annotations__)
FIT_KEYWORDS: tuple[str, ...] = tuple(FitOptions.__annotations__)
# The keywords that describe the sample, which a prepared problem has fixed.
SAMPLE_KEYWORDS: tuple[str, ...] = ("weights", "weight_type", "offset", "exposure")
VARIANCES = "a working variance: NB2Variance (pm.NB2) or PowerVariance"
START = "one finite real value per coefficient of the design"
SNAPSHOTS = (
    "complete SolverOptions records, or mappings of solver option names to values"
)
# The marker of a field that a snapshot record does not hold.
_ABSENT = object()


def _unknown(kind: str, key: object, valid: Sequence[str]) -> SpecificationError:
    """The refusal of a keyword that a collector does not declare."""
    return SpecificationError(
        "specification.unknown_option",
        {"kind": kind, "key": key, "valid": messages.format_keys(valid)},
    )


def _invalid(argument: str, value: object, valid: str) -> SpecificationError:
    """The refusal of a value outside the domain of its argument."""
    return SpecificationError(
        "specification.invalid_value",
        {"argument": argument, "value": value, "valid": valid},
    )


def _conflicting(first: str, second: str) -> SpecificationError:
    """The refusal of two arguments that cannot be given together."""
    return SpecificationError(
        "specification.conflicting", {"first": first, "second": second}
    )


def _unimplemented(feature: str) -> SpecificationError:
    """The refusal of a feature that is not available."""
    return SpecificationError("specification.unimplemented", {"feature": feature})


def requests_named_hc(
    options: Mapping[str, object], default: Inference | None = None
) -> bool:
    """Whether the covariance requested by a call is a named HC method.

    Read from the keywords as given, before they are collected: a ``vcov`` given is
    the exact string of a method; with neither ``vcov`` nor ``cluster``, the request
    is ``default``, the prepared problem's, where there is one.
    """
    vcov = options.get("vcov")
    if vcov is not None:
        return isinstance(vcov, str) and vcov in HC_METHODS
    if options.get("cluster") is not None:
        return False
    return default is not None and default.covariance == "hc"


def requests_cluster(
    options: Mapping[str, object], default: Inference | None = None
) -> bool:
    """Whether the covariance requested by a call is a cluster covariance.

    Read from the keywords as given, before they are collected: a ``vcov`` given is a
    mapping with the key ``"cluster"``; a ``cluster`` given is one; with neither, the
    request is ``default``, the prepared problem's, where there is one.  One or more
    dimensions alike: the call's ``ssc`` is then that request's convention.
    """
    vcov = options.get("vcov")
    if vcov is not None:
        return isinstance(vcov, Mapping) and "cluster" in vcov
    if options.get("cluster") is not None:
        return True
    return default is not None and default.covariance == "cluster"


def requests_robust(
    options: Mapping[str, object], default: Inference | None = None
) -> bool:
    """Whether the covariance requested by a call is the robust covariance.

    Read from the keywords as given, before they are collected: a ``vcov`` given is the
    exact string ``"robust"``; a ``cluster`` given is not; with neither, the request is
    ``default``, the prepared problem's, where there is one, else the robust default.
    The call's ``ssc`` is then that request's convention.
    """
    vcov = options.get("vcov")
    if vcov is not None:
        return isinstance(vcov, str) and vcov == "robust"
    if options.get("cluster") is not None:
        return False
    return default is None or default.covariance == "robust"


def collect(
    options: Mapping[str, object],
    kind: str,
    valid: Sequence[str],
    *,
    has_fixed_effects: bool = False,
    named_hc: bool = False,
    clustered: bool = False,
    robust: bool = False,
) -> Mapping[str, object]:
    """The keywords given, checked against the keywords of a collector.

    Parameters
    ----------
    options : mapping
        The keywords as received, owned by the caller and read once.
    kind : str
        The name of the call in a refusal, as ``"keyword of fit"``.
    valid : sequence of str
        The keywords that the collector declares.
    has_fixed_effects : bool, default False
        Whether the formula has a fixed-effects part.
    named_hc : bool, default False
        Whether the call requests a named HC method (:func:`requests_named_hc`):
        its ``ssc`` and ``bread`` are then the keywords of that request, validated
        with it, and not later features.
    clustered : bool, default False
        Whether the call requests a cluster covariance (:func:`requests_cluster`):
        its ``ssc`` is then the keyword of that request, validated with it, and not
        a later feature; its ``bread`` stays one.
    robust : bool, default False
        Whether the call requests the robust covariance (:func:`requests_robust`):
        its ``ssc`` is then the keyword of that request, validated with it, and not
        a later feature; its ``bread`` stays one.

    Returns
    -------
    mapping
        A read-only copy of the keywords given, in their order.

    Raises
    ------
    SpecificationError
        ``specification.unknown_option`` for the first key, in the order given,
        that the collector does not declare; ``specification.unimplemented`` for
        the first explicit request of a later feature (:func:`later_feature`).
    """
    if not has_fixed_effects:
        valid = tuple(key for key in valid if key != "retain")
    for key in options:
        if key not in valid:
            raise _unknown(kind, key, valid)
    for key, value in options.items():
        if key in LATER_FEATURES:
            if options.get("solver") == "iols":
                _iols_later_feature(key, value)
            if has_fixed_effects and key in FE_ACTIVE_FEATURES:
                continue
            if named_hc and key in NAMED_HC_FEATURES:
                continue
            if clustered and key in CLUSTER_FEATURES:
                continue
            if robust and key in ROBUST_FEATURES:
                continue
            later_feature(key, value)
    return MappingProxyType(dict(options))


def _iols_later_feature(key: str, value: object) -> None:
    """Preserve the accepted IOLS unsupported-feature capability boundary."""
    if key == "cancel" and not _same_value(value, INACTIVE_CANCEL):
        refuse_iols("cancelled fixed effects")
    if key == "difference" and not _same_value(value, INACTIVE_DIFFERENCE):
        refuse_iols("difference moments")
    if key == "bias_correction" and not _same_value(value, INACTIVE_BIAS_CORRECTION):
        refuse_iols("bias corrections")
    if key == "gmm" or (key == "moment" and not _same_value(value, DEFAULT_MOMENT)):
        refuse_iols("instruments (i2SLS)")


def later_feature(key: str, value: object) -> None:
    """Refuse the explicit request of an unavailable feature; accept its inactive value.

    ``cancel=None``, ``difference=False``, ``bias_correction=False``,
    ``moment="multiplicative"``, ``ssc="stata"`` and ``bread="observed"`` request
    nothing; every other value, and any value of
    ``gmm``, ``advise`` and ``threads``, is refused.
    """
    inactive: dict[str, object] = {
        "cancel": INACTIVE_CANCEL,
        "difference": INACTIVE_DIFFERENCE,
        "bias_correction": INACTIVE_BIAS_CORRECTION,
        "moment": DEFAULT_MOMENT,
        "ssc": DEFAULT_SSC,
        "bread": DEFAULT_BREAD,
        "retain": None,
    }
    if key in inactive and _same_value(value, inactive[key]):
        return
    raise _unimplemented(f"{key}={value!r}")


def _same_value(value: object, inactive: object) -> bool:
    """Whether ``value`` is the inactive value, by exact type and equality."""
    return type(value) is type(inactive) and bool(value == inactive)


# ------------------------------------------------------------------ the estimator


def _finite_real(value: object, argument: str, valid: str) -> float:
    """A finite real number other than a Boolean, as ``float``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise _invalid(argument, value, valid)
    number = float(value)
    if not np.isfinite(number):
        raise _invalid(argument, value, valid)
    return number


def resolve_omega(value: object) -> float:
    """The mean order: any finite real number, ``-0.0`` as ``0.0``."""
    return _mean_order(_finite_real(value, "omega", "finite real numbers"))


def _power_variance(
    tau: object, *, solver: Solver = "newton", omega: float = DEFAULT_OMEGA
) -> PowerVariance:
    """The power variance of ``tau``: any finite ``tau`` for the root route.

    Outside ``[1, 2]`` the Newton fit at a nonzero order is the joint root route;
    ``solver="iols"`` refuses it, and the order zero, whose least-squares
    route has no working variance to select a root of, refuses it as not available.
    """
    valid = "finite real numbers"
    if solver == "iols":
        valid = "finite real numbers in [1, 2]"
    value = _finite_real(tau, "tau", valid)
    if not 1.0 <= value <= 2.0:
        if solver == "iols":
            refuse_iols(f"Power(tau={tau!r}) outside [1, 2]")
        if omega == 0.0:
            raise _unimplemented(f"tau={tau!r} outside [1, 2]")
    return PowerVariance(tau=value)


def _variance_object(
    variance: object, *, solver: Solver = "newton", omega: float = DEFAULT_OMEGA
) -> PowerVariance | NB2Variance:
    """A working-variance object of an accepted exact type, as given."""
    if type(variance) is NB2Variance:
        if solver == "iols":
            refuse_iols("NB2 working variance")
        return variance
    if type(variance) is PowerVariance:
        return _power_variance(variance.tau, solver=solver, omega=omega)
    raise _invalid("variance", variance, VARIANCES)


def resolve_variance(
    tau: object | None,
    variance: object | None,
    *,
    solver: Solver = "newton",
    omega: float = DEFAULT_OMEGA,
) -> PowerVariance | NB2Variance:
    """The working variance of ``tau`` or ``variance``, or the default of neither.

    Parameters
    ----------
    tau : object or None
        The explicit ``tau``, or ``None`` where omitted.
    variance : object or None
        The explicit ``variance``, or ``None`` where omitted.
    solver : {"newton", "iols"}, default "newton"
        The solver of the fit.
    omega : float, default 1.0
        The resolved mean order of the fit.

    Returns
    -------
    PowerVariance or NB2Variance
        ``PowerVariance(tau=1)`` where neither is given; the power variance of an
        explicit finite ``tau`` (outside ``[1, 2]`` for the Newton solver at a nonzero
        order only); the object given as ``variance``.

    Raises
    ------
    SpecificationError
        Both given (``specification.conflicting``); ``tau`` outside ``[1, 2]`` at
        ``omega = 0`` (``specification.unimplemented``); a value that is not a
        finite real number or an object of another type
        (``specification.invalid_value``).
    SolverCapabilityError
        ``tau`` outside ``[1, 2]`` or NB2 with ``solver="iols"``.
    """
    if tau is not None and variance is not None:
        raise _conflicting(f"tau={tau!r}", f"variance={variance!r}")
    if variance is not None:
        return _variance_object(variance, solver=solver, omega=omega)
    if tau is not None:
        return _power_variance(tau, solver=solver, omega=omega)
    return PowerVariance(tau=DEFAULT_TAU)


def resolve_estimator(
    options: Mapping[str, object], *, solver: Solver = "newton"
) -> Estimator:
    """The estimator record of the keywords ``omega``, ``tau`` and ``variance``."""
    omega = resolve_omega(options.get("omega", DEFAULT_OMEGA))
    variance = resolve_variance(
        options.get("tau"), options.get("variance"), solver=solver, omega=omega
    )
    return Estimator(mean=PowerMean(omega), variance=variance)


# ------------------------------------------------------------------ the inference


def cluster_dimensions(value: object) -> tuple[str, ...]:
    """The cluster dimensions of the public spelling, in order (module docstring).

    One variable as given, ``a^b`` for an interaction of two, or several dimensions
    joined by ``+`` (``"a + b"``), each stripped and checked by the canonical
    :class:`~powermean.data.inference_requests.ClusterRequest`: nonempty names,
    repetitions kept, no number of dimensions fixed.
    """
    if not isinstance(value, str) or not value.strip():
        raise _invalid("vcov", {"cluster": value}, "the name of a cluster variable")
    if "+" not in value:
        return (value,)
    return ClusterRequest(tuple(part.strip() for part in value.split("+"))).dimensions


def cluster_inference(
    dimensions: tuple[str, ...],
    multiway: object = OMITTED,
    psd: object = OMITTED,
) -> Inference:
    """The inference of cluster dimensions with the keys ``multiway`` and ``psd``.

    One dimension without either key is the plain one cluster variable,
    unchanged; every other request is the typed
    :class:`~powermean.data.inference_requests.ClusterRequest`, which checks the form,
    the ``"psd"`` treatment and where it is valid (``specification.invalid_value``,
    ``specification.conflicting``).
    """
    if len(dimensions) == 1 and not given(multiway) and not given(psd):
        return Inference(covariance="cluster", cluster=dimensions[0])
    record: dict[str, object] = {"type": "ClusterRequest", "dimensions": dimensions}
    for key, value in (("multiway", multiway), ("psd", psd)):
        if given(value):
            record[key] = value
    return Inference(covariance="cluster", clusters=ClusterRequest.from_dict(record))


def _vcov_mapping(spec: Mapping[str, object]) -> Inference:
    """The inference of a ``vcov`` mapping: the keys of a cluster request."""
    check_deferred_vcov(spec)
    for key in spec:
        if key == "adjustment":
            raise _unimplemented(f"the vcov key {key!r}")
        if key not in CLUSTER_KEYS:
            raise _unknown("vcov key", key, CLUSTER_KEYS)
    if "cluster" not in spec:
        raise _invalid("vcov", dict(spec), "'robust', or a mapping with 'cluster'")
    return cluster_inference(
        cluster_dimensions(spec["cluster"]),
        spec.get("multiway", OMITTED),
        spec.get("psd", OMITTED),
    )


def _vcov(spec: object) -> Inference:
    """The inference of a ``vcov`` value: ``"robust"``, a named method or a mapping."""
    check_deferred_vcov(spec)
    if isinstance(spec, str):
        if spec == "robust":
            return Inference()
        if spec in HC_METHODS:
            return Inference(covariance="hc", method=cast(HCMethod, spec))
        raise _invalid("vcov", spec, VALID_VCOV)
    if isinstance(spec, Mapping):
        return _vcov_mapping(spec)
    raise _invalid("vcov", spec, VALID_VCOV)


def resolve_inference(vcov: object | None, cluster: object | None) -> Inference:
    """The inference of ``vcov`` and of its shorthand ``cluster``, consistent.

    Parameters
    ----------
    vcov : object or None
        The explicit ``vcov``, or ``None`` where omitted.
    cluster : object or None
        The explicit ``cluster``, or ``None`` where omitted.

    Returns
    -------
    Inference
        The robust covariance where neither is given; the one that ``vcov`` names,
        a named HC method by its exact string included; the cluster covariance of
        ``cluster``, one or several dimensions.  Both given must name the same
        dimensions, in the same order.

    Raises
    ------
    SpecificationError
        A cluster given beside ``vcov="robust"``, beside a named HC method or beside
        other dimensions (``specification.conflicting``);
        ``"psd"`` outside a signed multiway request (``specification.conflicting``);
        the key ``"adjustment"`` (``specification.unimplemented``); another value
        (``specification.invalid_value``, ``specification.unknown_option``).
    """
    named = None if cluster is None else cluster_dimensions(cluster)
    if vcov is None:
        if named is None:
            return Inference()
        return cluster_inference(named)
    inference = _vcov(vcov)
    if named is not None and inference.dimensions != named:
        raise _conflicting(f"cluster={cluster!r}", f"vcov={vcov!r}")
    return inference


def ordinary_inference(
    options: Mapping[str, object], default: Inference | None = None
) -> Inference:
    """The covariance requested by a call without fixed effects (module docstring).

    Parameters
    ----------
    options : mapping
        The collected keywords of the call.
    default : Inference, optional
        The prepared problem's request, for a fit; ``None`` at a preparation.

    Returns
    -------
    Inference
        The request of the ``vcov`` and ``cluster`` given, else ``default``, else the
        robust covariance; a named HC method takes the call's explicit ``ssc`` and
        ``bread`` over those of the request it keeps, and a cluster or robust request
        the call's explicit ``ssc`` (:func:`cluster_convention`,
        :func:`robust_convention`).  For the ``bread`` of a robust or cluster
        request, the collector accepted the inactive value only, which requests
        nothing.

    Raises
    ------
    SpecificationError
        The refusals of :func:`resolve_inference`; an explicit ``ssc`` or ``bread`` of
        a named method, or an explicit ``ssc`` of a cluster or robust request, outside
        its domain, ``None`` included (``specification.invalid_value``).
    """
    if options.get("vcov") is not None or options.get("cluster") is not None:
        found = resolve_inference(options.get("vcov"), options.get("cluster"))
    else:
        found = Inference() if default is None else default
    if found.covariance == "cluster":
        return cluster_convention(found, options)
    if found.covariance != "hc":
        return robust_convention(found, options)
    ssc = explicit_ssc(options["ssc"]) if "ssc" in options else found.ssc
    bread = explicit_bread(options["bread"]) if "bread" in options else found.bread
    return replace(found, ssc=ssc, bread=bread)


def cluster_convention(found: Inference, options: Mapping[str, object]) -> Inference:
    """A cluster request with the explicit ``ssc`` of its call.

    The call's ``ssc``, beside its cluster request or alone at a fit of a prepared
    one, replaces the convention of the request it keeps, validated by its domain
    (``specification.invalid_value``, ``None`` included) and recorded with the typed
    request; omitted, the request keeps its own.  An explicit ``"stata"`` is
    recorded as explicit too, even though its factor is unchanged.
    """
    if "ssc" not in options:
        return found
    value = options["ssc"]
    clusters = found.clusters
    if clusters is None:
        if found.cluster is None:
            raise ValueError("a cluster request names its dimensions")
        clusters = ClusterRequest((found.cluster,))
    return Inference(covariance="cluster", clusters=clusters, ssc=explicit_ssc(value))


def robust_convention(found: Inference, options: Mapping[str, object]) -> Inference:
    """The robust request with the explicit ``ssc`` of its call.

    The call's ``ssc``, beside the robust request or alone at a fit of a prepared one,
    replaces the convention of the request it keeps, validated by its domain
    (``specification.invalid_value``, ``None`` included) and recorded with the request;
    omitted, the request keeps its own, none for the plain robust default.  An
    explicit ``"stata"`` is recorded as explicit too, even though its factor is
    unchanged.
    """
    if found.covariance != "robust":
        raise ValueError("the request is not the robust covariance")
    if "ssc" not in options:
        return found
    return Inference(covariance="robust", ssc=explicit_ssc(options["ssc"]))


# --------------------------------------------------------------------- the solver


def resolve_solver(value: object | None) -> Solver:
    """Newton remains the default; IOLS is an explicit optional no-FE solver."""
    if value is None or value == DEFAULT_SOLVER:
        return DEFAULT_SOLVER
    if value == "iols":
        return "iols"
    raise _invalid("solver", value, messages.format_keys(SOLVERS))


def resolve_iols_solver_options(
    solver_options: object | None, tol: object | None
) -> tuple[IolsOptions, SolverOptions]:
    """Reuse the accepted IOLS and common certificate option mapping."""
    if solver_options is None:
        iols_opts = IolsOptions()
        cert_opts = resolve_solver_options(None, tol)
    elif isinstance(solver_options, IolsOptions):
        iols_opts = resolve_iols_options(solver_options)
        cert_opts = resolve_solver_options(None, tol)
    elif isinstance(solver_options, Mapping):
        cert_sub = {k: v for k, v in solver_options.items() if k in _CERT_KEYS}
        iols_sub = {k: v for k, v in solver_options.items() if k not in _CERT_KEYS}
        iols_opts = IolsOptions.from_dict(iols_sub)
        cert_opts = resolve_solver_options(cert_sub, tol)
    else:
        raise _invalid(
            "solver_options",
            solver_options,
            "complete IolsOptions records, or mappings of solver option "
            "names to values",
        )
    options_to_kernel_control(iols_opts)
    return iols_opts, cert_opts


def _top_level_tol(tol: object) -> float:
    """The top-level ``tol`` validated by the canonical domain of the option."""
    resolved = DOMAINS["tol"].resolve(tol)
    if resolved is None:
        raise _invalid("tol", tol, DOMAINS["tol"].valid)
    return float(resolved)


def _snapshot_values(record: SolverOptions) -> dict[str, object]:
    """The stored values of a snapshot record, by option, before their validation.

    Read field by field from the instance's own stored state (``vars``), never from
    the defaults that the class declares, so that the canonical table validates what
    the record holds and not what its constructor once accepted; a record whose
    stored state lacks a field is not a complete snapshot and is refused.
    """
    stored = vars(record)
    values: dict[str, object] = {}
    for name in OPTION_NAMES:
        value = stored.get(name, _ABSENT)
        if value is _ABSENT:
            raise _invalid("solver_options", record, SNAPSHOTS)
        values[name] = value
    return values


def resolve_solver_options(
    solver_options: object | None, tol: object | None
) -> SolverOptions:
    """The resolved options of ``solver_options`` and the shorthand ``tol``.

    Parameters
    ----------
    solver_options : mapping, SolverOptions or None
        The explicit ``solver_options``, or ``None`` where omitted: a flat mapping
        of option names to values, or a complete resolved record.
    tol : object or None
        The explicit ``tol``, or ``None`` where omitted (the caller reads the
        presence of the keyword: :func:`_explicit_tol`).

    Returns
    -------
    SolverOptions
        The canonical record (the module docstring): the defaults; the mapping over
        them; the record validated again, value by value, against the canonical
        table; the top-level ``tol`` over them where the mapping has no ``tol``
        key; the one value where both are given and equal.

    Raises
    ------
    SpecificationError
        An unknown key or a value outside its domain, by
        :func:`~powermean.solve.options.resolve_options` and the canonical domains,
        for a mapping and for the stored values of a record alike; a record that
        lacks a field (``specification.invalid_value``);
        ``specification.conflicting`` where both give a valid, unequal ``tol``.
    """
    supplied = False
    if solver_options is None:
        base = resolve_options({})
    elif isinstance(solver_options, SolverOptions):
        base, supplied = resolve_options(_snapshot_values(solver_options)), True
    else:
        base = resolve_options(solver_options)
        supplied = isinstance(solver_options, Mapping) and "tol" in solver_options
    if tol is None:
        return base
    top = _top_level_tol(tol)
    if not supplied:
        return replace(base, tol=top)
    if top == base.tol:
        return base
    raise _conflicting(f"tol={tol!r}", f"solver_options tol={base.tol!r}")


def _explicit_tol(options: Mapping[str, object]) -> object | None:
    """The ``tol`` keyword where it is present, whatever it holds; ``None`` if absent.

    Presence, not value, tells a supplied ``tol`` from an omitted one: a keyword
    given as ``None`` is a supplied value, outside the canonical domain, and is
    refused here before any comparison with ``solver_options``.
    """
    if "tol" not in options:
        return None
    tol = options["tol"]
    if tol is None:
        raise _invalid("tol", tol, DOMAINS["tol"].valid)
    return tol


# ----------------------------------------------------------- the other keywords


def resolve_flag(value: object, argument: str) -> bool:
    """A keyword that must be a ``bool``."""
    if not isinstance(value, bool):
        raise _invalid(argument, value, "True or False")
    return value


def resolve_on_failure(value: object | None) -> OnFailure:
    """``"raise"`` (the default) or ``"return"``."""
    if value is None:
        return "raise"
    for choice in ON_FAILURE:
        if value == choice:
            return choice
    raise _invalid("on_failure", value, messages.format_keys(ON_FAILURE))


def resolve_start(value: object | None) -> FloatArray | None:
    """One starting vector on the reported scale, owned; ``None`` where omitted.

    Accepted at every order, with the same domain and shape rules: at ``omega = 0``
    the vector is mapped and recorded, and the least-squares solve does not read it
    (the module docstring).  A list of vectors is the multistart of the
    instrumental systems, which are not available.
    """
    if value is None:
        return None
    if (
        isinstance(value, Sequence)
        and value
        and all(
            isinstance(item, Sequence | np.ndarray) and not isinstance(item, str)
            for item in value
        )
    ):
        raise _unimplemented("a list of starts (multistart)")
    try:
        start = np.asarray(cast("ArrayLike", value), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise _invalid("start", value, START) from error
    if start.ndim != 1 or not start.size or not np.isfinite(start).all():
        raise _invalid("start", value, START)
    return _immutable(start)


def resolve_weight_kind(
    weights: object | None, weight_type: object | None
) -> WeightKind:
    """The stated kind of the weights; ``weight_type`` needs ``weights``."""
    if weight_type is None:
        return "analytic"
    if weights is None:
        raise _conflicting(f"weight_type={weight_type!r}", "weights=None")
    for kind in WEIGHT_KINDS:
        if weight_type == kind:
            return kind
    raise _invalid("weight_type", weight_type, messages.format_keys(WEIGHT_KINDS))


def resolve_name(value: object | None, argument: str) -> str | None:
    """A column name or an expression: a nonempty string, or ``None`` where omitted."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _invalid(argument, value, "a nonempty column name or expression")
    return value


# ------------------------------------------------------------------- the records


@dataclass(frozen=True)
class Preparation:
    """What the sample construction reads, normalized (the module docstring)."""

    positive_only: bool
    weights: str | None
    weight_kind: WeightKind
    offset: str | None
    exposure: str | None
    inference: Inference
    subset: object
    retain: RetainedColumns = field(default_factory=lambda: RetainedColumns(()))
    fe_vcov: FEVcovRequest = "robust"
    ssc: str | ConventionValues | Omitted = OMITTED
    bread: FEBread = "observed"
    explicit_vcov: bool = False
    cancel: str | Literal[True] | None = None
    bias_correction: BiasRequest = None


@dataclass(frozen=True)
class Specification:
    """The typed records of one fit, normalized (the module docstring)."""

    estimator: Estimator
    inference: Inference
    options: SolverOptions
    solver: Solver
    start: FloatArray | None
    on_failure: OnFailure
    keep_influence: bool
    positive_only: bool | None
    cancel: str | Literal[True] | None = None
    fe_vcov: FEVcovRequest = "robust"
    ssc: str | ConventionValues | Omitted = OMITTED
    bread: FEBread = "observed"
    explicit_vcov: bool = False
    fe_solver_options: ResolvedFESolverOptions | None = None
    iols_options: IolsOptions | None = None
    bias_correction: BiasRequest = None
    root_options: RootOptions | None = None


def _root_solver_options(
    estimator: Estimator, solver_options: object | None, tol: object | None
) -> tuple[SolverOptions, RootOptions | None]:
    """The resolved options, and the root-Newton options of a fit outside ``[1, 2]``.

    A convex or log-OLS fit resolves ``solver_options`` as before and has no root
    options, so that a root key stays unknown there.  A fit outside ``[1, 2]`` reads
    the six root keys from a mapping (each validated by
    the domains of :class:`~powermean.solve.cancelled_records.RootOptions`, the
    defaults where omitted) and every other key as before; a snapshot record leaves
    the root options at their defaults.  An unknown key is refused with every valid
    key of the route.
    """
    if estimator.route != "power" or not outside_convex(estimator.variance):
        return resolve_solver_options(solver_options, tol), None
    if not isinstance(solver_options, Mapping):
        return resolve_solver_options(solver_options, tol), RootOptions()
    valid = (*OPTION_NAMES, *ROOT_DOMAINS)
    for key in solver_options:
        if key not in valid:
            raise _unknown("solver option", key, valid)
    root = {key: value for key, value in solver_options.items() if key in ROOT_DOMAINS}
    other = {
        key: value for key, value in solver_options.items() if key not in ROOT_DOMAINS
    }
    return resolve_solver_options(other, tol), RootOptions.from_dict(root)


def _iols_specification(
    options: Mapping[str, object], inference: Inference
) -> Specification:
    """Normalize only the new IOLS branch; leave current Newton/FE ordering intact."""
    estimator = resolve_estimator(options, solver="iols")
    requested = ordinary_inference(options, inference)
    positive_only = options.get("positive_only")
    positive = (
        None if positive_only is None else resolve_flag(positive_only, "positive_only")
    )
    on_failure = resolve_on_failure(options.get("on_failure"))
    keep = resolve_flag(options.get("keep_influence", True), "keep_influence")
    start = resolve_start(options.get("start"))
    if estimator.mean.omega <= 0.0:
        refuse_iols(
            f"nonpositive mean order omega={estimator.mean.omega!r} "
            "(including log OLS at omega=0)"
        )
    if start is not None:
        refuse_iols("user-supplied start values")
    iols_opts, cert_opts = resolve_iols_solver_options(
        options.get("solver_options"), _explicit_tol(options)
    )
    return Specification(
        estimator=estimator,
        inference=requested,
        options=cert_opts,
        solver="iols",
        start=None,
        on_failure=on_failure,
        keep_influence=keep,
        positive_only=positive,
        iols_options=iols_opts,
    )


def preparation_of(
    options: Mapping[str, object], *, has_fixed_effects: bool = False
) -> Preparation:
    """The :class:`Preparation` of the collected keywords of ``prepare`` or ``fit``."""
    weights = resolve_name(options.get("weights"), "weights")
    offset = resolve_name(options.get("offset"), "offset")
    exposure = resolve_name(options.get("exposure"), "exposure")
    if offset is not None and exposure is not None:
        raise _conflicting(f"offset={offset!r}", f"exposure={exposure!r}")
    pos_only = resolve_flag(options.get("positive_only", False), "positive_only")
    w_kind = resolve_weight_kind(weights, options.get("weight_type"))
    if not has_fixed_effects:
        return Preparation(
            positive_only=pos_only,
            weights=weights,
            weight_kind=w_kind,
            offset=offset,
            exposure=exposure,
            inference=ordinary_inference(options),
            subset=options.get("subset"),
        )
    cancel = resolve_cancel(options.get("cancel"), has_fixed_effects=True)
    retain = resolve_retain(options.get("retain"))
    fe_vcov, ssc_val, bread_val, explicit_vcov = resolve_fe_vcov(
        options.get("vcov"),
        options.get("cluster"),
        ssc=options.get("ssc", OMITTED),
        bread=options.get("bread", "observed"),
        cancel=cancel,
    )
    inf = Inference()
    return Preparation(
        positive_only=pos_only,
        weights=weights,
        weight_kind=w_kind,
        offset=offset,
        exposure=exposure,
        inference=inf,
        subset=options.get("subset"),
        retain=retain,
        fe_vcov=fe_vcov,
        ssc=ssc_val,
        bread=bread_val,
        explicit_vcov=explicit_vcov,
        cancel=cancel,
        bias_correction=resolve_bias_correction(options.get("bias_correction", False)),
    )


def specification_of(
    options: Mapping[str, object],
    inference: Inference,
    *,
    has_fixed_effects: bool = False,
    prepared_fe: tuple[
        FEVcovRequest,
        str | ConventionValues | Omitted,
        FEBread,
        bool,
        str | Literal[True] | None,
    ]
    | None = None,
    prepared_bias: BiasRequest = None,
) -> Specification:
    """The :class:`Specification` of the collected keywords of a fit.

    ``inference`` is the default of the prepared problem, which an explicit ``vcov``
    or ``cluster`` of the fit replaces, and whose named HC method takes an explicit
    ``ssc`` or ``bread`` of the fit (:func:`ordinary_inference`).
    """
    if options.get("solver") == "iols":
        if has_fixed_effects:
            refuse_iols("fixed effects")
        return _iols_specification(options, inference)
    if not has_fixed_effects:
        estimator = resolve_estimator(options)
        requested = ordinary_inference(options, inference)
        positive_only = options.get("positive_only")
        solver_options, root_options = _root_solver_options(
            estimator, options.get("solver_options"), _explicit_tol(options)
        )
        return Specification(
            estimator=estimator,
            inference=requested,
            options=solver_options,
            solver=resolve_solver(options.get("solver")),
            start=resolve_start(options.get("start")),
            on_failure=resolve_on_failure(options.get("on_failure")),
            keep_influence=resolve_flag(
                options.get("keep_influence", True), "keep_influence"
            ),
            positive_only=None
            if positive_only is None
            else resolve_flag(positive_only, "positive_only"),
            root_options=root_options,
        )
    omega = resolve_omega(options.get("omega", DEFAULT_OMEGA))
    prep_vcov, prep_ssc, prep_bread, prep_explicit, prep_cancel = (
        ("robust", OMITTED, "observed", False, None)
        if prepared_fe is None
        else prepared_fe
    )
    cancel = (
        resolve_cancel(options["cancel"], has_fixed_effects=True)
        if "cancel" in options
        else prep_cancel
    )
    variance = resolve_fe_variance(
        omega, options.get("tau"), options.get("variance"), cancel=cancel
    )
    estimator = Estimator(mean=PowerMean(omega), variance=variance)
    if (
        options.get("vcov") is not None
        or options.get("cluster") is not None
        or "ssc" in options
        or "bread" in options
    ):
        raw_vcov = (
            options.get("vcov")
            if (options.get("vcov") is not None or options.get("cluster") is not None)
            else (None if not prep_explicit else prep_vcov)
        )
        fe_vcov, ssc_val, bread_val, explicit_vcov = resolve_fe_vcov(
            raw_vcov,
            options.get("cluster"),
            ssc=options.get("ssc", prep_ssc),
            bread=options.get("bread", prep_bread),
            cancel=cancel,
        )
        if not explicit_vcov and prep_explicit:
            fe_vcov = prep_vcov
            explicit_vcov = True
    else:
        fe_vcov, ssc_val, bread_val, explicit_vcov = (
            prep_vcov,
            prep_ssc,
            prep_bread,
            prep_explicit,
        )
    if cancel is not None and bread_val != "observed":
        raise _invalid("bread", bread_val, "'observed'")
    solver = resolve_fe_solver(options.get("solver"))
    start = resolve_start(options.get("start"))
    on_failure = resolve_on_failure(options.get("on_failure"))
    keep_influence = resolve_flag(options.get("keep_influence", True), "keep_influence")
    positive_only = options.get("positive_only")
    pos_flag = (
        None if positive_only is None else resolve_flag(positive_only, "positive_only")
    )
    fe_solver_opts = resolve_fe_solver_options(
        options.get("solver_options"),
        _explicit_tol(options),
        cancel=cancel,
        selected_root=cancel is None and omega != 0 and outside_convex(variance),
    )
    return Specification(
        estimator=estimator,
        inference=inference,
        options=fe_solver_opts.solver_options,
        solver=solver,
        start=start,
        on_failure=on_failure,
        keep_influence=keep_influence,
        positive_only=pos_flag,
        cancel=cancel,
        fe_vcov=fe_vcov,
        ssc=ssc_val,
        bread=bread_val,
        explicit_vcov=explicit_vcov,
        fe_solver_options=fe_solver_opts,
        bias_correction=(
            resolve_bias_correction(options["bias_correction"])
            if "bias_correction" in options
            else prepared_bias
        ),
    )


def fixed_values(options: Mapping[str, object], fixed: Mapping[str, object]) -> None:
    """A convenience wrapper's fixed values against the explicit ones of the call.

    An explicit value equal to the fixed one is accepted; a contradictory one is
    refused (``specification.conflicting``), never overridden silently.  ``tau``
    fixed by a wrapper also refuses an explicit ``variance``.
    """
    for key, value in fixed.items():
        if key in options and not _same_number(options[key], value):
            raise _conflicting(f"{key}={options[key]!r}", f"{key}={value!r} (fixed)")
    if "tau" in fixed and "variance" in options:
        raise _conflicting(
            f"variance={options['variance']!r}", f"tau={fixed['tau']!r} (fixed)"
        )


def _same_number(value: object, fixed: object) -> bool:
    """Whether an explicit number equals the fixed one: a real, not a Boolean."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return False
    return bool(value == fixed)
