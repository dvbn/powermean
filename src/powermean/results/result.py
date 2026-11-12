"""The result object of a fit.

A :class:`Result` holds the :class:`~powermean.results.retained.RetainedFit` that
:func:`powermean.results.handoff.bind` formed, and exposes the names ``coef``, ``se``,
``vcov``, ``confint``, ``tidy``, ``summary``, ``predict`` and ``influence``, and the
attributes ``certificate``, ``sample``, ``estimator`` and ``provenance``.  It is an
internal class: the package does not export its name.

``effect`` and a separate ``effects`` target table use a shared numeric reporting
context, without changing the coefficient panel.  The additive ``contrast`` method
replays two explicit raw regressor configurations in the fitted design, holding fixed
effects and offsets fixed. Local derivatives have internal typed support; a shared
table/export API is not exposed.  Coefficient combinations and restrictions use the same
context on demand; redundant restrictions count once under the machine-precision rank
rule, and a discrepancy with an untestable covariance-range component receives no
p-value.

Every quantity is bound to the same point, equation, rows and coordinate basis, and
mapped once by the order scale: the coefficients ``beta = theta / d / omega``, the
covariance ``V / (d d') / omega^2`` entry by entry, the standard errors ``s / d /
|omega|``, the coefficient influences ``A s_i / d / omega`` after the bread map; a
negative order reverses the signed maps and not the standard errors; at ``omega = 0``
the order multiplier is absent and the inverse column scaling remains.  Tables and
series are fresh copies whose edition changes nothing retained; no method hands out a
retained array, the fitted ``ModelSpec`` or a mutable buffer, and no attribute or
refusal hands out a retained record: the certificate, and the last state and certificate
that a :class:`~powermean.errors.ConvergenceError` carries, are re-formed on every
access with arrays of their own (:mod:`powermean.results.owned`), since the shape of an
array over immutable bytes is not immutable, and an edit of a record handed out must
reach nothing retained.  The labels of a prediction are owned copies of the caller's
Index, by the rule of the preparation (``powermean._prepare.compose``): the same type,
names, levels, categories, duplicates and order, never the caller's label or name
objects.

Inference is withheld where the point is not certified: a failed solve keeps its last
state and its diagnostics, and ``se``, ``vcov``, ``confint``, ``tidy``, ``influence``
and ``predict`` raise :class:`~powermean.errors.ConvergenceError` with the status, the
last state and the certificate, while ``coef`` reports the point and ``summary`` prints
it as uncertified.  On the linear route the fresh certificate
(:mod:`powermean.solve.linear_certificate`) decides as the joint one does; a bare linear
candidate handed over without its conclusion has no certificate and no status, and its
inference and prediction are refused as not available without that certificate, while
its coefficients and retained components are reported.  Where no point exists, because
the least-squares solve formed none or the automatic start had no finite value, ``coef``
is withheld as well, with the last state that exists (the candidate with its reason, or
the start failure) and no certificate.  Degenerate inference at an exact root reports
its zero standard errors, undefined ratios and missing intervals as such: nothing is
floored.

A later covariance (``vcov``) is formed again from the retained primitives by the
convention of the fit, for ``"robust"`` or one classification retained with the fit, or,
for the exact string of a named HC method with the explicit ``ssc`` and ``bread`` of its
call, and for a typed cluster request of retained dimensions (``{"cluster": "a + b"}``
with ``"multiway"`` and ``"psd"``) with the explicit ``ssc`` of its call, by the
retained adapters at the certified point (:mod:`powermean.results.active_covariance`);
the covariance requested at the fit is cached and served without them, so that
``keep_influence=False`` keeps the default available while a request that needs the
discarded contributions is refused and nothing is rebuilt from the certificate.
``with_vcov`` returns a view whose standard errors, tables, summary, effects and tests
read the covariance it names, under that covariance's reference (normal, or ``t`` with
its degrees of freedom), while its coefficients, point, status and certificate are the
fit's.  A prediction replays the owned fitted right-hand side on the caller's rows and
never learns from them (:mod:`powermean.results.prediction`), and an offset or exposure
given as an expression is replayed the same way from the caller's original columns
(:func:`powermean.results.prediction.expression_offset`); its one keyword, ``type``, as
in ``predict(type="power_mean")``, is declared as the typed record
:class:`PredictionOptions` of the keyword collector and validated there, so that an
unknown keyword and a kind that is not available are refused with their identifiers.
The reference of the active covariance gives the statistics, the p-values and the
intervals: the normal reference of the ``"stata"`` convention, or the ``t`` of the
convention in force of a named method or of a cluster request.
"""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict, Unpack

import numpy as np
import pandas as pd

from powermean import messages
from powermean._prepare.compose import _owned_index
from powermean.data.request_values import OMITTED, given
from powermean.data.specs import Estimator
from powermean.effects.targets import IntervalKind, ReportingContext, ReportingPoint
from powermean.errors import ConvergenceError, SpecificationError
from powermean.model.variance import FloatArray
from powermean.results import covariance, naming, prediction
from powermean.results.absorbed_owned import RetainedCertificateSnapshot
from powermean.results.active_covariance import (
    answer,
    fe_keywords,
    t_columns,
    view_provenance,
)
from powermean.results.contrast_reporting import ContrastResult, contrast_result
from powermean.results.effect_design import simple_columns
from powermean.results.effect_reporting import (
    EffectResult,
    context_snapshot,
    effect_result,
    effects_table,
)
from powermean.results.fe_result_backends import FEBackend
from powermean.results.fe_sample import FESampleView
from powermean.results.owned import (
    owned_any_certificate,
    owned_candidate,
    owned_linear,
)
from powermean.results.provenance import Provenance, provenance
from powermean.results.restriction_reporting import (
    CombinationResult,
    RestrictionResult,
    equality_result,
    linear_result,
    nonlinear_result,
    restriction_result,
)
from powermean.results.retained import AnyCertificate, CovarianceRecord, RetainedFit
from powermean.results.sample import SampleView
from powermean.results.summary import render
from powermean.solve.cancelled_records import CancelledCertificate

PredictionKind = Literal["power_mean"]
PREDICTION_KINDS: tuple[PredictionKind, ...] = ("power_mean",)
DEFAULT_PREDICTION: PredictionKind = "power_mean"
# The keywords of ``predict``: the one key of ``PredictionOptions``.
PREDICTION_KEYWORDS: tuple[str, ...] = ("type",)
INFERENCE_STATUS: tuple[str, ...] = ("ok", "degenerate", "missing")
LINEAR_FEATURE = "the certificate of the linear path, which {need} needs"


class PredictionOptions(TypedDict, total=False):
    """The keyword options of :meth:`Result.predict`.

    ``type`` is the kind of prediction, ``"power_mean"`` by default: the fitted
    conditional power mean of the order of the fit, the only kind available.  No other
    key exists; :func:`prediction_kind` refuses one at run time, as the type checker
    refuses it at the call.
    """

    type: PredictionKind


def prediction_kind(options: Mapping[str, object]) -> PredictionKind:
    """The kind of prediction that the keywords of ``predict`` name.

    Parameters
    ----------
    options : mapping
        The keywords given, as the collector received them.

    Returns
    -------
    str
        ``"power_mean"``, the default and the only kind.

    Raises
    ------
    SpecificationError
        A keyword other than ``type`` (``specification.unknown_option``, with the
        valid keyword), or a ``type`` that is not available
        (``specification.invalid_value``, with the valid kinds).
    """
    for key in options:
        if key not in PREDICTION_KEYWORDS:
            raise SpecificationError(
                "specification.unknown_option",
                {
                    "kind": "keyword of predict",
                    "key": key,
                    "valid": messages.format_keys(PREDICTION_KEYWORDS),
                },
            )
    kind = options.get("type", DEFAULT_PREDICTION)
    for valid in PREDICTION_KINDS:
        if kind == valid:
            return valid
    raise SpecificationError(
        "specification.invalid_value",
        {
            "argument": "type",
            "value": kind,
            "valid": messages.format_keys(PREDICTION_KINDS),
        },
    )


def normal_quantile(level: float) -> float:
    """``q`` with ``P(|Z| <= q) = level`` under the standard normal distribution."""
    if not isinstance(level, float) or not 0 < level < 1:
        raise ValueError("level must be a float strictly between zero and one")
    return statistics.NormalDist().inv_cdf((1 + level) / 2)


def two_sided_p(statistic: FloatArray) -> FloatArray:
    """``P(|Z| >= |z|)`` under the standard normal, NaN where ``z`` is undefined."""
    values = np.full(statistic.shape, np.nan)
    for position, value in enumerate(statistic.tolist()):
        if not math.isnan(value):
            values[position] = math.erfc(abs(value) / math.sqrt(2.0))
    return values


def not_certified(result: Result) -> ConvergenceError:
    """The refusal of inference of a result that is not certified.

    The error that the result's inference raises, for the dispatch of the fit with
    ``on_failure="raise"``: the status, the last state (the solver's candidate, the
    linear candidate, the start failure or the supplied certificate) and the
    certificate, as owned copies.
    """
    return result._withheld()


class Result:
    """The result of a fit (module docstring).

    Parameters
    ----------
    retained : RetainedFit
        The state that :func:`powermean.results.handoff.bind` formed.

    Attributes
    ----------
    certificate : Certificate, LinearCertificate or None
        The authentic certificate of the point, as supplied: the joint one on the
        power route, the fresh linear one on the linear route; ``None`` where no
        point was certified (no point exists, or a bare linear candidate).
    sample : SampleView
        The sample of the fit: positions, labels, accounting and counts.
    estimator : Estimator
        The estimator record.
    provenance : Provenance
        What ran.
    """

    __slots__ = ("_fe", "_fit", "_provenance", "_sample", "_view")

    def __init__(self, retained: RetainedFit | FEBackend) -> None:
        """Hold a typed retained fit or its FE capability adapter."""
        self._fit: RetainedFit | None = None
        self._fe: FEBackend | None = None
        # The active covariance of a view (with_vcov); None: the fit's own.
        self._view: CovarianceRecord | None = None
        self._sample: SampleView | FESampleView
        if isinstance(retained, RetainedFit):
            self._fit = retained
            self._sample = SampleView(
                retained.sample,
                retained.labels,
                retained.columns,
                retained.subset_missing,
            )
            self._provenance = provenance(retained)
        else:
            self._fe = retained
            self._sample = retained.sample
            self._provenance = retained.provenance

    @property
    def _retained(self) -> RetainedFit:
        """The no-FE state, available only on its unchanged branch."""
        assert self._fit is not None
        return self._fit

    # -------------------------------------------------------------- attributes

    @property
    def certificate(
        self,
    ) -> AnyCertificate | RetainedCertificateSnapshot | CancelledCertificate | None:
        """The authentic certificate of the point, as supplied; a fresh owned copy.

        The same values, quantities, gates and diagnostics on every access, with
        arrays of their own: an edit of a copy handed out reaches nothing retained.
        """
        if self._fe is not None:
            return self._fe.certificate
        certificate = self._retained.certificate
        return None if certificate is None else owned_any_certificate(certificate)

    @property
    def sample(self) -> SampleView | FESampleView:
        """The sample of the fit."""
        return self._sample

    @property
    def estimator(self) -> Estimator:
        """The estimator record."""
        return self._fe.estimator if self._fe is not None else self._retained.estimator

    @property
    def provenance(self) -> Provenance:
        """What ran."""
        return self._provenance

    # ---------------------------------------------------------------- gates

    def _names(self) -> pd.Index[str]:
        """The coefficient names as an index labeled ``term``."""
        if self._fe is not None:
            return self._fe._names()
        return pd.Index(list(self._retained.names), name="term")

    def _withheld(self) -> ConvergenceError:
        """The refusal of inference where the point is not certified.

        The last state (the solver's candidate, the linear candidate with its
        reason, the start failure, or the supplied certificate) and the certificate
        travel as owned copies: the error is the caller's to inspect, and an edit of
        what it carries reaches nothing retained.
        """
        if self._fe is not None:
            return self._fe._withheld()
        retained = self._retained
        status = retained.status
        if status is None:
            raise ValueError("a point without a certificate has no status")
        certificate = self.certificate
        last_state: object = certificate
        if retained.candidate is not None:
            last_state = owned_candidate(retained.candidate)
        elif retained.linear is not None:
            last_state = owned_linear(retained.linear)
        elif retained.failure is not None:
            last_state = retained.failure
        identifier = "convergence.not_certified"
        fields: dict[str, object] | None = None
        selected_certificate = retained.certificate
        if (
            selected_certificate is not None
            and selected_certificate.secondary == ("coefficient",)
            and selected_certificate.reported is not None
        ):
            errors = selected_certificate.reported.standard_errors
            unusable = np.flatnonzero(~np.isfinite(errors) | (errors < 0))
            if unusable.size:
                identifier = "convergence.unusable_covariance"
                fields = {
                    "coefficients": messages.format_keys(
                        [retained.names[int(i)] for i in unusable]
                    )
                }
        return ConvergenceError(
            identifier,
            fields,
            status=status,
            last_state=last_state,
            certificate=certificate,
        )

    def _linear_refusal(self, need: str) -> SpecificationError:
        """The refusal of what a bare linear candidate cannot give: no certificate."""
        return SpecificationError(
            "specification.unimplemented",
            {"feature": LINEAR_FEATURE.format(need=need)},
        )

    def _theta(self) -> FloatArray:
        """The point in the coordinates of the solver, where one was formed."""
        retained = self._retained
        if retained.theta is None:
            if retained.status is not None:
                raise self._withheld()
            raise self._linear_refusal("a point of a rank-deficient linear solve")
        if not np.isfinite(retained.theta).all():
            raise self._withheld()
        return retained.theta

    def _certified_default(self, need: str) -> CovarianceRecord:
        """The default covariance of a certified point, for ``need``."""
        retained = self._retained
        if retained.estimator.route == "linear" and retained.status is None:
            raise self._linear_refusal(need)
        if not retained.certified:
            raise self._withheld()
        if retained.default is None:
            raise self._withheld()
        return retained.default

    def _active(self, need: str) -> CovarianceRecord:
        """The active covariance of a certified point: a view's, else the fit's own."""
        default = self._certified_default(need)
        return default if self._view is None else self._view

    # ---------------------------------------------------------------- reports

    def _beta(self) -> FloatArray:
        """The reported coefficients ``theta / d / omega``."""
        return self._retained.sample.order_scale.coefficients(self._theta())

    def coef(self) -> pd.Series[float]:
        """The reported coefficients, by name.

        Returns
        -------
        pandas.Series
            ``beta``, a fresh series named ``estimate`` and indexed by ``term``.

        Raises
        ------
        ConvergenceError
            The point is not finite (status ``nonfinite``), or no point exists: the
            least-squares solve formed none, or the automatic start had no finite
            value.
        SpecificationError
            A bare linear candidate without its conclusion formed no coefficients.
        """
        if self._fe is not None:
            return self._fe.coef()
        return pd.Series(np.array(self._beta()), index=self._names(), name="estimate")

    def se(self) -> pd.Series[float]:
        """The standard errors of the active covariance, by name.

        The covariance requested at the fit, or that of a view (``with_vcov``).  Zero at
        an exact root with degenerate inference, NaN where a variance is negative; never
        floored.
        """
        if self._fe is not None:
            return self._fe.se()
        record = self._active("a standard error")
        errors = np.array(record.standard_errors)
        return pd.Series(errors, index=self._names(), name="std_error")

    def _reporting_context(self) -> ReportingContext:
        """Snapshot reported coordinates without refit, recovery or influence.

        The active covariance with its reference (normal, or ``t`` with its degrees
        of freedom) and its records; the coefficients are the fit's.
        """
        if self._fe is not None:
            return self._fe._reporting_context()
        record = self._active("a coefficient target")
        return context_snapshot(
            self._retained.names,
            self._beta(),
            record.reported,
            record.distribution,
            self.estimator.mean.omega,
            point=ReportingPoint(str(self._retained.status)),
            metadata=record.to_dict(),
        )

    def effect(
        self,
        term: str,
        *,
        contrast: tuple[float, float] | None = None,
        level: float = 0.95,
        ci: IntervalKind = "transform",
    ) -> EffectResult:
        """A finite proportional change on the fitted conditional power mean.

        Parameters
        ----------
        term : str
            A standalone direct level regressor. Ambiguous nonlinear/interacted or
            categorical selectors require a full learned-design contrast.
        contrast : tuple of float, optional
            Values (d0, d1); omission means a one-unit change.
        level : float, default 0.95
            Pointwise confidence level using the active covariance's actual reference.
        ci : {"transform", "delta"}, default "transform"
            Transform log-contrast endpoints, or use a delta interval.

        Returns
        -------
        EffectResult
            Detached estimate, SE, interval, order, label, metadata and optional
            on-demand influence from genuinely retained coefficient primitives.
        """
        context = self._reporting_context()
        design = self._retained.design if self._fe is None else self._fe.design
        return effect_result(
            context,
            simple_columns(design),
            term,
            contrast=contrast,
            level=level,
            ci=ci,
            influence_provider=self.influence,
        )

    def contrast(
        self,
        before: Mapping[str, object],
        after: Mapping[str, object],
        *,
        level: float = 0.95,
        ci: IntervalKind = "transform",
    ) -> ContrastResult:
        """Compare two explicit configurations on the conditional power mean.

        Parameters
        ----------
        before, after : mapping of str to object
            Complete raw RHS regressor rows, with exactly the fitted design's raw
            variable names and scalar values. Fitted categorical coding and
            stateful transforms are replayed. No values are filled or averaged.
            A categorical level outside the fitted encoding is refused, whatever
            the warning filters.
            Extra outcome, fixed-effect, offset or exposure inputs are refused.
            Fixed effects and offsets are held fixed, even if an RHS variable also
            supplies one of those components. No fixed effects need be recovered.
        level : float, default 0.95
            Pointwise confidence level under the active covariance's reference.
        ci : {"transform", "delta"}, default "transform"
            Transform log-contrast endpoints, or use the existing delta policy.

        Returns
        -------
        ContrastResult
            Detached proportional effect expm1((X(after)-X(before))'beta), standard
            error, interval, full coefficient gradient, raw configurations, target
            table and metadata. Optional influence requires retained contributions.
        """
        context = self._reporting_context()
        design = self._retained.design if self._fe is None else self._fe.design
        return contrast_result(
            context,
            design,
            before,
            after,
            level=level,
            ci=ci,
            influence_provider=self.influence,
        )

    def effects(
        self,
        terms: Sequence[str] | None = None,
        *,
        level: float = 0.95,
        ci: IntervalKind = "transform",
    ) -> pd.DataFrame:
        """A separate target table, in requested order, using one numeric snapshot.

        Parameters
        ----------
        terms : sequence of str, optional
            Selected standalone direct level regressors. Omission keeps all reported
            coefficients as exp(beta)-1 coordinate transformations, not as full
            regressor changes. Selected ambiguous or omitted regressors are refused.
        level : float, default 0.95
            Pointwise confidence level; rows do not imply simultaneous coverage.
        ci : {"transform", "delta"}, default "transform"
            Interval transformation, as in :meth:`effect`.

        Returns
        -------
        pandas.DataFrame
            Fresh proportional target rows and detached inference/point metadata.
            The ordinary coefficient table remains available through ``tidy()``.
        """
        context = self._reporting_context()
        design = self._retained.design if self._fe is None else self._fe.design
        return effects_table(context, design, terms, level=level, ci=ci)

    def wald(self, r: object, q: object) -> RestrictionResult:
        """Test R beta = q with the active reported point, covariance and reference.

        R is an explicit m by k real matrix in reported coefficient order; q is an
        explicit m-vector. Redundant rows count once: the effective count r is the
        machine-precision rank of R V R' (NumPy's documented matrix_rank rule), shared
        by the generalized inverse. Normal gives chi-square(r); t gives F(r, df) = Q/r
        with the fit's denominator df. The report carries the requested count, the
        independent row count and the reduction. A discrepancy with a component outside
        the covariance range is degenerate with no p-value; exact-zero covariance is
        degenerate; negative or undefined covariance is missing. Empty or inconsistent
        requests refuse.
        """
        return restriction_result(self._reporting_context(), r, q)

    def test(self, expression: str) -> RestrictionResult:
        """Test one affine coefficient equality, for example ``"x1 = x2"``.

        Uses the same restrictions and reference as :meth:`wald`. Identifiers
        name reported coefficients; arithmetic and real constants are allowed.
        Multi-equation strings and Python evaluation are not supported.
        """
        return equality_result(self._reporting_context(), expression)

    def lincom(self, expression: str, *, level: float = 0.95) -> CombinationResult:
        """Report an affine combination, for example ``"x1 + 2*x2"``.

        The detached scalar record has estimate, standard error, statistic,
        p-value, ordinary interval, full gradient, table and active metadata.
        Variance is the full c'Vc; normal/t inference uses the active reference.
        """
        return linear_result(self._reporting_context(), expression, level=level)

    def nlcom(self, expression: str, *, level: float = 0.95) -> CombinationResult:
        """Report smooth coefficient algebra by the analytic delta method.

        Arithmetic and ``exp`` are supported, for example ``"exp(x1 + 2*x2)-1"``.
        Value and full analytic gradient use the reported (possibly corrected)
        coefficients. The active full covariance is used once. Identifiers name
        reported coefficients; arbitrary Python evaluation is never used.
        """
        return nonlinear_result(self._reporting_context(), expression, level=level)

    def with_vcov(
        self, spec: object, *, ssc: object = OMITTED, bread: object = OMITTED
    ) -> Result:
        """A view of the result under another covariance, without a new fit.

        Parameters
        ----------
        spec, ssc, bread
            The request, as :meth:`vcov` takes it.

        Returns
        -------
        Result
            A result over the same retained fit whose standard errors, tables,
            summary, effects, contrasts and tests read the requested covariance with
            its reference.  The coefficients, the point, the status and the
            certificate are the fit's: the certificate continues to describe the
            covariance requested at fitting, and nothing is certified again.  A
            fixed-effects result keeps its own views, coefficients and any explicit
            BBP shift at their original points.

        Raises
        ------
        ConvergenceError, SpecificationError
            As :meth:`vcov`.
        """
        if self._fe is not None:
            fe_keywords(ssc, bread)
            return Result(self._fe.with_vcov(spec))
        default = self._certified_default("a covariance")
        record = answer(self._retained, default, spec, ssc=ssc, bread=bread)
        view = copy.copy(self)
        view._view = record
        view._provenance = view_provenance(self._provenance, record)
        return view

    def vcov(
        self, spec: object = None, *, ssc: object = OMITTED, bread: object = OMITTED
    ) -> pd.DataFrame:
        """A covariance of the coefficients on the reported scale, by name.

        Parameters
        ----------
        spec : None, str, mapping or Inference, optional
            ``None`` for the active covariance (the one requested at the fit, or a
            view's); ``"robust"``; ``{"cluster": name}`` for a classification
            retained with the fit; ``{"cluster": "a + b"}`` for several retained
            dimensions, with the optional keys ``"multiway"`` (``"cgm"``, the
            default signed form, or ``"ddg"``) and ``"psd"`` (``"cgm"`` or
            ``"none"``, on a signed multiway request only); the exact string of a
            named HC method, ``"HC0"`` to ``"HC5"`` or ``"HC4m"``; or an
            :class:`~powermean.data.specs.Inference` record.
        ssc : str or ConventionValues, optional
            The explicit convention of a named HC method: ``"stata"``, ``"none"``,
            ``"regression"`` or a custom record.  Omitted, the fit's convention in
            force, whose factor a named method never takes and whose reference it
            keeps.  Robust and cluster requests take that convention's factor,
            reference and ``psd`` treatment, with no second scaling; never a
            key of the mapping.
        bread : {"observed", "expected"}, optional
            The bread of a named HC method; omitted, the observed bread.  Beside
            ``"robust"`` or a cluster only ``"observed"`` is accepted.

        Returns
        -------
        pandas.DataFrame
            The covariance, a fresh table whose ``attrs`` record the method, the
            cluster, the factor, ``N``, ``G``, the rank, the diagnostics, the bread,
            the coordinate and the convention; for a named method or a recomputed
            convention also the convention requested and in force with its source,
            the factor's owner, the reference's degrees of freedom and, for a named
            method, the leverage geometry; for a multiway covariance also its form,
            the cluster count of each dimension and the ``psd`` treatment.

        Raises
        ------
        ConvergenceError
            The point is not certified.
        SpecificationError
            A request that is not available (HAC and Conley: a later release), a
            classification that was not retained or has a missing value on the
            estimation sample, ``"psd"`` outside a signed multiway request, a request
            that needs the discarded influence contributions, HC1 with ``N = K``, a
            second correction beside a named method, structural unit leverage.
        """
        if self._fe is not None:
            fe_keywords(ssc, bread)
            return self._fe.vcov(spec)
        default = self._certified_default("a covariance")
        if spec is None and not given(ssc) and not given(bread):
            record = default if self._view is None else self._view
        else:
            record = answer(self._retained, default, spec, ssc=ssc, bread=bread)
        return covariance.frame(record, self._retained.names)

    def confint(self, level: float = 0.95) -> pd.DataFrame:
        """The intervals ``beta_j -/+ q se_j`` of the active covariance.

        Parameters
        ----------
        level : float, default 0.95
            The confidence level, strictly between zero and one.

        Returns
        -------
        pandas.DataFrame
            Columns ``lower`` and ``upper``, by name; NaN where the standard error
            is zero or missing.  ``q`` is the quantile of the active
            covariance's reference: normal, or ``t`` with its degrees of freedom.
        """
        if self._fe is not None:
            return self._fe.confint(level)
        table = self.tidy(level)
        frame = table[["conf_low", "conf_high"]].to_numpy(dtype=np.float64)
        return pd.DataFrame(frame, index=self._names(), columns=["lower", "upper"])

    def tidy(self, level: float = 0.95) -> pd.DataFrame:
        """One row per coefficient: estimate, standard error, ratio, p-value, interval.

        Parameters
        ----------
        level : float, default 0.95
            The confidence level of the interval.

        Returns
        -------
        pandas.DataFrame
            ``term``, ``estimate``, ``std_error``, ``statistic``, ``p_value``,
            ``conf_low``, ``conf_high`` and ``inference`` (``ok``, ``degenerate`` for
            a zero standard error, whose ratio is undefined, or ``missing``), under
            the active covariance and its reference: normal, or ``t`` with its
            degrees of freedom.
        """
        if self._fe is not None:
            return self._fe.tidy(level)
        quantile = normal_quantile(level)
        record = self._active("an interval")
        estimate = np.array(self._beta())
        errors = np.array(record.standard_errors)
        statistic = np.full(estimate.shape, np.nan)
        usable = np.isfinite(errors) & (errors > 0)
        with np.errstate(over="ignore", invalid="ignore"):
            statistic[usable] = estimate[usable] / errors[usable]
        reference = record.distribution
        if reference.kind == "normal":
            p_values = two_sided_p(statistic)
        else:
            quantile, p_values = t_columns(reference, level, statistic)
        with np.errstate(over="ignore", invalid="ignore"):
            half = np.where(usable, quantile * errors, np.nan)
        status = np.where(usable, "ok", np.where(errors == 0, "degenerate", "missing"))
        return pd.DataFrame(
            {
                "term": list(self._retained.names),
                "estimate": estimate,
                "std_error": errors,
                "statistic": statistic,
                "p_value": p_values,
                "conf_low": estimate - half,
                "conf_high": estimate + half,
                "inference": status.astype(object),
            }
        )

    def summary(self, level: float = 0.95) -> str:
        """The printed summary, as text; no warning."""
        if self._fe is not None:
            return self._fe.summary(level)
        normal_quantile(level)  # the level is checked whether or not a table forms
        retained = self._retained
        table: pd.DataFrame | None = None
        beta: pd.Series[float] | None = None
        record: CovarianceRecord | None = None
        if retained.certified and retained.default is not None:
            record = self._active("an interval")
            table = self.tidy(level)
        elif retained.theta is not None and np.isfinite(retained.theta).all():
            beta = self.coef()
        return render(
            retained,
            self._provenance,
            table=table,
            beta=beta,
            level=level,
            record=record,
        )

    def influence(self) -> pd.DataFrame:
        """The coefficient influences ``A s_i`` on the reported scale, by row.

        Returns
        -------
        pandas.DataFrame
            One row per row of positive weight, indexed by its original position
            (``position``), one column per coefficient: the contribution of the row
            to the first-order change of ``beta`` (sum normalization); frequency
            weights give one row per stored row.

        Raises
        ------
        ConvergenceError
            The point is not certified.
        SpecificationError
            The influence contributions were discarded (``keep_influence=False``), or
            a bare linear candidate has no certificate.
        """
        if self._fe is not None:
            return self._fe.influence()
        self._certified_default("an influence")
        retained = self._retained
        influence = retained.influence
        if influence is None:
            raise SpecificationError(
                "specification.influence_discarded", {"request": "the influences"}
            )
        rows = covariance.influences(influence, retained.sample.order_scale)
        positions = retained.sample.positions[influence.units.identifiers]
        return pd.DataFrame(
            np.array(rows),
            index=pd.Index(positions.tolist(), name="position"),
            columns=self._names(),
        )

    def fixef(self) -> dict[str, pd.Series[float]]:
        """The normalized absorbed fixed effects per block on the reported scale."""
        if self._fe is not None:
            return self._fe.fixef()
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "fixef",
                "value": "no fixed effects",
                "valid": "a fitted model with absorbed fixed effects",
            },
        )

    # ------------------------------------------------------------- prediction

    def predict(
        self, data: pd.DataFrame | None = None, **options: Unpack[PredictionOptions]
    ) -> pd.Series[float]:
        """The fitted conditional power mean ``exp(X beta + o)``."""
        if self._fe is not None:
            return self._fe.predict(data, **options)
        kind = prediction_kind(options)
        retained = self._retained
        if retained.estimator.route == "linear" and retained.status is None:
            raise self._linear_refusal("a prediction")
        if not retained.certified:
            raise self._withheld()
        beta = self._beta()
        sample = retained.sample
        if data is None:
            fitted = prediction.power_mean(sample.data.design, beta, sample.data.offset)
            index: pd.Index[Any] = self._sample.labels
        else:
            if not isinstance(data, pd.DataFrame):
                raise TypeError("data must be a pandas DataFrame")
            matrix = prediction.replayed_matrix(retained.design, data)
            offset = self._prediction_offset(data)
            fitted = prediction.power_mean(matrix, beta, offset)
            index = _owned_index(data.index)
        series = pd.Series(fitted, index=index, name=kind)
        series.attrs["label"] = naming.prediction_label(retained.estimator.mean.omega)
        series.attrs["omega"] = retained.estimator.mean.omega
        return series

    def _prediction_offset(self, data: pd.DataFrame) -> FloatArray:
        """The raw offset of the prediction rows: a column read, or an expression."""
        retained = self._retained
        columns = retained.columns
        design = retained.offset_design
        if design is not None:
            return prediction.expression_offset(
                design, data, exposure=columns.exposure is not None
            )
        return prediction.offset_of(
            data, offset=columns.offset, exposure=columns.exposure
        )
