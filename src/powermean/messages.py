"""Every text shown to the user, with an identifier and one template.

Each message of the package is a :class:`Message`: an identifier such as
``"specification.unknown_option"``, the name of the exception or warning class
that carries it, and one template in the syntax of :meth:`str.format` whose fields
are plain names.  :data:`REGISTRY` maps identifiers to messages, and :func:`render`
formats one deterministically.  :mod:`powermean.errors` consumes this module;
nothing here imports beyond the standard library.

Contract (``DESIGN.md``, Section 4.6; ``design/11_engineering.md``, Section 5,
rule 8; decision D45):

- An identifier is ``"<prefix>.<name>"``.  The prefix names the class that carries
  the message: ``specification``, ``sample``, ``identification``, ``existence``,
  ``convergence``, ``solver``, ``coverage``, ``numerical`` and ``covariance``.
  Identifiers are stable: a test asserts on them, and the ports print the same texts.
- A template is rendered with exactly the fields it names, in insertion-free,
  locale-free :meth:`str.format` formatting; ``specification.hac_replicas`` names
  none.  A missing field raises :class:`KeyError`, an unexpected one
  :class:`TypeError`, an unknown identifier :class:`KeyError`.  These are internal
  errors of the caller, not messages for the user.
- A field converted with ``!r`` whose type is exactly ``int`` or
  :class:`fractions.Fraction` is written with every decimal digit, in the syntax of
  :func:`repr`, however many digits it has: :func:`repr` refuses an integer with
  more digits than :func:`sys.get_int_max_str_digits` allows, and a refused value
  is still reported as supplied.  Every other field is formatted as
  :meth:`str.format` formats it.
- A field that lists valid keys (``valid``) receives the string returned by
  :func:`format_keys`, which sorts the keys and quotes each one, so that an
  unknown-option message always lists the valid keys in one order.
- The messages of ``existence``, ``convergence`` and ``numerical`` name the field
  ``status``: the exception or warning that carries them fills it with its ``status``
  attribute (``design/02_computation.md``, Section 7), so that text and payload
  cannot disagree.

The catalog covers the foundation and the error cases documented for M0: the
refusals of decision D41, the sample rules of ``design/02_computation.md``, Section
2, steps 3 and 4, the unknown-option error of Section 5.7, and the statuses of
Section 7.  The identification and existence messages exist with their classes;
their dispatch is the existence analysis and collinearity resolution of M1 (D41,
item 4).  A later milestone adds its messages here.

Decision D65 adds the seventeen templates of the expanded covariance menu, worded
as ``plan/evidence/M1_contract.txt``, Section 6(i), states them: two notes carried
by ``CovarianceWarning`` (prefix ``covariance``), fourteen refusals carried by
``SpecificationError``, and ``convergence.unusable_covariance``, whose ``status`` the
``ConvergenceError`` that carries it fills.  They are registered, and exported with
the interface description, before any fit, request or result issues them: their
dispatch belongs to the tasks that implement the requests of milestone M1.  The
wording of ``specification.hac_lag_too_long`` fixes no bound of an explicit lag.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from string import Formatter
from types import MappingProxyType

# The class that carries a message, by identifier prefix.  Each name is a class of
# `powermean.errors`; the test of that module checks the correspondence.
CATEGORIES: Mapping[str, str] = MappingProxyType(
    {
        "specification": "SpecificationError",
        "sample": "SampleError",
        "identification": "IdentificationError",
        "existence": "ExistenceError",
        "convergence": "ConvergenceError",
        "solver": "SolverCapabilityError",
        "coverage": "CoverageWarning",
        "numerical": "NumericalWarning",
        "covariance": "CovarianceWarning",
    }
)


@dataclass(frozen=True)
class Message:
    """One message: its identifier, the class that carries it, and its template."""

    identifier: str
    category: str
    template: str

    @property
    def fields(self) -> tuple[str, ...]:
        """The names of the fields of the template, in order of first appearance."""
        return field_names(self.template)


def field_names(template: str) -> tuple[str, ...]:
    """The field names of a template, each a plain identifier, without duplicates.

    Raises :class:`ValueError` for a positional, attribute or index field, which the
    registry does not accept because a message is rendered from named values only.
    """
    names: list[str] = []
    for _, name, _, _ in Formatter().parse(template):
        if name is None:
            continue
        if not name.isidentifier():
            msg = f"template field {name!r} is not a plain name: {template!r}"
            raise ValueError(msg)
        if name not in names:
            names.append(name)
    return tuple(names)


def format_keys(keys: Iterable[str]) -> str:
    """The keys, sorted, each quoted, joined by commas: ``'a', 'b', 'c'``.

    Used to fill the field ``valid`` of a message that lists valid keys or values.
    Raises :class:`ValueError` when there is no key, because a message that offers
    no valid key is malformed.
    """
    sorted_keys = sorted(keys)
    if not sorted_keys:
        msg = "a list of valid keys cannot be empty"
        raise ValueError(msg)
    return ", ".join(repr(key) for key in sorted_keys)


def _catalog(entries: Iterable[tuple[str, str]]) -> Mapping[str, Message]:
    """Build the registry from ``(identifier, template)`` pairs, checking each."""
    registry: dict[str, Message] = {}
    for identifier, template in entries:
        prefix, _, name = identifier.partition(".")
        if prefix not in CATEGORIES or not name:
            msg = f"identifier {identifier!r} has no known prefix"
            raise ValueError(msg)
        if identifier in registry:
            msg = f"identifier {identifier!r} is registered twice"
            raise ValueError(msg)
        field_names(template)
        registry[identifier] = Message(identifier, CATEGORIES[prefix], template)
    return MappingProxyType(registry)


REGISTRY: Mapping[str, Message] = _catalog(
    [
        # SpecificationError: inconsistent arguments, and the refusals of D41.
        (
            "specification.unimplemented",
            "Not available in this version: {feature}.",
        ),
        (
            "specification.conflicting",
            "{first} cannot be given together with {second}.",
        ),
        (
            "specification.invalid_value",
            "{argument}={value!r} is not accepted; the valid values are {valid}.",
        ),
        (
            "specification.unknown_option",
            "Unknown {kind} {key!r}; the valid ones are {valid}.",
        ),
        # SpecificationError: the refusals of a later request on a result (task
        # M0.21): a cluster variable that the fit did not retain, one with a missing
        # value on the estimation sample (no row is dropped after the fit), and a
        # request that needs the influence contributions discarded by
        # keep_influence=False, which requires a refit (design/04, Section 1).
        (
            "specification.cluster_not_retained",
            "Cluster variable {name!r} was not retained with the fit; the retained "
            "cluster variables are {retained}.",
        ),
        (
            "specification.cluster_missing_values",
            "Cluster variable {name!r} has {count} missing value(s) on the estimation "
            "sample; no row is dropped after the fit.",
        ),
        (
            "specification.influence_discarded",
            "The influence contributions were discarded (keep_influence=False); "
            "{request} needs them and requires a refit.",
        ),
        # SpecificationError: the refusals of the expanded covariance menu of milestone
        # M1, approved by decision D65 (plan/evidence/M1_contract.txt, Section 6(i)):
        # dependence metadata named after the fit, the time of a HAC request, its
        # replicated rows and its lag, the residual degrees of freedom of HC1, unit
        # leverage, the coordinates, locations and pooling of a Conley request, and a
        # method that a case does not support.  No request issues them yet.
        (
            "specification.metadata_not_retained",
            "Variable {name!r} was not retained with the fit; the retained variables "
            "are {retained}.",
        ),
        (
            "specification.metadata_missing_values",
            "Variable {name!r} has {count} missing value(s) on the estimation sample; "
            "no row is dropped after the fit.",
        ),
        (
            "specification.time_ties",
            "Time variable {name!r} repeats {count} period(s) within a unit; {request} "
            "needs one row per unit and period.",
        ),
        (
            "specification.time_gaps",
            "Time variable {name!r} has {count} interior gap(s) at step {step}; "
            "{request} needs consecutive periods.",
        ),
        (
            "specification.hac_replicas",
            "Frequency counts above one leave the order of replicated rows undefined; "
            "HAC with a positive or automatic lag is refused; an explicit lag 0 is "
            "available.",
        ),
        (
            "specification.hac_lag_undefined",
            "The automatic HAC lag is undefined: {reason}; pass an explicit lag.",
        ),
        (
            "specification.hac_lag_too_long",
            "lag={lag} is not below the series length {length}.",
        ),
        (
            "specification.no_residual_dof",
            "{method} needs more observations than parameters; N = {count} equals the "
            "rank {rank}.",
        ),
        (
            "specification.hac_automatic_unavailable",
            "Automatic HAC lag selection is not available for {case}; pass an explicit "
            "lag.",
        ),
        (
            "specification.unit_leverage",
            "{method} is undefined: {count} observation(s) have leverage one.",
        ),
        (
            "specification.coordinates_invalid",
            "Coordinates {name!r} are outside the geographic range: {detail}.",
        ),
        (
            "specification.conley_locations",
            "The automatic Conley cutoff needs at least four distinct locations; "
            "{count} found; pass an explicit cutoff.",
        ),
        (
            "specification.conley_pooling",
            "{count} location(s) hold several observations; pass 'pool_times': True to "
            "confirm pooling across times at one location.",
        ),
        (
            "specification.unsupported_inference",
            "{method} is not available for {case}.",
        ),
        # Table arguments and public sample alignment.
        (
            "specification.table_column",
            "{argument} names {name!r}, which is not a column of the data.",
        ),
        (
            "specification.table_not_numeric",
            "Column {name!r} is not numeric; in a list of variables, pass "
            "expand=True to expand a categorical column into indicators.",
        ),
        (
            "specification.table_argument",
            "{argument}={value!r} is not accepted; expected {expected}.",
        ),
        (
            "specification.table_sample",
            "The data do not match the fit given as sample: {detail}.",
        ),
        # SampleError: steps 3 and 4 of sample construction, and an empty sample.
        (
            "sample.negative_outcome",
            "{count} row(s) have a negative outcome; the outcome must be nonnegative.",
        ),
        (
            "sample.zero_outcome_at_order",
            "{count} row(s) have a zero outcome and omega={omega} is not positive; "
            "pass positive_only=True to drop them.",
        ),
        (
            "sample.negative_weight",
            "{count} row(s) have a negative weight; weights must be nonnegative.",
        ),
        (
            "sample.empty",
            "No usable observation remains of the {count} input row(s).",
        ),
        (
            "sample.table_arms",
            "Treatment {name!r} has {count} distinct value(s) on the usable rows; a "
            "comparison of arms needs at least two.",
        ),
        (
            "sample.jochmans",
            "Jochmans requires a complete observed array ({detail}).",
        ),
        (
            "identification.jochmans_no_products",
            "No admissible rectangle contains a positive outcome product.",
        ),
        (
            "identification.jochmans_jacobian",
            "The Jochmans moment Jacobian lacks resolved full rank at the trial point.",
        ),
        (
            "convergence.jochmans",
            "The Jochmans fit could not return a usable local solution: {status}.",
        ),
        (
            "convergence.prediction_recovery",
            "Fixed effects could not be recovered for prediction ({status}): {reason}.",
        ),
        # IdentificationError: dispatched by the collinearity resolution of M1.
        (
            "identification.rank_deficient",
            "The design has rank {rank} for {columns} columns, and the deficiency "
            "cannot be resolved by dropping collinear columns.",
        ),
        # IdentificationError: the sample steps for zeros and singletons (owner's
        # ruling 70) left no observation, on any route.
        (
            "identification.not_identified",
            "No slope is identified once separated observations and singletons are "
            "removed: of the {count} usable row(s), {separated} are separated (zero "
            "outcomes that the fixed effects and regressors fit exactly; "
            "{separated_by_fixed_effect} of them by a fixed effect alone) and "
            "{singleton} are alone in a fixed-effect level; none remains.",
        ),
        (
            "identification.table_rank",
            "The regression of {target} on the variables has not full column rank "
            "({columns} columns, {rows} rows); remove the collinear variables.",
        ),
        # ExistenceError: dispatched by the existence analysis of M1.
        (
            "existence.no_finite_root",
            "No finite root of the estimating equation on this sample: the "
            "certificate reports status {status!r} and is attached to this error.",
        ),
        # ConvergenceError: any status other than the two verified ones, with
        # on_failure="raise".  The text names the status only: a certificate may not
        # exist (a solver_error before certification), so it does not claim one.
        (
            "convergence.not_certified",
            "No certified solution within the budget: status {status!r}; the last "
            "state is attached to this error.",
        ),
        # ConvergenceError (decision D65): the root gates hold, but the covariance
        # requested at the fit gives a coefficient no usable standard error, so the
        # coefficient gate of D49 fails.  The status is that of the accepted status
        # composition, which the error supplies; no status is added.
        (
            "convergence.unusable_covariance",
            "The candidate meets the root gates, but the covariance requested at the "
            "fit gives {coefficients} no usable standard error: status {status!r}. "
            "Fit with a usable covariance, then request this one.",
        ),
        # SolverCapabilityError.
        (
            "solver.unsupported_case",
            "Solver {solver!r} does not cover {case}.",
        ),
        # CoverageWarning: outside the results proved in BBP.
        (
            "coverage.outside_bbp",
            "{case} is computed but lies outside the results proved in BBP.",
        ),
        # CoverageWarning: a formula term that Formulaic reads as another one.
        (
            "coverage.power_collapsed",
            "In the formula, {written} is read as {variable}: a power of one variable "
            "is its interaction with itself and adds no column. Write {fix} for the "
            "power.",
        ),
        # CoverageWarning: a regressor of the formula that the fit omits, one note per
        # regressor, as fixest ("The variable 'x' has been removed because of
        # collinearity") and ppmlhdfe ("note: x omitted because of collinearity") do.
        (
            "coverage.collinear_omitted",
            "The regressor {name!r} is omitted because of collinearity with the other "
            "regressors or the fixed effects on the estimation sample; it has no "
            "coefficient.",
        ),
        (
            "numerical.small_mean_order",
            "At omega={omega}, rounding in the power transformation can affect "
            "reported coefficients and standard errors even when the transformed "
            "fit has status {status!r}.",
        ),
        # NumericalWarning: the status verified_secondary of the certificate.
        (
            "numerical.secondary_convergence",
            "The solution meets the secondary tolerance tol_secondary={tol_secondary} "
            "but not tol={tol}: the certificate reports status {status!r}.",
        ),
        (
            "numerical.zero_row_false_alarm",
            "Low fitted means for zero outcomes triggered a separation check at "
            "original row positions {positions}. Fresh support checks confirmed "
            "overlap on the same sample, and the polished fit passed strict "
            "certification (status {status!r}).",
        ),
        # NumericalWarning: a merit metric that the caller stated, replaced at the zero
        # start of a cancelled root solve (solve_cancelled_root), once per fit.
        (
            "numerical.merit_metric_replaced",
            "The cancelled root solve started from zero, as the Gamma (tau = 2) fit "
            "certified no root on this sample, and used merit_metric='gram' in place "
            "of the requested {requested!r}: the Gamma metric there is the tau = 2 "
            "Jacobian at a point that equation did not certify, and whether its "
            "factor forms can be decided by rounding, hence by the unit of the "
            "outcome. The fit reports status {status!r}.",
        ),
        # NumericalWarning (owner's ruling 78): a certified root at which the
        # certificate found the curvature not positive definite, so that it read the
        # score in the solve's fixed merit metric; one note per fit, at the fit.
        (
            "numerical.indefinite_curvature",
            "The estimating equation may have other roots: its curvature at the "
            "certified root (status {status!r}) is not positive definite.",
        ),
        # CovarianceWarning (decision D65): the eigenvalue adjustment of decision D4
        # applied to a signed multiway estimate, and an indefinite covariance whose
        # dependent inference is withheld (decision D62).  No covariance issues them
        # yet.
        # ``reference``: the Stata command of the same estimator that applies the
        # same adjustment, ``ppmlhdfe``, or ``reghdfe`` for fixed-effect log-OLS
        # ("VCV matrix was non-positive semi-definite; adjustment from Cameron,
        # Gelbach & Miller applied").
        (
            "covariance.psd_adjusted",
            "The signed multiway covariance was not positive semidefinite; {count} "
            "{repair}, as {reference} does.",
        ),
        (
            "covariance.indefinite",
            "The {method} covariance is not positive semidefinite (smallest eigenvalue "
            "{minimum}); inference that needs a valid variance is withheld.",
        ),
        # Named HC covariances of joint fixed-effect fits (owner's rulings 63 and 65).
        # The second follows the warning of R sandwich 3.1-3, meatHC (R/vcovHC.R).
        (
            "covariance.singletons_excluded",
            "{count} singleton observation(s), each alone in a fixed-effect level "
            "(leverage one, zero score), are excluded from the {method} covariance; "
            "N = {kept}.",
        ),
        (
            "covariance.unstable_leverage",
            "{method} covariances are numerically unstable for hat values close to 1 "
            "(and undefined if exactly 1), as for {count} observation(s).",
        ),
        # A saturated fit, as many observations as parameters (owner's rulings 65 and
        # 74): every covariance of the fit is missing, with this one note.
        (
            "covariance.saturated",
            "The fit is saturated: its N = {count} observations equal its {rank} "
            "parameters, so no residual degrees of freedom remain and its standard "
            "errors are missing.",
        ),
    ]
)


def _digits(integer: int) -> str:
    """The sign and the decimal digits of an integer, however many digits it has.

    ``str`` refuses an integer with more digits than
    :func:`sys.get_int_max_str_digits` allows (4300 by default).  A
    :class:`~decimal.Decimal` constructed from an integer is exact whatever the
    precision of the context, and its exponent is 0, so its text is the plain
    digits; CPython's C implementation of :mod:`decimal` writes them without that
    limit.  Neither the limit nor the context is changed.
    """
    return str(Decimal(integer))


class _Formatter(Formatter):
    """The formatting of :meth:`str.format`, but ``!r`` writes an exact number in full.

    The conversion ``!r`` of a value whose type is exactly ``int`` or
    :class:`~fractions.Fraction` returns the text of :func:`repr`, written from
    :func:`_digits`, so that no number of digits makes it raise
    :class:`ValueError`.  A subclass, such as ``bool``, keeps its own
    representation, and any other value, conversion or format specification is
    left to :class:`string.Formatter`, which formats as :meth:`str.format` does.
    Nothing is caught: a container that holds such a number, or a type whose own
    representation fails, raises as it would with :meth:`str.format`.
    """

    def convert_field(self, value: object, conversion: str | None) -> object:
        """The value converted for its field, as :meth:`str.format` converts it."""
        if conversion == "r":
            if type(value) is int:
                return _digits(value)
            if type(value) is Fraction:
                numerator = _digits(value.numerator)
                denominator = _digits(value.denominator)
                return f"Fraction({numerator}, {denominator})"
        return super().convert_field(value, conversion)


_FORMATTER = _Formatter()


def render(identifier: str, fields: Mapping[str, object]) -> str:
    """The text of a message, from its identifier and exactly its fields.

    Raises :class:`KeyError` for an unknown identifier or a missing field, and
    :class:`TypeError` for a field that the template does not name.  The fields are
    formatted as :meth:`str.format` formats them, except that ``!r`` writes an exact
    integer or fraction in full, whatever its size (:class:`_Formatter`).
    """
    message = REGISTRY[identifier]
    expected = message.fields
    missing = [name for name in expected if name not in fields]
    if missing:
        msg = f"{identifier}: missing field(s) {format_keys(missing)}"
        raise KeyError(msg)
    unexpected = [name for name in fields if name not in expected]
    if unexpected:
        msg = f"{identifier}: unexpected field(s) {format_keys(unexpected)}"
        raise TypeError(msg)
    return _FORMATTER.vformat(message.template, (), fields)
