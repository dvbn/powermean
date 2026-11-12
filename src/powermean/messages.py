"""Every text shown to the user, with an identifier and one template.

Each message of the package is a :class:`Message`: an identifier such as
``"specification.unknown_option"``, the name of the exception or warning class
that carries it, and one template in the syntax of :meth:`str.format` whose fields
are plain names.  :data:`REGISTRY` maps identifiers to messages, and :func:`render`
formats one deterministically.  :mod:`powermean.errors` consumes this module;
nothing here imports beyond the standard library.

Contract:

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
  attribute, so that text and payload
  cannot disagree.

The catalog covers the refusals of features that are not available, the sample
rules of steps 3 and 4 of the sample construction, the unknown-option error and the
statuses.  The identification and existence messages exist with their classes;
their dispatch is the existence analysis and the collinearity resolution.

The expanded covariance menu has seventeen templates: two notes carried
by ``CovarianceWarning`` (prefix ``covariance``), fourteen refusals carried by
``SpecificationError``, and ``convergence.unusable_covariance``, whose ``status`` the
``ConvergenceError`` that carries it fills.  The
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
        # SpecificationError: inconsistent arguments, and the refusals of features
        # that are not available.
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
        # SpecificationError: the refusals of a later request on a result:
        # a cluster variable that the fit did not retain, one with a missing
        # value on the estimation sample (no row is dropped after the fit), and a
        # request that needs the influence contributions discarded by
        # keep_influence=False, which requires a refit.
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
        # SpecificationError: the refusals of the expanded covariance menu:
        # dependence metadata named after the fit, the time of a HAC request, its
        # replicated rows and its lag, the residual degrees of freedom of HC1, unit
        # leverage, the coordinates, locations and pooling of a Conley request, and a
        # method that a case does not support.
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
        # IdentificationError: dispatched by the collinearity resolution.
        (
            "identification.rank_deficient",
            "The design has rank {rank} for {columns} columns, and the deficiency "
            "cannot be resolved by dropping collinear columns.",
        ),
        # ExistenceError: dispatched by the existence analysis.
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
        # ConvergenceError: the root gates hold, but the covariance
        # requested at the fit gives a coefficient no usable standard error, so the
        # coefficient gate fails.  The status is that of the status
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
        # NumericalWarning: the status verified_secondary of the certificate.
        (
            "numerical.secondary_convergence",
            "The solution meets the secondary tolerance tol_secondary={tol_secondary} "
            "but not tol={tol}: the certificate reports status {status!r}.",
        ),
        # CovarianceWarning: the eigenvalue adjustment
        # applied to a signed multiway estimate, and an indefinite covariance whose
        # dependent inference is withheld.
        (
            "covariance.psd_adjusted",
            "The signed multiway covariance was not positive semidefinite; {count} "
            "negative eigenvalue(s) were set to zero, as ppmlhdfe does.",
        ),
        (
            "covariance.indefinite",
            "The {method} covariance is not positive semidefinite (smallest eigenvalue "
            "{minimum}); inference that needs a valid variance is withheld.",
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
