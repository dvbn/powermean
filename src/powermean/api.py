"""The public functions of the package.

``fit``, ``ppml``, ``gpml``, ``logols``, ``prepare`` and ``solver_defaults``, with
the public types ``NB2`` and ``SolverOptions``.  The estimator of a call is fixed
before anything is computed: the mean order ``omega`` and
the working variance (``tau`` in ``[1, 2]``, or ``variance=pm.NB2(kappa)``; without
fixed effects, at a nonzero ``omega`` and with the Newton solver, also any finite
``tau`` outside ``[1, 2]``), on the sample that the preparation constructs
once from the formula's first part, the weights, the offset or exposure, the subset
and the outcome domain.  The joint system is solved by the convex Newton solver from
the package's start, outside ``[1, 2]`` by root Newton from the certified nearest
convex fit of the same sample (PPML below 1, Gamma above 2) or the caller's start,
whose certified root is a selected local root and not claimed unique, or at ``omega
= 0`` by the least-squares solve of the log response, and the point is certified
afresh against the original equation before it is reported; only a certified point
carries inference.  The keywords of
each call are declared by a closed typed collector
(:mod:`powermean.interface.options`) and validated at the entry point: an unknown
keyword, a value outside its domain, two arguments that contradict each other and
the explicit request of a feature that is not available are refused with the
``SpecificationError`` and the identifier of the catalog
(:mod:`powermean.messages`).

``fit(formula, data, **options)`` prepares and fits in one call, its formula and its
data given positionally or by keyword (``data=df``);
``prepare(formula, data, **options)`` returns a :class:`Prepared` problem whose
``fit`` reuses the frozen sample across mean orders and working variances, on the
one outcome domain fixed at the preparation; ``ppml``, ``gpml`` and ``logols`` fix
``(omega, tau)`` at ``(1, 1)``, ``(1, 2)`` and ``omega = 0`` and refuse a
contradictory explicit value; ``solver_defaults`` returns the canonical typed record
of the options of the certificate and of the convex Newton solver.  ``NB2`` is the
NB2 working variance ``V(m) = m + kappa m^2`` with any ``kappa >= 0``, and
``SolverOptions`` the record that ``solver_defaults`` returns. The ordinary fit
returns a :class:`~powermean.results.result.Result`. Selecting
``estimator="gmm1"`` or ``"gmm2"`` fits the two-way difference equation and
returns a :class:`~powermean.results.jochmans.JochmansResult`.

Fixed effects: the formula and prepared-fit routes default to `absorbed="profiled"`.
Choose `"index"` explicitly in `solver_options` to use that organization. Joint
fits accept an optional `sample_request` record; cancelled fits accept an optional
`controls` record (CancelledFitControls). Preparation stays order neutral.
A joint fit with `tau` outside `[1, 2]` at a nonzero `omega` is a selected local
root: rank checks only on its sample (no existence analysis, a `CoverageWarning`
where the root is not characterized), the certified nearest convex fit of the same
sample by the requested organization (PPML below 1, Gamma above 2) or the caller's
start, then sparse full-joint root Newton; the existing nested `root_options`
record of `solver_options` holds its root-Newton options.
Joint fits admit expanded requested inference, including HC and multiway controls.
Cancelled fit-time inference is observed/Stata robust or one-way clustering.
Unsupported requests refuse before solving.  No default of a fit without fixed
effects is changed.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Unpack, cast, overload

import pandas as pd

from powermean import messages
from powermean.errors import CoverageWarning, SpecificationError
from powermean.formula.grammar import collapsed_powers, parse_formula
from powermean.interface.dispatch import fit_prepared
from powermean.interface.jochmans import fit_difference_gmm
from powermean.interface.options import (
    DEFAULT_SOLVER,
    FIT_KEYWORDS,
    PREPARE_KEYWORDS,
    PREPARED_FIT_KEYWORDS,
    DifferenceGMMOptions,
    FitOptions,
    PreparedFitOptions,
    PrepareOptions,
    collect,
    fixed_values,
    preparation_of,
    requests_cluster,
    requests_named_hc,
    requests_robust,
    specification_of,
)
from powermean.interface.preparation import PreparedProblem, prepare_problem
from powermean.model.variance import NB2Variance
from powermean.results.jochmans import JochmansResult
from powermean.results.result import Result
from powermean.solve.iols.options import IolsOptions
from powermean.solve.options import SolverOptions

NB2 = NB2Variance
# The frames between the dispatch and the caller of a public function, at which the
# secondary warning names the caller's line: the helper, the public function, the
# caller.
_CALLER_DEPTH = 3


def solver_defaults(solver: str = DEFAULT_SOLVER) -> SolverOptions | IolsOptions:
    """The canonical defaults of the options of ``solver``, as a typed record.

    Parameters
    ----------
    solver : {"newton", "iols"}, default "newton"
        The solver whose options are returned.

    Returns
    -------
    SolverOptions or IolsOptions
        The resolved record: the four
        thresholds of the certificate and the nine settings of the convex Newton
        solver, or the no-FE IOLS options, each at its default.
        A record given as ``solver_options`` is a
        complete snapshot: its values, the default ones included, are supplied.

    Raises
    ------
    SpecificationError
        Another name (``specification.invalid_value``).
    """
    if solver == "iols":
        return IolsOptions()
    if solver != DEFAULT_SOLVER:
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "solver",
                "value": solver,
                "valid": messages.format_keys([DEFAULT_SOLVER, "iols"]),
            },
        )
    return SolverOptions()


class Prepared:
    """A prepared problem: the sample and design built once, fitted at any order.

    Returned by :func:`prepare`; not constructed directly.  ``fit`` derives the
    sample of the requested mean order from the frozen base, on the outcome domain
    that the preparation fixed, and fits it under the keywords given; the sample
    keywords (``weights``, ``weight_type``, ``offset``, ``exposure``, ``subset``)
    belong to the preparation and are not keywords of ``fit``.
    """

    __slots__ = ("_problem",)

    def __init__(self, problem: PreparedProblem) -> None:
        """Hold the prepared problem."""
        if type(problem) is not PreparedProblem:
            raise TypeError("problem must be a PreparedProblem")
        self._problem = problem

    @property
    def positive_only(self) -> bool:
        """The outcome domain of every fit of the problem."""
        return self._problem.positive_only

    def fit(self, **options: Unpack[PreparedFitOptions]) -> Result:
        """Fit the prepared problem under the keywords given.

        Parameters
        ----------
        **options : PreparedFitOptions
            The estimation keywords of ``fit`` (:class:`~powermean.interface.
            options.PreparedFitOptions`): ``omega``, ``tau`` or ``variance``,
            ``vcov`` or ``cluster`` (``"robust"`` with ``ssc``, which also applies
            alone to a prepared robust request; every cluster dimension must have
            been retained by the preparation, with ``ssc``, which also applies alone
            to a prepared cluster request; a named HC method by its exact string,
            with ``ssc`` and ``bread``, which also apply alone to a prepared named
            method),
            ``solver``, ``solver_options``, ``tol``, ``start``, ``on_failure``,
            ``keep_influence``; an explicit ``positive_only`` must equal the
            prepared domain.

        Returns
        -------
        Result
            The result of the fit.

        Raises
        ------
        SpecificationError
            An unknown keyword, a value outside its domain, contradictory
            arguments, or a feature that is not available.
        SampleError
            An order at or below zero on an outcome with zeros, without
            ``positive_only`` at the preparation.
        ConvergenceError
            No certified solution, with ``on_failure="raise"``.
        """
        return _fit_problem(
            self._problem, options, "keyword of Prepared.fit", _CALLER_DEPTH
        )


def _has_fixed_effects(formula: object) -> bool:
    """Whether ``formula`` is a valid formula string with a fixed-effects part."""
    if not isinstance(formula, str):
        return False
    try:
        return bool(parse_formula(formula).fixed_effects)
    except SpecificationError:
        return False


def _note_collapsed_powers(formula: str, depth: int) -> None:
    """Warn, at the caller ``depth`` frames up, of each power that adds no column."""
    try:
        regressors = parse_formula(formula).regressors
    except (SpecificationError, TypeError, AttributeError):
        return  # a formula that cannot be read is refused where it is prepared
    for written, variable, fix in collapsed_powers(regressors):
        fields = {"written": written, "variable": variable, "fix": fix}
        warnings.warn(
            CoverageWarning("coverage.power_collapsed", fields), stacklevel=depth + 1
        )


def _fit_problem(
    problem: PreparedProblem,
    options: Mapping[str, object],
    kind: str,
    stacklevel: int,
) -> Result:
    """Normalize the fit keywords of a prepared problem and dispatch the fit."""
    has_fe = problem.has_fixed_effects
    collected = collect(
        options,
        kind,
        PREPARED_FIT_KEYWORDS,
        has_fixed_effects=has_fe,
        named_hc=requests_named_hc(options, problem.inference),
        clustered=requests_cluster(options, problem.inference),
        robust=requests_robust(options, problem.inference),
    )
    specification = specification_of(
        collected,
        problem.inference,
        has_fixed_effects=has_fe,
        prepared_fe=problem.fe_defaults if has_fe else None,
        prepared_bias=problem.bias_correction,
    )
    return fit_prepared(problem, specification, stacklevel=stacklevel)


def prepare(
    formula: str, data: pd.DataFrame, **options: Unpack[PrepareOptions]
) -> Prepared:
    """Build the sample and the design once, for fits at any order and variance.

    Parameters
    ----------
    formula : str
        ``outcome ~ regressors``, in Formulaic syntax.  Positional or by keyword.
    data : pandas.DataFrame
        The data, read by position and never modified.  Positional or by keyword.
    **options : PrepareOptions
        The sample keywords (:class:`~powermean.interface.options.PrepareOptions`):
        ``positive_only``, ``weights``, ``weight_type``, ``offset`` or ``exposure``
        (a column or an expression), ``subset``, and the default ``vcov`` or
        ``cluster`` of the fits, whose cluster dimensions (``"a + b"`` for several,
        with the keys ``"multiway"`` and ``"psd"`` and the keyword ``ssc``) are
        retained with the sample.

    Returns
    -------
    Prepared
        The prepared problem, whose ``fit`` reuses the frozen sample.

    Raises
    ------
    SpecificationError
        An unknown keyword, a value outside its domain, contradictory arguments, a
        malformed formula, or a feature that is not available.
    SampleError
        A finite negative outcome among complete rows or no usable observation,
        including an empty frame. Non-finite outcomes are excluded as missing.
    """
    _note_collapsed_powers(formula, 2)
    has_fe = _has_fixed_effects(formula)
    collected = collect(
        options,
        "keyword of prepare",
        PREPARE_KEYWORDS,
        has_fixed_effects=has_fe,
        named_hc=requests_named_hc(options),
        clustered=requests_cluster(options),
        robust=requests_robust(options),
    )
    return Prepared(
        prepare_problem(
            formula, data, preparation_of(collected, has_fixed_effects=has_fe)
        )
    )


def _split(
    options: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """The keywords of ``fit`` by stage: those of ``prepare`` and those of the fit."""
    preparing = {k: v for k, v in options.items() if k in PREPARE_KEYWORDS}
    fitting = {k: v for k, v in options.items() if k in PREPARED_FIT_KEYWORDS}
    if "cancel" in options:
        preparing["cancel"] = options["cancel"]
    return preparing, fitting


def _fit(
    formula: str,
    data: pd.DataFrame,
    options: Mapping[str, object],
    fixed: Mapping[str, object],
    stacklevel: int,
) -> Result:
    """Prepare once and fit once: ``fit`` and the three convenience wrappers."""
    _note_collapsed_powers(formula, stacklevel)
    has_fe = _has_fixed_effects(formula)
    collected = collect(
        options,
        "keyword of fit",
        FIT_KEYWORDS,
        has_fixed_effects=has_fe,
        named_hc=requests_named_hc(options),
        clustered=requests_cluster(options),
        robust=requests_robust(options),
    )
    fixed_values(collected, fixed)
    preparing, fitting = _split(collected)
    problem = prepare_problem(
        formula,
        data,
        preparation_of(preparing, has_fixed_effects=has_fe),
        solver="iols" if fitting.get("solver") == "iols" else "newton",
        _single_fit=True,
    )
    specification = specification_of(
        {**fitting, **fixed},
        problem.inference,
        has_fixed_effects=has_fe,
        prepared_fe=problem.fe_defaults if has_fe else None,
        prepared_bias=problem.bias_correction,
    )
    return fit_prepared(problem, specification, stacklevel=stacklevel)


@overload
def fit(formula: str, data: pd.DataFrame, **options: Unpack[FitOptions]) -> Result: ...


@overload
def fit(
    formula: str, data: pd.DataFrame, **options: Unpack[DifferenceGMMOptions]
) -> JochmansResult: ...


def fit(formula: str, data: pd.DataFrame, **options: object) -> Result | JochmansResult:
    """Power-mean regression, with or without fixed effects.

    Parameters
    ----------
    formula : str
        ``outcome ~ regressors``, in Formulaic syntax; the outcome untransformed.
        Positional or by keyword.
    data : pandas.DataFrame
        The data, read by position and never modified.  Positional or by keyword
        (``data=df``).
    **options : FitOptions
        The keywords of :class:`~powermean.interface.options.FitOptions`: ``omega``
        (1 by default; 0 is log OLS), ``tau`` (in ``[1, 2]``; any finite value for a
        no-FE Newton fit at a nonzero ``omega``) or ``variance=pm.NB2(
        kappa)``, ``positive_only``, ``weights`` and ``weight_type``, ``offset`` or
        ``exposure``, ``subset``, ``vcov`` or ``cluster`` (without fixed effects,
        ``"robust"``, the default, with the keyword ``ssc``, several dimensions as
        ``{"cluster": "a + b"}`` with ``"multiway"`` and ``"psd"`` and the keyword
        ``ssc``, or a HC method by its exact string, ``"HC0"`` to ``"HC5"`` or
        ``"HC4m"``, with its ``ssc`` and ``bread``). Joint fixed-effect fits also
        accept these named HC methods. Their leverage calculation includes the fixed
        effects. Other options include ``solver``, ``solver_options``, ``tol``,
        ``start``, ``on_failure`` and ``keep_influence``.

        Set ``estimator="gmm1"`` or ``"gmm2"`` for the Jochmans exogenous
        two-way exponential regression. Use an intercept-free formula with
        two fixed-effect identifiers, such as ``"patents ~ 0 + log_rd | firm + year"``.
        The default ``fe_layout="rectangular"`` requires one row per cell of a
        complete crossing (Stata's ``twexp``). Set ``fe_layout="directed"`` for
        all ordered non-self links among at least four agents
        (Stata's ``twgravity``). Observed zeros are retained. These estimators
        accept ``vcov="paper"`` (default) or ``"stata"``, ``start`` in coefficient
        order (default zero), and ``max_iter`` (default 100).

    Returns
    -------
    Result or JochmansResult
        An ordinary fit supplies ``coef``, ``se``, ``vcov``, ``confint``,
        ``tidy``, ``summary``, ``predict`` and ``influence``, and the attributes
        ``certificate``, ``sample``, ``estimator`` and ``provenance``.
        A two-way difference fit reports slopes and full cell-projection
        inference with its retained sample and local root certificate.

    Raises
    ------
    SpecificationError
        An unknown keyword, a value outside its domain, contradictory arguments, a
        malformed formula, or a feature that is not available.
    SampleError
        A finite negative outcome among complete rows, zeros at ``omega <= 0``
        without ``positive_only=True``, or no usable observation, including an
        empty frame. A two-way difference fit also refuses incomplete arrays,
        including missing or non-finite model values.
    IdentificationError
        A two-way difference fit has insufficient identifying contrasts,
        outcome products or resolved Jacobian rank.
    ConvergenceError
        No certified solution, with ``on_failure="raise"``: the last state and the
        certificate, where one was formed, are attached.
    """
    if "estimator" in options:
        return fit_difference_gmm(
            formula, data, **cast("DifferenceGMMOptions", options)
        )
    return _fit(formula, data, options, {}, _CALLER_DEPTH)


def ppml(formula: str, data: pd.DataFrame, **options: Unpack[FitOptions]) -> Result:
    """Poisson pseudo-maximum likelihood: ``fit`` with ``omega = 1`` and ``tau = 1``.

    The keywords are those of :func:`fit`; an explicit ``omega`` or ``tau`` must
    equal the fixed value, and ``variance`` is refused (``specification.conflicting``).
    """
    return _fit(formula, data, options, {"omega": 1.0, "tau": 1.0}, _CALLER_DEPTH)


def gpml(formula: str, data: pd.DataFrame, **options: Unpack[FitOptions]) -> Result:
    """Gamma pseudo-maximum likelihood: ``fit`` with ``omega = 1`` and ``tau = 2``.

    The keywords are those of :func:`fit`; an explicit ``omega`` or ``tau`` must
    equal the fixed value, and ``variance`` is refused (``specification.conflicting``).
    """
    return _fit(formula, data, options, {"omega": 1.0, "tau": 2.0}, _CALLER_DEPTH)


def logols(formula: str, data: pd.DataFrame, **options: Unpack[FitOptions]) -> Result:
    """Log OLS: ``fit`` with ``omega = 0``, least squares of ``log Y`` on regressors.

    The keywords are those of :func:`fit`; an explicit ``omega`` must be zero
    (``specification.conflicting`` otherwise), and ``positive_only=True`` is needed
    where the outcome has zeros.
    """
    return _fit(formula, data, options, {"omega": 0.0}, _CALLER_DEPTH)
