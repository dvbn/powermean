"""The printed summary of a result, for a no-FE fit.

The layout, without fixed-effect and bias-correction lines: the estimator and the
package version, the outcome, the mean order and the working variance, the observations
with the accounting of the sample, the zeros, the weights, the standard errors with
their convention and counts, the certificate, the coverage label of the theory, the
table of the coefficients under the normal reference, then the notes and the references
that every result holds (:mod:`powermean.results.citations`).

A result whose inference is withheld prints the point with its status and no standard
error, so that a failed or supplied uncertified point is visibly not certified; a result
without a point (a least-squares solve that formed none, an automatic start without a
finite value) prints its status and the reason, with no coefficients; a certified log
OLS fit prints its fresh linear certificate and the least-squares solve that reached the
point; degenerate inference at an exact root prints its zero standard errors and
undefined ratios as such.  The standard errors are those of the active covariance: the
one requested at the fit, or a view's (``with_vcov``); a named HC method prints its
method, bread, leverage geometry, factor with its owner, convention in force with its
source and reference; a multiway cluster covariance prints its dimensions, its form, the
cluster count of each dimension, its factor with its owner, its convention with its
source and reference, and the ``psd`` treatment (whether the adjustment was active, the
matrix reported being then the adjusted one); and a ``t`` reference labels the table's
ratio and p-value.  The summary emits no warning: the secondary convergence warning
belongs to the fit, and a summary is a retrieval.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from powermean.results import naming
from powermean.results.provenance import Provenance
from powermean.results.retained import CovarianceRecord, RetainedFit

WIDTH = 84
RULE = "-" * WIDTH


def _number(value: float, digits: int = 4) -> str:
    """A number for the table: fixed digits, ``nan`` where undefined."""
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def _scientific(value: float) -> str:
    """A diagnostic of the certificate: two significant digits, or ``nan``."""
    return "nan" if math.isnan(value) else f"{value:.1e}"


def _header(
    retained: RetainedFit, provenance: Provenance, record: CovarianceRecord | None
) -> list[str]:
    """The lines before the table: estimator, sample, weights, errors, certificate."""
    estimator, sample = retained.estimator, retained.sample
    version = str(provenance.versions["powermean"])
    title = "Power-mean regression"
    lines = [f"{title:<{WIDTH - len(version) - 10}}powermean {version}"]
    lines.append(f"Estimator: {naming.estimator_name(estimator)}")
    omega = estimator.mean.omega
    lines.append(
        f"Outcome: {retained.parts.outcome}    Mean order: omega = "
        f"{naming.order_text(omega)} ({naming.mean_label(omega)})    "
        f"Working variance: {naming.variance_text(estimator)}"
    )
    counts = sample.counts
    excluded = ", ".join(
        f"{reason.replace('_', ' ')} {count}"
        for reason, count in counts.items()
        if reason != "retained" and count
    )
    accounting = excluded if excluded else "none dropped"
    lines.append(
        f"Observations: {sample.count} of {sample.nobs} input rows ({accounting})"
    )
    zeros = int(np.count_nonzero(sample.data.y == 0))
    zero_line = f"Zeros: {zeros} of {sample.count} ({100 * zeros / sample.count:.1f}%)"
    if sample.conditioned:
        zero_line += "; conditioned on positive outcomes (zero outcomes removed)"
    lines.append(zero_line)
    lines.append(_weights_line(retained))
    lines.append(_errors_line(retained, record))
    lines.append(_certificate_line(retained))
    label = naming.coverage(estimator)
    lines.append(f"Coverage: {label.label} ({label.basis})")
    return lines


def _weights_line(retained: RetainedFit) -> str:
    """The weights of the fit and the counts that they imply."""
    sample, columns = retained.sample, retained.columns
    weights = sample.data.weights
    if weights is None:
        return "Weights: none"
    line = f"Weights: {weights.kind}, column {columns.weights!r}"
    if weights.frequency:
        line += f" (N = {sample.effective_n} observations in {sample.count} rows)"
    return line


def _named_line(record: CovarianceRecord) -> str:
    """The method, bread, leverage, factor, convention and reference of a named HC."""
    applied = record.applied
    if applied is None or applied.factor is None:
        raise ValueError("a named HC covariance records its applied convention")
    reference = applied.reference
    referenced = (
        "normal reference"
        if reference.kind == "normal"
        else f"t reference with {reference.df} degrees of freedom"
    )
    resolved = applied.convention
    return (
        f"Std. errors: {record.method} ({record.bread} bread, {record.hat_geometry} "
        f"information leverage), factor {applied.factor.formula} owned by the "
        f"method with N = {record.count}; convention {resolved.label} "
        f"({resolved.source}); {referenced}"
    )


def _referenced(record: CovarianceRecord) -> str:
    """The reference distribution of a record, in words."""
    reference = record.distribution
    if reference.kind == "normal":
        return "normal reference"
    return f"t reference with {reference.df} degrees of freedom"


def _multiway_line(record: CovarianceRecord) -> str | None:
    """The line of a multiway cluster covariance, or ``None`` for another record."""
    answer, applied = record.answer, record.applied
    if answer is None or answer.multiway is None or applied is None:
        return None
    multiway = answer.multiway
    form = "signed multiway (CGM)" if multiway.form == "cgm" else "sum of one-way (DDG)"
    counts = ", ".join(str(count) for count in applied.counts.clusters)
    factor = applied.factor
    formula = (
        f"{factor.formula} with G_min = {applied.counts.g_min}"
        if factor is not None
        else f"{applied.convention.values.cluster} with each component's own G"
    )
    line = (
        f"Std. errors: {form} clustered by {record.cluster!r} ({counts} clusters), "
        f"factor {formula} owned by the convention; convention "
        f"{applied.convention.label} ({applied.convention.source}); "
        f"{_referenced(record)}"
    )
    if answer.psd == "cgm":
        state = "active" if answer.adjusted else "inactive"
        line += f"; eigenvalue adjustment {state}"
    elif answer.psd == "none":
        line += "; no eigenvalue adjustment (psd 'none')"
    return line


def _errors_line(retained: RetainedFit, record: CovarianceRecord | None) -> str:
    """The convention of the active covariance, or why none is reported."""
    default = retained.default if record is None else record
    if retained.estimator.route == "linear" and retained.status is None:
        return "Std. errors: not available before the certificate of the linear path"
    if not retained.certified or default is None:
        return "Std. errors: withheld, the solution is not certified"
    multiway = _multiway_line(default)
    if multiway is not None:
        line = multiway
    elif default.method is not None:
        line = _named_line(default)
    elif default.applied is not None:
        applied = default.applied
        factor = applied.factor
        if factor is None:
            raise ValueError("a conventional covariance records its factor")
        reference = applied.reference
        referenced = (
            "normal reference"
            if reference.kind == "normal"
            else f"t reference with {reference.df} degrees of freedom"
        )
        line = (
            f"Std. errors: {default.kind} ({default.bread} bread), factor "
            f"{factor.formula} owned by the convention; convention "
            f"{applied.convention.label} ({applied.convention.source}); {referenced}"
        )
    elif default.kind == "cluster":
        line = (
            f"Std. errors: clustered by {default.cluster!r} ({default.clusters} "
            f"clusters), factor G/(G-1); normal reference"
        )
    else:
        line = (
            f"Std. errors: robust (observed bread), factor N/(N-1) with "
            f"N = {default.count}; normal reference"
        )
    if default.diagnostics:
        line += f"; covariance diagnostics: {', '.join(default.diagnostics)}"
    if default.degenerate:
        line += "; degenerate inference at an exact root"
    return line


def _no_point_line(retained: RetainedFit) -> str:
    """Why no certificate exists: no point, or a bare linear candidate."""
    failure, linear = retained.failure, retained.linear
    if failure is not None:
        return (
            f"Certified solution: no (status {retained.status!r}; no candidate: "
            f"{failure.message})"
        )
    if linear is not None and retained.status is not None:
        detail = linear.reason.replace("_", " ")
        if linear.message:
            detail += f", {linear.message}"
        return (
            f"Certified solution: no (status {retained.status!r}; no point of the "
            f"least-squares solve: {detail})"
        )
    return (
        "Certified solution: not available for the linear path in this version"
    )


def _solve_text(retained: RetainedFit) -> str:
    """How the point was reached: the Newton trace, the least-squares solve, neither."""
    candidate, linear = retained.candidate, retained.linear
    if candidate is not None:
        count = candidate.iterations
        return f"{count} iteration{'s' if count != 1 else ''}, {retained.reason}"
    if linear is not None and linear.factorization is not None:
        factorization = linear.factorization
        return (
            f"least-squares solve ({factorization.driver}), rank "
            f"{factorization.rank} of {linear.dimension}"
        )
    return "supplied point, no solve"


def _certificate_line(retained: RetainedFit) -> str:
    """The certificate: its level, its quantities and the trace of the solve."""
    certificate = retained.certificate
    if certificate is None:
        return _no_point_line(retained)
    verdict = "yes" if certificate.accepted else "no"
    level = certificate.level
    pieces = [
        f"status {retained.status!r}" if level is None else f"{level} level",
        f"scaled score {_scientific(certificate.scaled_score)}",
        f"index correction {_scientific(certificate.index_correction)}",
    ]
    coefficients = certificate.coefficients
    if coefficients is not None:
        pieces.append(
            f"largest correction/SE {_scientific(coefficients.largest_ratio)}"
        )
    if not certificate.accepted:
        failed = ", ".join(certificate.strict)
        pieces.append(f"failed gates: {failed}")
    pieces.append(_solve_text(retained))
    return f"Certified solution: {verdict} ({'; '.join(pieces)})"


def _table(table: pd.DataFrame, level: float, *, statistic: str = "z") -> list[str]:
    """The coefficient table: the ratio ``z`` of the normal reference, or ``t``."""
    percent = f"{100 * level:g}% conf. interval"
    names = [str(term) for term in table["term"]]
    width = max(len(name) for name in names) + 2
    head = (
        f"{'':<{width}}{'coef.':>12}{'std. err.':>12}{statistic:>9}"
        f"{'P>|' + statistic + '|':>9}{'[' + percent + ']':>24}"
    )
    lines = [RULE, head, RULE]
    columns = ["estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"]
    values = table[columns].to_numpy(dtype=np.float64)
    for name, row in zip(names, values.tolist(), strict=True):
        estimate, error, ratio, p_value, low, high = row
        lines.append(
            f"{name:<{width}}{_number(estimate):>12}{_number(error):>12}"
            f"{_number(ratio, 2):>9}{_number(p_value, 3):>9}"
            f"{_number(low):>12}{_number(high):>12}"
        )
    lines.append(RULE)
    return lines


def _point(beta: pd.Series[float] | None) -> list[str]:
    """The coefficients alone, where inference is withheld."""
    lines = [RULE, f"{'':<12}{'coef.':>12}    (inference withheld)", RULE]
    if beta is None:
        lines.append("no coefficients were formed")
    else:
        for name, value in zip(beta.index.tolist(), beta.tolist(), strict=True):
            lines.append(f"{name!s:<12}{_number(float(value)):>12}")
    lines.append(RULE)
    return lines


def render(
    retained: RetainedFit,
    provenance: Provenance,
    *,
    table: pd.DataFrame | None,
    beta: pd.Series[float] | None,
    level: float,
    record: CovarianceRecord | None = None,
) -> str:
    """The summary text: the header, the table or the point, the references.

    ``record`` is the active covariance of the table; ``None`` reads the fit's own.
    """
    lines = _header(retained, provenance, record)
    if table is not None:
        active = retained.default if record is None else record
        statistic = "z"
        if active is not None and active.distribution.kind == "t":
            statistic = "t"
        lines.extend(_table(table, level, statistic=statistic))
    else:
        lines.extend(_point(beta))
    if retained.sample.conditioned:
        lines.append(
            "Note: the fit conditions on positive outcomes; dropping zeros does not "
            "resolve the composition problem (BBP, subsec:targetparm)."
        )
    lines.append("References:")
    lines.extend(f"  {reference.text()}" for reference in provenance.references)
    return "\n".join(lines)
