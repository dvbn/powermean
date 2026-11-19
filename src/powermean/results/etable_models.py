"""What ``etable`` reads from one fitted result, through its public interface.

The table uses public result projections and provenance:

==================  ================================================================
``coef()``          reported points without inference, and authoritative term order
``tidy(level)``     term, estimate, standard error, p-value, interval and the status
                    of the inference; ``attrs["coefficient_point"]`` where the
                    reported coefficients are corrected
``effects(level=)``  the same rows as ``exp(b) - 1``, with the delta-method standard
                    error and the transformed interval
``sample``          ``count`` (rows of the estimation sample) and ``zeros``
``provenance``      ``certified`` and ``status``, and the mappings ``estimator``
                    (mean order, working variance, fixed-effect terms and the
                    cancelled one) and ``covariance``
``with_vcov(spec)``  only where ``etable`` is given ``vcov=``: the view of the result
                    under that covariance, which the result computes from what it
                    retains, or refuses
==================  ================================================================

Every coefficient, standard error, p-value and interval of the table is the one that
these methods return; nothing is estimated, corrected or tested again here, and no
private attribute is read.

Two cases print a status in place of numbers, so that a table of several results is
not lost to one of them:

- A result that is not certified (``provenance.certified`` is false) carries no
  inference: the package raises ``convergence.not_certified`` from ``tidy``.  Its
  column prints no coefficient, and the row "Status" reads ``not certified`` with
  the status of the certificate.
- A covariance request refused with a recognized inference ``SpecificationError`` (for
  example ``specification.unsupported_inference`` for a cancelled fit under a
  multiway request) leaves the coefficients of that column, which do not depend on
  the covariance, without standard errors, intervals, p-values or stars, and the row
  "Covariance" reads ``refused`` with the identifier of the refusal.

The covariance record of a result without fixed effects names its kind under
``"kind"`` and its cluster count under ``"clusters"`` (``"cluster_counts"`` for
several dimensions); that of a fixed-effects result under ``"method"`` and ``"G"``.
Both spellings are read.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import pandas as pd

from powermean.errors import SpecificationError
from powermean.results.etable_options import EtableOptions, refuse

COLUMNS = ["estimate", "std_error", "p_value", "conf_low", "conf_high"]
VERIFIED = "verified"


class _Sample(Protocol):
    """The two counts that ``etable`` reads from the sample view of a result."""

    @property
    def count(self) -> int:
        """The number of rows of the estimation sample."""
        ...

    @property
    def zeros(self) -> int:
        """The number of zero outcomes among them."""
        ...


class _Provenance(Protocol):
    """The two records that ``etable`` reads from the provenance of a result."""

    @property
    def estimator(self) -> Mapping[str, object]:
        """The estimator: mean order, working variance, fixed-effect formulation."""
        ...

    @property
    def covariance(self) -> Mapping[str, object] | None:
        """The active covariance: its kind, clusters and convention."""
        ...

    @property
    def certified(self) -> bool:
        """Whether the point is certified, so that it carries inference."""
        ...

    @property
    def status(self) -> str | None:
        """The status of the certificate."""
        ...


class FittedResult(Protocol):
    """The public interface of a result that ``etable`` reads."""

    @property
    def sample(self) -> _Sample:
        """The sample view."""
        ...

    @property
    def provenance(self) -> _Provenance:
        """What ran."""
        ...

    def tidy(self, level: float = 0.95) -> pd.DataFrame:
        """One row per coefficient."""
        ...

    def coef(self) -> pd.Series:
        """The reported points and their order, without reading inference."""
        ...

    def effects(self, *, level: float = 0.95) -> pd.DataFrame:
        """One row per transformed coefficient coordinate."""
        ...

    def with_vcov(self, spec: object) -> FittedResult:
        """The view of the result under another covariance, without a new fit."""
        ...


@dataclass(frozen=True)
class Model:
    """What one column prints, read from one result.

    ``terms`` is indexed by term with the columns ``estimate``, ``std_error``,
    ``p_value``, ``conf_low`` and ``conf_high`` on the requested scale; it is empty
    for a result that is not certified.  ``status`` is the status of the
    certificate, prefixed by ``not certified`` where that is the case.
    """

    terms: pd.DataFrame
    estimator: str
    omega: float
    effects: Mapping[str, str]
    correction: str
    covariance: str
    clusters: str
    status: str
    nobs: int
    zeros: int
    convention: str
    reference: str
    projection: Mapping[str, object]


def index_text(value: object) -> str:
    """An index of the estimator as printed: an integer without a fraction."""
    number = float(cast("float", value))
    return str(int(number)) if number == int(number) else repr(number)


def _estimator(record: Mapping[str, object]) -> tuple[str, float]:
    """The short name of the estimator and its mean order."""
    omega = float(cast("float", cast("Mapping[str, object]", record["mean"])["omega"]))
    variance = cast("Mapping[str, object]", record["variance"])
    if omega == 0:
        return "log OLS", omega
    if "kappa" in variance:
        return f"NB2, kappa = {index_text(variance['kappa'])}", omega
    tau = float(cast("float", variance["tau"]))
    if omega == 1 and tau in (1.0, 2.0):
        return ("PPML" if tau == 1.0 else "GPML"), omega
    return f"tau = {index_text(tau)}", omega


def _effects(record: Mapping[str, object]) -> dict[str, str]:
    """``{term: "Estimated" | "Cancelled"}`` for the fixed-effect terms of a result."""
    formulation = cast("Mapping[str, object]", record["fixed_effects"])
    blocks = cast("Sequence[str]", formulation.get("blocks", ()))
    cancelled = formulation.get("cancelled")
    return {b: "Cancelled" if b == cancelled else "Estimated" for b in blocks}


def _covariance(record: Mapping[str, object]) -> tuple[str, str]:
    """The printed covariance and its cluster counts.

    The record of a result without fixed effects names its kind under ``"kind"``
    and its clusters under ``"clusters"``; that of a fixed-effects result under
    ``"method"`` and ``"G"``.  Both are read.
    """
    cluster = record.get("cluster")
    method = record.get("method")
    if cluster:
        counts = (
            record.get("cluster_counts") or record.get("clusters") or record.get("G")
        )
        if isinstance(counts, Sequence) and not isinstance(counts, str):
            printed = ", ".join(f"{int(count):,d}" for count in counts)
        else:
            printed = "" if counts is None else f"{int(cast('int', counts)):,d}"
        return f"cluster: {cluster}", printed
    if isinstance(method, str) and method.upper().startswith("HC"):
        return method, ""
    return str(record.get("kind") or method or "robust"), ""


def _detached(value: object) -> object:
    """Copy nested read-only provenance mappings into plain owned metadata."""
    if isinstance(value, Mapping):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detached(item) for item in value)
    if isinstance(value, list):
        return [_detached(item) for item in value]
    return deepcopy(value)


def _correction(point: object, provenance: _Provenance) -> str:
    """The correction attached to the reported point, even without inference."""
    text = str(point)
    if " corrected" in text:
        return text.split(" corrected", maxsplit=1)[0]
    options = getattr(provenance, "options", {})
    bias = options.get("bias_correction", {})
    if isinstance(bias, Mapping) and bias.get("status") == "applied":
        return "WZ" if bias.get("method") == "wz" else "BBP"
    return ""


def _terms(result: FittedResult, options: EtableOptions) -> tuple[pd.DataFrame, str]:
    """Read one projection; scale quantities with units and align status by term."""
    if options.scale == "coefficient":
        projection = result.tidy(options.level)
        order = None
    else:
        coef = getattr(result, "coef", None)
        order = list(coef().index) if callable(coef) else None
        if order == []:
            projection = pd.DataFrame(columns=["term", *COLUMNS, "inference"])
        else:
            projection = result.effects(level=options.level)
    terms = projection.set_index("term")[COLUMNS].astype("float64")
    if order is not None:
        terms = terms.reindex(order)
    status = projection.set_index("term")["inference"].reindex(terms.index)
    if options.scale == "effect":
        terms[["estimate", "std_error", "conf_low", "conf_high"]] *= 100.0
    terms["inference"] = status
    terms.loc[status != "ok", COLUMNS[2:]] = np.nan
    terms.loc[~status.isin(["ok", "degenerate"]), "std_error"] = np.nan
    terms.attrs = cast("dict[Hashable | None, object]", _detached(projection.attrs))
    correction = _correction(
        projection.attrs.get("coefficient_point", ""), result.provenance
    )
    return terms, correction


def _refused_terms(result: FittedResult, options: EtableOptions) -> pd.DataFrame:
    """A refused covariance never reads the old covariance's inference."""
    coef = getattr(result, "coef", None)
    if not callable(coef):
        raise refuse(
            "results",
            type(result).__name__,
            "a fitted result with reported coefficients",
        )
    read_coef = cast("Callable[[], pd.Series[float]]", coef)
    terms = read_coef().rename("estimate").to_frame()
    terms.index.name = "term"
    if options.scale == "effect":
        terms["estimate"] = 100.0 * np.expm1(terms["estimate"])
    terms[COLUMNS[1:]] = np.nan
    terms["inference"] = "refused"
    return terms


def _inference_labels(record: Mapping[Hashable | None, object]) -> tuple[str, str]:
    """Display only the actual exposed convention and reference."""
    convention = str(record.get("convention") or "unavailable")
    reference = str(record.get("reference") or "unavailable")
    df = record.get("reference_df")
    if reference == "t" and df is not None:
        reference += f" (df = {index_text(df)})"
    return convention, reference


_REFUSALS = {
    "specification." + name
    for name in (
        "unsupported_inference",
        "influence_discarded",
        "cluster_not_retained",
        "cluster_missing_values",
        "no_residual_dof",
        "unit_leverage",
    )
}


def model_of(result: FittedResult, options: EtableOptions) -> Model:
    """Read one result through its public interface (module docstring)."""
    provenance = result.provenance
    estimator, omega = _estimator(provenance.estimator)
    status = str(provenance.status)
    terms = pd.DataFrame(
        columns=COLUMNS, index=pd.Index([], name="term"), dtype="float64"
    )
    correction = covariance = clusters = ""
    convention = reference = "unavailable"
    projection: dict[str, object] = {}
    if not provenance.certified:
        status = f"not certified ({status})"
    else:
        viewed, refusal = result, ""
        if options.vcov is not None:
            try:
                viewed = result.with_vcov(options.vcov)
            except SpecificationError as error:
                if error.identifier not in _REFUSALS:
                    raise
                refusal = error.identifier
        if refusal:
            terms = _refused_terms(result, options)
            correction = _correction("", provenance)
            covariance = f"refused ({refusal})"
        else:
            terms, correction = _terms(viewed, options)
            record = dict(viewed.provenance.covariance or {})
            covariance, clusters = _covariance(record)
            projection = cast("dict[str, object]", _detached(terms.attrs))
            projection["covariance"] = _detached(record)
            attrs = cast("Mapping[Hashable | None, object]", record)
            convention, reference = _inference_labels({**attrs, **terms.attrs})
    return Model(
        terms=terms,
        estimator=estimator,
        omega=omega,
        effects=_effects(provenance.estimator),
        correction=correction,
        covariance=covariance,
        clusters=clusters,
        status=status,
        nobs=int(result.sample.count),
        zeros=int(result.sample.zeros),
        convention=convention,
        reference=reference,
        projection=projection,
    )
