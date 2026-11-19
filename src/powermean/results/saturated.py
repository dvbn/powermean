"""The covariance of a saturated fit: missing, with one note (``covariance.saturated``).

A saturated fit has as many observations as parameters, its design identified: rows of
positive weight, or their frequency total with frequency weights, equal to the dimension
of its estimating equation (:func:`powermean.solve.reported.saturated_rows`).  Only a
certificate decides it: the joint, cancelled, no-FE and linear certificates record the
decision, and a fit that runs no certificate is never decided saturated and never
reaches this module.  The root of a saturated fit is exact and every residual is zero:
every sandwich meat is zero in exact arithmetic and its computed value rounding noise,
every hat value is one, and the factor ``N/(N - K)`` of HC1 is undefined.

Owner's rulings 65 and 74 (``briefs/SATFIX.md``).  On a certified saturated fit (an
accepted certificate), each covariance request that its route accepts, at the fit or
after it, is answered here by a missing covariance (every entry NaN) and missing
standard errors, with one ``CovarianceWarning``, ``covariance.saturated``, and no other
covariance note beside it.  The route reads and validates the request first: a request
it refuses on any fit (an unknown key, a classification that was not retained, a
second correction beside a named method) is refused as before; a refusal for zero
residual degrees of freedom is not raised on a saturated fit (the routes convert it
where they apply a convention before the solve).  No covariance, convention factor,
leverage or influence row is formed for the answer.  A saturated fit that is not
certified forms no covariance and issues no note: its inference is withheld.  R's
``sandwich`` returns NaN matrices for HC1 to HC5 with its hat warning on the panel of
the brief (reported: ``reports/codex/r-refs-v1/work/REPORT.md``, Part C).

:class:`SaturatedCovariance` answers the readers of both record families: the
standard errors, the reported covariance (``reported`` and
:meth:`~SaturatedCovariance.require_reported_covariance`), the normal reference
(``distribution``, ``reference``), the degeneracy flags and the metadata of a
covariance table in the schema of each family (:meth:`~SaturatedCovariance.to_dict`
without fixed effects, :meth:`~SaturatedCovariance.fe_attributes` with them), each
with the key ``"note"``, the rendered text that the summary and ``etable`` print.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from powermean import messages
from powermean.errors import CovarianceWarning
from powermean.inference.ssc import ReferenceDistribution
from powermean.model.variance import FloatArray
from powermean.results.owned import owned_array

NOTE = "covariance.saturated"
NORMAL = ReferenceDistribution("normal", None)


@dataclass(frozen=True, eq=False)
class SaturatedCovariance:
    """The missing covariance of one request on a saturated fit.

    Attributes
    ----------
    method : str
        ``"robust"``, ``"cluster"`` or the exact string of a named HC method.
    cluster : str, tuple of str or None
        The requested classification(s), or ``None``.
    bread : str
        The bread of the request.
    count : int
        ``N``, the rows of positive weight of the fit.
    rank : int
        Its parameters, equal to ``count`` (the fit is saturated).
    standard_errors : array
        Missing (NaN), one per reported coefficient.
    reported_covariance : array
        Missing (NaN), square.
    """

    method: str
    cluster: str | tuple[str, ...] | None
    bread: str
    count: int
    rank: int
    standard_errors: FloatArray
    reported_covariance: FloatArray

    @property
    def reported(self) -> FloatArray:
        """The reported covariance, every entry missing (the no-FE record's name)."""
        return self.reported_covariance

    @property
    def distribution(self) -> ReferenceDistribution:
        """The normal reference: no residual degrees of freedom remain for a ``t``."""
        return NORMAL

    @property
    def reference(self) -> ReferenceDistribution:
        """The normal reference (the FE record's name)."""
        return NORMAL

    @property
    def degenerate(self) -> bool:
        """No standard error is zero: every one is missing."""
        return False

    @property
    def missing(self) -> bool:
        """Every standard error is missing."""
        return True

    @property
    def note(self) -> str:
        """The rendered text of the note."""
        return messages.render(NOTE, {"count": self.count, "rank": self.rank})

    def require_reported_covariance(self) -> FloatArray:
        """The reported covariance, an owned copy, every entry missing."""
        return owned_array(self.reported_covariance)

    def to_dict(self) -> dict[str, object]:
        """The metadata of a covariance table without fixed effects, and the note."""
        named = self.method not in ("robust", "cluster")
        cluster = self.cluster
        if isinstance(cluster, tuple):
            cluster = " + ".join(cluster)
        return {
            "kind": "hc" if named else self.method,
            "cluster": cluster,
            "factor": None,
            "count": self.count,
            "clusters": None,
            "rank": self.rank,
            "diagnostics": [NOTE],
            "bread": self.bread,
            "coordinate": "reported",
            "convention": None,
            "reference": "normal",
            "degenerate": False,
            "method": self.method if named else None,
            "saturated": True,
            "note": self.note,
        }

    def fe_attributes(self) -> dict[str, object]:
        """The metadata of a covariance table with fixed effects, and the note."""
        return {
            "method": self.method,
            "cluster": self.cluster,
            "factor": None,
            "N": self.count,
            "G": None,
            "rank": self.rank,
            "diagnostics": (NOTE,),
            "bread": self.bread,
            "coordinate": "reported",
            "convention": None,
            "saturated": True,
            "note": self.note,
        }


def saturated_certificate(certificate: object) -> bool:
    """Whether a fixed-effects certificate (or its snapshot) is that of a saturated fit.

    Its coefficient gate does not apply exactly then
    (:attr:`powermean.solve.absorbed_certificate.FEPointCertificate.not_applicable`).
    """
    return "coefficient" in getattr(certificate, "not_applicable", ())


def saturated_covariance(
    method: str,
    cluster: str | tuple[str, ...] | None,
    bread: str,
    count: int,
    k: int,
    *,
    stacklevel: int = 2,
) -> SaturatedCovariance:
    """The missing covariance of one request, with its note issued once.

    ``k`` is the number of reported coefficients; ``count`` the rows of positive
    weight of the fit, which equal its parameters.  The note is issued at every
    answer, as the other covariance notes are: at the fit, and at each request after it.
    """
    missing = np.full(k, np.nan)
    record = SaturatedCovariance(
        method,
        cluster,
        bread,
        int(count),
        int(count),
        owned_array(missing),
        owned_array(np.full((k, k), np.nan)),
    )
    warnings.warn(
        CovarianceWarning(NOTE, {"count": record.count, "rank": record.rank}),
        stacklevel=stacklevel + 1,
    )
    return record
