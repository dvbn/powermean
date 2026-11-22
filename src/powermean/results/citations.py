"""The references that a result retains and that its summary prints.

``DESIGN.md``, Sections 1 and 4.5; ``design/06_reuse_map.md``, the row ``results/``
(S6 ``references.py`` is a reference for the form of a citation record, not a port:
its registry of twenty entries, its resolver and its public bibliography methods are
not carried).  The author requires both manuscripts in every result, with the method
references of the estimator and of the covariance convention that the result reports:

- Benatia, Bellégo and Pape (2026), "Dealing with Logs and Zeros in Regression
  Models", the reference manuscript of the package (BBP), arXiv:2203.11820v4,
  posted on 16 November 2026;
- Benatia (2026), "Efficient Estimation of Proportional Treatment Effects", the
  manuscript of the efficiency layer, whose current title block carries no revision
  date, so none is attached;
- the pseudo-maximum-likelihood results that the power-mean estimators rest on
  (Gouriéroux, Monfort and Trognon, 1984), the quasi-likelihood of the power family
  of working variances (Wedderburn, 1974), PPML (Santos Silva and Tenreyro, 2006),
  and the program whose small-sample convention the default standard errors reproduce
  (Correia, Guimarães and Zylkin, 2020; file 04, Section 3).

No journal, DOI, month or publication status is inferred for the manuscripts: each
carries its authors, its year, its title and its supplied citation note,
and no kind; the metadata of the published references, their kind ``"article"``
included, is that of the verified S6 registry.  A reference is a frozen record whose
``text`` is plain text; nothing here reads a file or the network, and the
fixed-effect, instrument and targeting citations of S6 are not carried.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from powermean.data.specs import Estimator

Kind = Literal["article"]


@dataclass(frozen=True)
class Reference:
    """One citation: its key, authors as written, year, title and venue.

    ``authors`` are the names as they appear on the work, given name first; ``kind``
    is ``"article"`` for a published article, as the verified registry states it,
    and absent (``None``) for a manuscript, whose publication status is not
    inferred; ``note`` is the supplied manuscript citation note; ``journal``,
    ``volume``, ``number``, ``pages`` and ``doi`` are those of a published article,
    and absent otherwise.
    """

    key: str
    authors: tuple[str, ...]
    year: int
    title: str
    kind: Kind | None = None
    journal: str | None = None
    volume: str | None = None
    number: str | None = None
    pages: str | None = None
    note: str | None = None
    doi: str | None = None

    def text(self) -> str:
        """The citation as one line of plain text, without invented fields."""
        names = self.authors
        prefix = (
            names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
        )
        pieces = [f"{prefix} ({self.year}).", f"{self.title}."]
        if self.journal is not None:
            venue = self.journal
            if self.volume is not None:
                venue += f" {self.volume}"
            if self.number is not None:
                venue += f"({self.number})"
            if self.pages is not None:
                venue += f", {self.pages}"
            pieces.append(venue + ".")
        if self.note is not None:
            pieces.append(self.note + ".")
        if self.doi is not None:
            pieces.append(f"https://doi.org/{self.doi}")
        return " ".join(pieces)

    def to_dict(self) -> dict[str, object]:
        """The metadata as values that JSON holds; absent fields are ``None``."""
        return {
            "key": self.key,
            "authors": list(self.authors),
            "year": self.year,
            "title": self.title,
            "kind": self.kind,
            "journal": self.journal,
            "volume": self.volume,
            "number": self.number,
            "pages": self.pages,
            "note": self.note,
            "doi": self.doi,
        }


BBP = "bbp2026"
PROPORTIONAL = "benatia2026"
PML_THEORY = "gourieroux1984a"
PML_POISSON = "gourieroux1984b"
QUASI_LIKELIHOOD = "wedderburn1974"
PPML = "santos_silva_tenreyro2006"
CONVENTION = "ppmlhdfe2020"
JOCHMANS = "jochmans2017"

_ENTRIES = (
    Reference(
        BBP,
        ("David Benatia", "Christophe Bellégo", "Louis-Daniel Pape"),
        2026,
        "Dealing with Logs and Zeros in Regression Models",
        note="arXiv:2203.11820v4, 16 November 2026",
    ),
    Reference(
        PROPORTIONAL,
        ("David Benatia",),
        2026,
        "Efficient Estimation of Proportional Treatment Effects",
        note="Manuscript",
    ),
    Reference(
        PML_THEORY,
        ("Christian Gouriéroux", "Alain Monfort", "Alain Trognon"),
        1984,
        "Pseudo Maximum Likelihood Methods: Theory",
        kind="article",
        journal="Econometrica",
        volume="52",
        number="3",
        pages="681-700",
        doi="10.2307/1913471",
    ),
    Reference(
        PML_POISSON,
        ("Christian Gouriéroux", "Alain Monfort", "Alain Trognon"),
        1984,
        "Pseudo Maximum Likelihood Methods: Applications to Poisson Models",
        kind="article",
        journal="Econometrica",
        volume="52",
        number="3",
        pages="701-720",
        doi="10.2307/1913472",
    ),
    Reference(
        QUASI_LIKELIHOOD,
        ("Robert W. M. Wedderburn",),
        1974,
        "Quasi-Likelihood Functions, Generalized Linear Models, and the Gauss-Newton "
        "Method",
        kind="article",
        journal="Biometrika",
        volume="61",
        number="3",
        pages="439-447",
        doi="10.1093/biomet/61.3.439",
    ),
    Reference(
        PPML,
        ("João M. C. Santos Silva", "Silvana Tenreyro"),
        2006,
        "The Log of Gravity",
        kind="article",
        journal="The Review of Economics and Statistics",
        volume="88",
        number="4",
        pages="641-658",
        doi="10.1162/rest.88.4.641",
    ),
    Reference(
        CONVENTION,
        ("Sergio Correia", "Paulo Guimarães", "Thomas Zylkin"),
        2020,
        "Fast Poisson Estimation with High-Dimensional Fixed Effects",
        kind="article",
        journal="The Stata Journal",
        volume="20",
        number="1",
        pages="95-115",
        doi="10.1177/1536867X20909691",
    ),
    Reference(
        JOCHMANS,
        ("Koen Jochmans",),
        2017,
        "Two-Way Models for Gravity",
        kind="article",
        journal="The Review of Economics and Statistics",
        volume="99",
        pages="478-485",
        doi="10.1162/REST_a_00620",
    ),
)

REGISTRY = MappingProxyType({entry.key: entry for entry in _ENTRIES})
# Both manuscripts, in every result (the author's requirement).
MANUSCRIPTS: tuple[str, ...] = (BBP, PROPORTIONAL)


def reference_keys(estimator: Estimator) -> tuple[str, ...]:
    """The keys of the references of a result for ``estimator``, in citation order.

    Both manuscripts first; then, on the power route, the pseudo-maximum-likelihood
    theory that every quasi-score estimator of the package rests on, the
    quasi-likelihood of the power family for a power working variance, and the two
    PPML references at ``omega = 1``, ``tau = 1``; last, the program whose convention
    the default standard errors reproduce.  Log OLS cites the manuscripts and the
    convention only.
    """
    from powermean.model.variance import PowerVariance  # noqa: PLC0415

    keys = list(MANUSCRIPTS)
    if estimator.route == "power":
        keys.append(PML_THEORY)
        variance = estimator.variance
        if isinstance(variance, PowerVariance):
            keys.append(QUASI_LIKELIHOOD)
            if variance.tau == 1 and estimator.mean.omega == 1:
                keys.extend((PML_POISSON, PPML))
    keys.append(CONVENTION)
    return tuple(keys)


def references(estimator: Estimator) -> tuple[Reference, ...]:
    """The references of :func:`reference_keys`, as records."""
    return tuple(REGISTRY[key] for key in reference_keys(estimator))
