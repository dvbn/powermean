"""The public BBP paper that alpha results cite.

Citation metadata is fixed to the public arXiv record. Formatting reads no files
and does not access the network.
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

    ``authors`` are the names as they appear on the work, given name first.
    Journal fields and ``kind`` are unset for the arXiv preprint. The ``note``
    identifies the public version and ``doi`` is the identifier in its record.
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


BBP = "bbp2025"
REGISTRY = MappingProxyType(
    {
        BBP: Reference(
            BBP,
            ("David Benatia", "Christophe Bellégo", "Louis Pape"),
            2025,
            "Dealing with Logs and Zeros in Regression Models",
            note="arXiv:2203.11820v3",
            doi="10.48550/arXiv.2203.11820",
        ),
    }
)


def reference_keys(estimator: Estimator) -> tuple[str, ...]:
    """Return the alpha's citation key, regardless of the estimator."""
    return (BBP,)


def references(estimator: Estimator) -> tuple[Reference, ...]:
    """The references of :func:`reference_keys`, as records."""
    return tuple(REGISTRY[key] for key in reference_keys(estimator))
