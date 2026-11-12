"""The exceptions and warnings of the package.

Each class carries one message of :mod:`powermean.messages`: it is constructed
from an identifier and a mapping of the fields of that template, renders the text
once, and keeps three attributes:

- ``identifier``: the identifier of the message, stable across versions;
- ``fields``: the fields that filled the template, read-only;
- ``message``: the rendered text, also ``str(error)``.

The numerical classes keep what the solver and the certificate supplied, for
inspection, exactly as given: no copy, no repair, no verdict.

- :class:`ExistenceError` carries the ``certificate`` and its ``status``;
- :class:`ConvergenceError` carries the ``last_state``, its ``status`` and, when
  one was computed, the ``certificate``;
- :class:`NumericalWarning` carries the ``status`` and the ``certificate``.

For these three, ``status`` is also the field ``status`` of the template, so that
the text and the attribute cannot disagree; the mapping passed as ``fields`` holds
the other fields only.  This module does not interpret the status strings and
decides nothing about a solution.

Meanings:

===========================  =========================================================
``SpecificationError``       Inconsistent arguments; a combination that is not
                             available
``SampleError``              Negative outcomes; zeros with ``omega <= 0`` without
                             ``positive_only=True``; no usable observations
``IdentificationError``      Rank deficiency that cannot be resolved by dropping
                             collinear columns
``ExistenceError``           A certificate shows that the estimating equation has no
                             finite root
``ConvergenceError``         No certified solution within the budget
``SolverCapabilityError``    The requested solver does not cover the case
``CoverageWarning``          Computed, but outside the results proved in BBP
``NumericalWarning``         The secondary convergence criteria are met but the
                             requested tolerance is not: status ``verified_secondary``
``CovarianceWarning``        A note about a reported covariance: an indefinite signed
                             multiway estimate adjusted, or an indefinite covariance
                             whose inference is withheld
===========================  =========================================================

The three warnings are distinct categories, none a subclass of another, so that each
can be filtered without the others.  No class derives from another class of the
package.  A malformed construction (unknown identifier, wrong fields, a message of
another class) raises :class:`KeyError`, :class:`TypeError` or :class:`ValueError`:
it is a defect of the caller, not a message for the user.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from powermean import messages


def _rendered(
    identifier: str, fields: Mapping[str, object] | None, category: str
) -> tuple[Mapping[str, object], str]:
    """The read-only fields and the text of a message carried by ``category``."""
    message = messages.REGISTRY[identifier]
    if message.category != category:
        msg = f"{identifier!r} belongs to {message.category}, not {category}"
        raise ValueError(msg)
    filled = MappingProxyType(dict(fields or {}))
    return filled, messages.render(identifier, filled)


def _with_status(
    fields: Mapping[str, object] | None, status: str
) -> Mapping[str, object]:
    """The fields with ``status`` added; the mapping must not name it itself."""
    if fields and "status" in fields:
        msg = "'status' is the status attribute, not a field of the mapping"
        raise TypeError(msg)
    return {**(fields or {}), "status": status}


class SpecificationError(Exception):
    """Inconsistent arguments, or a combination that is not available."""

    def __init__(
        self, identifier: str, fields: Mapping[str, object] | None = None
    ) -> None:
        self.identifier = identifier
        self.fields, self.message = _rendered(identifier, fields, type(self).__name__)
        super().__init__(self.message)


class SampleError(Exception):
    """Negative outcomes; zeros at a nonpositive order; no usable observation."""

    def __init__(
        self, identifier: str, fields: Mapping[str, object] | None = None
    ) -> None:
        self.identifier = identifier
        self.fields, self.message = _rendered(identifier, fields, type(self).__name__)
        super().__init__(self.message)


class IdentificationError(Exception):
    """Rank deficiency that cannot be resolved by dropping collinear columns."""

    def __init__(
        self, identifier: str, fields: Mapping[str, object] | None = None
    ) -> None:
        self.identifier = identifier
        self.fields, self.message = _rendered(identifier, fields, type(self).__name__)
        super().__init__(self.message)


class ExistenceError(Exception):
    """A certificate shows that the estimating equation has no finite root.

    Carries the ``certificate`` as supplied, its ``status``, and the ``last_state``
    when the solver left one.
    """

    def __init__(
        self,
        identifier: str,
        fields: Mapping[str, object] | None = None,
        *,
        status: str,
        certificate: object,
        last_state: object | None = None,
    ) -> None:
        self.identifier = identifier
        self.status = status
        self.certificate = certificate
        self.last_state = last_state
        self.fields, self.message = _rendered(
            identifier, _with_status(fields, status), type(self).__name__
        )
        super().__init__(self.message)


class ConvergenceError(Exception):
    """No certified solution within the budget.

    Carries the ``last_state`` as supplied, the ``status`` of the certificate, and
    the ``certificate`` itself when one was computed.
    """

    def __init__(
        self,
        identifier: str,
        fields: Mapping[str, object] | None = None,
        *,
        status: str,
        last_state: object,
        certificate: object | None = None,
    ) -> None:
        self.identifier = identifier
        self.status = status
        self.last_state = last_state
        self.certificate = certificate
        self.fields, self.message = _rendered(
            identifier, _with_status(fields, status), type(self).__name__
        )
        super().__init__(self.message)


class SolverCapabilityError(Exception):
    """The requested solver does not cover the case."""

    def __init__(
        self, identifier: str, fields: Mapping[str, object] | None = None
    ) -> None:
        self.identifier = identifier
        self.fields, self.message = _rendered(identifier, fields, type(self).__name__)
        super().__init__(self.message)


class CoverageWarning(UserWarning):
    """The requested combination is computed but outside the results proved in BBP."""

    def __init__(
        self, identifier: str, fields: Mapping[str, object] | None = None
    ) -> None:
        self.identifier = identifier
        self.fields, self.message = _rendered(identifier, fields, type(self).__name__)
        super().__init__(self.message)


class NumericalWarning(UserWarning):
    """The secondary convergence criteria are met but the requested tolerance is not.

    Carries the ``status`` reported by the certificate (``verified_secondary``, as
    the caller reports it) and the ``certificate`` as supplied.  Distinct from
    :class:`CoverageWarning`, which concerns theory, not numerics.
    """

    def __init__(
        self,
        identifier: str,
        fields: Mapping[str, object] | None = None,
        *,
        status: str,
        certificate: object,
    ) -> None:
        self.identifier = identifier
        self.status = status
        self.certificate = certificate
        self.fields, self.message = _rendered(
            identifier, _with_status(fields, status), type(self).__name__
        )
        super().__init__(self.message)


class CovarianceWarning(UserWarning):
    """A note about a reported covariance.

    An indefinite signed multiway estimate whose negative eigenvalues were set to zero,
    or an indefinite covariance whose dependent inference is withheld.  Distinct from
    :class:`CoverageWarning` and :class:`NumericalWarning`, and from the diagnostic
    warning of the ported covariance primitives (:mod:`powermean.inference.vcov`),
    which carries no identifier.
    """

    def __init__(
        self, identifier: str, fields: Mapping[str, object] | None = None
    ) -> None:
        self.identifier = identifier
        self.fields, self.message = _rendered(identifier, fields, type(self).__name__)
        super().__init__(self.message)
