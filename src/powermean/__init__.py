"""powermean: power-mean regression for outcomes with zeros.

The public interface is the list ``__all__`` below: the functions ``fit``, ``ppml``,
``gpml``, ``logols``, ``prepare``, ``solver_defaults`` and the
convention factory ``ssc``, the working variance ``NB2``, the typed record
``SolverOptions`` that ``solver_defaults`` returns, and the exceptions and warnings,
which a caller catches
or filters by these names. The table functions ``etable``,
``summary_statistics``, ``balance_table``, ``balance_regression`` and
``outcome_table`` share the same lazy public interface.
:mod:`powermean.testing` is the synthetic data used by tests, examples and fixtures,
with its own ``__all__``.

The names are resolved on first access (``__getattr__``), so that importing the
package, or a module of a lower layer, loads neither the interface, the result layer
nor the solvers until a public function is used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from powermean.api import NB2, fit, gpml, logols, ppml, prepare, solver_defaults
    from powermean.data.inference_conventions import ssc
    from powermean.describe.balance import balance_table
    from powermean.describe.balance_regression import balance_regression
    from powermean.describe.outcome import outcome_table
    from powermean.describe.summary import summary_statistics
    from powermean.errors import (
        ConvergenceError,
        CovarianceWarning,
        CoverageWarning,
        ExistenceError,
        IdentificationError,
        NumericalWarning,
        SampleError,
        SolverCapabilityError,
        SpecificationError,
    )
    from powermean.results.etable import etable
    from powermean.solve.options import SolverOptions

__all__ = [
    "NB2",
    "ConvergenceError",
    "CovarianceWarning",
    "CoverageWarning",
    "ExistenceError",
    "IdentificationError",
    "NumericalWarning",
    "SampleError",
    "SolverCapabilityError",
    "SolverOptions",
    "SpecificationError",
    "balance_regression",
    "balance_table",
    "etable",
    "fit",
    "gpml",
    "logols",
    "outcome_table",
    "ppml",
    "prepare",
    "solver_defaults",
    "ssc",
    "summary_statistics",
]

# The module that defines each public name, resolved lazily.
_SOURCES: dict[str, str] = {
    "NB2": "powermean.api",
    "fit": "powermean.api",
    "gpml": "powermean.api",
    "logols": "powermean.api",
    "ppml": "powermean.api",
    "prepare": "powermean.api",
    "solver_defaults": "powermean.api",
    "ssc": "powermean.data.inference_conventions",
    "summary_statistics": "powermean.describe.summary",
    "balance_table": "powermean.describe.balance",
    "balance_regression": "powermean.describe.balance_regression",
    "outcome_table": "powermean.describe.outcome",
    "etable": "powermean.results.etable",
    "SolverOptions": "powermean.solve.options",
    "ConvergenceError": "powermean.errors",
    "CovarianceWarning": "powermean.errors",
    "CoverageWarning": "powermean.errors",
    "ExistenceError": "powermean.errors",
    "IdentificationError": "powermean.errors",
    "NumericalWarning": "powermean.errors",
    "SampleError": "powermean.errors",
    "SolverCapabilityError": "powermean.errors",
    "SpecificationError": "powermean.errors",
}


def __getattr__(name: str) -> object:
    """A public name, imported from its module on first access."""
    source = _SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module 'powermean' has no attribute {name!r}")
    value: object = getattr(importlib.import_module(source), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """The public names beside the attributes of the module."""
    return sorted(set(globals()) | set(__all__))
