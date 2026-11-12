"""Options of the optional no-FE IOLS solver and lossless kernel-control mapping.

Each option is a field with its default, its valid values and one sentence of
documentation:

    option                  default                   valid values
    eps                     1e-12                     positive finite real numbers
    anderson_m              5                         nonnegative integers
    max_phase1_iterations   100000                    positive integers
    stopcrit                1000000                   positive integers
    max_phase2_evaluations  5000000                   positive integers
    delta_grid              exp(linspace(-5, 5, 5))   nonempty lists, tuples or
                                                      one-dimensional arrays of
                                                      positive finite real numbers
    rho_log_min             -7                        finite real numbers
    rho_log_max             14                        finite real numbers
    iols_backend            'cpp'                     'cpp', 'numpy'

When ``iols_backend`` is omitted by the caller, ``'cpp'`` is the preferred backend and
automatically falls back to ``'numpy'`` only if the compiled extension module
``powermean._native.iols._kernel_cpp`` is genuinely absent on disk. When
``iols_backend='cpp'`` is explicitly passed by the caller, strict C++ selection is
enforced without fallback.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Literal, Self, get_args

import numpy as np

from powermean.errors import SpecificationError
from powermean.solve.iols.direct_route import DirectNoFEBackend
from powermean.solve.iols.kernel import IOLSKernelControl
from powermean.solve.absorbed import (
    POSITIVE_INTEGER,
    POSITIVE_REAL,
    Setting,
    changes,
    resolved,
)
from powermean.solve.options import (
    OPTION_MAPPINGS,
    Choices,
    IntegerRange,
    RealInterval,
)

Backend = Literal["cpp", "numpy"]
NONNEGATIVE_INTEGERS = "nonnegative integers"
FINITE_REALS = "finite real numbers"
POSITIVE_GRIDS = (
    "nonempty lists, tuples or one-dimensional arrays of positive finite real numbers"
)

DELTA_GRID: tuple[float, ...] = tuple(
    float(delta) for delta in np.exp(np.linspace(-5.0, 5.0, 5))
)


@dataclass(frozen=True)
class PositiveGrid:
    """A nonempty ordered one-dimensional collection of positive finite real numbers.

    A list, a tuple or a one-dimensional array, each element a positive finite real
    by the rules of :class:`powermean.solve.options.RealInterval`; stored as a tuple
    of the nearest ``float`` of each element, in the order given, repetitions kept.  A
    string, a mapping, a set, a scalar, an array of another dimension and an empty
    collection are not grids.
    """

    valid: str = POSITIVE_GRIDS

    def resolve(self, value: object) -> tuple[float, ...] | None:
        """The tuple of floats, or ``None`` if ``value`` is not such a grid."""
        if isinstance(value, np.ndarray):
            items: list[object] = list(value) if value.ndim == 1 else []
        elif isinstance(value, list | tuple):
            items = list(value)
        else:
            return None
        if not items:
            return None
        stored: list[float] = []
        for item in items:
            number = POSITIVE_REAL.resolve(item)
            if number is None:
                return None
            stored.append(number)
        return tuple(stored)


NONNEGATIVE_INTEGER = IntegerRange(lower=0, upper=None, valid=NONNEGATIVE_INTEGERS)
FINITE_REAL = RealInterval(lower=-math.inf, upper=math.inf, valid=FINITE_REALS)

IOLS_DOMAINS: Mapping[str, Setting] = MappingProxyType(
    {
        "eps": POSITIVE_REAL,
        "anderson_m": NONNEGATIVE_INTEGER,
        "max_phase1_iterations": POSITIVE_INTEGER,
        "stopcrit": POSITIVE_INTEGER,
        "max_phase2_evaluations": POSITIVE_INTEGER,
        "delta_grid": PositiveGrid(),
        "rho_log_min": FINITE_REAL,
        "rho_log_max": FINITE_REAL,
        "iols_backend": Choices(values=get_args(Backend)),
    }
)
IOLS_OPTION_NAMES: tuple[str, ...] = tuple(IOLS_DOMAINS)


@dataclass(frozen=True, kw_only=True)
class IolsOptions:
    """The options of the optional no-FE IOLS solver.

    Keyword arguments only.  Each value is validated against :data:`IOLS_DOMAINS` and
    stored as resolved.

    Parameters
    ----------
    eps : float, default 1e-12
        Internal stopping threshold on the largest coefficient change, before the
        residual check.  The common
        certificate decides convergence; a positive finite real.
    anderson_m : int, default 5
        Initial depth of the Anderson mixing; a nonnegative integer, zero included,
        stored as given and not a statement about a later adaptation of the
        depth.
    max_phase1_iterations : int, default 100000
        Budget of pilot updates; a positive integer, 1 included.
    stopcrit : int, default 1000000
        Budget of updates per step size; a positive integer, 1 included.
    max_phase2_evaluations : int, default 5000000
        Budget of evaluations of the exact map; a positive integer, 1 included.
    delta_grid : tuple of float, default exp(linspace(-5, 5, 5))
        The shifts of the pilot, in their order and with their repetitions; a nonempty
        list, tuple or one-dimensional array of positive finite reals, stored as a
        tuple of floats; neither sorted, deduplicated nor rebuilt from endpoints.
    rho_log_min : float, default -7
        Logarithmic lower bound of the step grid; a finite real, independent of
        ``rho_log_max`` and not exponentiated here.
    rho_log_max : float, default 14
        Logarithmic upper bound of the step grid; a finite real, independent of
        ``rho_log_min`` and not exponentiated here.
    iols_backend : {'cpp', 'numpy'}, default 'cpp'
        The implementation: the compiled full loop or its NumPy twin, an explicit and
        recorded choice.
    """

    eps: float = 1e-12
    anderson_m: int = 5
    max_phase1_iterations: int = 100000
    stopcrit: int = 1000000
    max_phase2_evaluations: int = 5000000
    delta_grid: tuple[float, ...] = DELTA_GRID
    rho_log_min: float = -7.0
    rho_log_max: float = 14.0
    iols_backend: Backend = "cpp"

    def __new__(cls, *_args: object, **kwargs: object) -> Self:
        """Record whether ``iols_backend`` was explicitly supplied by the caller."""
        instance = super().__new__(cls)
        object.__setattr__(instance, "_iols_backend_explicit", "iols_backend" in kwargs)
        return instance

    def __getnewargs_ex__(self) -> tuple[tuple[object, ...], dict[str, object]]:
        """Keep backend presence when copying or unpickling this typed record."""
        values = self.to_dict()
        if not self.backend_explicit:
            values.pop("iols_backend")
        return (), values

    def __post_init__(self) -> None:
        """Validate every option, in the order of the fields, and store it resolved."""
        for option in fields(self):
            value = getattr(self, option.name)
            stored = resolved(IOLS_DOMAINS, option.name, value)
            object.__setattr__(self, option.name, stored)

    @property
    def backend_explicit(self) -> bool:
        """Whether ``iols_backend`` was explicitly passed rather than defaulted."""
        return bool(getattr(self, "_iols_backend_explicit", False))

    def to_dict(self) -> dict[str, object]:
        """The resolved options by name, in the order of the fields, as a new dict.

        The grid is a new plain list of its floats, in their order; every other value
        is the scalar stored. This complete mapping supplies the backend explicitly
        when passed back as options. The typed record retains omission separately;
        Computation stores that presence as metadata beside its option snapshot.
        """
        values: dict[str, object] = {}
        for option in fields(self):
            value = getattr(self, option.name)
            values[option.name] = list(value) if isinstance(value, tuple) else value
        return values

    @classmethod
    def from_dict(cls, options: object) -> IolsOptions:
        """The record of a flat mapping of options: its values over the defaults."""
        return cls(**changes(options, tuple(IOLS_DOMAINS)))


def resolve_iols_options(options: object = None) -> IolsOptions:
    """Resolve ``solver_options`` for ``solver='iols'``."""
    if options is None:
        return IolsOptions()
    if isinstance(options, IolsOptions):
        stored = vars(options)
        presence = stored.get("_iols_backend_explicit")
        if (
            type(presence) is not bool
            or any(name not in stored for name in IOLS_OPTION_NAMES)
            or (not presence and stored["iols_backend"] != "cpp")
        ):
            raise SpecificationError(
                "specification.invalid_value",
                {
                    "argument": "solver_options",
                    "value": options,
                    "valid": "complete IolsOptions records with valid backend presence",
                },
            )
        values = {name: stored[name] for name in IOLS_OPTION_NAMES}
        if not presence:
            values.pop("iols_backend")
        return IolsOptions.from_dict(values)
    if isinstance(options, Mapping):
        return IolsOptions.from_dict(options)
    raise SpecificationError(
        "specification.invalid_value",
        {
            "argument": "solver_options",
            "value": options,
            "valid": f"IolsOptions or {OPTION_MAPPINGS}",
        },
    )


def effective_backend_request(options: IolsOptions) -> DirectNoFEBackend:
    """Internal backend request preserving explicit ``'cpp'`` vs preferred ``'cpp'``.

    - Explicit ``iols_backend='numpy'`` -> ``'numpy'``
    - Explicit ``iols_backend='cpp'`` -> ``'cpp'`` (strict C++, no fallback)
    - Default/omitted ``iols_backend`` -> ``'auto'`` (preferred ``'cpp'``, falls back to
      ``'numpy'`` only when ``powermean._native.iols._kernel_cpp`` is genuinely absent)
    """
    if options.iols_backend == "numpy":
        return "numpy"
    if options.backend_explicit:
        return "cpp"
    return "auto"


def options_to_kernel_control(options: IolsOptions) -> IOLSKernelControl:
    """Map :class:`IolsOptions` to :class:`IOLSKernelControl` without lossy coercion.

    The paper requires at least two pilot/per-step iterations, but permits one
    exact-map evaluation. Its curvature grid does not order the rho bounds (the
    legacy exponential-grid helper does). The default pilot follows the original
    endpoint/size path. Other ordered vectors use the internal exact-vector input,
    retaining every float, repetition and position without endpoint reconstruction.
    """
    if options.max_phase1_iterations < 2:
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "max_phase1_iterations",
                "value": options.max_phase1_iterations,
                "valid": (
                    "integers >= 2 (the direct fixed-tau IOLS kernel requires "
                    "max_phase1_iterations >= 2; budget 1 is not clamped)"
                ),
            },
        )
    if options.stopcrit < 2:
        raise SpecificationError(
            "specification.invalid_value",
            {
                "argument": "stopcrit",
                "value": options.stopcrit,
                "valid": (
                    "integers >= 2 (the direct fixed-tau IOLS kernel starts its "
                    "step counter at k = 1 with while k < stopcrit; stopcrit 1 is "
                    "not clamped)"
                ),
            },
        )
    return IOLSKernelControl(
        anderson_m=int(options.anderson_m),
        eps=float(options.eps),
        stopcrit=int(options.stopcrit),
        max_phase1_iterations=int(options.max_phase1_iterations),
        max_phase2_evaluations=int(options.max_phase2_evaluations),
        delta_log_min=-5.0,
        delta_log_max=5.0,
        delta_grid_size=len(options.delta_grid),
        _pilot_deltas=None if options.delta_grid == DELTA_GRID else options.delta_grid,
        rho_log_min=float(options.rho_log_min),
        rho_log_max=float(options.rho_log_max),
        rho_grid_size=150,
        transformed_design_normalization="none",
        direct_restart_start="phase1_start",
    )
