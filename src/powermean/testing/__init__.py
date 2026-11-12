"""Synthetic data with known parameters, for tests, examples and fixtures.

Four functions return a :class:`Synthetic`: a data frame, a correctly specified
formula, the true coefficients, the role of each column and the seed recipe.

==================  ============================================================
``solver_data``     the 20 designs without fixed effects of the solver study
``twoway_data``     the five two-way fixed-effect cells of the direct solver study
``threeway_data``   the four designs of the three-way panel simulation
``showcase_data``   one firm-year panel with a column for every feature
==================  ============================================================

The first three are ports of the simulation code of Benatia, Bellego and Pape,
"Dealing with Logs and Zeros in Regression Models".  With their default seeds they
regenerate the datasets of those studies, up to the rounding of elementary functions
and matrix products, which differs across platforms.

Every function takes ``zeros``, a share of outcomes set to zero with the conditional
mean kept.
"""

from powermean.testing.designs_solver import SOLVER_DESIGNS, SOLVER_SEED, solver_data
from powermean.testing.designs_threeway import (
    THREEWAY_DESIGNS,
    THREEWAY_SEED,
    threeway_data,
)
from powermean.testing.designs_twoway import TWOWAY_DESIGNS, TWOWAY_SEED, twoway_data
from powermean.testing.showcase import (
    SHOWCASE_LAWS,
    SHOWCASE_TRUTH,
    showcase_data,
    showcase_treat_coefficient,
)
from powermean.testing.synthetic import Synthetic, save

__all__ = [
    "SHOWCASE_LAWS",
    "SHOWCASE_TRUTH",
    "SOLVER_DESIGNS",
    "SOLVER_SEED",
    "THREEWAY_DESIGNS",
    "THREEWAY_SEED",
    "TWOWAY_DESIGNS",
    "TWOWAY_SEED",
    "Synthetic",
    "save",
    "showcase_data",
    "showcase_treat_coefficient",
    "solver_data",
    "threeway_data",
    "twoway_data",
]
