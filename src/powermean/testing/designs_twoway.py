"""The five cells with two-way fixed effects of the direct solver study.

Cells ``d12`` to ``d16`` of that study.  Every cell
has ``E(y | x, g1, g2) = exp(x'b + a[g1] + c[g2])`` with a binary treatment of
coefficient ``log(1.08)`` and standard normal regressors.

=====  ============  =========  ==========================================
Cell   Topology      Levels     Outcome law
=====  ============  =========  ==========================================
d12    balanced      20 by 15   Poisson
d13    balanced      20 by 15   lognormal, log-scale standard deviation 1
d14    unbalanced    40 by 30   negative binomial, size 0.2
d15    unbalanced    40 by 30   lognormal, log-scale standard deviation 2.5
d16    communities   40 by 40   lognormal, log-scale standard deviation 1
=====  ============  =========  ==========================================

In the unbalanced cells the frequency of level ``j`` is proportional to
``j^-1.25`` and every level has at least three observations.  In ``d16`` the levels
form two communities joined by 40 observations.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from powermean.testing.synthetic import (
    TREATMENT_COEFFICIENT,
    FloatArray,
    IntArray,
    Synthetic,
    describe_law,
    draw_outcome,
    inflate_zeros,
    require_design,
    seeded,
)

TWOWAY_SOURCE = (
    "direct solver study of Benatia, Bellego and Pape: cells d12 to d16"
)
TWOWAY_SEED = 20260926
"""Master seed of the direct study (split 40)."""


@dataclass(frozen=True)
class TwoWayDesign:
    """One cell with two-way fixed effects of the direct solver study."""

    number: int
    topology: str
    levels: tuple[int, int]
    fe_sd: float
    law: str
    parameters: Mapping[str, float] = field(default_factory=dict)
    index_sd: float = 0.6
    minimum_per_level: int = 0
    frequency_power: float = 0.0

    def describe(self) -> str:
        """One line for listings: topology, levels, outcome law and parameters."""
        levels = f"{self.levels[0]} by {self.levels[1]} levels"
        return f"{self.topology}, {levels}; {describe_law(self.law, self.parameters)}"


TWOWAY_DESIGNS: dict[str, TwoWayDesign] = {
    "d12": TwoWayDesign(12, "balanced", (20, 15), 0.5, "poisson"),
    "d13": TwoWayDesign(
        13, "balanced", (20, 15), 0.5, "lognormal_dmc", {"residual_sd": 1.0}
    ),
    "d14": TwoWayDesign(
        14,
        "unbalanced",
        (40, 30),
        0.5,
        "nb2",
        {"size": 0.2},
        minimum_per_level=3,
        frequency_power=1.25,
    ),
    "d15": TwoWayDesign(
        15,
        "unbalanced",
        (40, 30),
        1.0,
        "lognormal_dmc",
        {"residual_sd": 2.5},
        index_sd=1.5,
        minimum_per_level=3,
        frequency_power=1.25,
    ),
    "d16": TwoWayDesign(
        16,
        "communities",
        (40, 40),
        1.0,
        "lognormal_dmc",
        {"residual_sd": 1.0},
        index_sd=1.5,
    ),
}


def _twoway_groups(
    design: TwoWayDesign, n: int, rng: np.random.Generator
) -> tuple[tuple[IntArray, IntArray], FloatArray]:
    """Two classifications and the sum of their effects (``_groups``).

    Each block of effects is shifted so that the unweighted average of its
    exponentials over levels is one.
    """
    if design.topology == "communities":
        # Two communities of 20 levels on each side.  The first 120 rows are 40
        # same-level edges, 40 ring edges within a community and 40 edges across,
        # which connect the two communities.
        base = np.arange(40)
        ring = (base // 20) * 20 + (base + 1) % 20
        cross = (base + 20) % 40
        g = np.r_[base, base, base]
        h = np.r_[base, ring, cross]
        remainder = n - len(g)
        community = rng.binomial(1, 0.5, remainder)
        g = np.r_[g, 20 * community + rng.integers(0, 20, remainder)]
        h = np.r_[h, 20 * community + rng.integers(0, 20, remainder)]
        permutation = rng.permutation(n)
        blocks = [g[permutation], h[permutation]]
    else:
        blocks = []
        for count in design.levels:
            if design.topology == "balanced":
                g = np.arange(n) % count
            else:
                minimum = design.minimum_per_level
                probability = (1 + np.arange(count)) ** (-design.frequency_power)
                tail = rng.choice(
                    count, n - minimum * count, p=probability / probability.sum()
                )
                g = np.r_[np.tile(np.arange(count), minimum), tail]
            rng.shuffle(g)
            blocks.append(g)
    fe_index = np.zeros(n)
    for g, count in zip(blocks, design.levels, strict=True):
        effect = rng.normal(0, design.fe_sd, count)
        effect -= np.log(np.mean(np.exp(effect)))
        fe_index += effect[g]
    first, second = (np.asarray(g, dtype=np.int64) for g in blocks)
    return (first, second), fe_index


def twoway_data(
    design: str,
    *,
    n: int = 2000,
    k: int = 9,
    zeros: float = 0.0,
    seed: int = TWOWAY_SEED,
    draw: int = 0,
    split: int = 40,
) -> Synthetic:
    """One dataset of a two-way fixed-effect cell of the direct solver study.

    The model is ``E(y | x, g1, g2) = exp(x'b + a[g1] + c[g2])`` with a binary
    treatment (coefficient ``log(1.08)``) and ``k - 1`` standard normal regressors.
    The topologies are balanced (``d12``, ``d13``), unbalanced with skewed level
    frequencies (``d14``, ``d15``) and two weakly linked communities (``d16``).
    With the default arguments and ``draw`` below 10000, the result is a dataset of
    the study.

    Parameters
    ----------
    design : str
        A key of ``TWOWAY_DESIGNS``.
    n, k : int
        Observations and regressors; the study used 2000 and 9.
    zeros : float
        Share of outcomes set to zero, with the mean kept.
    seed, draw, split : int
        Master seed, replication number and split.  The study seeds each stream
        with ``SeedSequence([seed, split, cell number, draw, stream])``.

    Returns
    -------
    Synthetic
        Columns ``y``, ``treat``, ``x1`` to ``x{k-1}``, ``g1``, ``g2``, ``true_mean``.
    """
    require_design(TWOWAY_DESIGNS, design, "twoway")
    spec = TWOWAY_DESIGNS[design]
    floor = 120 if spec.topology == "communities" else 0
    floor = max(floor, spec.minimum_per_level * max(spec.levels), 1)
    if n < floor or k < 2:
        raise ValueError(f"design {design} needs n >= {floor} and k >= 2")
    key = [seed, split, spec.number, draw]

    rng = seeded([*key, 0])
    treatment = rng.binomial(1, 0.5, n)
    z = rng.normal(size=(n, k - 1))
    slopes = (-1.0) ** np.arange(k - 1)
    slopes *= spec.index_sd / math.sqrt(np.sum(slopes**2))
    intercept = math.log(2.0) - 0.5 * spec.index_sd**2
    x = np.column_stack((treatment, z))
    beta = np.r_[TREATMENT_COEFFICIENT, slopes]
    index = intercept + x @ beta

    (g1, g2), fe_index = _twoway_groups(spec, n, seeded([*key, 1]))
    index = index + fe_index
    mean = np.exp(index)
    y = draw_outcome(spec.law, spec.parameters, index, mean, seeded([*key, 2]))
    y = inflate_zeros(y, zeros, seeded([*key, 3]))

    names = ["treat", *(f"x{j}" for j in range(1, k))]
    frame = pd.DataFrame(x, columns=names)
    frame["treat"] = frame["treat"].astype(int)
    frame.insert(0, "y", y)
    frame["g1"], frame["g2"], frame["true_mean"] = g1, g2, mean
    roles = {
        "y": "outcome",
        "treat": "binary treatment with mean 1/2, coefficient log(1.08)",
    }
    roles.update(dict.fromkeys(names[1:], "standard normal"))
    roles.update(
        g1=f"first classification, {spec.levels[0]} levels, {spec.topology}",
        g2=f"second classification, {spec.levels[1]} levels, {spec.topology}",
        true_mean="E(y | regressors, effects)",
    )
    fingerprint = hashlib.blake2b(digest_size=8)
    for array in (x, y, g1, g2):
        fingerprint.update(np.ascontiguousarray(array).view(np.uint8))
    return Synthetic(
        design=f"twoway/{design}",
        frame=frame,
        formula="y ~ " + " + ".join(names) + " | g1 + g2",
        truth=dict(zip(names, beta.tolist(), strict=True)),
        columns=roles,
        seed=(
            f"SeedSequence([{seed}, {split}, {spec.number}, {draw}, stream]); "
            "streams: regressors 0, effects 1, outcome 2, zeros 3"
        ),
        source=TWOWAY_SOURCE,
        fingerprint=str(int.from_bytes(fingerprint.digest(), "little")),
        notes=("No draw is conditioned on the outcome; a level can have only zeros.",),
    )
