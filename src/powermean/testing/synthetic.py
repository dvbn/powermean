"""The record that every data generator returns, and the pieces they share.

A generator draws regressors, computes the conditional mean ``exp(index)`` and draws
the outcome from one of the laws of :func:`draw_outcome`.  The laws are those of the
simulation code of Benatia, Bellego and Pape, "Dealing with Logs and Zeros in
Regression Models".
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

TREATMENT_COEFFICIENT = math.log(1.08)
"""Coefficient of the binary treatment in every design."""

CONTAMINATION: Mapping[str, float] = {
    "index_sd": 1.5,
    "residual_sd": 0.5,
    "contamination_probability": 0.001,
    "contamination_multiplier": 500.0,
}
"""Parameters of the stress design with rare large outcomes (design 19)."""

LAW_LABELS: Mapping[str, str] = {
    "poisson": "Poisson",
    "gamma": "Gamma",
    "nb2": "negative binomial",
    "heterogeneous_nb2": "negative binomial with two sizes",
    "tweedie15": "compound Poisson-Gamma",
    "ig3": "inverse Gaussian",
    "lognormal_cp11": "lognormal",
    "zero_lognormal_cp11": "lognormal with half of the outcomes at zero",
    "lognormal_dmc": "lognormal",
    "contamination": "lognormal with rare large outcomes",
}
"""Plain name of each outcome law of :func:`draw_outcome`, for listings."""


@dataclass(frozen=True)
class Synthetic:
    """One synthetic dataset and what generated it.

    Attributes
    ----------
    design : str
        Family and design, for example ``"solver/G_nb2"``.
    frame : pandas.DataFrame
        One row per observation.  ``true_mean`` is ``E(y | regressors, effects)``.
    formula : str
        A correctly specified model for ``y`` in the formula syntax of the package.
    truth : dict of str to float
        True coefficient of each regressor.
    columns : dict of str to str
        Role of each column of ``frame``.
    seed : str
        The seed recipe with its values.
    source : str
        The simulation study that the design reproduces; empty for the showcase.
    fingerprint : str
        Checksum of the data in the convention of the archive of the study,
        so that a dataset can be matched to an archived draw.
    notes : tuple of str
        Properties of the design that a user or a test can rely on.
    """

    design: str
    frame: pd.DataFrame
    formula: str
    truth: dict[str, float]
    columns: dict[str, str]
    seed: str
    source: str
    fingerprint: str
    notes: tuple[str, ...] = ()

    @property
    def zero_share(self) -> float:
        """Share of zero outcomes."""
        return float((self.frame["y"] == 0).mean())

    def digest(self) -> dict[str, object]:
        """Row count, column sums and the first and last rows of the numeric columns.

        A reader of the data in another program can compare its own sums with these.
        The test suite compares them with the values of the original modules.
        """
        numeric = self.frame.select_dtypes("number")
        return {
            "rows": len(numeric),
            "sums": numeric.sum().to_dict(),
            "first": numeric.iloc[0].to_dict(),
            "last": numeric.iloc[-1].to_dict(),
        }

    def metadata(self) -> dict[str, object]:
        """Everything except the data, as plain types that JSON can store.

        A missing value in the digest (``x_miss`` of the showcase can be missing in
        its first or last row) becomes ``None``, hence JSON ``null``: JSON has no
        ``NaN`` token.  Finite values are unchanged.
        """
        digest = self.digest()
        for part in ("sums", "first", "last"):
            values = cast(dict[str, float], digest[part])
            digest[part] = {
                name: None if math.isnan(value) else value
                for name, value in values.items()
            }
        return {
            "design": self.design,
            "zero_share": self.zero_share,
            "formula": self.formula,
            "truth": self.truth,
            "columns": self.columns,
            "seed": self.seed,
            "source": self.source,
            "fingerprint": self.fingerprint,
            "notes": list(self.notes),
            "numpy": np.__version__,
            "digest": digest,
        }


def save(data: Synthetic, path: Path) -> None:
    """Write the data, and the metadata to a JSON file with the same stem.

    The format follows the suffix of ``path``: ``.csv`` or ``.dta``; Parquet is not
    supported.  The JSON file is strict JSON: ``allow_nan=False`` refuses the ``NaN``
    and ``Infinity`` tokens that other languages do not parse.
    """
    if path.suffix == ".csv":
        data.frame.to_csv(path, index=False)
    elif path.suffix == ".dta":
        data.frame.to_stata(path, write_index=False, version=118)
    else:
        raise ValueError("the output file must end in .csv or .dta")
    text = json.dumps(data.metadata(), indent=2, allow_nan=False)
    path.with_suffix(".json").write_text(text + "\n")


def describe_law(law: str, parameters: Mapping[str, float]) -> str:
    """Plain name of a law followed by its parameters."""
    detail = ", ".join(f"{name} {value:g}" for name, value in parameters.items())
    return LAW_LABELS[law] + (f" ({detail})" if detail else "")


def seeded(key: Sequence[int]) -> np.random.Generator:
    """Generator of the stream identified by a list of integers."""
    return np.random.default_rng(np.random.SeedSequence(list(key)))


def require_design(designs: Mapping[str, object], design: str, family: str) -> None:
    """Raise a ``ValueError`` listing the valid designs if ``design`` is not one."""
    if design not in designs:
        valid = ", ".join(designs)
        raise ValueError(f"unknown {family} design {design!r}; valid designs: {valid}")


def draw_outcome(
    law: str,
    parameters: Mapping[str, float],
    index: FloatArray,
    mean: FloatArray,
    rng: np.random.Generator,
    *,
    h: IntArray | None = None,
) -> FloatArray:
    """Draw outcomes with conditional mean ``mean = exp(index)``.

    Parameters
    ----------
    law : str
        ``"poisson"``; ``"gamma"`` (shape one); ``"nb2"`` (``size``);
        ``"heterogeneous_nb2"`` (``size_h1`` where ``h`` is one, ``size_h0``
        elsewhere); ``"tweedie15"`` (compound Poisson-Gamma, variance
        ``2 mean^1.5``); ``"ig3"`` (inverse Gaussian, variance ``mean^3``);
        ``"lognormal_cp11"`` (log-scale standard deviation one);
        ``"zero_lognormal_cp11"`` (the same, set to zero with probability one half
        and doubled otherwise); ``"lognormal_dmc"`` (``residual_sd``);
        ``"contamination"`` (lognormal times ``contamination_multiplier`` with
        probability ``contamination_probability``, mean kept).
    parameters : mapping
        The parameters named above.
    index, mean : array
        Linear index and its exponential.
    rng : numpy.random.Generator
        The outcome stream.
    h : array of int, optional
        Indicator used by ``"heterogeneous_nb2"`` only.

    Notes
    -----
    The two ``cp11`` laws and ``lognormal_dmc`` are the same distribution at unit
    standard deviation but different floating-point expressions.  Both are kept so
    that each design regenerates the numbers of its study.
    """
    n = len(mean)
    if law == "poisson":
        return rng.poisson(mean).astype(float)
    if law == "gamma":
        return rng.gamma(1.0, mean)
    if law == "tweedie15":
        count = rng.poisson(np.sqrt(mean))
        y = np.zeros_like(mean)
        active = count > 0
        y[active] = rng.gamma(count[active], np.sqrt(mean[active]))
        return y
    if law == "ig3":
        return rng.wald(mean, 1.0)
    if law == "nb2":
        size = parameters["size"]
        return rng.negative_binomial(size, size / (size + mean)).astype(float)
    if law == "heterogeneous_nb2":
        if h is None:
            raise ValueError("heterogeneous_nb2 needs the indicator h")
        sizes = np.where(h == 1, parameters["size_h1"], parameters["size_h0"])
        return rng.negative_binomial(sizes, sizes / (sizes + mean)).astype(float)
    if law in ("lognormal_cp11", "zero_lognormal_cp11"):
        y = mean * np.exp(rng.normal(size=n) - 0.5)
        if law == "zero_lognormal_cp11":
            y *= rng.binomial(1, 0.5, n) / 0.5
        return y
    if law in ("lognormal_dmc", "contamination"):
        sd = parameters["residual_sd"]
        log_y = index + rng.normal(0, sd, n) - 0.5 * sd * sd
        if law == "contamination":
            q = parameters["contamination_probability"]
            high = parameters["contamination_multiplier"]
            low = (1 - q * high) / (1 - q)
            log_y = log_y + np.log(np.where(rng.binomial(1, q, n), high, low))
        return np.exp(log_y)
    raise ValueError(f"unknown outcome law {law!r}")


def inflate_zeros(y: FloatArray, share: float, rng: np.random.Generator) -> FloatArray:
    """Set a share of outcomes to zero and divide the others by one minus the share.

    The conditional mean is unchanged, so the coefficients of an exponential mean
    are unchanged.  With ``share = 0`` nothing is drawn and ``y`` is returned.
    """
    if not 0.0 <= share < 1.0:
        raise ValueError("zeros must be a share in [0, 1)")
    if share == 0.0:
        return y
    return y * rng.binomial(1, 1.0 - share, len(y)) / (1.0 - share)


def checksum(x: FloatArray, y: FloatArray) -> str:
    """SHA-256 of a regressor matrix and an outcome.

    The value is the ``dataset_sha256`` stored in the archive of the solver study.
    """
    x = np.ascontiguousarray(x, dtype="<f8")
    y = np.ascontiguousarray(y, dtype="<f8")
    xs = hashlib.sha256(x.tobytes()).hexdigest()
    ys = hashlib.sha256(y.tobytes()).hexdigest()
    shapes = {"x_shape": list(x.shape), "y_shape": list(y.shape), "dtype": "<f8"}
    header = json.dumps(shapes, sort_keys=True)
    return hashlib.sha256(f"{header}|{xs}|{ys}".encode()).hexdigest()
