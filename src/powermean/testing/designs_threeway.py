"""The four designs of the three-way panel simulation.

The design is that of the tables of Weidner and Zylkin (2021).  A panel of
exporters ``i``, importers ``j`` and years ``t`` has

    x[i,j,t] = x[i,j,t-1] / 2 + a[i,t] + g[j,t] + nu[i,j,t]
    E(y[i,j,t] | x, effects) = exp(x[i,j,t] + a[i,t] + g[j,t] + e[i,j])

with effects of standard deviation 1/16 and ``nu`` of standard deviation 1/2.  The
outcome is the mean times a lognormal disturbance of mean one whose log is a
stationary first-order autoregression within the pair (coefficient 0.3) and whose
variance gives the conditional variance of the design.

All random numbers of a replication come from one master array, keyed by the seed
and the replication number only.  A panel is the corner of that array, so panels
of different sizes and the four designs share random numbers.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from powermean.testing.synthetic import (
    FloatArray,
    Synthetic,
    inflate_zeros,
    require_design,
    seeded,
)

THREEWAY_SOURCE = (
    "three-way panel simulation of Benatia, Bellego and Pape, "
    "after Weidner and Zylkin (2021)"
)
THREEWAY_SEED = 20261002
"""Base seed of the main study (packet ``M``, master array 100 by 100 by 10)."""
THREEWAY_DESIGNS: dict[str, str] = {
    "wz1": "conditional variance 1",
    "wz2": "conditional variance equal to the mean",
    "wz3": "conditional variance equal to the squared mean",
    "wz4": "conditional variance 0.5 mean + 0.5 exp(2 x) mean^2",
}
"""The four variance functions of the tables of Weidner and Zylkin (2021)."""
_PACKET = "M"
_INNOVATION_SCHEME = "philox4x64-sha256key-boxmuller/v1"
_SD_EFFECT, _SD_NU, _AR_RHO, _BETA0, _X_LAG = 1.0 / 16.0, 0.5, 0.3, 1.0, 0.5


def _philox_key(seed: int, draw: int) -> npt.NDArray[np.uint64]:
    """Key of the random stream of one replication (``philox_key``)."""
    message = f"wz_fe_sim/innovations/v1|{int(seed)}|{_PACKET}|{int(draw)}".encode()
    digest = hashlib.sha256(message).digest()
    return np.frombuffer(digest[:16], dtype="<u8").astype(np.uint64)


def _box_muller(bitgen: np.random.Philox, n: int) -> FloatArray:
    """Standard normals from raw 64-bit words.

    The stream depends on the Philox bit generator and not on the algorithm of
    ``Generator.normal``.
    """
    m = (n + 1) // 2
    words = bitgen.random_raw(2 * m)
    u = ((words >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0**-53
    u1, u2 = u[0::2], u[1::2]
    rad = np.sqrt(-2.0 * np.log(u1))
    ang = 2.0 * np.pi * u2
    z = np.empty(2 * m)
    z[0::2] = rad * np.cos(ang)
    z[1::2] = rad * np.sin(ang)
    return z[:n]


def master_innovations(
    seed: int, draw: int, countries: int, years: int
) -> tuple[dict[str, FloatArray], str]:
    """The five innovation arrays of one replication and their hash.

    Port of ``master_innovations``.  The hash is the ``draw_hash`` stored in the
    archive of the study.
    """
    bitgen = np.random.Philox(key=_philox_key(seed, draw))
    shapes = {
        "alpha": (countries, years),
        "gamma": (countries, years),
        "eta": (countries, countries),
        "nu": (countries, countries, years),
        "eps": (countries, countries, years),
    }
    arrays = {}
    digest = hashlib.sha256(f"{_INNOVATION_SCHEME}|{countries}|{years}".encode())
    for name, shape in shapes.items():
        arrays[name] = _box_muller(bitgen, math.prod(shape)).reshape(shape)
        digest.update(np.ascontiguousarray(arrays[name], dtype="<f8").tobytes())
    return arrays, digest.hexdigest()


def _threeway_master(
    innovations: Mapping[str, FloatArray],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Regressor, log mean and standardized disturbance.

    The regressor follows ``x[t] = x[t-1] / 2 + a[i,t] + g[j,t] + nu[i,j,t]`` and the
    disturbance is a stationary AR(1) with coefficient 0.3 and unit variance.
    """
    a = _SD_EFFECT * innovations["alpha"]
    g = _SD_EFFECT * innovations["gamma"]
    e = _SD_EFFECT * innovations["eta"]
    nu = _SD_NU * innovations["nu"]
    eps = innovations["eps"]
    years = nu.shape[2]
    x = np.empty(nu.shape)
    z = np.empty(nu.shape)
    x[:, :, 0] = a[:, None, 0] + g[None, :, 0] + nu[:, :, 0]
    z[:, :, 0] = eps[:, :, 0]
    sq = np.sqrt(1.0 - _AR_RHO**2)
    for t in range(1, years):
        x[:, :, t] = (
            _X_LAG * x[:, :, t - 1] + a[:, None, t] + g[None, :, t] + nu[:, :, t]
        )
        z[:, :, t] = _AR_RHO * z[:, :, t - 1] + sq * eps[:, :, t]
    loglam = a[:, None, :] + g[None, :, :] + e[:, :, None] + _BETA0 * x
    return x, loglam, z


def _variance_ratio(design: str, lam: FloatArray, x: FloatArray) -> FloatArray:
    """``Var(y | .) / mean^2`` of the four designs (``variance_ratio``, tables)."""
    if design == "wz1":
        return lam**-2.0
    if design == "wz2":
        return 1.0 / lam
    if design == "wz3":
        return np.ones_like(lam)
    return 0.5 / lam + 0.5 * np.exp(2.0 * x)


def _long_frame(x: FloatArray, y: FloatArray, lam: FloatArray) -> pd.DataFrame:
    """Arrays indexed by exporter, importer and year, as one row per observation."""
    countries, _, years = x.shape
    labels = np.arange(1, countries + 1)
    exporter = np.repeat(labels, countries * years)
    importer = np.tile(np.repeat(labels, years), countries)
    return pd.DataFrame(
        {
            "exporter": exporter,
            "importer": importer,
            "year": np.tile(np.arange(1, years + 1), countries * countries),
            "pair": (exporter - 1) * countries + importer,
            "x": x.ravel(),
            "y": y.ravel(),
            "true_mean": lam.ravel(),
        }
    )


def threeway_data(
    design: str,
    *,
    countries: int = 20,
    years: int = 5,
    zeros: float = 0.0,
    seed: int = THREEWAY_SEED,
    draw: int = 0,
    master: tuple[int, int] = (100, 10),
    self_pairs: bool = True,
) -> Synthetic:
    """Draw one dyadic panel of the three-way fixed-effect simulation.

    The panel takes the first ``countries`` exporters and importers and the first
    ``years`` periods of the master array, as the study does.  With the default seed
    and master array and ``draw`` from 1 to 5000, the result is a panel of the main
    study.  The extension with 500 countries used seed 20261004 and a master array
    of 500 countries and 2 years.

    Parameters
    ----------
    design : str
        A key of ``THREEWAY_DESIGNS``.
    countries, years : int
        Size of the panel; at most the size of the master array.
    zeros : float
        Share of outcomes set to zero, with the mean kept.  The designs themselves
        have positive outcomes.
    seed, draw : int
        The Philox key is the first 16 bytes of the SHA-256 of
        ``"wz_fe_sim/innovations/v1|<seed>|M|<draw>"``.
    master : tuple of int
        Countries and years of the master array.  Changing it changes the random
        numbers.
    self_pairs : bool
        Keep the pairs of a country with itself, as the study does.

    Returns
    -------
    Synthetic
        Long format with ``exporter``, ``importer``, ``year``, ``pair``, ``x``,
        ``y`` and ``true_mean``, sorted by exporter, then importer, then year.  The
        true coefficient of ``x`` is one.
    """
    require_design(THREEWAY_DESIGNS, design, "threeway")
    if not (1 <= countries <= master[0] and 1 <= years <= master[1]):
        raise ValueError("countries and years must be between 1 and the master sizes")
    innovations, draw_hash = master_innovations(seed, draw, master[0], master[1])
    x_master, loglam_master, z_master = _threeway_master(innovations)
    cell = (slice(0, countries), slice(0, countries), slice(0, years))
    x, loglam, z = x_master[cell], loglam_master[cell], z_master[cell]
    lam = np.exp(loglam)
    s2 = np.log1p(_variance_ratio(design, lam, x))
    y = lam * np.exp(-0.5 * s2 + np.sqrt(s2) * z)

    frame = _long_frame(x, y, lam)
    if not self_pairs:
        frame = frame[frame["exporter"] != frame["importer"]].reset_index(drop=True)
    number = list(THREEWAY_DESIGNS).index(design) + 1
    zero_key = [seed, draw, number, countries, years, 3]
    frame["y"] = inflate_zeros(frame["y"].to_numpy(), zeros, seeded(zero_key))
    effects = (
        "pair + exporter^year + importer^year" if years > 1 else "exporter + importer"
    )
    return Synthetic(
        design=f"threeway/{design}",
        frame=frame,
        formula=f"y ~ x | {effects}",
        truth={"x": _BETA0},
        columns={
            "exporter": "origin country",
            "importer": "destination country",
            "year": "period",
            "pair": "exporter-importer pair",
            "x": "regressor, autoregressive within pair, true coefficient 1",
            "y": f"outcome; {THREEWAY_DESIGNS[design]}",
            "true_mean": "E(y | x, effects)",
        },
        seed=(
            f"Philox key from sha256('wz_fe_sim/innovations/v1|{seed}|{_PACKET}|"
            f"{draw}'); master array {master[0]} x {master[0]} x {master[1]}; "
            "Box-Muller normals"
        ),
        source=THREEWAY_SOURCE,
        fingerprint=draw_hash,
        notes=(
            "The fingerprint is the hash of the master innovations of the "
            "replication; it does not depend on the design or on the panel size.",
            "With one year the panel is a complete two-way array.",
        ),
    )
