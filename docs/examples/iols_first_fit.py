"""Illustrate no-FE IOLS on simulated data and report the actual implementation.

Run with an installed compatible Powermean build. The fixed-seed example is
not a coverage or speed study.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import powermean as pm


def main() -> None:
    """Fit the documented example without concealing failures or backend choice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cpp", "numpy"), default="auto")
    args = parser.parse_args()

    rng = np.random.default_rng(20261028)
    n = 240
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    mean = np.exp(0.3 + 0.25 * x1 - 0.15 * x2)
    data = pd.DataFrame({"y": rng.poisson(mean), "x1": x1, "x2": x2})
    options: dict[str, object] = {}
    if args.backend != "auto":
        options["iols_backend"] = args.backend

    result = pm.fit(
        "y ~ 1 + x1 + x2", data, omega=1, tau=1, solver="iols",
        solver_options=options, vcov="robust",
    )
    print(result.tidy().to_string(index=False))
    provenance = result.provenance
    print(f"Solver: {provenance.solver}")
    if provenance.trace is None:
        raise RuntimeError("The IOLS example returned no solver trace")
    for name in ("requested_backend", "actual_backend", "fallback_used"):
        print(f"{name}: {provenance.trace[name]}")


if __name__ == "__main__":
    main()
