"""Check the installed distribution and the public examples on each wheel."""

import os
import runpy
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

# Keep test time and reduction order independent of hardware thread count.
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

import powermean as pm
from powermean._native.basis import _echelon_cpp
from powermean._native.iols import _kernel_cpp


def main():
    package = distribution("powermean")
    assert package.version == "0.1.1"
    installed = Path(package.locate_file("powermean")).resolve()
    for module in (pm, _kernel_cpp, _echelon_cpp):
        assert Path(module.__file__).resolve().is_relative_to(installed)

    data = pd.DataFrame({"y": [1., 3., 2., 6.], "x": [0., 0., 1., 1.]})
    for tau in (1, 1.5, 2):
        fit = pm.fit("y ~ x", data, omega=1, tau=tau)
        assert fit.provenance.certified and fit.sample.count == 4
        np.testing.assert_allclose(fit.coef().loc["x"], np.log(2), rtol=1e-7)
        np.testing.assert_allclose(fit.predict(), [2., 2., 4., 4.], rtol=1e-7)

    gravity = runpy.run_path("examples/first_fit.py")
    trade = gravity["gravity_data"]()
    fit = pm.fit(gravity["FORMULA"], trade, omega=1, tau=1, cluster="pair")
    assert fit.provenance.certified and fit.sample.count == 1056 and fit.sample.zeros == 63
    np.testing.assert_allclose(
        [fit.coef().loc["agreement"], fit.se().loc["agreement"]],
        [0.431454536113, 0.0563498608], rtol=1e-7,
    )
    short = runpy.run_path("examples/short_panel.py", run_name="__main__")
    firms = short["firm_data"]()
    cancelled = pm.fit("sales ~ 0 + price_index | firm + year", firms,
                       omega=1, tau=2, cancel="firm", cluster="firm")
    assert len(cancelled.predict()) == cancelled.sample.count
    assert "firm" in cancelled.predict(type="fixed_effects")
    runpy.run_path("examples/nonlinear_did.py", run_name="__main__")
    jochmans = runpy.run_path("examples/jochmans.py", run_name="__main__")
    panel = jochmans["trade_panel"](14, 20, directed=False)
    gmm = pm.fit("trade ~ 0 + cost + agreement | origin + destination",
                 panel, estimator="gmm2")
    np.testing.assert_allclose(
        gmm.predict(type="residual").groupby(panel["origin"]).sum(), 0, atol=1e-8,
    )
    example = next(path for path in (
        Path("iols_first_fit.py"), Path("docs/examples/iols_first_fit.py"),
    ) if path.is_file())
    iols = subprocess.run([sys.executable, "-I", example, "--backend", "cpp"],
                          check=True, text=True, stdout=subprocess.PIPE).stdout
    assert "actual_backend: cpp" in iols and "fallback_used: False" in iols
    print("Verified the native extensions, three working variances, gravity, DiD, Jochmans and strict C++ IOLS.")


if __name__ == "__main__":
    main()
