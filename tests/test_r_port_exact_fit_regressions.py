"""Public fits with independent roots preserved by the R port review."""

from itertools import product
from math import atanh, log, log1p
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import powermean as pm


@pytest.mark.parametrize(
    ("direction", "retry", "bias"),
    [(1, None, False), (1, 0, False), (1, 0, True), (-1, 0, False)],
    ids=["default", "no-retry", "wz-bias", "negative"],
)
@pytest.mark.filterwarnings(
    "ignore:Covariance diagnostics:powermean.inference.vcov.CovarianceWarning"
)
def test_cancelled_ppml_wz_certifies_a_dyadic_exact_fit(direction, retry, bias):
    sign = [1, 1, -1, -1]
    epsilon = direction * 2.0**-16
    data = pd.DataFrame(
        [
            (i, j, t, 4 * i + j, sign[i] * sign[j] * sign[t])
            for i, j, t in product(range(4), repeat=3)
        ],
        columns=["exporter", "importer", "year", "pair", "x"],
    )
    data["y"] = 1 + epsilon * data["x"]
    options = {
        "tau": 1,
        "cancel": "pair",
        "vcov": {"cluster": "pair", "adjustment": "wz"},
        "ssc": "none",
        "on_failure": "return",
    }
    if retry is not None:
        options["solver_options"] = {"retry": retry}
    if bias:
        options["bias_correction"] = "wz"
    fit = pm.fit("y ~ 0 + x | pair + exporter^year + importer^year", data, **options)

    assert fit.provenance.certified, fit.provenance.status
    assert fit.sample.positions.tolist() == list(range(64))
    assert fit.coef()["x"] == pytest.approx(atanh(epsilon), abs=2e-14)
    np.testing.assert_array_equal(
        fit.certificate.measures.coefficients.corrections, [0.0]
    )
    np.testing.assert_array_equal(fit.vcov(), [[0.0]])
    if bias:
        predicted = fit.predict()
        assert predicted.attrs["nuisance_correction"] == "not_supplied"
        np.testing.assert_allclose(predicted, data["y"], rtol=1e-12)


@pytest.mark.parametrize(
    ("case", "effects", "options", "excluded", "expected"),
    [
        pytest.param(
            "00242",
            "g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8",
            {"omega": 1.0, "variance": pm.NB2(0.3), "vcov": {"cluster": "pair + year"}},
            (23, 51, 79),
            [0.0706849342431531, -0.6526284445659194],
            id="fuzz-00242",
        ),
        pytest.param(
            "00467",
            "g8 + g7 + g6 + g5 + g4 + g3 + g2 + g1",
            {"omega": 0.5, "tau": 2.0},
            (1, 42, 75),
            [-117.07742422662969, 0.35908491761604555],
            id="fuzz-00467",
        ),
        pytest.param(
            "00908",
            "g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8",
            {"omega": 0.5, "tau": 1.25, "vcov": {"cluster": "pair"}},
            (),
            [-0.35165259096786017, -0.6715948463203323],
            id="fuzz-00908",
        ),
        pytest.param(
            "02964",
            "g3 + g2 + g1",
            {"omega": 1.0, "tau": 1.75, "vcov": {"cluster": "pair + year"}},
            (),
            [0.05709007404520906, 0.07619480620464354],
            id="fuzz-02964",
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:Low fitted means:powermean.errors.NumericalWarning")
@pytest.mark.filterwarnings(
    "ignore:The signed multiway covariance:powermean.errors.CovarianceWarning"
)
def test_joint_convex_fit_certifies_the_explicit_dummy_root(
    case, effects, options, excluded, expected
):
    data = pd.read_csv(
        Path(__file__).parent / "data" / f"fuzz-{case}.csv",
        dtype={"pair": str},
        index_col="_row_label",
        converters={"x": float, "z": float, "offset": float, "y": float},
    )
    fit = pm.fit(
        f"y ~ 0 + x + z | {effects}",
        data,
        offset="offset",
        on_failure="return",
        **options,
    )
    assert fit.provenance.certified, fit.provenance.status
    assert fit.sample.positions.tolist() == [
        row for row in range(len(data)) if row not in excluded
    ]
    np.testing.assert_allclose(
        fit.coef().loc[["x", "z"]], expected, rtol=1e-7, atol=1e-10
    )


@pytest.mark.parametrize("change", [0.0, -(2.0**-48)], ids=["original", "perturbed"])
@pytest.mark.filterwarnings(
    "ignore:Covariance diagnostics:powermean.inference.vcov.CovarianceWarning"
)
def test_dense_ppml_certifies_the_frequency_weighted_exposure_root(change):
    data = pd.DataFrame(
        {
            "y": [0, 1, 2, 3, 2, 3, 4],
            "x": [1, 0, 0, 0, 1, 1, 1],
            "w": [1, 1, 2, 3, 4, 1, 3],
            "exposure": [3.0, 1.0 + change, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    fit = pm.fit(
        "y ~ x",
        data,
        exposure="exposure",
        weights="w",
        weight_type="frequency",
        vcov="robust",
        ssc="none",
        on_failure="return",
    )
    assert fit.provenance.certified, fit.provenance.status
    assert fit.sample.positions.tolist() == list(range(7))
    intercept = -log1p(change / 14)
    expected = [intercept, log(23 / 18) - intercept]
    np.testing.assert_allclose(fit.coef().loc[["Intercept", "x"]], expected, atol=1e-12)
