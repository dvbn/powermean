"""Quantities computed from a fit, against their definitions."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import powermean as pm

from conftest import design

FORMULA = "y ~ 1 + x1 + x2 + d"
NAMES = ["x1", "x2", "d"]


@pytest.fixture(scope="module")
def result(counts):
    return pm.ppml(FORMULA, counts)


def test_effect_is_the_proportional_change(result):
    effect = result.effect("d")
    assert effect.estimate == pytest.approx(np.expm1(result.coef()["d"]), rel=1e-12)
    # delta method: the derivative of exp(b) - 1 is exp(b)
    assert effect.std_error == pytest.approx(np.exp(result.coef()["d"]) * result.se()["d"], rel=1e-9)


def test_predict_in_sample_and_on_new_data(result, counts):
    X = design(counts, NAMES)
    np.testing.assert_allclose(result.predict().to_numpy(), np.exp(X @ result.coef().to_numpy()), rtol=1e-12)
    new = pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, 0.0], "d": [0.0, 1.0]})
    beta = result.coef()
    expected = np.exp(beta["Intercept"] + beta["x1"] * new["x1"] + beta["x2"] * new["x2"] + beta["d"] * new["d"])
    np.testing.assert_allclose(result.predict(new).to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_linear_combination(result):
    weights = np.array([0.0, 2.0, 0.0, 1.0])
    table = result.lincom("d + 2*x1").to_frame()
    assert table["estimate"].iloc[0] == pytest.approx(weights @ result.coef().to_numpy(), rel=1e-12)
    expected_se = np.sqrt(weights @ result.vcov().to_numpy() @ weights)
    assert table["std_error"].iloc[0] == pytest.approx(expected_se, rel=1e-10)


def test_wald_test_of_two_coefficients(result):
    names = list(result.coef().index)
    R = np.zeros((2, len(names)))
    R[0, names.index("x1")] = 1.0
    R[1, names.index("x2")] = 1.0
    value = R @ result.coef().to_numpy()
    chi2 = float(value @ np.linalg.solve(R @ result.vcov().to_numpy() @ R.T, value))
    table = result.wald(R, np.zeros(2)).to_frame()
    row = table.iloc[0]
    if str(row["distribution"]).lower().startswith("f"):
        assert row["statistic"] == pytest.approx(chi2 / 2, rel=1e-9)
    else:
        assert row["statistic"] == pytest.approx(chi2, rel=1e-9)
        assert row["p_value"] == pytest.approx(stats.chi2.sf(chi2, 2), rel=1e-8, abs=1e-300)


def test_changing_the_covariance_keeps_the_coefficients(counts):
    clustered = pm.ppml(FORMULA, counts, cluster="firm")
    robust = clustered.with_vcov("robust")
    np.testing.assert_array_equal(robust.coef().to_numpy(), clustered.coef().to_numpy())
    np.testing.assert_allclose(robust.se().to_numpy(), pm.ppml(FORMULA, counts).se().to_numpy(), rtol=1e-12)


def test_heteroskedasticity_consistent_variants_are_ordered(result):
    hc0 = np.diag(result.vcov("HC0").to_numpy())
    hc1 = np.diag(result.vcov("HC1").to_numpy())
    hc3 = np.diag(result.vcov("HC3").to_numpy())
    assert np.all(hc1 > hc0)
    assert np.all(hc3 > hc0)


def test_influence_contributions_reproduce_hc0(result):
    contributions = result.influence().to_numpy()
    np.testing.assert_allclose(contributions.T @ contributions, result.vcov("HC0").to_numpy(), rtol=1e-9, atol=1e-14)


def test_tidy_table(result):
    table = result.tidy()
    assert list(table["term"]) == ["Intercept", "x1", "x2", "d"]
    np.testing.assert_allclose(table["estimate"].to_numpy(), result.coef().to_numpy())
    np.testing.assert_allclose(table["std_error"].to_numpy(), result.se().to_numpy())
