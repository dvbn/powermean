"""Estimates against computations written out by hand in NumPy."""

import numpy as np
import pandas as pd
import pytest

import powermean as pm

from conftest import design

FORMULA = "y ~ 1 + x1 + x2 + d"
NAMES = ["x1", "x2", "d"]


def poisson_newton(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Poisson pseudo-maximum likelihood by Newton's method, from the log of the mean."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.mean())
    for _ in range(100):
        mu = np.exp(X @ beta)
        step = np.linalg.solve(X.T @ (X * mu[:, None]), X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-13:
            break
    return beta


def test_ppml_coefficients_and_robust_standard_errors(counts):
    X, y = design(counts, NAMES), counts["y"].to_numpy()
    beta = poisson_newton(X, y)
    mu = np.exp(X @ beta)
    bread = np.linalg.inv(X.T @ (X * mu[:, None]))
    scores = X * (y - mu)[:, None]
    n = len(y)
    expected = bread @ (scores.T @ scores) @ bread * n / (n - 1)

    result = pm.ppml(FORMULA, counts)

    np.testing.assert_allclose(result.coef().to_numpy(), beta, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(result.vcov().to_numpy(), expected, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(result.se().to_numpy(), np.sqrt(np.diag(expected)), rtol=1e-7)


def test_ppml_first_order_condition(counts):
    X, y = design(counts, NAMES), counts["y"].to_numpy()
    result = pm.ppml(FORMULA, counts)
    mu = np.exp(X @ result.coef().to_numpy())
    assert np.max(np.abs(X.T @ (y - mu))) < 1e-7


def test_gpml_first_order_condition_and_robust_covariance(positive):
    X, y = design(positive, NAMES), positive["y"].to_numpy()
    result = pm.gpml(FORMULA, positive)
    mu = np.exp(X @ result.coef().to_numpy())
    residual = y / mu - 1.0
    assert np.max(np.abs(X.T @ residual)) < 1e-7

    jacobian = X.T @ (X * (y / mu)[:, None])
    bread = np.linalg.inv(jacobian)
    scores = X * residual[:, None]
    n = len(y)
    expected = bread @ (scores.T @ scores) @ bread.T * n / (n - 1)
    np.testing.assert_allclose(result.vcov().to_numpy(), expected, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("tau", [1.0, 1.5, 2.0])
def test_power_variance_first_order_condition(counts, tau):
    X, y = design(counts, NAMES), counts["y"].to_numpy()
    result = pm.fit(FORMULA, counts, omega=1, tau=tau)
    mu = np.exp(X @ result.coef().to_numpy())
    score = X.T @ ((y - mu) * mu ** (1.0 - tau))
    assert np.max(np.abs(score)) < 1e-6


@pytest.mark.parametrize("tau", [1.0, 1.5, 2.0])
def test_iols_agrees_with_newton(counts, tau):
    newton = pm.fit(FORMULA, counts, omega=1, tau=tau)
    iols = pm.fit(FORMULA, counts, omega=1, tau=tau, solver="iols")
    np.testing.assert_allclose(iols.coef().to_numpy(), newton.coef().to_numpy(), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(iols.se().to_numpy(), newton.se().to_numpy(), rtol=1e-6)


def test_log_ols_is_least_squares_on_the_log_outcome(positive):
    X, y = design(positive, NAMES), positive["y"].to_numpy()
    expected, *_ = np.linalg.lstsq(X, np.log(y), rcond=None)
    result = pm.logols(FORMULA, positive)
    np.testing.assert_allclose(result.coef().to_numpy(), expected, rtol=1e-9, atol=1e-11)


def test_log_ols_drops_zeros_only_on_request(counts):
    with pytest.raises(pm.SampleError):
        pm.logols(FORMULA, counts)
    result = pm.logols(FORMULA, counts, positive_only=True)
    assert result.sample.count == int((counts["y"] > 0).sum())


def test_cluster_covariance(counts):
    X, y = design(counts, NAMES), counts["y"].to_numpy()
    beta = poisson_newton(X, y)
    mu = np.exp(X @ beta)
    bread = np.linalg.inv(X.T @ (X * mu[:, None]))
    scores = pd.DataFrame(X * (y - mu)[:, None]).groupby(counts["firm"].to_numpy()).sum().to_numpy()
    groups = scores.shape[0]
    expected = bread @ (scores.T @ scores) @ bread

    result = pm.ppml(FORMULA, counts, cluster="firm")
    reported = result.vcov()

    np.testing.assert_allclose(reported.to_numpy(), expected * reported.attrs["factor"], rtol=1e-7, atol=1e-12)
    assert reported.attrs["factor"] == pytest.approx(groups / (groups - 1))


def test_prepared_sample_gives_the_same_fit(counts):
    prepared = pm.prepare(FORMULA, counts)
    direct = pm.gpml(FORMULA, counts)
    np.testing.assert_array_equal(prepared.fit(omega=1, tau=2).coef().to_numpy(), direct.coef().to_numpy())


def test_readme_example():
    rng = np.random.default_rng(20261028)
    n = 240
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    mean = np.exp(0.3 + 0.25 * x1 - 0.15 * x2)
    data = pd.DataFrame({"y": rng.poisson(mean), "x1": x1, "x2": x2})
    result = pm.ppml("y ~ 1 + x1 + x2", data, vcov="robust")
    assert int((data["y"] == 0).sum()) == 66
    assert result.coef()["x1"] == pytest.approx(0.224509, abs=5e-7)
    assert result.se()["x1"] == pytest.approx(0.053047, abs=5e-7)
