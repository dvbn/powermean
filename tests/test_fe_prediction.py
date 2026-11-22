"""Public predictions reuse the fitted slopes and estimation sample."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

import powermean as pm
from powermean.results import prediction
from powermean.results.prediction import _adjust_index
from powermean.results.two_way_prediction import TwoWayPrediction


def _panel(*, directed=False):
    rng = np.random.default_rng(82)
    rows, cols = (20, 20) if directed else (14, 20)
    cells = [
        (i, j) for i, j in product(range(rows), range(cols)) if not directed or i != j
    ]
    i, j = np.asarray(cells).T
    x = rng.normal(size=len(i))
    d = rng.binomial(1, 0.4, size=len(i))
    a, b = rng.normal(0, 0.3, rows), rng.normal(0, 0.3, cols)
    y = rng.poisson(np.exp(0.85 - 0.27 * x + 0.18 * d + a[i] + b[j]))
    return pd.DataFrame({"y": y, "x": x, "d": d, "i": i, "j": j})


@pytest.mark.parametrize("method", ["gmm1", "gmm2"])
@pytest.mark.parametrize("directed", [False, True], ids=["panel", "dyads"])
def test_gmm_recovery_matches_both_observed_margins(method, directed):
    data = _panel(directed=directed)
    formula = "y ~ 0 + center(x) + d | i + j"
    indices = np.arange(len(data)) // 2
    data.index = pd.Index(indices, name="original")
    if not directed:
        data.loc[data.i == 0, "y"] = 0
    layout = "directed" if directed else "rectangular"
    fit = pm.fit(formula, data, estimator=method, fe_layout=layout)
    beta = fit.coef()
    covariance = fit.vcov()
    y = data.y.to_numpy().copy()
    yhat = fit.predict()
    effects = fit.predict(type="fixed_effects")
    for key in ("i", "j"):
        pd.testing.assert_series_equal(effects[key], fit.fixef()[key])
    assert yhat.index.equals(data.index)
    np.testing.assert_allclose(fit.predict(type="residual"), y - yhat)
    for key in ("i", "j"):
        totals = pd.Series(np.asarray(yhat)).groupby(data[key].to_numpy()).sum()
        np.testing.assert_allclose(
            totals, data.groupby(key).y.sum(), rtol=1e-10, atol=1e-9
        )
    zero = (data.i.to_numpy() == 0) & (not directed)
    assert tuple(np.flatnonzero(zero)) == yhat.attrs["zero_boundary_rows"]
    if not directed:
        assert np.isnan(effects["i"].loc[0])
    offset = (data.x - data.x.mean()) * beta["center(x)"] + data.d * beta["d"]
    mu = np.exp(offset + data.i.map(effects["i"]) + data.j.map(effects["j"]))
    np.testing.assert_allclose(yhat[~zero], mu[~zero], rtol=1e-12)

    covariates = data[["x", "d"]].assign(x=data.x + 0.4)
    counterfactual = fit.predict(covariates)
    np.testing.assert_allclose(counterfactual, yhat * np.exp(0.4 * beta["center(x)"]))
    data["y"] = 1
    pd.testing.assert_series_equal(fit.predict(), yhat)
    pd.testing.assert_frame_equal(fit.vcov(), covariance)
    pd.testing.assert_series_equal(fit.with_vcov("stata").predict(), yhat)


@pytest.mark.parametrize(
    ("variance", "omega"),
    [
        ({"tau": 1}, 1.0),
        ({"tau": 1.5}, 0.7),
        ({"tau": 2}, 1.0),
        ({"variance": pm.NB2(0.3)}, 1.0),
        ({"tau": 2.3}, 1.0),
    ],
    ids=["ppml", "interior", "gpml", "nb2", "signed"],
)
def test_cancelled_predict_uses_existing_scales_and_retains_offsets(variance, omega):
    rng = np.random.default_rng(260)
    data = pd.DataFrame(
        [(i, j) for i, j in product(range(40), range(4))], columns=["i", "j"]
    )
    data["x"] = rng.normal(size=len(data))
    data["offset"] = rng.normal(0.0, 0.2, len(data))
    offset = (data.offset + 0.05 * data.x).to_numpy()
    index = 0.2 * data.x + offset + rng.normal(0, 0.4, 40)[data.i]
    data["y"] = np.exp(index) * rng.gamma(8, 1 / 8, len(data))
    data.index = pd.Index(np.arange(len(data)) + 1000, name="original")
    model = pm.fit(
        "y ~ 0 + x | i + j",
        data,
        omega=omega,
        cancel="i",
        offset="I(offset + 0.05 * x)",
        cluster="i",
        ssc="none",
        **variance,
    )
    yhat = model.predict()
    effects = model.predict(type="fixed_effects")
    for key in ("i", "j"):
        pd.testing.assert_series_equal(effects[key], model.fixef()[key])
    assert yhat.index.equals(data.index)
    expected = np.exp(
        data.x * model.coef()["x"]
        + offset
        + data.i.map(effects["i"])
        + data.j.map(effects["j"])
    )
    np.testing.assert_allclose(yhat, expected, rtol=5e-13)
    residual = model.predict(type="residual")
    np.testing.assert_allclose(residual, data.y**omega - expected**omega, atol=1e-12)
    cf = model.predict(data[["x"]].assign(x=data.x + 0.5))
    np.testing.assert_allclose(cf, yhat * np.exp(0.5 * model.coef()["x"]), rtol=5e-13)
    with pytest.raises(pm.SpecificationError, match="fitted identifiers"):
        model.predict(data.assign(j=-1))


def test_joint_counterfactual_uses_fitted_rows_and_frozen_formula():
    data = _panel()
    data["y"] += 1
    data.loc[data.i == 0, "y"] = 0
    data.loc[0, "y"] = np.nan
    data.loc[data.index[-1], "i"] = 99
    fit = pm.fit("y ~ 0 + center(x) + d | i + j", data, tau=1, ssc="none")
    kept = data.iloc[fit.sample.positions]
    assert 0 not in kept.i.to_numpy()
    assert 99 not in kept.i.to_numpy()
    yhat = fit.predict()
    changed = kept.assign(x=kept.x + 0.25)
    np.testing.assert_allclose(
        fit.predict(changed), yhat * np.exp(0.25 * fit.coef()["center(x)"])
    )
    np.testing.assert_allclose(fit.predict(type="residual"), kept.y - yhat)
    with pytest.raises(pm.SpecificationError, match="retained index"):
        fit.predict(changed.iloc[::-1])
    with pytest.raises(pm.SpecificationError, match="None for residual"):
        fit.predict(changed, type="residual")


@pytest.mark.parametrize("with_fe", [False, True])
def test_log_ols_residual_uses_log_outcome(with_fe):
    data = _panel()
    data["y"] += 1
    formula = "y ~ x + d" + (" | i + j" if with_fe else "")
    fit = pm.logols(formula, data)
    np.testing.assert_allclose(
        fit.predict(type="residual"), np.log(data.y) - np.log(fit.predict()), atol=1e-14
    )


def test_uncertified_cancelled_fit_withholds_all_prediction_types():
    data = _panel()
    fit = pm.fit(
        "y ~ 0 + x + d | i + j",
        data,
        tau=2,
        cancel="i",
        cluster="i",
        solver_options={"max_iter": 1, "retry": 0},
        on_failure="return",
    )
    assert not fit.provenance.certified
    for kind in ("power_mean", "fixed_effects", "residual"):
        with pytest.raises(pm.ConvergenceError):
            fit.predict(type=kind)


def test_counterfactual_differences_covariates_before_summing():
    original = np.array([[2.0**54, 0.0]])
    changed = np.array([[2.0**54, 1.0]])
    np.testing.assert_array_equal(
        _adjust_index(np.zeros(1), changed, original, np.ones(2), 1.0), [1.0]
    )
    np.testing.assert_allclose(
        _adjust_index(
            np.zeros(1),
            np.array([[1e308]]),
            np.array([[-1e308]]),
            np.array([1e-308]),
            1.0,
        ),
        [2.0],
        rtol=1e-15,
    )


@pytest.mark.parametrize(
    ("q", "original", "changed", "beta", "expected"),
    [
        (-1.0, [0.0, 1.0], [2.0**54, 2.0**54], [1.0, -1.0], 0.0),
        (0.0, [-8e307, 8e307], [8e307, -8e307], [2.0, 2.0], 0.0),
        (-(2.0**54), [0.0, 0.0], [2.0**54, 1.0], [1.0, 1.0], 1.0),
    ],
)
def test_counterfactual_accounts_for_cancellation_and_intermediate_overflow(
    q, original, changed, beta, expected
):
    np.testing.assert_array_equal(
        _adjust_index(
            np.array([q]),
            np.array([changed]),
            np.array([original]),
            np.array(beta),
            1.0,
        ),
        [expected],
    )


def test_wide_counterfactual_uses_vectorized_arithmetic(monkeypatch):
    def unexpected_fallback(*_args):
        pytest.fail("a well-conditioned wide design should use the vectorized path")

    monkeypatch.setattr(prediction, "Fraction", unexpected_fallback)
    original = np.zeros((32, 253))
    changed = original.copy()
    changed[:, 0] = 0.5
    q = np.full(len(original), 2.0)
    beta = np.ones(original.shape[1])
    np.testing.assert_array_equal(_adjust_index(q, original, original, beta, 1.0), q)
    np.testing.assert_array_equal(
        _adjust_index(q, changed, original, beta, 1.0), q + 0.5
    )


def test_counterfactual_scales_subnormal_order_before_rounding():
    tiny = np.nextafter(0.0, 1.0)
    np.testing.assert_array_equal(
        _adjust_index(
            np.array([17 * tiny]),
            np.array([[1 / 32]]),
            np.zeros((1, 1)),
            np.ones(1),
            16 * tiny,
        ),
        [18 * tiny],
    )


def test_counterfactual_detects_overflow_hidden_by_rounding():
    limit = np.finfo(np.float64).max
    with pytest.raises(FloatingPointError, match="not representable"):
        _adjust_index(
            np.zeros(1),
            np.ones((1, 1)),
            np.array([[-(2.0**-53)]]),
            np.array([limit]),
            1.0,
        )


def test_two_way_zero_boundary_ignores_an_overflowing_slope_index():
    data = _panel()
    data["x"] *= 2.0**-20
    zero = data.i == 0
    data.loc[zero, "y"] = 0
    model = pm.fit("y ~ 0 + x + d | i + j", data, estimator="gmm2")
    changed = data.copy()
    changed.loc[zero, "x"] = 1e308
    np.testing.assert_array_equal(model.predict(changed), model.predict())


def test_gmm_fixed_effect_labels_keep_subnanosecond_identifiers():
    data = _panel()
    data["i"] = pd.Series(
        [np.datetime64(int(code) + 1, "ps") for code in data.i],
        dtype=object,
    )
    model = pm.fit("y ~ 0 + x + d | i + j", data, estimator="gmm1")
    labels = model.fixef()["i"].index
    assert labels.is_unique
    assert labels.dtype == object
    assert list(labels) == data.i.drop_duplicates().tolist()
    np.testing.assert_array_equal(model.predict(data), model.predict())


def test_invalid_gmm_counterfactual_is_rejected_before_recovery(monkeypatch):
    def fail(_):
        raise AssertionError("invalid counterfactual triggered FE recovery")

    data = _panel()
    model = pm.fit("y ~ 0 + x + d | i + j", data, estimator="gmm1")
    monkeypatch.setattr(TwoWayPrediction, "recovered", property(fail))
    with pytest.raises(pm.SpecificationError, match="retained index"):
        model.predict(data.iloc[::-1])
    with pytest.raises(pm.SpecificationError, match="fitted identifiers"):
        model.predict(data.assign(j=-1))


def test_gmm_predictions_are_invariant_to_a_large_common_regressor_level():
    data = _panel(directed=True)
    data["x"] = np.round(data.x * 32) / 32
    formula = "y ~ 0 + x + d | i + j"
    options = {"estimator": "gmm2", "fe_layout": "directed"}
    fitted = pm.fit(formula, data, **options)
    shifted = data.assign(x=data.x + 2**40)
    same_fit = pm.fit(formula, shifted, **options)
    np.testing.assert_allclose(same_fit.predict(), fitted.predict(), rtol=1e-12)
    np.testing.assert_allclose(
        same_fit.predict(shifted.assign(x=shifted.x + 0.5)),
        fitted.predict(data.assign(x=data.x + 0.5)),
        rtol=1e-12,
    )
