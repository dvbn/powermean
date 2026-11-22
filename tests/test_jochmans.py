"""Independent literal products, derivatives and projections on tiny arrays."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import root
from scipy.stats import norm

import powermean as pm
from powermean.solve.jochmans import DifferenceMoments

FORMULA = "y ~ 0 + x2 + x1 | left + right"
METHODS = ("gmm1", "gmm2")
LAYOUTS = ("rectangular", "directed")


def example(layout):
    i, j = np.indices((6, 6) if layout == "directed" else (5, 7))
    if layout == "directed":
        observed = i != j
        i, j = i[observed], j[observed]
    else:
        i, j = i.ravel(), j.ravel()
    t = np.arange(len(i))
    x1 = np.sin(0.73 * t) + 0.04 * i
    x2 = np.cos(0.91 * t) - 0.03 * j
    y = np.exp(0.4 * x1 - 0.31 * x2 + 0.07 * i - 0.02 * j)
    y *= 1.0 + 0.36 * np.sin(1.21 * t)
    y[7] = 0
    left = [f"site-{3 * k + 2}" for k in i]
    right = [f"site-{3 * k + 2}" for k in j] if layout == "directed" else 11 * j + 7
    return pd.DataFrame(
        {"left": left, "right": right, "x1": x1, "x2": x2, "y": y},
        index=pd.Index((t + 6) // 2, name="duplicate_index"),
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("method", METHODS)
def test_large_additive_effect_preserves_dependent_contrast_rank(layout, method):
    shape = (7, 7) if layout == "directed" else (6, 7)
    i, j = np.indices(shape)
    keep = i != j if layout == "directed" else np.ones(shape, dtype=bool)
    i, j = i[keep], j[keep]
    t = np.arange(i.size)
    x = np.round(8 * np.sin(1.7 * t + 0.3)) / 8
    row_effect = (2.0**28) * ((3 * i) % 5 - 2)
    x2 = x + row_effect
    assert_array_equal(x2 - row_effect, x)
    y = np.exp(0.4 * x + 0.1 * i - 0.05 * j) * (1 + 0.3 * np.cos(2.3 * t))
    data = pd.DataFrame({"left": i, "right": j, "x1": x, "x2": x2, "y": y})

    with pytest.raises(pm.IdentificationError) as error:
        pm.fit(FORMULA, data, estimator=method, fe_layout=layout)
    assert error.value.identifier == "identification.rank_deficient"


@pytest.mark.parametrize("method", METHODS)
def test_single_rectangle_cannot_report_residual_as_covariance(method):
    frame = pd.DataFrame(
        {
            "left": [0, 0, 1, 1],
            "right": [0, 1, 0, 1],
            "x": [0, 0, 0, 1],
            "y": [1, 1, 1, 2],
        }
    )
    for start in (np.log(2), np.log(2) + 5e-11):
        with pytest.raises(pm.ConvergenceError, match="unusable_covariance"):
            pm.fit("y ~ 0 + x | left + right", frame, estimator=method, start=[start])


def test_object_arrays_do_not_discard_imaginary_components():
    frame = example("rectangular")
    outcome = frame["y"].to_numpy(dtype=object, copy=True)
    outcome[0] = np.complex128(outcome[0] + 2j)
    with pytest.raises(pm.SampleError, match="must be real numbers"):
        pm.fit(FORMULA, frame.assign(y=outcome), estimator="gmm1")

    start = np.array([np.complex128(2j), 0.0], dtype=object)
    with pytest.raises(pm.SpecificationError, match="a finite real vector"):
        pm.fit(FORMULA, frame, estimator="gmm1", start=start)


def literal(y, x, left, right, beta, *, method, instrument=None, shift=0):
    """Enumerate only rectangles whose four cells occur in the input dictionary."""
    index = dict(zip(zip(left, right, strict=True), range(len(y)), strict=True))
    z = x if instrument is None else instrument
    x = x - shift
    moment = np.zeros(x.shape[1])
    jacobian = np.zeros((x.shape[1], x.shape[1]))
    incidence = np.zeros_like(x)
    count = 0
    for i, k in combinations(dict.fromkeys(left), 2):
        for j, other in combinations(dict.fromkeys(right), 2):
            cells = [(i, j), (k, other), (i, other), (k, j)]
            if not all(cell in index for cell in cells):
                continue
            a, b, c, d = (index[cell] for cell in cells)
            contrast = z[a] + z[b] - z[c] - z[d]
            p, q = x[a] + x[b], x[c] + x[d]
            positive_index, negative_index = (-p, -q) if method == "gmm1" else (q, p)
            u = y[a] * y[b] * np.exp(positive_index @ beta)
            v = y[c] * y[d] * np.exp(negative_index @ beta)
            kernel = contrast * (u - v)
            moment += kernel
            jacobian += np.outer(contrast, u * positive_index - v * negative_index)
            incidence[[a, b, c, d]] += kernel
            count += 1
    return moment / count, jacobian / count, incidence * (len(y) / count), count


@pytest.mark.parametrize(
    ("shape", "directed"), [((4, 5), False), ((5, 4), False), ((5, 5), True)]
)
@pytest.mark.parametrize("method", METHODS)
def test_focal_moment_actual_jacobian_and_incident_rows(shape, directed, method):
    rng = np.random.default_rng(59)
    i, j = np.indices(shape)
    mask = i != j if directed else np.ones_like(i, dtype=bool)
    x = rng.uniform(-1, 1, (*i.shape, 2)) * mask[..., None]
    y = rng.uniform(0.2, 1, i.shape) * mask
    y[1, 2] = 0
    beta = np.array([0.12, -0.37])
    system = DifferenceMoments(y, x, mask, gmm1=method == "gmm1")
    state = system.evaluate(beta)
    derivative, incident = system.derivatives(state, project=True)
    instruments = np.moveaxis(system.t, 0, -1)
    expected, expected_j, expected_h, count = literal(
        system.y[mask],
        x[mask],
        i[mask],
        j[mask],
        beta,
        method=method,
        instrument=instruments[mask],
    )
    assert count == (30 if directed else 60)
    assert_allclose(state.moment / count, expected, atol=3e-15)
    assert_allclose(derivative / count, expected_j, atol=3e-15)
    assert_allclose(
        incident.T.reshape(x.shape)[mask] * (mask.sum() / count), expected_h, atol=3e-15
    )
    assert_allclose(expected_h.mean(axis=0), 4 * expected, atol=3e-15)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_public_fit_and_literal_full_covariance(method, layout):
    data = example(layout).sample(frac=1, random_state=9)
    y = data.y.to_numpy()
    x = data[["x2", "x1"]].to_numpy()
    left, right = list(data.left), list(data.right)
    fit = pm.fit(FORMULA, data, estimator=method, fe_layout=layout)

    def calculate(beta):
        return literal(y, x, left, right, beta, method=method)

    expected = root(lambda b: calculate(b)[0], np.zeros(2), tol=1e-11)
    assert expected.success
    assert_allclose(fit.coef(), expected.x, atol=2e-10, rtol=0)
    assert list(fit.coef().index) == ["x2", "x1"]
    assert fit.sample.retained_index.equals(data.index)
    assert_array_equal(fit.sample.positions, np.arange(len(data)))
    assert fit.sample.count == len(data)
    assert fit.sample.zeros == 1
    moment, jacobian, projected, count = calculate(expected.x)
    assert count == (90 if layout == "directed" else 210)
    assert_allclose(moment, 0, atol=1e-12)
    loading = np.linalg.solve(jacobian, projected.T)
    paper = (loading @ loading.T) / len(data) ** 2
    assert abs(paper[0, 1]) > 1e-7
    assert_allclose(fit.vcov(), paper, atol=2e-11, rtol=1e-9)
    assert_allclose(fit.se(), np.sqrt(paper.diagonal()))
    assert_allclose(
        fit.confint(0.90),
        expected.x[:, None]
        + np.sqrt(paper.diagonal())[:, None] * [-norm.ppf(0.95), norm.ppf(0.95)],
    )
    assert_allclose(fit.tidy().p_value, 2 * norm.sf(abs(expected.x) / fit.se()))
    assert fit.certificate.status == "local_solution"
    assert "local root" in fit.summary()
    assert f"Jochmans {method.upper()}" in repr(pm.etable(fit))
    assert "Differenced" in repr(pm.etable(fit, scale="effect"))
    # Centering used by author programs alters a common scale on valid cells.
    # Square padding affects that center, and twgravity additionally uses n²
    # in the projection meat but still divides the final variance by observed N.
    slots = (
        max(fit.dimensions) ** 2 if layout == "directed" or method == "gmm2" else len(y)
    )
    _, centered_j, centered_h, _ = literal(
        y, x, left, right, expected.x, method=method, shift=x.sum(axis=0) / slots
    )
    source = np.linalg.solve(centered_j, centered_h.T)
    factor = len(y) / slots if layout == "directed" else 1.0
    author_convention = fit.with_vcov("stata")
    assert_allclose(
        author_convention.vcov(), factor * (source @ source.T) / len(y) ** 2
    )
    assert_allclose(fit.vcov("stata"), author_convention.vcov())
    assert author_convention.provenance.covariance["factor"] == factor
    refit = pm.fit(
        FORMULA, data.iloc[::-1], fe_layout=layout, estimator=method, vcov="stata"
    )
    assert refit.sample.fingerprint != fit.sample.fingerprint
    assert_allclose(refit.coef(), fit.coef(), atol=2e-12, rtol=0)
    assert_allclose(refit.vcov(), author_convention.vcov(), atol=2e-12, rtol=0)


def test_gmm1_and_gmm2_are_distinct_and_units_are_respected():
    data = example("rectangular")
    a = pm.fit(FORMULA, data, estimator="gmm1")
    b = pm.fit(FORMULA, data, estimator="gmm2")
    assert np.max(np.abs(a.coef() - b.coef())) > 0.01
    scales = [2.0**-40, 2.0**25]
    rescaled = data.assign(
        x2=data.x2 * scales[0], x1=data.x1 * scales[1], y=data.y * 2.0**700
    )
    changed = pm.fit(FORMULA, rescaled, estimator="gmm2")
    assert_allclose(changed.coef() * scales, b.coef(), atol=1e-12, rtol=0)
    assert_allclose(
        changed.vcov() * np.outer(scales, scales), b.vcov(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_sample_geometry_and_missing_values(layout):
    data = example(layout)
    with pytest.raises(pm.SampleError, match="expected every"):
        pm.fit(FORMULA, data.iloc[:-1], estimator="gmm1", fe_layout=layout)
    repeated = pd.concat([data.iloc[:-1], data.iloc[:1]])
    with pytest.raises(pm.SampleError, match="duplicate"):
        pm.fit(FORMULA, repeated, estimator="gmm1", fe_layout=layout)
    for column, value in [("x1", np.nan), ("left", None), ("y", np.inf)]:
        invalid = data.copy()
        invalid.iloc[3, data.columns.get_loc(column)] = value
        with pytest.raises(pm.SampleError, match="missing or non-finite"):
            pm.fit(FORMULA, invalid, estimator="gmm1", fe_layout=layout)
    with pytest.raises(pm.SampleError, match="negative outcome"):
        pm.fit(FORMULA, data.assign(y=-data.y), estimator="gmm1", fe_layout=layout)


def test_directed_common_set_self_links_and_independent_rectangular_axes():
    data = example("directed")
    with pytest.raises(pm.SampleError, match="self-links"):
        pm.fit(
            FORMULA,
            data.assign(right=data.left),
            estimator="gmm1",
            fe_layout="directed",
        )
    with pytest.raises(pm.SampleError, match="expected every"):
        pm.fit(FORMULA, data, estimator="gmm1", fe_layout="rectangular")
    with pytest.raises(pm.SampleError, match="common agents"):
        pm.fit(
            FORMULA,
            data.assign(right="new-" + data.right),
            estimator="gmm1",
            fe_layout="directed",
        )
    rectangle = example("rectangular")
    rectangle["right"] = [f"site-{3 * ((j - 7) // 11) + 2}" for j in rectangle.right]
    fit = pm.fit(FORMULA, rectangle, estimator="gmm1")
    assert fit.dimensions == (5, 7)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_identification_and_covariance_failures(layout):
    data = example(layout)
    with pytest.raises(pm.IdentificationError, match="design has rank"):
        pm.fit("y ~ x1 + x2 | left + right", data, estimator="gmm1", fe_layout=layout)
    with pytest.raises(pm.IdentificationError, match="design has rank"):
        pm.fit(FORMULA, data.assign(x2=2 * data.x1), estimator="gmm1", fe_layout=layout)
    additive = pd.factorize(data.left)[0] + 3 * pd.factorize(data.right)[0]
    with pytest.raises(pm.IdentificationError, match="design has rank"):
        pm.fit(FORMULA, data.assign(x2=additive), estimator="gmm1", fe_layout=layout)
    with pytest.raises(pm.IdentificationError, match="positive outcome product"):
        pm.fit(
            FORMULA,
            data.assign(y=(data.left == data.left.iloc[0])),
            estimator="gmm1",
            fe_layout=layout,
        )
    with pytest.raises(pm.ConvergenceError, match="unusable_covariance"):
        pm.fit(FORMULA, data.assign(y=2.0), estimator="gmm1", fe_layout=layout)


def test_specification_and_solver_failures():
    data = example("rectangular")
    for formula in (
        "y ~ x1",
        FORMULA + " | x1 ~ x2",
        "y ~ 0 + x1 | left^right + right",
    ):
        with pytest.raises(pm.SpecificationError):
            pm.fit(formula, data, estimator="gmm1")
    with pytest.raises(pm.SpecificationError):
        pm.fit(FORMULA, data, estimator="gmm1", vcov="robust")
    with pytest.raises(pm.SpecificationError):
        pm.fit(FORMULA, data, estimator="gmm1", start=[0])
    with pytest.raises(pm.ConvergenceError, match="numerical_domain"):
        pm.fit(FORMULA, data, estimator="gmm1", start=[1e90, 0])
    with pytest.raises(pm.ConvergenceError, match="iteration_limit"):
        pm.fit(FORMULA, data, estimator="gmm1", max_iter=1)
    fit = pm.fit(FORMULA, data, estimator="gmm1")
    with pytest.raises(ValueError, match="strictly between"):
        fit.confint(1.0)


def test_tiny_moment_in_a_tail_is_not_an_acceptable_root():
    separated = pd.DataFrame(
        {"i": [0, 0, 1, 1], "j": [0, 1, 0, 1], "y": [1, 0, 0, 1], "x": [1, 0, 0, 1]}
    )
    with pytest.raises((pm.IdentificationError, pm.ConvergenceError)):
        pm.fit("y ~ 0 + x | i + j", separated, estimator="gmm1", start=[48])


def test_formula_transforms_and_quoted_columns_keep_names_and_input_rows():
    data = example("rectangular").rename(columns={"x2": "other x", "left": "left ID"})
    result = pm.fit(
        "y ~ 0 + I(x1**2) + `other x` | `left ID` + right", data, estimator="gmm1"
    )
    assert list(result.coef().index) == ["I(x1 ** 2)", "other x"]
    assert result.sample.retained_index.equals(data.index)
    data.iloc[1, data.columns.get_loc("other x")] = np.nan
    with pytest.raises(pm.SampleError, match="missing"):
        pm.fit(
            'y ~ 0 + np.nan_to_num(Q("other x")) + x1 | `left ID` + right',
            data,
            estimator="gmm1",
        )


def test_formula_categorical_interaction_keeps_the_fitted_design():
    data = example("rectangular")
    data["regime"] = pd.Series(
        np.where(np.arange(len(data)) % 3 == 0, "treated", "control"),
        dtype="str",
    ).to_numpy()
    categorical = pm.fit("y ~ 0 + x1:regime | left + right", data, estimator="gmm2")
    data["control"] = data.x1 * (data.regime == "control")
    data["treated"] = data.x1 * (data.regime == "treated")
    slope = pm.fit("y ~ 0 + control + treated | left + right", data, estimator="gmm2")
    # Formulaic 1.0 includes the T. prefix for this full-rank interaction.
    assert list(categorical.coef().index) in (
        ["x1:regime[control]", "x1:regime[treated]"],
        ["x1:regime[T.control]", "x1:regime[T.treated]"],
    )
    assert_allclose(categorical.coef(), slope.coef())
    assert_allclose(categorical.vcov(), slope.vcov())


def test_no_admissible_positive_dyad_product_and_rank_of_actual_jacobian():
    dyads = example("directed")
    shares_agent = (dyads.left == "site-2") & (dyads.right == "site-5") | (
        dyads.left == "site-5"
    ) & (dyads.right == "site-8")
    with pytest.raises(pm.IdentificationError, match="positive outcome product"):
        pm.fit(
            FORMULA,
            dyads.assign(y=shares_agent),
            estimator="gmm1",
            fe_layout="directed",
        )
    data = example("rectangular")
    support = np.zeros(len(data))
    support[[0, 8]] = 1
    with pytest.raises(pm.IdentificationError, match="moment Jacobian"):
        pm.fit(FORMULA, data.assign(y=support), estimator="gmm1")


def test_unsupported_fit_options_and_covariance_refusal_certificate():
    data = example("rectangular")
    with pytest.raises(pm.SpecificationError, match="estimator"):
        pm.fit(FORMULA, data, estimator="gmm3")
    with pytest.raises(pm.SpecificationError, match="fe_layout"):
        pm.fit(FORMULA, data, estimator="gmm1", fe_layout="unbalanced")
    with pytest.raises(pm.SpecificationError, match="fe_layout"):
        pm.fit(FORMULA, data, fe_layout="directed")
    with pytest.raises(pm.SpecificationError, match="Unknown keyword"):
        pm.fit(FORMULA, data, estimator="gmm1", cluster="left")
    extreme = data.copy()
    extreme.iloc[0, extreme.columns.get_loc("y")] = 1e-120
    with pytest.raises(pm.ConvergenceError, match="numerical_domain"):
        pm.fit(FORMULA, extreme, estimator="gmm1")
    with pytest.raises(pm.ConvergenceError, match="unusable_covariance") as failure:
        pm.fit(FORMULA, data.assign(y=2), estimator="gmm1")
    assert failure.value.certificate.status == "local_solution"


@pytest.mark.parametrize(("layout", "levels"), [("rectangular", 2), ("directed", 4)])
def test_narrowest_supported_array_width(layout, levels):
    data = example(layout)
    agents = data.left.unique()[:levels]
    support = data.left.isin(agents)
    if layout == "directed":
        support &= data.right.isin(agents)
    data = data.loc[support]
    fit = pm.fit(FORMULA, data, fe_layout=layout, estimator="gmm2")
    y, x = data.y.to_numpy(), data[["x2", "x1"]].to_numpy()
    moment, jacobian, projected, _ = literal(
        y, x, list(data.left), list(data.right), fit.coef().to_numpy(), method="gmm2"
    )
    assert_allclose(moment, 0, atol=1e-10)
    loading = np.linalg.solve(jacobian, projected.T)
    assert_allclose(fit.vcov(), (loading @ loading.T) / len(data) ** 2, atol=2e-11)
