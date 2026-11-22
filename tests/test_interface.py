"""Public entry points, supported fits, and option errors."""

from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

import powermean as pm

FORMULA = "y ~ 1 + x1 + x2 + d"


def test_public_names_resolve():
    for name in pm.__all__:
        assert getattr(pm, name) is not None


def test_fixed_effects_fit_across_formula_entry_points():
    data = pd.DataFrame(
        {
            "firm": ["a"] * 4 + ["b"] * 4,
            "x": [0, 0, 1, 1] * 2,
            "y": [1.0, 3.0, 3.0, 5.0, 2.0, 4.0, 8.0, 10.0],
        }
    )
    formula = "y ~ x | firm"
    # The firms have different outcome ratios, so the working variance matters.
    ppml_ratio = 2.6
    gpml_ratio = np.sqrt(6.0)
    ppml_baselines = np.array([6.0, 12.0]) / (1 + ppml_ratio)
    gpml_baselines = (np.array([2.0, 3.0]) + np.array([4.0, 9.0]) / gpml_ratio) / 2
    results = (
        (pm.ppml(formula, data), ppml_ratio, ppml_baselines),
        (pm.gpml(formula, data), gpml_ratio, gpml_baselines),
        (pm.fit(formula, data), ppml_ratio, ppml_baselines),
        (pm.prepare(formula, data).fit(), ppml_ratio, ppml_baselines),
    )
    for result, ratio, baselines in results:
        assert result.provenance.certified
        assert result.sample.count == len(data)
        assert list(result.coef().index) == ["x"]
        np.testing.assert_allclose(result.coef()["x"], np.log(ratio), rtol=1e-7)
        means = np.repeat((baselines[:, None] * [1, ratio]).ravel(), 2)
        np.testing.assert_allclose(result.predict(), means, rtol=1e-7)


def test_unknown_keyword_is_refused(counts):
    with pytest.raises(pm.SpecificationError) as caught:
        pm.ppml(FORMULA, counts, nonsense=1)
    assert caught.value.identifier == "specification.unknown_option"


def test_negative_outcome_is_refused(counts):
    with pytest.raises(pm.SampleError) as caught:
        pm.ppml(FORMULA, counts.assign(y=counts["y"] - 100.0))
    assert caught.value.identifier == "sample.negative_outcome"


def test_iols_outside_its_domain_is_refused(counts):
    with pytest.raises(pm.SolverCapabilityError):
        pm.fit(FORMULA, counts, omega=1, tau=3, solver="iols")


def test_outside_the_convex_range_warns(counts):
    with pytest.warns(pm.CoverageWarning):
        pm.fit(FORMULA, counts, omega=1, tau=3)


def test_the_data_are_not_modified(counts):
    before = counts.copy(deep=True)
    pm.ppml(FORMULA, counts, cluster="firm")
    pd.testing.assert_frame_equal(counts, before)


def test_summary_is_text(counts):
    text = pm.ppml(FORMULA, counts).summary()
    assert "PPML" in text
    assert "x1" in text


def test_iols_backend_selection(counts):
    module = "powermean._native.iols._kernel_cpp"
    has_cpp = find_spec(module) is not None

    numpy_result = pm.fit(
        FORMULA, counts, solver="iols", solver_options={"iols_backend": "numpy"}
    )
    assert numpy_result.provenance.certified
    numpy_trace = numpy_result.provenance.trace
    assert numpy_trace is not None
    assert numpy_trace["actual_backend"] == "numpy"
    assert numpy_trace["fallback_used"] is False

    auto_result = pm.fit(FORMULA, counts, solver="iols")
    auto_trace = auto_result.provenance.trace
    assert auto_result.provenance.certified
    assert auto_trace is not None
    assert auto_trace["actual_backend"] == ("cpp" if has_cpp else "numpy")
    assert auto_trace["fallback_used"] is (not has_cpp)

    cpp_options = {"iols_backend": "cpp"}
    if not has_cpp:
        with pytest.raises(ModuleNotFoundError) as caught:
            pm.fit(FORMULA, counts, solver="iols", solver_options=cpp_options)
        assert caught.value.name == module
        return

    cpp_result = pm.fit(FORMULA, counts, solver="iols", solver_options=cpp_options)
    assert cpp_result.provenance.certified
    cpp_trace = cpp_result.provenance.trace
    assert cpp_trace is not None
    assert cpp_trace["actual_backend"] == "cpp"
    assert cpp_trace["fallback_used"] is False
    np.testing.assert_allclose(cpp_result.coef(), numpy_result.coef(), rtol=1e-7)
    np.testing.assert_allclose(cpp_result.vcov(), numpy_result.vcov(), rtol=1e-6)
