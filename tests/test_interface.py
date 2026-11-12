"""The public names, and what the package refuses."""

import pandas as pd
import pytest

import powermean as pm

FORMULA = "y ~ 1 + x1 + x2 + d"


def test_public_names_resolve():
    for name in pm.__all__:
        assert getattr(pm, name) is not None


def test_fixed_effects_are_refused_in_this_version(counts):
    for call in (pm.ppml, pm.gpml, pm.fit, pm.prepare):
        with pytest.raises(pm.SpecificationError) as caught:
            call("y ~ x1 | firm", counts)
        assert caught.value.identifier == "specification.unimplemented"
        assert "fixed effects" in str(caught.value)


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
    assert "PPML" in text and "x1" in text


def test_compiled_iols_kernel_is_not_in_this_version(counts):
    import dataclasses

    options = dataclasses.replace(pm.solver_defaults("iols"), iols_backend="cpp")
    with pytest.raises(pm.SolverCapabilityError) as caught:
        pm.fit(FORMULA, counts, omega=1, tau=1, solver="iols", solver_options=options)
    assert "iols_backend" in str(caught.value)
    numpy_kernel = dataclasses.replace(options, iols_backend="numpy")
    pm.fit(FORMULA, counts, omega=1, tau=1, solver="iols", solver_options=numpy_kernel)
