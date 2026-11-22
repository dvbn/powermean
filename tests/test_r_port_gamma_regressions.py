"""Regression cases from the R port, exercised through public Powermean calls."""

from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from math import log

import pandas as pd
import pytest

import powermean as pm


@pytest.mark.filterwarnings(
    "ignore:clustering with no dimension:powermean.errors.CoverageWarning"
)
def test_cancelled_gamma_certifies_the_centered_equation():
    delta = 2.0**-28
    shift = 0.25
    x = [0.5 + delta, 0.5 - delta, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    data = pd.DataFrame(
        {
            "y": [1, 1, 0, 0, 1, 1, 1, 1],
            "x": x,
            "offset": [shift * value for value in x],
            "firm": ["a"] * 4 + ["b"] * 4,
            "period": ["j"] * 6 + ["k"] * 2,
            "rows": range(1, 9),
        }
    )
    fit = pm.fit(
        "y ~ 0 + x | firm + period",
        data,
        tau=2,
        cancel="firm",
        offset="offset",
        cluster="rows",
        ssc="none",
        on_failure="return",
    )
    assert fit.provenance.certified, fit.provenance.status
    assert fit.sample.positions.tolist() == list(range(len(data)))
    slope = float(fit.coef()["x"])
    with localcontext() as context:
        context.prec = 70
        width = Decimal(delta)
        index = width * (Decimal(slope) + Decimal(shift))
        positive, negative = (-index).exp(), index.exp()
        score = width * (positive - negative)
        curvature = width**2 * (positive + negative)
        correction = score / curvature
    assert float(curvature) == pytest.approx(2.0**-55, rel=1e-12, abs=0.0)
    assert abs(float(correction)) < 1e-8, (slope, score, curvature, correction)
    assert slope == pytest.approx(-shift, abs=1e-8)


def fit_joint_gamma(epsilon, prepared):
    slopes = (0.0, 1.0, 2.0 - epsilon)
    data = pd.DataFrame(
        [
            {"x": slopes[row], "y": float(row < 2), "g": g, "h": h, "period": period}
            for period, h, g, row in product("ab", "fm", "ew", range(len(slopes)))
        ]
    )
    formula = "y ~ 0 + x | g^h + period"
    options = {"tau": 2, "vcov": "robust", "ssc": "none", "on_failure": "return"}
    if prepared:
        return pm.prepare(formula, data).fit(**options)
    return pm.fit(formula, data, **options)


@pytest.mark.parametrize("prepared", [False, True], ids=["formula", "prepared"])
@pytest.mark.parametrize("exponent", [39, 40])
@pytest.mark.filterwarnings("ignore:Low fitted means:powermean.errors.NumericalWarning")
def test_joint_gamma_certifies_the_finite_root_near_the_support_boundary(
    exponent, prepared
):
    epsilon = 2.0**-exponent
    fit = fit_joint_gamma(epsilon, prepared)
    assert fit.provenance.status == "verified"
    assert fit.sample.positions.tolist() == list(range(24))
    # Exact row scores at the finite root balance the cell and slope equations.
    e = Fraction(epsilon)
    scores = [e - 1, 2 - e, -1]
    assert sum(scores) == 0
    assert scores[1] + (2 - e) * scores[2] == 0
    assert fit.coef()["x"] == pytest.approx(
        log(epsilon) - log(3.0 - epsilon), rel=0, abs=1e-12
    )
    certificate = fit.certificate
    defaults = pm.solver_defaults()
    assert certificate.strict == ()
    for name in ("tol", "tol_secondary", "rcond_min", "coef_tol"):
        assert getattr(certificate.controls, name) == getattr(defaults, name)
    assert certificate.measures.rcond > certificate.controls.rcond_min

    # All cells have the same profile, so removing the weighted cell mean
    # gives the efficient slope influence for either working curvature.
    x = [0, 1, 2 - e]
    for bread, weights in (("observed", [e, 3 - e, 0]), ("expected", [1, 1, 1])):
        center = sum(w * v for w, v in zip(weights, x, strict=True)) / Fraction(
            sum(weights)
        )
        information = 8 * sum(
            w * (v - center) ** 2 for w, v in zip(weights, x, strict=True)
        )
        covariance = 8 * sum(
            ((v - center) * score / information) ** 2
            for v, score in zip(x, scores, strict=True)
        )
        view = fit.with_vcov({"method": "HC0", "bread": bread})
        assert view.vcov().loc["x", "x"] == pytest.approx(float(covariance), rel=1e-12)
        if bread == "observed":
            assert fit.vcov().loc["x", "x"] == pytest.approx(
                float(covariance), rel=1e-12
            )


@pytest.mark.parametrize("prepared", [False, True], ids=["formula", "prepared"])
def test_joint_gamma_at_the_boundary_has_genuine_recession(prepared):
    fit = fit_joint_gamma(0.0, prepared)
    assert fit.provenance.status == "no_finite_root"
    assert not fit.provenance.certified
    assert fit.certificate is None
