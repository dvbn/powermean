"""Strict certification of two-way fixed effects at interior powers."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import qr

import powermean as pm
from powermean.fe.certificate_basis import certificate_basis
from powermean.fe.groups import Grouping, encode_blocks
from powermean.fe.profile import ProfileState
from powermean.fe.profile_arithmetic import prepare_profile
from powermean.model.variance import PowerVariance
from powermean.solve.reduced_nonlinear import ConditionalSpace
from powermean.systems.profiled import ProfiledProblem


def _panel():
    data = pd.DataFrame(
        (
            (g, h, z)
            for g, h, z in product(range(3), range(4), range(3))
            if (g, h) not in ((0, 0), (1, 3))
        ),
        columns=["g", "h", "z"],
    )
    g, h, z = (data[name].to_numpy() for name in ("g", "h", "z"))
    x = np.cos(g + 2 * h + 3 * z)
    data["x"] = x
    data["y"] = np.exp(
        0.3 * x + 0.1 * g - 0.15 * h + 0.05 * np.sin(2 * z + 0.2 * g + 0.4 * h)
    )
    return data


def _dummy_reference(data, tau):
    y = data["y"].to_numpy()
    # All eight columns include one dependent dummy. Keep the slope first.
    dummies = pd.get_dummies(data[["g", "h"]], columns=["g", "h"]).to_numpy(float)
    _, triangular, pivot = qr(dummies, mode="economic", pivoting=True)
    rank = np.linalg.matrix_rank(triangular)
    assert rank == 6
    full = np.column_stack((data["x"], dummies))
    design = full[:, np.r_[0, 1 + pivot[:rank]]]
    beta = np.linalg.lstsq(design, np.log(y), rcond=None)[0]
    for _ in range(6):
        eta = design @ beta
        mean = np.exp(eta)
        unit = np.exp((1 - tau) * eta)
        score = unit * (y - mean)
        observed = unit * ((tau - 1) * y + (2 - tau) * mean)
        bread = (design.T * observed) @ design
        correction = np.linalg.solve(bread, design.T @ score)
        score_error = np.max(np.abs(full.T @ score))
        index_error = np.max(np.abs(design @ correction))
        if max(score_error, index_error) < 1e-14:
            break
        beta += correction
    assert max(score_error, index_error) < 1e-14
    slope_influence = score * np.linalg.solve(bread, design.T)[0]
    return beta[0], mean, slope_influence @ slope_influence


@pytest.mark.parametrize("prepared", [False, True], ids=["formula", "prepared"])
@pytest.mark.parametrize("tau", [1.1, 1.6, 1.9])
def test_joint_interior_power_certifies_the_full_dummy_root(tau, prepared):
    data = _panel()
    formula = "y ~ 0 + x | g + h"
    tol = 1e-12
    options = dict(tau=tau, tol=tol, vcov="robust", ssc="none", on_failure="return")
    if prepared:
        fit = pm.prepare(formula, data).fit(**options)
    else:
        fit = pm.fit(formula, data, **options)

    assert fit.provenance.status == "verified"
    assert fit.sample.positions.tolist() == list(range(len(data)))
    certificate = fit.certificate
    assert certificate.controls.tol == tol
    assert certificate.strict == ()
    assert certificate.measures.scaled_score <= tol
    assert certificate.measures.index_correction <= tol
    slope, mean, variance = _dummy_reference(data, tau)
    assert fit.coef()["x"] == pytest.approx(slope, rel=0, abs=tol)
    np.testing.assert_allclose(fit.predict(), mean, rtol=tol, atol=0)
    assert fit.vcov().loc["x", "x"] == pytest.approx(variance, rel=1e-10, abs=0)


def test_conditional_refinement_preserves_the_passing_canonical_index():
    grouping = encode_blocks({"h": [1, 0, 0, 1, 1], "g": [2, 0, 1, 0, 1]})
    basis = certificate_basis(grouping)
    scale = float(2**20)
    ulp = np.spacing(scale)
    offset = np.array([0, scale, 2 * scale, scale, 2 * scale])
    y = np.exp([0, 1.52 * ulp, 1.52 * ulp, 0, 0])
    weights = np.ones(5)
    variance = PowerVariance(1.6)
    # Construct just the conditional subproblem, with h eliminated and g fixed.
    space = object.__new__(ConditionalSpace)
    space.problem = ProfiledProblem(
        z=y, design=np.zeros((5, 1)), offset=offset, weights=weights,
        grouping=grouping, variance=variance, columns=("x",),
    )
    space.basis = basis
    space.block = 0
    space.remaining = np.r_[0, 1 + basis.positions[1][1:]]
    conditional = Grouping((grouping.blocks[0],), grouping.positions)
    space.initial = prepare_profile(conditional, y, offset, weights, variance=variance)
    retained = np.array([0, -scale, -2 * scale])
    warm = ProfileState(grouping, (np.array([0, 1.48 * ulp]), retained))
    values = np.r_[0, retained[1:]]
    controls = dict(tolerance=2e-11, max_sweeps=1, warm=warm)

    ordinary = space.solve(values, **controls)
    refined = space.solve(values, refine_tol=1e-12, **controls)
    assert ordinary.result.work.sweeps_attempted == 0
    assert refined.result.work.sweeps_attempted == 1
    # The sweep crosses a rounding boundary in (offset + h) + g.
    # Its conditional equation improves while its canonical equation fails.
    np.testing.assert_array_equal(
        space.reconstruct(refined).index, space.reconstruct(ordinary).index
    )
