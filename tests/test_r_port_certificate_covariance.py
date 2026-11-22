"""Cross-platform certificate and working covariance regressions from the R port."""

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
import pytest

import powermean as pm
from powermean.inference.vcov import hc_covariance
from powermean.results.absorbed_root_binding import _observed
from powermean.solve.absorbed_certificate import FEPointCertificate
from powermean.solve.absorbed_root import _StageCertifier


@pytest.mark.filterwarnings(
    "ignore:Covariance diagnostics:powermean.inference.vcov.CovarianceWarning"
)
@pytest.mark.filterwarnings("ignore:The joint fit:powermean.errors.CoverageWarning")
def test_signed_fe_zero_se_certifies_at_the_paired_slope_root(monkeypatch):
    certificates = []
    certify = _StageCertifier.certify

    def capture(self, variance, point):
        found = certify(self, variance, point)
        if (
            variance.tau == 2.3
            and isinstance(found, FEPointCertificate)
            and found.accepted
        ):
            certificates.append(found)
        return found

    monkeypatch.setattr(_StageCertifier, "certify", capture)
    unit = 2.0**-1070
    data = pd.DataFrame(
        {
            "y": np.ones(8),
            "x": np.tile([-unit, -unit, unit, unit], 2),
            "g": np.repeat([1, 2], 4),
            "offset": np.tile([-0.125, 0.125], 4),
        }
    )
    fit = pm.fit(
        "y ~ 0 + x | g",
        data,
        tau=2.3,
        offset="offset",
        cluster="g",
        on_failure="return",
    )

    assert fit.provenance.certified, fit.provenance.status
    assert fit.sample.positions.tolist() == list(range(8))
    assert abs(fit.coef()["x"] * unit) < 1e-12
    np.testing.assert_array_equal(fit.se(), [0.0])
    check = fit.certificate.coefficients
    assert check is not None
    assert check.accepted
    np.testing.assert_array_equal(check.corrections, [0.0])
    evidence = certificates[-1]
    forged = replace(
        evidence,
        geometry=replace(evidence.geometry, compensated_mixed=True),
        _mixed_retry_scale=None,
    )
    with pytest.raises(ValueError, match="observed evidence is not bound"):
        _observed(forged, [])
    if evidence._mixed_retry_scale is not None:
        # The original Linux ARM cancellation exercises the compensated chart.
        # Keep the stored pass flags while substituting a different correction.
        certificate_check = evidence.coefficients
        assert certificate_check is not None
        forged = replace(
            evidence,
            coefficients=replace(certificate_check, corrections=np.ones(1)),
        )
        with pytest.raises(ValueError, match="observed evidence is not bound"):
            _observed(forged, [])


@pytest.mark.parametrize("fit_time", [True, False], ids=["direct", "retained-view"])
@pytest.mark.parametrize(
    "amplitude", [0.5, 1.0], ids=["finite", "overflowing-spectrum"]
)
@pytest.mark.filterwarnings(
    "ignore:Covariance diagnostics:powermean.inference.vcov.CovarianceWarning"
)
def test_ddg_rank_uses_the_working_covariance_scale(fit_time, amplitude):
    data = pd.DataFrame(
        list(product((-1, 1), repeat=3)), columns=["z", "x", "replicate"]
    )
    data["y"] = np.where(data.x == data.z, amplitude * 1.5 * 2.0**512, 1.0)
    data["g"] = data.x > 0
    data["h"] = data.z > 0
    data[["x", "z"]] *= 2**10
    request = {"cluster": "g+h", "multiway": "ddg"}
    prepared = pm.prepare("y ~ 0 + x + z", data, cluster="g+h")
    if fit_time:
        fit = prepared.fit(vcov=request, ssc="none", start=[0.0, 0.0])
    else:
        fit = prepared.fit(vcov="robust", ssc="none", start=[0.0, 0.0]).with_vcov(
            request, ssc="none"
        )

    assert fit.provenance.certified
    covariance = fit.vcov().to_numpy()
    assert np.isfinite(covariance).all()
    assert covariance[0, 0] > 0
    np.testing.assert_array_equal(covariance, np.full((2, 2), covariance[0, 0]))
    assert fit.provenance.covariance["rank"] == 1


@pytest.mark.filterwarnings(
    "ignore:Covariance diagnostics:powermean.inference.vcov.CovarianceWarning"
)
def test_overflowing_working_spectrum_keeps_small_reportable_eigenvalues():
    large = 3.0 * 2.0**511
    result = hc_covariance([[large, large, 0.0], [0.0, 0.0, 2.0**-29]])

    assert result.rank == 1
    np.testing.assert_array_equal(result.eigenvalues, [0.0, 2.0**-60, np.inf])
