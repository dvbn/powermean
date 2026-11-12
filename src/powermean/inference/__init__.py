"""Inference: influence contributions, covariance conventions and the reported scale.

Its modules include the influence maps (:mod:`powermean.inference.influence`) and
covariance estimators (:mod:`powermean.inference.vcov`), with their numerical helpers
(:mod:`powermean.inference.numeric`); the influence record of a candidate
(:mod:`powermean.inference.contributions`); the robust and one-way cluster covariances
with the factors of the ``"stata"`` convention and the
covariance on a reported scale (:mod:`powermean.inference.convention`); and the
provider of the reported scale of the certificate
(:mod:`powermean.inference.provider`).  The package is internal: nothing here is a
public name.
"""
