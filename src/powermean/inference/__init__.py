"""Inference: influence contributions, covariance conventions and the reported scale.

``DESIGN.md``, Sections 5.1 and 5.2; ``design/04_inference_and_effects.md``, Sections
1 to 3.  Task M0.16 adds the ported influence maps
(:mod:`powermean.inference.influence`) and covariance estimators
(:mod:`powermean.inference.vcov`) of pyiols, with their numerical helpers
(:mod:`powermean.inference.numeric`); the influence record of a candidate
(:mod:`powermean.inference.contributions`); the robust and one-way cluster covariances
of the first milestone with the factors of the ``"stata"`` convention and the
covariance on a reported scale (:mod:`powermean.inference.convention`); and the
provider of the reported scale of the certificate
(:mod:`powermean.inference.provider`).  The package is internal: nothing here is a
public name.
"""
