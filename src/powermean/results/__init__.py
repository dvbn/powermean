"""The result of a fit: reporting, prediction, sample and provenance views (M0.21).

``DESIGN.md``, Sections 4.4 and 4.5; ``design/04_inference_and_effects.md``, Sections
1 to 3 and 5 and its retained-state specification; ``design/02_computation.md``,
Section 9; ``design/01_statistical_core.md``, Sections 3, 6 and 7; decisions D41, D49,
D55, D57 and D58.  The modules of this package are the result layer of ``DESIGN.md``,
Section 5.1: they consume what the earlier stages formed, a sample of the preparation
layer, an estimator record, an authentic certificate or conclusion of the solve stage,
and bind them to one coefficient point, one equation, one set of rows and one
coordinate basis (:mod:`powermean.results.handoff`).  Nothing here solves, selects a
row, trains a formula again or certifies a candidate: the public ``fit`` and
``prepare`` wrappers that do so are task M0.22, and this package exports no public
name.

================  ===================================================================
``citations``     the two manuscripts and the method references that a summary prints
``naming``        the names of the estimators, the orders and the coverage labels
``retained``      the owned state bound to the fitted point (file 04, Section 2.3)
``owned``         owned copies of the records retained and handed out
``binding``       the binding of a sample to its preparation and its estimator
``handoff``       the internal handoff of M0.22 and its validation against the inputs
``point``         the point with its influence record and default covariance
``covariance``    covariances from the retained primitives, on the reported scale
``prediction``    the fitted power mean from the owned fitted right-hand side
``provenance``    what ran: versions, provider, settings, trace, sample, convention
``sample``        the sample view: positions, labels, accounting and counts
``summary``       the printed summary
``result``        the result object: ``coef``, ``se``, ``vcov``, ``confint``,
                  ``tidy``, ``summary``, ``predict`` and ``influence``
================  ===================================================================
"""

__all__: list[str] = []
