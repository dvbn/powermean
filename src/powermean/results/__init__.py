"""The result of a fit: reporting, prediction, sample and provenance views.

The modules of this package are the result layer: they consume what the earlier stages
formed, a sample of the preparation layer, an estimator record, an authentic certificate
or conclusion of the solve stage, and bind them to one coefficient point, one equation,
one set of rows and one coordinate basis (:mod:`powermean.results.handoff`).  Nothing
here solves, selects a row, trains a formula again or certifies a candidate: the public
``fit`` and ``prepare`` wrappers do so, and this package exports no public name.

================  ===================================================================
``citations``     the two manuscripts and the method references that a summary prints
``naming``        the names of the estimators, the orders and the coverage labels
``retained``      the owned state bound to the fitted point
``owned``         owned copies of the records retained and handed out
``binding``       the binding of a sample to its preparation and its estimator
``handoff``       the internal handoff and its validation against the inputs
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
