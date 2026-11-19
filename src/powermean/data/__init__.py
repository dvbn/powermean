"""Data: the specification records and the typed inputs of an estimation.

``DESIGN.md``, Sections 5.1 to 5.3; ``design/11_engineering.md``, Section 3.1 (the
preparation layer).  Task M0.19 adds the specification records
(:mod:`powermean.data.specs`: ``PowerMean``, ``Estimator`` and ``Inference``;
:mod:`powermean.data.computation`: ``Computation``), the typed observation weights of
the three kinds
(:mod:`powermean.data.weights`) and the owned record of the arrays of an estimation
with its offset or exposure (:mod:`powermean.data.inputs`), ported from the expreg
foundation (``PROVENANCE.md``, Section 13).  Sample construction and accounting
(subset, missing values, the outcome domain, zero weights, the design and its
scaling, the fingerprint of the sample) are task M0.20: nothing here selects a row.
The package works on arrays and small records, imports neither pandas nor Formulaic
nor any later stage, and is internal: ``powermean`` exports none of its names.
"""
