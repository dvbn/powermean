"""Data: the specification records and the typed inputs of an estimation.

The preparation layer.  It holds the specification records (:mod:`powermean.data.specs`:
``PowerMean``, ``Estimator`` and ``Inference``; :mod:`powermean.data.computation`:
``Computation``), the typed observation weights of the three kinds
(:mod:`powermean.data.weights`) and the owned record of the arrays of an estimation with
its offset or exposure (:mod:`powermean.data.inputs`).  Sample construction and
accounting (subset, missing values, the outcome domain, zero weights, the design and its
scaling, the fingerprint of the sample) are a separate step: these records select no
row.  The package works on arrays and small records, imports neither pandas nor
Formulaic nor any later stage, and is internal: ``powermean`` exports none of its names.
"""
