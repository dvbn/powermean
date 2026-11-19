"""Descriptive and balance tables for the work that precedes estimation.

Four functions, exported by the root package and resolved there on first access:
``summary_statistics``, ``balance_table``, ``balance_regression`` and
``outcome_table``.  They have no formula, no solver and no advisory, and they return
the table object of :mod:`powermean.results.tables`, whose exports they share with
``etable``.

==================  ================================================================
``options``         the keywords of the four functions, typed and validated
``frame``           rows, variables, arms, weights and subgroups read from a frame
``numeric``         moments, quantiles, the two-sample comparison, least squares,
                    sandwich covariances, conventions, joint tests, power moments
``regression``      the linear regression of the two balance functions
``common``          the pieces of layout that the tables share
``summary``         ``summary_statistics``
``balance``         ``balance_table``
``balance_regression``  ``balance_regression``
``outcome``         ``outcome_table``
==================  ================================================================

Importing this package loads none of its modules, and no module of the estimation
layers.
"""

__all__: list[str] = []
