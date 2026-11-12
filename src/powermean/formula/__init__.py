"""The formula front end: the three-part grammar and the model frame of its first part.

:mod:`powermean.formula.grammar`
splits a formula into its parts with the standard library only;
:mod:`powermean.formula.frame` evaluates the first part with Formulaic on every input
row, through the typed views of :mod:`powermean.formula.formulaic_api`,
:mod:`powermean.formula.names` reads the names that a Python factor loads from the
data, and :mod:`powermean.formula.validity` marks the rows whose values are missing or
not finite.  Nothing here selects the estimation sample.  The package is
internal: ``powermean`` exports none of its names.
"""
