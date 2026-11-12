"""Internal coordination of the formula front end with the sample construction.

:mod:`powermean.formula` evaluates a formula on every row with pandas and
Formulaic and may import no later stage; :mod:`powermean.data` constructs the sample
on arrays and may import neither pandas nor Formulaic nor the interface.  The modules
of this package stand between them.  :mod:`powermean._prepare.compose` reads an
evaluated :class:`~powermean.formula.frame.ModelFrame` and the columns of its data,
feeds arrays and masks to :func:`powermean.data.sample.build_sample`, and keeps the
user's pandas Index in a state of its own, selected by position only;
:mod:`powermean._prepare.effects` reads the raw labels of a formula's fixed-effect terms
by position and marks their missing values for step 2 of the sample construction.
They evaluate no formula, transform no outcome and fit nothing, and this package
exports no public name.  No stage of the pipeline imports it.
"""
