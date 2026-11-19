"""The two scikit-learn files of the Newton iteration, carried by the package.

``design/02_computation.md``, Section 5.1; decisions D24, D40 and D48.  The modules
:mod:`powermean._vendor.vendor_newton` (``NewtonSolver`` and ``NewtonCholeskySolver``)
and :mod:`powermean._vendor.vendor_linear_loss` (``LinearModelLoss`` and
``sandwich_dot``) are the files ``vendor_newton.py`` and ``vendor_linear_loss.py`` of
the solver campaign, under their names and with the scikit-learn notice
``SKLEARN_COPYING`` of this folder (BSD-3-Clause).  They import nothing from
scikit-learn: the L-BFGS fallback and its two helpers are removed, and the four other
names that they imported are replaced by local equivalents in
:mod:`powermean._vendor._compat`.  Their types are in :mod:`powermean._vendor._typing`,
and the SciPy objects that they use are reached through
:mod:`powermean._vendor._scipy`.  ``PROVENANCE.md``, Section 6, lists every edit.

Rule 4 of ``design/11_engineering.md``, Section 3.1: only the Newton adapter,
``powermean.solve.newton_convex`` (task M0.10), imports this package; its modules
import one another.  Nothing here is public.
"""
