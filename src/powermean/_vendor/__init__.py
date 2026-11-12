"""The two scikit-learn files of the Newton iteration, carried by the package.

The modules
:mod:`powermean._vendor.vendor_newton` (``NewtonSolver`` and ``NewtonCholeskySolver``)
and :mod:`powermean._vendor.vendor_linear_loss` (``LinearModelLoss`` and
``sandwich_dot``) are carried from scikit-learn, with the scikit-learn notice
``SKLEARN_COPYING`` of this folder (BSD-3-Clause).  They import nothing from
scikit-learn: the L-BFGS fallback and its two helpers are removed, and the four other
names that they imported are replaced by local equivalents in
:mod:`powermean._vendor._compat`.  Their types are in :mod:`powermean._vendor._typing`,
and the SciPy objects that they use are reached through
:mod:`powermean._vendor._scipy`.

Only the Newton adapter, ``powermean.solve.newton_convex``, imports this package; its
modules import one another.  Nothing here is public.
"""
