"""Row kernels on ``(z, q)``: score residual, curvature weights and criterion.

For a transformed outcome ``z >= 0``, an index ``q`` and a working variance ``V``,
with ``m = exp(q)``, ``e`` the elasticity of ``V`` and ``w = m^2 / V(m)``:

    score residual         r = (z - m) m / V(m)
    curvature weight       c = w ((2 - e) + (e - 1) z / m),   c = -dr/dq
    information weight     w = m^2 / V(m)
    criterion              l(q; z),   dl/dq = -r

``c`` is the Hessian weight of BBP, ``eq:app-iols-exact-hessian``.  The criterion of the
power family is ``m - z q`` at ``tau = 1``, ``q + z / m`` at ``tau = 2`` and
``m^(2-tau)/(2-tau) + z m^(1-tau)/(tau-1)`` otherwise; that of NB2 is ``-z
log(m/(1+kappa m)) + log(1+kappa m)/kappa``, the Poisson criterion at ``kappa = 0``.  No
additive constant is removed from these forms.

Numerical precautions: log products are combined before exponentiating; the curvature
weight is a signed sum of two log terms and is never floored, since it is negative for
some rows when ``tau`` is outside ``[1, 2]``; a share that rounds to zero in ``e`` is
kept as a log, so that a tiny share times a large component survives; the score of a
positive row near its root is ``w log(z/m) expm1(d)/d``; zero rows are ``r = -w``, ``c =
(2 - e) w`` and ``l`` in closed form, so no product ``0 * exp(q)`` is formed.  A genuine
final underflow rounds to zero; a final value that is not representable raises
``FloatingPointError``.  Nothing is clipped, and no endpoint is substituted for a nearby
``tau`` or ``kappa``.  Validation errors are built-in exceptions; their public
translation is interface work.

The criterion is evaluated as written wherever its terms are representable.  A row whose
direct evaluation is not finite is evaluated again from the logs of its two terms
(``_log_sum``): a product such as ``exp(q) log1p(x)/x`` or ``z log(1/m + kappa)`` and a
signed sum of two terms that overflow while their sum does not are then formed without
the terms themselves.  This is the same analytic function in a second arithmetic, not a
substitute family; a sum that is itself not representable still raises.  A scaled term
``exp(a)/d`` of the power criterion whose exponential is not a normal double is formed
as ``exp(a - log|d|)``, a single rounding: at a ``tau`` adjacent to an endpoint,
``1/|d|`` is of order ``1e16`` and the direct quotient would otherwise lose the bits
that the subnormal ``exp(a)`` has lost, or return zero for a representable value
(``_scaled_exponential``).  The direct arithmetic, and so the zero-row form, is kept bit
for bit wherever ``exp(a)`` is normal.  The factor ``log(1/m + kappa)`` of the NB2 term
in ``z`` is formed, in the band where its root lies, from the exact difference ``kappa -
q`` and the series of ``exp(-q) - 1 + q``, so that its second-order part survives the
multiplication by a large outcome (``_nb2_pieces``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from powermean.model.variance import (
    FloatArray,
    NB2Variance,
    PowerVariance,
    Variance,
    _finite_array,
    _finite_result,
)

FIELDS = ("score", "observed_curvature", "fisher_curvature", "log_mean")
# log(2**-1022): exp(a) is a normal double, with its full 53 bits, for a at or above
# this value; below it, exp(a) is subnormal or zero and has lost bits before any
# further operation (the boundary of double-precision representability, not a
# tolerance on tau or on q).
LOG_SMALLEST_NORMAL = float(np.log(np.finfo(np.float64).tiny))
# log 2: where kappa m <= 1, the root of the NB2 factor log(1/m + kappa), at which
# exp(-q) + kappa is one, lies in the band 0 < q <= log 2, since kappa <= exp(-q)
# there makes exp(-q) at least 1/2 (a property of the factor, not a tolerance).
LOG_TWO = float(np.log(2.0))
# The terms of the tail exp(-q) - 1 + q kept on that band, through q^17/17!.  The
# series alternates with decreasing terms, so its truncation is below the first
# omitted term, q^18/18!, which is below 2^-59 of the tail on the band: beneath the
# rounding of a double, a truncation of the representation and not a tolerance.
EXP_TAIL_TERMS = 17


def _immutable(value: FloatArray) -> FloatArray:
    """An independent read-only copy of ``value``, in its shape."""
    array: FloatArray = np.frombuffer(
        np.asarray(value, dtype=np.float64).tobytes(), dtype=np.float64
    ).reshape(value.shape)
    return array


@dataclass(frozen=True)
class RowEvaluation:
    """The row terms at one index: ``r``, ``c``, ``w`` and the log mean ``q``.

    ``score`` is the score residual ``r``; ``observed_curvature`` the curvature weight
    ``c``, the exact negative derivative of ``r``; ``fisher_curvature`` the information
    weight ``w``, its conditional expectation.  The two curvatures are not substitutes
    for each other.  Each field is stored as an independent read-only array.
    """

    score: FloatArray
    observed_curvature: FloatArray
    fisher_curvature: FloatArray
    log_mean: FloatArray

    def __post_init__(self) -> None:
        """Check that every field is finite and store it immutably."""
        for name in FIELDS:
            value = _finite_array(getattr(self, name), name)
            object.__setattr__(self, name, _immutable(value))


def _exp(log_value: FloatArray) -> FloatArray:
    """``exp`` of a log quantity; an unrepresentable result is an error, not ``inf``."""
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        value = np.exp(log_value)
    return _finite_result(value, "row evaluation")


def _signed_sum(
    log_a: FloatArray, sign_a: float, log_b: FloatArray, sign_b: float
) -> FloatArray:
    """Signed two-term sum without requiring representable individual terms."""
    a, b = np.broadcast_arrays(log_a, log_b)
    if np.isnan(a).any() or np.isnan(b).any():
        raise FloatingPointError("nonfinite log product in row evaluation")
    if sign_a == 0:
        only_b: FloatArray = sign_b * _exp(b)
        return only_b
    if sign_b == 0:
        only_a: FloatArray = sign_a * _exp(a)
        return only_a
    if sign_a == sign_b:
        same_sign: FloatArray = sign_a * _exp(np.logaddexp(a, b))
        return same_sign
    result = np.zeros(a.shape)
    unequal = a != b
    aa, bb = a[unequal], b[unequal]
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        log_size = np.maximum(aa, bb) + np.log(-np.expm1(-np.abs(aa - bb)))
    result[unequal] = sign_a * np.sign(aa - bb) * _exp(log_size)
    if np.isposinf(a[~unequal]).any():
        raise FloatingPointError("unresolved infinite cancellation in row evaluation")
    return result


def _log_components(
    log_m: FloatArray, log_z: FloatArray, variance: PowerVariance | NB2Variance
) -> tuple[FloatArray, FloatArray, float, float, float, float]:
    """Return log(m**2/V), log(m*z/V), and curvature coefficients.

    The curvature weight is ``u m z / V + v m^2 / V`` with ``u = e - 1`` and
    ``v = 2 - e``.  Both coefficients are returned as a log and a sign: a rounded
    elasticity equal to one or two must not erase a tiny share times a large score
    component.  ``log_z`` is ``-inf`` at a zero outcome.
    """
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        if isinstance(variance, PowerVariance):
            tau = variance.tau
            log_b = (2 - tau) * log_m
            log_a = log_z + (1 - tau) * log_m
            u, v = tau - 1, 2 - tau
            return log_b, log_a, np.log(abs(u)), np.sign(u), np.log(abs(v)), np.sign(v)
        if variance.kappa == 0:
            return log_m, log_z, -np.inf, 0, 0.0, 1
        log_k = np.log(variance.kappa)
        odds = log_k + log_m
        log_b = -np.logaddexp(-log_m, log_k)
        log_a = log_z - np.logaddexp(0.0, odds)
        log_u = -np.logaddexp(0.0, -odds)
        log_v = -np.logaddexp(0.0, odds)
        return log_b, log_a, log_u, 1, log_v, 1


def _aligned_rows(
    z: ArrayLike, q: ArrayLike, variance: Variance
) -> tuple[FloatArray, FloatArray]:
    """The validation of the rows: built-in variance, aligned finite rows.

    Only the two verified built-in variances are enabled, by exact type: a subclass or
    another object of the protocol is refused, since a structural protocol alone is
    not a numerical certification.
    """
    if type(variance) not in (PowerVariance, NB2Variance):
        raise TypeError("variance must be a supported built-in working variance")
    z, q = _finite_array(z, "z"), _finite_array(q, "q")
    if z.ndim != 1 or q.ndim != 1 or z.shape != q.shape or not z.size:
        raise ValueError("z and q must be nonempty aligned one-dimensional arrays")
    if np.any(z < 0):
        raise ValueError("z must be nonnegative")
    return z, q


def evaluate_rows(
    z: ArrayLike, q: ArrayLike, *, variance: PowerVariance | NB2Variance
) -> RowEvaluation:
    """Evaluate ``r``, ``c`` and ``w`` on aligned one-dimensional rows.

    At mean order one, ``log_m`` is ``q`` itself and the normalized residual
    ``expm1(d)/omega`` is ``expm1(d)``, with ``d = log(z) - q``. Zero outcomes are valid
    and are evaluated in closed form.

    Parameters
    ----------
    z : array
        Transformed outcome, nonnegative and finite.
    q : array
        Index of the transformed mean, finite, aligned with ``z``.
    variance : PowerVariance or NB2Variance
        The working variance, by exact type.

    Returns
    -------
    RowEvaluation
        Read-only arrays ``score``, ``observed_curvature``, ``fisher_curvature`` and
        ``log_mean``.

    Raises
    ------
    FloatingPointError
        A final value is not representable in double precision.
    """
    z, q = _aligned_rows(z, q, variance)
    positive = z > 0
    log_z = np.zeros_like(z)
    log_z[positive] = np.log(z[positive])
    log_m = q
    log_zw = log_z.copy()
    log_zw[~positive] = -np.inf
    components = _log_components(log_m, log_zw, variance)
    log_b, log_a, log_u, sign_u, log_v, sign_v = components
    fisher = _exp(log_b)
    difference = log_z - q
    score = np.zeros_like(z)
    d = difference
    small = positive & (np.abs(d) <= 0.5)
    nonzero = small & (difference != 0)
    # expm1(d)/omega at omega = 1: (log(z) - q) * exprel(d), accurate near the root.
    dd = d[nonzero]
    relative = np.expm1(dd) / dd
    score[nonzero] = np.sign(difference[nonzero]) * _exp(
        log_b[nonzero] + np.log(np.abs(difference[nonzero])) + np.log(relative)
    )
    above = positive & ~small & (d > 0)
    below = positive & ~small & (d < 0)
    with np.errstate(under="ignore", divide="ignore"):
        score[above] = _exp(log_a[above] + np.log(-np.expm1(-d[above])))
        score[below] = -_exp(log_b[below] + np.log(-np.expm1(d[below])))
        score[~positive] = -_exp(log_b[~positive])
    curvature = _signed_sum(log_a + log_u, sign_u, log_b + log_v, sign_v)
    return RowEvaluation(score, curvature, fisher, log_m)


LogTerms = tuple[FloatArray, FloatArray, FloatArray, FloatArray]


def _log_sum(
    log_a: FloatArray, sign_a: FloatArray, log_b: FloatArray, sign_b: FloatArray
) -> FloatArray:
    """``sign_a exp(log_a) + sign_b exp(log_b)`` without forming either term.

    The result is ``s exp(L + t)`` with ``L`` the larger log, ``s`` the sign of its
    term and ``t`` the log of ``1 + exp(-|delta|)`` or of ``1 - exp(-|delta|)``, for
    terms of the same or of opposite signs, ``delta`` being the difference of the
    logs.  A term that is zero has log ``-inf``; two zero terms give zero; two terms
    of opposite signs and equal logs give zero; a sum that is not representable is
    ``inf`` or ``nan`` and is refused by the caller.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        first = log_a >= log_b
        large = np.where(first, log_a, log_b)
        small = np.where(first, log_b, log_a)
        sign = np.where(first, sign_a, sign_b)
        delta = small - large
        tail = np.where(
            sign_a == sign_b, np.log1p(np.exp(delta)), np.log(-np.expm1(delta))
        )
        value: FloatArray = sign * np.exp(large + tail)
        value[np.isneginf(large)] = 0.0
    return value


def _power_log_terms(z: FloatArray, q: FloatArray, tau: float) -> LogTerms:
    """The two terms of the power criterion as log magnitude and sign, row by row.

    The log products are those of :func:`_power_criterion`; the constant factors
    ``1/(2 - tau)`` and ``1/(tau - 1)`` enter as logs.  A zero row has ``log z = -inf``.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        log_z = np.log(z)
        ones = np.ones_like(q)
        if tau == 1:
            return q, ones, log_z + np.log(np.abs(q)), -np.sign(q)
        if tau == 2:
            return np.log(np.abs(q)), np.sign(q), log_z - q, ones
        log_a = (2 - tau) * q - np.log(abs(2 - tau))
        log_b = log_z + (1 - tau) * q - np.log(abs(tau - 1))
        return log_a, np.sign(2 - tau) * ones, log_b, np.sign(tau - 1) * ones


def _scaled_exponential(log_value: FloatArray, coefficient: float) -> FloatArray:
    """``exp(log_value) / coefficient``, row by row, without a premature underflow.

    The direct quotient is formed wherever ``exp(log_value)`` is a normal double
    (``log_value >= LOG_SMALLEST_NORMAL``): this is the direct arithmetic of the
    zero-row form and of the displayed form on positive rows, kept bit for bit.  Where
    the exponential is subnormal or zero it has already lost bits, or everything, before
    the division: at a ``tau`` adjacent to one or two, ``|coefficient|`` is ``2^-52`` or
    ``2^-53`` and ``exp(-746)/coefficient`` is a representable ``4.7e-309`` that the
    direct quotient returns as zero, while ``exp(-744)/coefficient`` is a normal
    ``3.5e-308`` that it returns with three correct bits.  Those rows are formed as
    ``sign(coefficient) exp(log_value - log|coefficient|)``, one rounding of the value
    itself; a value below half the smallest subnormal rounds to zero there, which is the
    genuine final underflow.  The coefficient is a nonzero double.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value: FloatArray = np.exp(log_value) / coefficient
        premature = log_value < LOG_SMALLEST_NORMAL
        if premature.any():
            value[premature] = np.sign(coefficient) * np.exp(
                log_value[premature] - np.log(abs(coefficient))
            )
    return value


def _power_criterion(z: FloatArray, q: FloatArray, tau: float) -> FloatArray:
    """The criterion of the power family, row by row.

    Zero rows take closed forms: ``q`` at ``tau = 2``, which forms no exponential, and
    ``exp((2 - tau) q) / (2 - tau)`` otherwise.  Positive rows write ``z m^(1-tau)`` as
    ``exp(log z + (1 - tau) q)`` so that a representable product of a large outcome and
    a small power is not lost to an underflow.  Each scaled term is formed by
    :func:`_scaled_exponential`, which keeps the direct quotient wherever its
    exponential is a normal double.  A row whose direct value is not finite is evaluated
    again by :func:`_log_sum` from the logs of its terms, which the direct arithmetic
    leaves untouched elsewhere.
    """
    positive = z > 0
    value = np.empty_like(q)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_z = np.log(z[positive])
        qp = q[positive]
        if tau == 1:
            value[~positive] = np.exp(q[~positive])
            value[positive] = np.exp(qp) - z[positive] * qp
        elif tau == 2:
            value[~positive] = q[~positive]
            value[positive] = qp + np.exp(log_z - qp)
        else:
            value[~positive] = _scaled_exponential((2 - tau) * q[~positive], 2 - tau)
            value[positive] = _scaled_exponential(
                (2 - tau) * qp, 2 - tau
            ) + _scaled_exponential(log_z + (1 - tau) * qp, tau - 1)
    indirect = ~np.isfinite(value)
    if indirect.any():
        value[indirect] = _log_sum(*_power_log_terms(z[indirect], q[indirect], tau))
    return value


def _exp_tail(q: FloatArray) -> FloatArray:
    """``exp(-q) - 1 + q`` on ``0 < q <= log 2``, row by row, by its Taylor series.

    The series ``q^2/2! - q^3/3! + q^4/4! - ...`` is evaluated in the nested form
    ``(q^2/2) (1 - (q/3) (1 - (q/4) (... (1 - q/17))))``, through ``q^17/17!``
    (``EXP_TAIL_TERMS``).  Each nested factor lies in ``(3/4, 1]``, so no difference of
    nearly equal quantities is formed and the tail has a relative error of a few
    units of ``2^-53``.  ``expm1(-q) + q`` loses the tail to the rounding of ``q``,
    and is zero once ``q^2/2`` is below half the spacing of the doubles at ``q``.  A
    tail below the smallest normal double, ``2^-1022``, is rounded to a multiple of
    the smallest subnormal, ``2^-1074``, with an absolute error of at most half that
    spacing rather than a relative one: a tail between ``2^-1075`` and ``2^-1074``
    rounds up to ``2^-1074``, and one below ``2^-1075``, half the smallest subnormal,
    rounds to zero, a genuine final underflow.
    """
    nested = np.ones_like(q)
    for k in range(EXP_TAIL_TERMS, 2, -1):
        nested = 1.0 - q / k * nested
    tail: FloatArray = 0.5 * q * q * nested
    return tail


def _nb2_pieces(
    q: FloatArray, kappa: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """``log x``, ``x``, ``log1p(x)/x``, the mask ``x > 1`` and ``log(1/m + kappa)``.

    ``x = kappa m``; the ratio is one where ``x`` underflows.  The last piece is the
    logarithmic factor ``log(exp(-q) + kappa)`` of the term ``z log(1/m + kappa)``,
    before its multiplication by ``z``: with ``z`` up to ``1e308``, an absolute error
    of the factor is multiplied by ``z`` in the criterion, so even a tiny factor needs
    a small relative error.  The factor vanishes where ``exp(-q) + kappa`` is one, a
    root that lies in the band ``0 < q <= log 2`` (``LOG_TWO``) where ``x <= 1``, and
    beyond ``q = log 2`` where ``x > 1``.  It is formed as

    - ``log1p((kappa - q) + g(q))`` in that band, with the tail
      ``g(q) = exp(-q) - 1 + q``, at most ``q^2/2``, of :func:`_exp_tail`.  Near the
      root ``kappa`` is within a factor two of ``q``, so ``kappa - q`` is exact
      (Sterbenz lemma): beyond its own final rounding, the factor carries that of
      ``g(q)`` only, a few units of ``2^-53 g(q)`` rather than of ``2^-53 q``, and
      its second-order part, ``q^2/2`` at ``kappa = q``, is kept.  The form
      ``-q + log1p(kappa exp(q))`` loses that part, and at ``kappa = q`` below
      ``2^-53``, where ``exp(q)`` rounds to one, it is zero.  Where ``kappa`` is not
      within a factor two of ``q`` the factor exceeds ``q/5`` in magnitude, and the
      rounding of ``kappa - q`` is not amplified;
    - ``-q + log1p(kappa m)`` elsewhere where ``x <= 1``: two terms of one sign for
      ``q <= 0``, and a factor at most ``log 2 - q`` beyond the band.  ``kappa m`` is
      the product of the given ``kappa`` and ``exp(q)``, not ``exp(log kappa + q)``,
      whose rounding of ``log kappa`` is ``|log kappa|`` times that of a double; it
      is ``x`` itself where ``exp(q)`` overflows while ``x`` does not (a subnormal
      ``kappa`` with ``q`` beyond 709);
    - ``logaddexp(-q, log kappa)``, that is ``log kappa + log1p(1/x)``, where
      ``x > 1``.

    A factor smaller than the rounding of the quantities it is formed from, as at
    the double ``kappa`` nearest the root for a given ``q``, is not resolved: that is
    a limit of double-precision arithmetic, not a tolerance.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        log_x = np.log(kappa) + q
        x = np.exp(log_x)
        ratio = np.divide(np.log1p(x), x, out=np.ones_like(x), where=x > 0)
        upper = log_x > 0
        product = kappa * np.exp(q)
        linear = -q + np.log1p(np.where(np.isfinite(product), product, x))
        band = ~upper & (q > 0) & (q <= LOG_TWO)
        linear[band] = np.log1p((kappa - q[band]) + _exp_tail(q[band]))
        linear[upper] = np.logaddexp(-q[upper], np.log(kappa))
    return log_x, x, ratio, upper, linear


def _nb2_log_terms(z: FloatArray, q: FloatArray, kappa: float) -> LogTerms:
    """The two terms of the NB2 criterion as log magnitude and sign, row by row.

    ``log(1 + kappa m)/kappa`` is ``q + log(log1p(x)/x)`` where ``x <= 1`` and
    ``log(logaddexp(0, log x)) - log kappa`` where ``x > 1``; ``z log(1/m + kappa)``
    is ``log z + log|log(1/m + kappa)|``, the last factor as in :func:`_nb2_pieces`,
    with the sign of that factor.
    """
    log_x, _, ratio, upper, linear = _nb2_pieces(q, kappa)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        log_a = q + np.log(ratio)
        log_a[upper] = np.log(np.logaddexp(0.0, log_x[upper])) - np.log(kappa)
        log_b = np.log(z) + np.log(np.abs(linear))
    return log_a, np.ones_like(q), log_b, np.sign(linear)


def _nb2_criterion(z: FloatArray, q: FloatArray, kappa: float) -> FloatArray:
    """The criterion of NB2, row by row.

    ``-z log(m/(1+kappa m))`` is ``z log(1/m + kappa)``, the factor formed by
    :func:`_nb2_pieces`: in the band of its root from the exact difference
    ``kappa - q`` and the tail ``exp(-q) - 1 + q``, so that its second-order part is
    kept, and elsewhere as ``-q + log1p(kappa m)`` from the given ``kappa`` and
    ``exp(q)`` where ``kappa m <= 1`` and as ``logaddexp(-q, log kappa)`` beyond.  In
    that band no rounding of ``log kappa`` enters the factor; for ``q > log 2`` its
    root ``kappa = 1 - exp(-q)`` lies where ``kappa m > 1``, and there
    ``logaddexp(-q, log kappa)`` carries the rounding of ``log kappa`` into a factor
    that nearly cancels.
    ``log(1 + kappa m)/kappa`` is ``m log1p(x)/x`` with ``x = kappa m`` where
    ``x <= 1``, the ratio being one where ``x`` underflows, and
    ``logaddexp(0, log x)/kappa`` where ``x > 1``; so a small or subnormal ``kappa``
    loses no accuracy and tends to the Poisson term ``m``.  At ``kappa = 0`` the
    criterion is the Poisson one of :func:`_power_criterion` at ``tau = 1``, exactly.
    A row whose direct value is not finite is evaluated again by :func:`_log_sum`
    from the logs of its terms.
    """
    if kappa == 0:
        return _power_criterion(z, q, 1.0)
    positive = z > 0
    log_x, _, ratio, upper, linear = _nb2_pieces(q, kappa)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        value: FloatArray = np.exp(q) * ratio
        value[upper] = np.logaddexp(0.0, log_x[upper]) / kappa
        value[positive] += z[positive] * linear[positive]
    indirect = ~np.isfinite(value)
    if indirect.any():
        value[indirect] = _log_sum(*_nb2_log_terms(z[indirect], q[indirect], kappa))
    return value


def evaluate_criterion(
    z: ArrayLike, q: ArrayLike, *, variance: PowerVariance | NB2Variance
) -> FloatArray:
    """Evaluate the criterion ``l(q; z)``, row by row.

    Its derivative in ``q`` is minus the score residual of :func:`evaluate_rows`, and
    its second derivative the observed curvature.  The function is separate from the
    row evaluation so that a criterion that is not representable, as that of NB2 at
    ``kappa = 1`` on the row ``z = 1e308, q = -800`` (whose term ``z log(1/m + 1)``
    exceeds the largest double while ``r``, ``c`` and ``w`` are representable), does
    not prevent the evaluation of the rows; and conversely a representable criterion,
    as ``exp(710)/2`` at ``tau = 0`` on the zero row ``q = 355``, is returned where the
    information weight ``exp(710)`` is not.

    Parameters
    ----------
    z : array
        Transformed outcome, nonnegative and finite.
    q : array
        Index of the transformed mean, finite, aligned with ``z``.
    variance : PowerVariance or NB2Variance
        The working variance, by exact type.

    Returns
    -------
    array
        Read-only row contributions ``l_i``.

    Raises
    ------
    FloatingPointError
        A contribution is not representable in double precision.
    """
    z, q = _aligned_rows(z, q, variance)
    if isinstance(variance, PowerVariance):
        value = _power_criterion(z, q, variance.tau)
    else:
        value = _nb2_criterion(z, q, variance.kappa)
    return _immutable(_finite_result(value, "row criterion"))
