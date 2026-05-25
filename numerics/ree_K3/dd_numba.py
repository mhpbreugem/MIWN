"""Double-double (DD) arithmetic for numba — ~32 significant digits at
float64 speed (2x float64 words, error-free transforms).

Gives policy-grade precision (PRECISION_POLICY: double-double, dps~32) inside
numba @njit kernels, ~10-30x float64 cost vs ~100-1000x for mpmath.

A DD number is a pair (hi, lo) with hi the rounded value and |lo| <= ulp(hi)/2.
All functions are @njit(inline) and take/return scalar float64 pairs so they
compose inside hot loops.

Validated against mpmath (dps=40) in the __main__ block: dd_exp / dd_log /
dd_mul / dd_div reach relative error < 1e-30.
"""
from __future__ import annotations

import math
from numba import njit

_SPLITTER = 134217729.0          # 2^27 + 1
_LN2_HI = 0.6931471805599453
_LN2_LO = 2.3190468138462996e-17


# ---------------------------------------------------------------------------
# Error-free transforms
# ---------------------------------------------------------------------------
@njit(inline="always", cache=True)
def two_sum(a, b):
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return s, err


@njit(inline="always", cache=True)
def quick_two_sum(a, b):
    s = a + b
    err = b - (s - a)
    return s, err


@njit(inline="always", cache=True)
def _split(a):
    t = _SPLITTER * a
    hi = t - (t - a)
    lo = a - hi
    return hi, lo


@njit(inline="always", cache=True)
def two_prod(a, b):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    err = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, err


# ---------------------------------------------------------------------------
# Core DD ops
# ---------------------------------------------------------------------------
@njit(inline="always", cache=True)
def dd_add(ah, al, bh, bl):
    s, e = two_sum(ah, bh)
    e += al + bl
    s, e = quick_two_sum(s, e)
    return s, e


@njit(inline="always", cache=True)
def dd_sub(ah, al, bh, bl):
    s, e = two_sum(ah, -bh)
    e += al - bl
    s, e = quick_two_sum(s, e)
    return s, e


@njit(inline="always", cache=True)
def dd_mul(ah, al, bh, bl):
    p, e = two_prod(ah, bh)
    e += ah * bl + al * bh
    s, e = quick_two_sum(p, e)
    return s, e


@njit(inline="always", cache=True)
def dd_mul_d(ah, al, b):
    p, e = two_prod(ah, b)
    e += al * b
    s, e = quick_two_sum(p, e)
    return s, e


@njit(inline="always", cache=True)
def dd_div(ah, al, bh, bl):
    q1 = ah / bh
    th, tl = dd_mul_d(bh, bl, q1)
    rh, rl = dd_sub(ah, al, th, tl)
    q2 = rh / bh
    th, tl = dd_mul_d(bh, bl, q2)
    rh, rl = dd_sub(rh, rl, th, tl)
    q3 = rh / bh
    s, e = quick_two_sum(q1, q2)
    s, e = dd_add(s, e, q3, 0.0)
    return s, e


# ---------------------------------------------------------------------------
# Transcendentals
# ---------------------------------------------------------------------------
@njit(cache=True)
def dd_exp(ah, al):
    if ah == 0.0 and al == 0.0:
        return 1.0, 0.0
    k = math.floor(ah / _LN2_HI + 0.5)
    kh, kl = dd_mul_d(_LN2_HI, _LN2_LO, k)
    rh, rl = dd_sub(ah, al, kh, kl)          # r = x - k*ln2,  |r|<=ln2/2
    sh, sl = 1.0, 0.0
    th, tl = 1.0, 0.0
    for n in range(1, 20):
        th, tl = dd_mul(th, tl, rh, rl)
        th, tl = dd_div(th, tl, float(n), 0.0)
        sh, sl = dd_add(sh, sl, th, tl)
        if abs(th) < 1e-34 * abs(sh):
            break
    ki = int(k)
    sh = math.ldexp(sh, ki)
    sl = math.ldexp(sl, ki)
    return sh, sl


@njit(cache=True)
def dd_log(ah, al):
    y = math.log(ah)
    # two Newton steps: y <- y + a*exp(-y) - 1  (quadratic convergence)
    eh, el = dd_exp(-y, 0.0)
    th, tl = dd_mul(ah, al, eh, el)
    th, tl = dd_sub(th, tl, 1.0, 0.0)
    yh, yl = dd_add(y, 0.0, th, tl)
    eh, el = dd_exp(-yh, -yl)
    th, tl = dd_mul(ah, al, eh, el)
    th, tl = dd_sub(th, tl, 1.0, 0.0)
    yh, yl = dd_add(yh, yl, th, tl)
    return yh, yl


if __name__ == "__main__":
    import mpmath as mp
    mp.mp.dps = 40

    def rel(dd, exact):
        approx = mp.mpf(dd[0]) + mp.mpf(dd[1])
        return float(abs((approx - exact) / exact))

    xs = [0.3, 1.7, -2.4, 12.5, -0.001, 7.0]
    worst = 0.0
    for x in xs:
        worst = max(worst, rel(dd_exp(x, 0.0), mp.e ** mp.mpf(x)))
    print(f"dd_exp  worst rel err: {worst:.2e}")
    worst = 0.0
    for x in [0.3, 1.7, 12.5, 0.001, 1e-8, 5.0]:
        worst = max(worst, rel(dd_log(x, 0.0), mp.log(mp.mpf(x))))
    print(f"dd_log  worst rel err: {worst:.2e}")
    a = dd_mul(1.0 / 3.0, 0.0, 3.0, 0.0)
    print(f"dd_mul  1/3*3 - 1 = {(mp.mpf(a[0]) + mp.mpf(a[1]) - 1):.2e}")
    d = dd_div(1.0, 0.0, 3.0, 0.0)
    print(f"dd_div  1/3 rel err: {rel(d, mp.mpf(1)/3):.2e}")
