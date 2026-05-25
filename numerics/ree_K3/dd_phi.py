"""Numba double-double smooth Phi map for K=3 (policy-grade, fast).

Same kernel-smoothed Bayes map as `code.contour_K3_halo.phi_K3_halo_smooth`
and `phi_mp.phi_K3_smooth_mp`, but evaluated in double-double arithmetic inside
numba — ~32 digits at a small multiple of float64 cost, vs mpmath's ~100-1000x.

Purpose: drive the residual ||F||inf < 1e-20 (PRECISION_POLICY) on the discrete
smooth map without the mpmath bottleneck.  Halo cells fixed; inner updated.

Prices are carried as two float64 arrays (Phi_hi, Phi_lo).
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit

from dd_numba import (dd_add, dd_sub, dd_mul, dd_mul_d, dd_div, dd_exp, dd_log)

EPS = 1e-30


@njit(cache=True)
def _dd_f(u, mean, tau):
    """Signal density f_v(u) in DD (sqrt-coeff cancels in Bayes ratio -> f64)."""
    dh, dl = dd_sub(u, 0.0, mean, 0.0)
    d2h, d2l = dd_mul(dh, dl, dh, dl)
    eh, el = dd_mul_d(d2h, d2l, -0.5 * tau)
    exh, exl = dd_exp(eh, el)
    c = math.sqrt(tau / (2.0 * math.pi))
    return dd_mul_d(exh, exl, c)


@njit(cache=True)
def _dd_logit(ph, pl):
    ah, al = dd_log(ph, pl)
    oh, ol = dd_sub(1.0, 0.0, ph, pl)
    bh, bl = dd_log(oh, ol)
    return dd_sub(ah, al, bh, bl)


@njit(cache=True)
def _dd_excess(ph, pl, lm_h, lm_l, gamma, W):
    """sum_k W*(E-1)/D, E=exp((logit_mu_k-logit_p)/gamma), D=(1-p)+p*E."""
    lp_h, lp_l = _dd_logit(ph, pl)
    omp_h, omp_l = dd_sub(1.0, 0.0, ph, pl)
    s_h, s_l = 0.0, 0.0
    for k in range(3):
        zh, zl = dd_sub(lm_h[k], lm_l[k], lp_h, lp_l)
        if gamma != 1.0:
            zh, zl = dd_mul_d(zh, zl, 1.0 / gamma)
        Eh, El = dd_exp(zh, zl)
        # D = (1-p) + p*E
        peh, pel = dd_mul(ph, pl, Eh, El)
        Dh, Dl = dd_add(omp_h, omp_l, peh, pel)
        # num = W*(E-1)
        em1h, em1l = dd_sub(Eh, El, 1.0, 0.0)
        nh, nl = dd_mul_d(em1h, em1l, W)
        qh, ql = dd_div(nh, nl, Dh, Dl)
        s_h, s_l = dd_add(s_h, s_l, qh, ql)
    return s_h, s_l


@njit(cache=True)
def _dd_clear(mu_h, mu_l, gamma, W):
    """Bisection for p with sum_k x_k(p)=0, in DD (~110 steps -> ~1e-33)."""
    lm_h = np.empty(3); lm_l = np.empty(3)
    for k in range(3):
        lm_h[k], lm_l[k] = _dd_logit(mu_h[k], mu_l[k])
    ah, al = EPS, 0.0
    bh, bl = dd_sub(1.0, 0.0, EPS, 0.0)
    fa_h, _ = _dd_excess(ah, al, lm_h, lm_l, gamma, W)
    if fa_h <= 0.0:
        return ah, al
    fb_h, _ = _dd_excess(bh, bl, lm_h, lm_l, gamma, W)
    if fb_h >= 0.0:
        return bh, bl
    for _ in range(112):
        sh, sl = dd_add(ah, al, bh, bl)
        ch, cl = dd_mul_d(sh, sl, 0.5)
        fc_h, _ = _dd_excess(ch, cl, lm_h, lm_l, gamma, W)
        if fc_h >= 0.0:
            ah, al = ch, cl
        else:
            bh, bl = ch, cl
    sh, sl = dd_add(ah, al, bh, bl)
    return dd_mul_d(sh, sl, 0.5)


@njit(cache=True, parallel=False)
def dd_phi(Phi, Plo, u_full, inner_lo, inner_hi,
           tau_vec, gamma_vec, W_vec, kernel_h):
    """One DD application of the smooth map. Returns (new_hi, new_lo)."""
    G = u_full.size
    f0h = np.empty((3, G)); f0l = np.empty((3, G))
    f1h = np.empty((3, G)); f1l = np.empty((3, G))
    for k in range(3):
        for i in range(G):
            f0h[k, i], f0l[k, i] = _dd_f(u_full[i], -0.5, tau_vec[k])
            f1h[k, i], f1l[k, i] = _dd_f(u_full[i], 0.5, tau_vec[k])

    h2h, h2l = dd_mul(kernel_h, 0.0, kernel_h, 0.0)
    t2h, t2l = dd_mul_d(h2h, h2l, 2.0)
    i2h, i2l = dd_div(1.0, 0.0, t2h, t2l)
    ni_h, ni_l = -i2h, -i2l                       # -inv2h2 in DD
    skip = kernel_h * math.sqrt(2.0 * 34.0 * math.log(10.0))

    Nh = Phi.copy(); Nl = Plo.copy()
    mu_h = np.empty(3); mu_l = np.empty(3)
    for i in range(inner_lo, inner_hi):
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                ph = Phi[i, j, l]; pl = Plo[i, j, l]
                # agent 0: slice over (a=axis1, b=axis2) at fixed i
                for ag in range(3):
                    A0h = 0.0; A0l = 0.0; A1h = 0.0; A1l = 0.0
                    for a in range(G):
                        for b in range(G):
                            if ag == 0:
                                qh = Phi[i, a, b]; ql = Plo[i, a, b]
                                i1 = a; i2 = b; k1 = 1; k2 = 2
                            elif ag == 1:
                                qh = Phi[a, j, b]; ql = Plo[a, j, b]
                                i1 = a; i2 = b; k1 = 0; k2 = 2
                            else:
                                qh = Phi[a, b, l]; ql = Plo[a, b, l]
                                i1 = a; i2 = b; k1 = 0; k2 = 1
                            dh, dl = dd_sub(qh, ql, ph, pl)
                            if dh > skip or dh < -skip:
                                continue
                            d2h, d2l = dd_mul(dh, dl, dh, dl)
                            eh2, el2 = dd_mul(d2h, d2l, ni_h, ni_l)
                            wh, wl = dd_exp(eh2, el2)
                            t0h, t0l = dd_mul(f0h[k1, i1], f0l[k1, i1],
                                              f0h[k2, i2], f0l[k2, i2])
                            t0h, t0l = dd_mul(t0h, t0l, wh, wl)
                            A0h, A0l = dd_add(A0h, A0l, t0h, t0l)
                            t1h, t1l = dd_mul(f1h[k1, i1], f1l[k1, i1],
                                              f1h[k2, i2], f1l[k2, i2])
                            t1h, t1l = dd_mul(t1h, t1l, wh, wl)
                            A1h, A1l = dd_add(A1h, A1l, t1h, t1l)
                    idx = i if ag == 0 else (j if ag == 1 else l)
                    f0o_h = f0h[ag, idx]; f0o_l = f0l[ag, idx]
                    f1o_h = f1h[ag, idx]; f1o_l = f1l[ag, idx]
                    numh, numl = dd_mul(f1o_h, f1o_l, A1h, A1l)
                    d0h, d0l = dd_mul(f0o_h, f0o_l, A0h, A0l)
                    denh, denl = dd_add(d0h, d0l, numh, numl)
                    if denh > 0.0:
                        mh, ml = dd_div(numh, numl, denh, denl)
                    else:
                        mh, ml = 0.5, 0.0
                    if mh < EPS:
                        mh, ml = EPS, 0.0
                    elif mh > 1.0 - 1e-16:
                        mh, ml = dd_sub(1.0, 0.0, EPS, 0.0)
                    mu_h[ag] = mh; mu_l[ag] = ml
                Nh[i, j, l], Nl[i, j, l] = _dd_clear(mu_h, mu_l,
                                                     gamma_vec[0], W_vec[0])
    return Nh, Nl
