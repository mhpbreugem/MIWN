"""Bandwidth-free SMOOTH co-area via a direct contour line integral.

A_v(p) = INT_{P=p} f_v / |grad P| ds, computed as a line integral using a C2
cubic-spline interpolant ALONG each scan line (a global spline through the whole
row, so the root moves smoothly with no stencil-shift kinks).  Averaged over the
two axes.  No kernel bandwidth.  Phi is smooth in the nodal prices (spline coeffs
are linear in them; the root is smooth by the implicit-function theorem wherever
dP/du != 0), so Newton/Anderson converge far below the crossing-count floor.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline

EPS = 1e-12

def _fsig(u, v, tau):
    m = 0.5 if v == 1 else -0.5
    return np.exp(-tau / 2.0 * (u - m) ** 2)   # coeff cancels in Bayes ratio

def _axis_evidence(slice2, p, u, tau_row, tau_free, axis):
    """Sum over scan lines along `axis`; spline the free axis, root-find at p."""
    G = u.size
    A0 = 0.0; A1 = 0.0
    for r in range(G):
        line = slice2[r, :] if axis == 0 else slice2[:, r]
        cs = CubicSpline(u, line)            # C2 interpolant along the line
        roots = cs.solve(p, extrapolate=False)
        if roots.size == 0:
            continue
        d = cs(roots, 1)
        u_row = u[r]
        f0r = _fsig(u_row, 0, tau_row); f1r = _fsig(u_row, 1, tau_row)
        for k in range(roots.size):
            dd = abs(d[k])
            if dd < EPS:
                continue
            uf = roots[k]
            A0 += f0r * _fsig(uf, 0, tau_free) / dd
            A1 += f1r * _fsig(uf, 1, tau_free) / dd
    return A0, A1

def evidence(slice2, p, u, tau_a, tau_b):
    """Two-axis-averaged co-area line integral. axis0 carries tau_a, axis1 tau_b."""
    # pass A: scan along axis0 (rows=tau_a on grid), spline axis1 (free=tau_b)
    A0a, A1a = _axis_evidence(slice2, p, u, tau_a, tau_b, axis=0)
    # pass B: scan along axis1 (cols=tau_b on grid), spline axis0 (free=tau_a)
    A0b, A1b = _axis_evidence(slice2, p, u, tau_b, tau_a, axis=1)
    return 0.5 * (A0a + A0b), 0.5 * (A1a + A1b)

def _bayes(u_own, tau_own, A0, A1):
    f0 = _fsig(u_own, 0, tau_own); f1 = _fsig(u_own, 1, tau_own)
    num = f1 * A1; den = f0 * A0 + num
    if den <= 0.0:
        return 0.5
    return min(max(num / den, EPS), 1.0 - EPS)

def _clear(mu, gamma, W):
    a, b = EPS, 1.0 - EPS
    lm = [np.log(m / (1 - m)) for m in mu]
    def excess(p):
        lp = np.log(p / (1 - p)); s = 0.0
        for q in lm:
            z = (q - lp) / gamma; e = np.exp(z)
            s += W * (e - 1.0) / ((1 - p) + p * e)
        return s
    if excess(a) <= 0: return a
    if excess(b) >= 0: return b
    for _ in range(80):
        c = 0.5 * (a + b)
        if excess(c) >= 0: a = c
        else: b = c
    return 0.5 * (a + b)

def phi_lineint(P_full, u_full, lo, hi, tau_vec, gamma_vec, W_vec):
    """Bandwidth-free smooth co-area Phi (inner updated, halo fixed)."""
    Pn = P_full.copy()
    for i in range(lo, hi):
        for j in range(lo, hi):
            for l in range(lo, hi):
                p = P_full[i, j, l]
                a0, a1 = evidence(P_full[i, :, :], p, u_full, tau_vec[1], tau_vec[2])
                m0 = _bayes(u_full[i], tau_vec[0], a0, a1)
                a0, a1 = evidence(P_full[:, j, :], p, u_full, tau_vec[0], tau_vec[2])
                m1 = _bayes(u_full[j], tau_vec[1], a0, a1)
                a0, a1 = evidence(P_full[:, :, l], p, u_full, tau_vec[0], tau_vec[1])
                m2 = _bayes(u_full[l], tau_vec[2], a0, a1)
                Pn[i, j, l] = _clear([m0, m1, m2], gamma_vec[0], W_vec[0])
    return Pn
