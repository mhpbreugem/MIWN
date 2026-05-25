"""Bandwidth-free, SMOOTH co-area map for K=3 (no kernel h).

The axis-scan contour maps (phi_K3_halo / coarea) are non-smooth: a crossing on
a grid LINE appears/disappears with finite co-area weight, so Phi jumps and the
fixed point is only defined to ~1e-3.  Here the evidence integral

    A_v(p) = INT f_v delta(P-p) dA = INT_contour f_v / |grad P| ds

is computed by MARCHING SQUARES: in each grid cell the level set P=p is a line
SEGMENT whose endpoints move continuously on the cell edges and whose length
grows from 0 as the contour enters a cell.  Hence Phi is continuous in the nodal
prices.

CAVEAT (measured): iteration is still unstable here because tail cells where
|grad P| -> 0 (P near 0 or 1) get unbounded co-area weight L/|grad P|; Anderson
and damped Picard diverge from the no-learning start (||F|| ~ 0.25 -> 0.46).
The converged bandwidth-free answer (1-R2 ~ 0.001-0.002, ||F|| ~ 1e-4) is
obtained instead from the cubic axis-scan map (code.phi_K3_halo_cubic) /
coarea.py.  Reaching ||F|| < 1e-20 bandwidth-free is not possible with a contour
map (non-smooth / unbounded weights) -- that requires the smoothed (kernel-h)
map, whose deficit then depends on h.  See deficit_vs_h.png.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange

from code.signals import f_signal           # type: ignore
from code.demand import clear_crra, EPS_PRICE  # type: ignore


@njit(cache=True)
def _ms_evidence(Pslice, p, u, du, tau_a, tau_b):
    """Marching-squares co-area integral over a (G,G) slice.

    Axis 0 (rows) carries tau_a, axis 1 (cols) carries tau_b.
    Returns (A0, A1).  The common signal-density coefficient cancels in the
    Bayes ratio, so f_signal's normalisation is irrelevant.
    """
    G = u.size
    A0 = 0.0
    A1 = 0.0
    xa = np.empty(4)
    xb = np.empty(4)
    for i in range(G - 1):
        for j in range(G - 1):
            P00 = Pslice[i, j]; P10 = Pslice[i + 1, j]
            P01 = Pslice[i, j + 1]; P11 = Pslice[i + 1, j + 1]
            n = 0
            # bottom edge (vary a at b=u[j]):  P00 -> P10
            if (P00 - p) * (P10 - p) < 0.0:
                t = (p - P00) / (P10 - P00)
                xa[n] = u[i] + t * du; xb[n] = u[j]; n += 1
            # top edge (b=u[j+1]):  P01 -> P11
            if (P01 - p) * (P11 - p) < 0.0:
                t = (p - P01) / (P11 - P01)
                xa[n] = u[i] + t * du; xb[n] = u[j + 1]; n += 1
            # left edge (a=u[i]):  P00 -> P01
            if (P00 - p) * (P01 - p) < 0.0:
                t = (p - P00) / (P01 - P00)
                xa[n] = u[i]; xb[n] = u[j] + t * du; n += 1
            # right edge (a=u[i+1]):  P10 -> P11
            if (P10 - p) * (P11 - p) < 0.0:
                t = (p - P10) / (P11 - P10)
                xa[n] = u[i + 1]; xb[n] = u[j] + t * du; n += 1
            if n != 2:
                continue                      # 0 = no contour; 4 = saddle (rare)
            uam = 0.5 * (xa[0] + xa[1]); ubm = 0.5 * (xb[0] + xb[1])
            L = math.sqrt((xa[0] - xa[1]) ** 2 + (xb[0] - xb[1]) ** 2)
            s = (uam - u[i]) / du; tt = (ubm - u[j]) / du
            dPds = (P10 - P00) * (1.0 - tt) + (P11 - P01) * tt
            dPdt = (P01 - P00) * (1.0 - s) + (P11 - P10) * s
            gmag = math.sqrt(dPds * dPds + dPdt * dPdt) / du
            if gmag < 1e-300:
                continue
            w = L / gmag
            f0a = f_signal(uam, 0, tau_a); f0b = f_signal(ubm, 0, tau_b)
            f1a = f_signal(uam, 1, tau_a); f1b = f_signal(ubm, 1, tau_b)
            A0 += f0a * f0b * w
            A1 += f1a * f1b * w
    return A0, A1


@njit(cache=True, inline="always")
def _bayes(u_own, tau_own, A0, A1):
    f0 = f_signal(u_own, 0, tau_own)
    f1 = f_signal(u_own, 1, tau_own)
    num = f1 * A1
    den = f0 * A0 + num
    if den <= 0.0:
        return 0.5
    mu = num / den
    if mu < EPS_PRICE:
        return EPS_PRICE
    if mu > 1.0 - EPS_PRICE:
        return 1.0 - EPS_PRICE
    return mu


@njit(cache=True, parallel=True)
def phi_K3_marching(P_full, u_full, inner_lo, inner_hi,
                    tau_vec, gamma_vec, W_vec):
    """Bandwidth-free smooth co-area Phi (inner updated, halo fixed)."""
    du = u_full[1] - u_full[0]
    P_new = P_full.copy()
    for i in prange(inner_lo, inner_hi):
        mu_vec = np.empty(3, dtype=np.float64)
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                p = P_full[i, j, l]
                A0, A1 = _ms_evidence(P_full[i, :, :], p, u_full, du,
                                      tau_vec[1], tau_vec[2])
                mu_vec[0] = _bayes(u_full[i], tau_vec[0], A0, A1)
                A0, A1 = _ms_evidence(P_full[:, j, :], p, u_full, du,
                                      tau_vec[0], tau_vec[2])
                mu_vec[1] = _bayes(u_full[j], tau_vec[1], A0, A1)
                A0, A1 = _ms_evidence(P_full[:, :, l], p, u_full, du,
                                      tau_vec[0], tau_vec[1])
                mu_vec[2] = _bayes(u_full[l], tau_vec[2], A0, A1)
                P_new[i, j, l] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P_new
