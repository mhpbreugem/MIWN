"""Bandwidth-free co-area contour map for K=3 (paper-side experimental driver).

The vendored `phi_K3_halo` / `phi_K3_halo_cubic` sum each price-contour
crossing with unit weight (`f_a * f_off`), which omits the co-area Jacobian
1/|dP/du_off|.  That biases the evidence integral

    A_v(p) = INT f_v(u_off) delta(P - p) du_off  =  SUM_crossings f_v / |dP/du_off|

and the bias does NOT vanish under grid refinement.  This module restores the
1/|dP/du_off| weight (the slope is already available from the Hermite root
find), giving a genuinely bandwidth-free estimator to grid-refine.

If this proves out it should be back-ported into `standards` via PR, not kept
as a private fork (see MIWN CLAUDE.md methods-sync rule).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from code.signals import f_signal           # type: ignore
from code.demand import clear_crra, EPS_PRICE  # type: ignore


@njit(cache=True, fastmath=False, inline="always")
def _hermite_root_slope(p0, p1, p2, p3, p_target):
    """Return (t, |dP/dt|) for the Hermite cubic root in [0,1].

    |dP/dt| is the contour slope in cell-units (uniform spacing); the physical
    |dP/du| = |dP/dt|/du, but the du cancels in the Bayes ratio so we keep
    cell-units throughout.
    """
    m0 = (p2 - p0) * 0.5
    m1 = (p3 - p1) * 0.5
    t = 0.5
    dPt = p2 - p1
    for _ in range(12):
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        Pt = h00 * p1 + h10 * m0 + h01 * p2 + h11 * m1
        dh00 = 6.0 * t2 - 6.0 * t
        dh10 = 3.0 * t2 - 4.0 * t + 1.0
        dh01 = -6.0 * t2 + 6.0 * t
        dh11 = 3.0 * t2 - 2.0 * t
        dPt = dh00 * p1 + dh10 * m0 + dh01 * p2 + dh11 * m1
        if dPt == 0.0:
            break
        delta = (Pt - p_target) / dPt
        t -= delta
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        if delta < 0.0:
            delta = -delta
        if delta < 1.0e-12:
            break
    if dPt < 0.0:
        dPt = -dPt
    return t, dPt


@njit(cache=True, fastmath=False)
def _scan_coarea(P_slice, p_target, axis, a_idx, u_full,
                 tau_a, tau_off, acc, slope_floor):
    """One axis scan with the 1/|dP/du| co-area weight per crossing."""
    G = u_full.size
    u_a = u_full[a_idx]
    f0_a = f_signal(u_a, 0, tau_a)
    f1_a = f_signal(u_a, 1, tau_a)

    prev_v = P_slice[0, a_idx] if axis == 0 else P_slice[a_idx, 0]
    for i in range(G - 1):
        next_v = (P_slice[i + 1, a_idx] if axis == 0
                  else P_slice[a_idx, i + 1])
        d_prev = prev_v - p_target
        d_next = next_v - p_target
        if d_prev == 0.0 and d_next == 0.0:
            prev_v = next_v
            continue
        if d_prev * d_next <= 0.0:
            denom = next_v - prev_v
            if denom == 0.0:
                prev_v = next_v
                continue
            if i >= 1 and i <= G - 3:
                if axis == 0:
                    p0 = P_slice[i - 1, a_idx]
                    p3 = P_slice[i + 2, a_idx]
                else:
                    p0 = P_slice[a_idx, i - 1]
                    p3 = P_slice[a_idx, i + 2]
                t, slope = _hermite_root_slope(p0, prev_v, next_v, p3,
                                               p_target)
            else:
                t = (p_target - prev_v) / denom
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                slope = denom if denom >= 0.0 else -denom
            if slope < slope_floor:
                slope = slope_floor
            u_off = (1.0 - t) * u_full[i] + t * u_full[i + 1]
            f0_off = f_signal(u_off, 0, tau_off)
            f1_off = f_signal(u_off, 1, tau_off)
            w = 1.0 / slope
            acc[0] += f0_a * f0_off * w
            acc[1] += f1_a * f1_off * w
        prev_v = next_v


@njit(cache=True, fastmath=False)
def _agent_evidence_coarea(P_slice, p_target, u_full,
                           tau_o0, tau_o1, acc, slope_floor):
    """Two-axis-averaged co-area evidence for one agent."""
    G = u_full.size
    a0 = 0.0
    a1 = 0.0

    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G):
        _scan_coarea(P_slice, p_target, 0, a_idx, u_full,
                     tau_o1, tau_o0, acc, slope_floor)
    a0 += acc[0]
    a1 += acc[1]

    acc[0] = 0.0
    acc[1] = 0.0
    for a_idx in range(G):
        _scan_coarea(P_slice, p_target, 1, a_idx, u_full,
                     tau_o0, tau_o1, acc, slope_floor)
    a0 += acc[0]
    a1 += acc[1]

    acc[0] = a0 / 2.0
    acc[1] = a1 / 2.0


@njit(cache=True, fastmath=False, inline="always")
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


@njit(cache=True, fastmath=False, parallel=True)
def phi_K3_coarea(P_full, u_full, inner_lo, inner_hi,
                  tau_vec, gamma_vec, W_vec, slope_floor=1.0e-9):
    """Bandwidth-free co-area Phi map (inner cells updated, halo fixed)."""
    P_new = P_full.copy()
    for i in prange(inner_lo, inner_hi):
        mu_vec = np.empty(3, dtype=np.float64)
        acc = np.empty(2, dtype=np.float64)
        for j in range(inner_lo, inner_hi):
            for l in range(inner_lo, inner_hi):
                p = P_full[i, j, l]

                _agent_evidence_coarea(P_full[i, :, :], p, u_full,
                                       tau_vec[1], tau_vec[2], acc,
                                       slope_floor)
                mu_vec[0] = _bayes(u_full[i], tau_vec[0], acc[0], acc[1])

                _agent_evidence_coarea(P_full[:, j, :], p, u_full,
                                       tau_vec[0], tau_vec[2], acc,
                                       slope_floor)
                mu_vec[1] = _bayes(u_full[j], tau_vec[1], acc[0], acc[1])

                _agent_evidence_coarea(P_full[:, :, l], p, u_full,
                                       tau_vec[0], tau_vec[1], acc,
                                       slope_floor)
                mu_vec[2] = _bayes(u_full[l], tau_vec[2], acc[0], acc[1])

                P_new[i, j, l] = clear_crra(mu_vec, gamma_vec, W_vec)
    return P_new
