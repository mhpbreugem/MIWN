#!/usr/bin/env python3
"""Warm-start the PROPER-WEIGHTED h=0 co-area map (phi_K3_coarea, 1/|grad P|)
from FPF kernel-PR.  If PR survives proper Bayesian weighting, deficit holds;
if it collapses to FR, PR is specific to the ξ-solver's smooth representation."""
import sys, time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
sys.path.insert(0, "/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import init_no_learning_K3
from code.halo import extract_inner, replace_inner
from code.f128 import revelation_deficit_f128
from coarea import phi_K3_coarea

CKPT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/checkpoints/g050_t0300.npz"
GAMMA, TAU, W = 0.5, 3.0, 1.0
G, pad, um = 20, 4, 5.0
ALPHA = 0.05; ITERS = 4000; REPORT = 10.0

du = 2 * um / (G - 1); Gf = G + 2 * pad
u_full = np.array([-um + (q - pad) * du for q in range(Gf)]); lo, hi = pad, pad + G
ui = u_full[lo:hi]
tv = np.full(3, TAU); gv = np.full(3, GAMMA); Wv = np.full(3, W)
halo = init_no_learning_K3(u_full, tv, gv, Wv)

a = np.load(CKPT, allow_pickle=True)
P_src = a["P_inner"].astype(float); u_src = a["u_grid_inner"].astype(float)
interp = RegularGridInterpolator((u_src, u_src, u_src), P_src,
                                 bounds_error=False, fill_value=None)
U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij")
pts = np.stack([U1.ravel(), U2.ravel(), U3.ravel()], axis=1)
P = np.clip(interp(pts).reshape((G, G, G)), 1e-9, 1 - 1e-9)
print(f"[coarea-warm] warm seed deficit = {float(revelation_deficit_f128(P,ui,tv,3)):.6f}", flush=True)

t0 = time.time(); last = 0.0; best = (P.copy(), 1e9)
for it in range(ITERS):
    Pf = replace_inner(halo, P, lo, hi)
    Q = extract_inner(phi_K3_coarea(Pf, u_full, lo, hi, tv, gv, Wv), lo, hi)
    F = float(np.max(np.abs(Q - P)))
    if F < best[1]: best = (P.copy(), F)
    P = (1 - ALPHA) * P + ALPHA * Q; P = np.clip(P, 1e-9, 1 - 1e-9)
    el = time.time() - t0
    if el - last >= REPORT or it == 0:
        d = float(revelation_deficit_f128(P, ui, tv, 3))
        print(f"[coarea-warm] g={GAMMA} tau={TAU} iter={it:4d} h=0.0000 (weighted) ||F||={F:.3e} 1-R2={d:.6f} best={best[1]:.3e} t={el:5.0f}s", flush=True); last = el

d_final = float(revelation_deficit_f128(P, ui, tv, 3))
print(f"[coarea-warm] DONE 1-R2={d_final:.6f} best||F||={best[1]:.3e} ({time.time()-t0:.0f}s)", flush=True)
np.savez_compressed("/home/user/MIWN/solutions/pool/ree_K3_G20_h0/coarea_warm_g050_t0300.npz",
                    P_inner=P, u_grid_inner=ui, gamma=GAMMA, tau=TAU, G=G, um=um, alpha=ALPHA)
