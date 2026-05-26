#!/usr/bin/env python3
"""Newton-Krylov polish on h=0 cubic from FPF kernel-PR warm seed.
If a PR fixed point exists for the cubic h=0 map, NK should converge to it.
If it diverges or returns to FR, PR is not a smooth fixed point here."""
import sys, time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import newton_krylov, NoConvergence
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_cubic
from code.halo import extract_inner, replace_inner
from code.f128 import revelation_deficit_f128

CKPT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/checkpoints/g050_t0300.npz"
GAMMA, TAU, W = 0.5, 3.0, 1.0
G, pad, um = 20, 4, 5.0

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
P0 = np.clip(interp(pts).reshape((G, G, G)), 1e-9, 1 - 1e-9)
print(f"[nk-warm] warm seed 1-R2={float(revelation_deficit_f128(P0,ui,tv,3)):.6f}", flush=True)

shape = (G, G, G); n_eval = [0]; t0 = time.time(); last = [0.0]
def F(x):
    n_eval[0] += 1
    Pf = replace_inner(halo, x.reshape(shape), lo, hi)
    Q = extract_inner(phi_K3_halo_cubic(Pf, u_full, lo, hi, tv, gv, Wv), lo, hi)
    el = time.time() - t0
    res = float(np.max(np.abs(Q.ravel() - x)))
    if el - last[0] >= 10.0:
        d = float(revelation_deficit_f128(x.reshape(shape), ui, tv, 3))
        print(f"[nk-warm] eval={n_eval[0]} ||F||={res:.3e} 1-R2={d:.6f} t={el:5.0f}s", flush=True); last[0] = el
    return Q.ravel() - x

try:
    sol = newton_krylov(F, P0.ravel(), f_tol=1e-8, maxiter=40,
                        method="lgmres", inner_maxiter=40, verbose=False)
    P_sol = sol.reshape(shape)
    res = float(np.max(np.abs(F(sol))))
    d = float(revelation_deficit_f128(P_sol, ui, tv, 3))
    print(f"[nk-warm] DONE 1-R2={d:.6f} ||F||={res:.3e} evals={n_eval[0]}", flush=True)
    np.savez_compressed("/home/user/MIWN/solutions/pool/ree_K3_G20_h0/nk_warm_g050_t0300.npz",
                        P_inner=P_sol, u_grid_inner=ui, gamma=GAMMA, tau=TAU, G=G, um=um)
except NoConvergence as e:
    print(f"[nk-warm] NK NoConvergence after {n_eval[0]} evals (NK failed -- not a smooth fixed point reachable here)", flush=True)
except Exception as e:
    print(f"[nk-warm] error: {type(e).__name__}: {str(e)[:200]}", flush=True)
