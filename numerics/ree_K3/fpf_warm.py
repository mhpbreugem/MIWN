#!/usr/bin/env python3
"""Warm-start h=0 cubic from an FPF KERNEL-PR field interpolated to G=20.
Test: does the curved PR seed survive under bandwidth-free cubic iteration?"""
import sys, time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_cubic
from code.halo import extract_inner, replace_inner
from code.f128 import revelation_deficit_f128

CKPT = "/home/user/FIXED-POINT-FACTORY/projects/REZN/checkpoints/g050_t0300.npz"
GAMMA, TAU, W = 0.5, 3.0, 1.0
G, pad, um = 20, 4, 5.0
ALPHA = 0.05; ITERS = 4000; REPORT = 10.0; CES = 1500

du = 2 * um / (G - 1); Gf = G + 2 * pad
u_full = np.array([-um + (q - pad) * du for q in range(Gf)]); lo, hi = pad, pad + G
ui = u_full[lo:hi]
tv = np.full(3, TAU); gv = np.full(3, GAMMA); Wv = np.full(3, W)


def deficit(P):
    return float(revelation_deficit_f128(P, ui, tv, 3))


def main():
    a = np.load(CKPT, allow_pickle=True)
    P_src = a["P_inner"].astype(float)
    u_src = a["u_grid_inner"].astype(float)
    G_src = P_src.shape[0]
    print(f"[fpf-warm] loaded {CKPT.split('/')[-1]} G_src={G_src} u_src=[{u_src[0]:.2f},{u_src[-1]:.2f}] gamma=0.5 tau=3", flush=True)
    interp = RegularGridInterpolator((u_src, u_src, u_src), P_src,
                                     bounds_error=False, fill_value=None)
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij")
    pts = np.stack([U1.ravel(), U2.ravel(), U3.ravel()], axis=1)
    P_inner = interp(pts).reshape((G, G, G))
    P_inner = np.clip(P_inner, 1e-9, 1 - 1e-9)
    print(f"[fpf-warm] warm seed deficit = {deficit(P_inner):.6f}", flush=True)
    halo = init_no_learning_K3(u_full, tv, gv, Wv)
    P = P_inner.copy()
    acc = None; cnt = 0; t0 = time.time(); last = 0.0; best = (P.copy(), 1e9)
    for it in range(ITERS):
        Pf = replace_inner(halo, P, lo, hi)
        Q = extract_inner(phi_K3_halo_cubic(Pf, u_full, lo, hi, tv, gv, Wv), lo, hi)
        F = float(np.max(np.abs(Q - P)))
        if F < best[1]: best = (P.copy(), F)
        P = (1 - ALPHA) * P + ALPHA * Q; P = np.clip(P, 1e-9, 1 - 1e-9)
        if it >= ITERS - CES:
            acc = Q.copy() if acc is None else acc + Q; cnt += 1
        el = time.time() - t0
        if el - last >= REPORT or it == 0:
            d_cur = deficit(P); d_ces = deficit(acc / cnt) if cnt else float("nan")
            print(f"[fpf-warm] g={GAMMA} tau={TAU} iter={it:4d} h=0.0000 ||F||={F:.3e} 1-R2={d_cur:.6f} 1-R2_ces={d_ces:.6f} best={best[1]:.3e} t={el:5.0f}s", flush=True); last = el
    Pces = acc / cnt
    d_ces = deficit(Pces)
    print(f"[fpf-warm] DONE 1-R2_ces={d_ces:.6f} best||F||={best[1]:.3e} ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed("/home/user/MIWN/solutions/pool/ree_K3_G20_h0/fpf_warm_g050_t0300.npz",
                        P_inner=Pces, u_grid_inner=ui, gamma=GAMMA, tau=TAU,
                        G=G, um=um, alpha=ALPHA, source=CKPT)


if __name__ == "__main__":
    main()
