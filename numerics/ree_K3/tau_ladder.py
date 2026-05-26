#!/usr/bin/env python3
"""tau-ladder continuation on h=0 cubic contour map at gamma=0.5 (REZN method).
Picard alpha=0.05 + Cesaro at each tau step, warm-started from the previous
step's Cesaro average.  Goal: track the PR branch from low tau up to tau=2."""
import sys, time
import numpy as np
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_cubic
from code.halo import extract_inner, replace_inner
from code.f128 import revelation_deficit_f128

G, pad, um, W = 20, 4, 5.0, 1.0
ALPHA = 0.05; CES = 800; ITERS = 2500       # per-tau budget (~3 min)
TAUS = [0.5, 0.8, 1.2, 1.5, 1.8, 2.0]
GAMMA = 0.5

du = 2 * um / (G - 1); Gf = G + 2 * pad
u_full = np.array([-um + (q - pad) * du for q in range(Gf)]); lo, hi = pad, pad + G
ui = u_full[lo:hi]


def deficit(P, tv):
    return float(revelation_deficit_f128(P, ui, tv, 3))


def step(P0_inner, tau, report=10.0):
    tv = np.full(3, tau); gv = np.full(3, GAMMA); Wv = np.full(3, W)
    halo = init_no_learning_K3(u_full, tv, gv, Wv)
    P = P0_inner.copy()
    acc = None; cnt = 0; t0 = time.time(); last = 0.0; best = (P.copy(), 1e9)
    print(f"[tau-ladder] tau={tau} start (warm) G={G} um={um} alpha={ALPHA}", flush=True)
    for it in range(ITERS):
        Pf = replace_inner(halo, P, lo, hi)
        Q = extract_inner(phi_K3_halo_cubic(Pf, u_full, lo, hi, tv, gv, Wv), lo, hi)
        F = float(np.max(np.abs(Q - P)))
        if F < best[1]: best = (P.copy(), F)
        P = (1 - ALPHA) * P + ALPHA * Q
        P = np.clip(P, 1e-9, 1 - 1e-9)
        if it >= ITERS - CES:
            acc = Q.copy() if acc is None else acc + Q; cnt += 1
        el = time.time() - t0
        if el - last >= report or it == 0 or it == ITERS - 1:
            d_cur = deficit(P, tv)
            d_ces = deficit(acc / cnt, tv) if cnt else float("nan")
            print(f"[tau-ladder] g={GAMMA} tau={tau} iter={it:4d} h=0.0000 ||F||={F:.3e} 1-R2={d_cur:.6f} 1-R2_ces={d_ces:.6f} best={best[1]:.3e} t={el:5.0f}s", flush=True)
            last = el
    Pces = acc / cnt
    d_ces = deficit(Pces, tv)
    print(f"[tau-ladder] tau={tau} DONE 1-R2_ces={d_ces:.6f} best||F||={best[1]:.3e}", flush=True)
    return Pces


def main():
    tv0 = np.full(3, TAUS[0]); gv = np.full(3, GAMMA); Wv = np.full(3, W)
    halo0 = init_no_learning_K3(u_full, tv0, gv, Wv)
    P = extract_inner(halo0, lo, hi)
    for tau in TAUS:
        P = step(P, tau)
    # save final
    np.savez_compressed("/home/user/MIWN/solutions/pool/ree_K3_G20_h0/tau_ladder_g050.npz",
                        P_inner=P, u_grid_inner=ui, gamma=GAMMA, taus=np.array(TAUS),
                        G=G, um=um, alpha=ALPHA, Ces=CES)
    print("[tau-ladder] all tau done; saved", flush=True)


if __name__ == "__main__":
    main()
