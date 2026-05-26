#!/usr/bin/env python3
"""h=0 (bandwidth-free cubic contour) with LOW-damping Picard + Cesaro tail
average, REZN-style branch tracking (anderson falls to FR; Picard with small
alpha holds the curved PR basin).  Live iter / ||F|| / 1-R2 / 1-R2_cesaro
every 10 s.  f128 metric.  NO kernel map."""
import csv, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_cubic
from code.halo import extract_inner, replace_inner
from code.f128 import revelation_deficit_f128

ROOT = Path("/home/user/MIWN")
QUEUE = ROOT / "todo" / "TASK_QUEUE.json"
OUT_DIR = ROOT / "solutions" / "pool" / "ree_K3_G20_h0"
CSV = OUT_DIR / "results.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

G, pad, um, W = 20, 4, 5.0, 1.0
ALPHA = 0.05            # low damping (REZN fig5 phase1)
CES = 1500              # Cesaro tail length
MAXIT = 8000

du = 2 * um / (G - 1); Gf = G + 2 * pad
u_full = np.array([-um + (q - pad) * du for q in range(Gf)]); lo, hi = pad, pad + G
ui = u_full[lo:hi]


def deficit(P, tv):
    return float(revelation_deficit_f128(P, ui, tv, 3))


def solve(gamma, tau, tid, report=10.0):
    tv = np.full(3, tau); gv = np.full(3, gamma); Wv = np.full(3, W)
    halo = init_no_learning_K3(u_full, tv, gv, Wv)
    P = extract_inner(halo, lo, hi).copy()
    acc = None; cnt = 0; t0 = time.time(); last = 0.0
    best = (P.copy(), 1e9)
    print(f"[picard] start g={gamma} tau={tau} {tid} alpha={ALPHA} Ces={CES} G={G} um={um}", flush=True)
    for it in range(MAXIT):
        Pf = replace_inner(halo, P, lo, hi)
        Q = extract_inner(phi_K3_halo_cubic(Pf, u_full, lo, hi, tv, gv, Wv), lo, hi)
        F = float(np.max(np.abs(Q - P)))
        if F < best[1]: best = (P.copy(), F)
        P = (1 - ALPHA) * P + ALPHA * Q
        P = np.clip(P, 1e-9, 1 - 1e-9)
        if it >= MAXIT - CES:
            acc = Q.copy() if acc is None else acc + Q
            cnt += 1
        el = time.time() - t0
        if el - last >= report or it == 0:
            d_cur = deficit(P, tv)
            d_ces = deficit(acc / cnt, tv) if cnt else float("nan")
            print(f"[picard] g={gamma} tau={tau} iter={it:4d} h=0.0000 ||F||={F:.3e} 1-R2={d_cur:.6f} 1-R2_ces={d_ces:.6f} best={best[1]:.3e} t={el:5.0f}s",
                  flush=True)
            last = el
    Pces = acc / cnt
    d_ces = deficit(Pces, tv); d_cur = deficit(P, tv)
    print(f"[picard] DONE g={gamma} tau={tau} {tid} 1-R2={d_cur:.6f} 1-R2_ces={d_ces:.6f} best||F||={best[1]:.3e} ({time.time()-t0:.0f}s)", flush=True)
    return Pces, d_ces, best[1]


def done_set():
    if not CSV.exists(): return set()
    return set((float(r["gamma"]), float(r["tau"]))
               for r in csv.DictReader(CSV.open()))


def main():
    q = json.loads(QUEUE.read_text())
    tasks = [(t["gamma"], t["tau"], t["id"]) for t in q["tasks"]]
    skip = done_set()
    first = not CSV.exists()
    fh = CSV.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=["gamma", "tau", "id", "G", "umax", "alpha",
                                       "1-R2_f128", "residual", "wall_s"])
    if first: w.writeheader()
    for gamma, tau, tid in tasks:
        if (float(gamma), float(tau)) in skip: continue
        try:
            t0 = time.time()
            P, d, res = solve(float(gamma), float(tau), tid)
            np.savez_compressed(OUT_DIR / f"{tid}.npz", P_inner=P, u_grid_inner=ui,
                                gamma=float(gamma), tau=float(tau), G=G, um=um, alpha=ALPHA)
            w.writerow({"gamma": gamma, "tau": tau, "id": tid, "G": G, "umax": um,
                        "alpha": ALPHA, "1-R2_f128": f"{d:.8f}",
                        "residual": f"{res:.2e}", "wall_s": f"{time.time()-t0:.0f}"})
            fh.flush()
        except Exception as e:
            print(f"[picard] FAIL g={gamma} tau={tau}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
