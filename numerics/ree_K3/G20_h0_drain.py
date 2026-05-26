#!/usr/bin/env python3
"""G=20 BANDWIDTH-FREE sweep (h=0, hardwired -- NO kernel/smooth map).

phi_K3_halo_cubic only.  f128 (longdouble) revelation_deficit to avoid the
float64 cancellation floor.  Anderson best-iterate (contour maps are
non-smooth; ||F|| floors at ~1e-3..1e-9, the regression metric still resolves
1-R^2 to many digits via f128).  Writes per-task .npz + appends results.csv.
"""
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
# h=0 HARDWIRE: import only the bandwidth-free contour map.
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_cubic
from code.halo import extract_inner, replace_inner
from code.f128 import revelation_deficit_f128
from ode_sweep import anderson_solve

# refuse to import any kernel function (defensive guard)
import code.contour_K3_halo as _ck3
assert "phi_K3_halo_smooth" not in dir() and "phi_K3_smooth_mp" not in dir(), \
    "kernel map must NOT be in this module's globals"

ROOT = Path("/home/user/MIWN")
QUEUE = ROOT / "todo" / "TASK_QUEUE.json"
OUT_DIR = ROOT / "solutions" / "pool" / "ree_K3_G20_h0"
CSV = OUT_DIR / "results.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

G, pad, um, W = 20, 4, 5.0, 1.0       # um=5 to match production REZN UMAX
du = 2 * um / (G - 1); Gf = G + 2 * pad
u = np.array([-um + (q - pad) * du for q in range(Gf)]); lo, hi = pad, pad + G
ui = u[lo:hi]


def solve_one(gamma, tau, tol=1e-12, mi=4000):
    tv, gv, Wv = np.full(3, tau), np.full(3, gamma), np.full(3, W)
    halo = init_no_learning_K3(u, tv, gv, Wv)
    def phi(P):
        return extract_inner(
            phi_K3_halo_cubic(replace_inner(halo, P, lo, hi), u, lo, hi, tv, gv, Wv),
            lo, hi)
    P, res = anderson_solve(phi, extract_inner(halo, lo, hi), tol=tol, max_iter=mi, m=8)
    one_minus_r2 = float(revelation_deficit_f128(P, ui, tv, 3))
    return P, one_minus_r2, res


def done_set():
    if not CSV.exists(): return set()
    out = set()
    for r in csv.DictReader(CSV.open()):
        out.add((float(r["gamma"]), float(r["tau"])))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=int, default=540)
    a = ap.parse_args()
    q = json.loads(QUEUE.read_text())
    tasks = [(t["gamma"], t["tau"], t["id"]) for t in q["tasks"]]
    skip = done_set()
    t0 = time.time(); done_cnt = 0
    first = not CSV.exists()
    fh = CSV.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=["gamma", "tau", "id", "G", "umax",
                                       "1-R2_f128", "residual", "wall_s"])
    if first: w.writeheader()
    for gamma, tau, tid in tasks:
        if (float(gamma), float(tau)) in skip:
            continue
        if time.time() - t0 > a.budget:
            print(f"[h=0] budget exhausted; solved {done_cnt}", flush=True); break
        print(f"[h=0] start g={gamma} tau={tau} {tid}", flush=True); t1 = time.time()
        try:
            P, one_minus_r2, res = solve_one(float(gamma), float(tau))
        except Exception as e:
            print(f"[h=0] FAIL g={gamma} tau={tau}: {type(e).__name__}: {e}", flush=True); continue
        dt = time.time() - t1
        np.savez_compressed(OUT_DIR / f"{tid}.npz", P_inner=P, u_grid_inner=ui,
                            gamma=float(gamma), tau=float(tau), G=G, um=um)
        w.writerow({"gamma": gamma, "tau": tau, "id": tid, "G": G, "umax": um,
                    "1-R2_f128": f"{one_minus_r2:.8f}", "residual": f"{res:.2e}",
                    "wall_s": f"{dt:.0f}"})
        fh.flush()
        print(f"[h=0] DONE g={gamma} tau={tau} 1-R2={one_minus_r2:.6f} res={res:.1e} ({dt:.0f}s)", flush=True)
        done_cnt += 1
    fh.close()
    print(f"[h=0] chunk done: {done_cnt} solved in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
