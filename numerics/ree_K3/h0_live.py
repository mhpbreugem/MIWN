#!/usr/bin/env python3
"""h=0 (bandwidth-free, cubic contour) solver with LIVE iter/F/1-R2 reports
every 10 s wall clock. f128 metric. NO kernel map. Loops through ready tasks
in the queue; one anderson solve per task; best-iterate kept."""
import csv, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_cubic
from code.halo import extract_inner, replace_inner
from code.f128 import revelation_deficit_f128
from ode_sweep import _anderson_step

ROOT = Path("/home/user/MIWN")
QUEUE = ROOT / "todo" / "TASK_QUEUE.json"
OUT_DIR = ROOT / "solutions" / "pool" / "ree_K3_G20_h0"
CSV = OUT_DIR / "results.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

G, pad, um, W = 20, 4, 5.0, 1.0
du = 2 * um / (G - 1); Gf = G + 2 * pad
u_full = np.array([-um + (q - pad) * du for q in range(Gf)]); lo, hi = pad, pad + G
ui = u_full[lo:hi]


def deficit(P, tv):
    return float(revelation_deficit_f128(P, ui, tv, 3))


def solve_live(gamma, tau, tid, report=10.0, max_iter=4000, tol=1e-12, m=8):
    tv, gv, Wv = np.full(3, tau), np.full(3, gamma), np.full(3, W)
    halo = init_no_learning_K3(u_full, tv, gv, Wv)
    shape = (G, G, G)
    P = extract_inner(halo, lo, hi).ravel()
    Fh, Ph = [], []; best = (P.copy(), 1e9, None); last = 0.0; t0 = time.time()
    nimp = 0; lastmean = 1e9
    print(f"[h=0] start g={gamma} tau={tau} {tid} G={G} um={um}", flush=True)
    for it in range(max_iter):
        Pf = replace_inner(halo, P.reshape(shape), lo, hi)
        Q = extract_inner(phi_K3_halo_cubic(Pf, u_full, lo, hi, tv, gv, Wv), lo, hi).ravel()
        F = Q - P
        res = float(np.max(np.abs(F)))
        if res < best[1]:
            best = (P.copy(), res, None); nimp = 0
        else:
            nimp += 1
        if res < lastmean * 0.999:
            lastmean = res
        el = time.time() - t0
        if el - last >= report or res < tol or it == 0:
            d = deficit(P.reshape(shape), tv)
            print(f"[h=0] g={gamma} tau={tau} iter={it:4d} h=0.0000 ||F||={res:.3e} 1-R2={d:.6f} best={best[1]:.3e} t={el:5.0f}s", flush=True)
            last = el
        if res < tol: break
        if best[1] < 1e-3 and nimp > 300: break          # plateau-stop
        Fh.append(F.copy()); Ph.append(P.copy())
        P = _anderson_step(Fh, Ph, m); P = np.clip(P, 1e-9, 1 - 1e-9)
    Pbest = best[0].reshape(shape)
    d_final = deficit(Pbest, tv)
    print(f"[h=0] DONE g={gamma} tau={tau} {tid} best||F||={best[1]:.3e} 1-R2={d_final:.6f} ({time.time()-t0:.0f}s)", flush=True)
    return Pbest, d_final, best[1]


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
    w = csv.DictWriter(fh, fieldnames=["gamma", "tau", "id", "G", "umax",
                                       "1-R2_f128", "residual", "wall_s"])
    if first: w.writeheader()
    for gamma, tau, tid in tasks:
        if (float(gamma), float(tau)) in skip:
            continue
        try:
            t0 = time.time()
            P, one_minus_r2, res = solve_live(float(gamma), float(tau), tid)
            np.savez_compressed(OUT_DIR / f"{tid}.npz", P_inner=P, u_grid_inner=ui,
                                gamma=float(gamma), tau=float(tau), G=G, um=um)
            w.writerow({"gamma": gamma, "tau": tau, "id": tid, "G": G, "umax": um,
                        "1-R2_f128": f"{one_minus_r2:.8f}",
                        "residual": f"{res:.2e}", "wall_s": f"{time.time()-t0:.0f}"})
            fh.flush()
        except Exception as e:
            print(f"[h=0] FAIL g={gamma} tau={tau}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
