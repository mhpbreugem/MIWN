#!/usr/bin/env python3
"""G=20 float64 anderson sweep of the (gamma,tau) tasks from the queue.

Per-task: ~7 min on this CPU (kernel map at G=20, anderson tol 1e-10, h=0.05*du).
NOT dd precision (~1e-10 residual, not <1e-20 policy) -- runs within the chunk
budget. Writes one .npz per task to solutions/pool/ree_K3_G20_f64/, appends to
results.csv (gamma, tau, 1-R2_lin, scatter, residual)."""
import argparse, csv, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from code.halo import extract_inner, replace_inner
from ode_sweep import anderson_solve

ROOT = Path("/home/user/MIWN")
QUEUE = ROOT / "todo" / "TASK_QUEUE.json"
OUT_DIR = ROOT / "solutions" / "pool" / "ree_K3_G20_f64"
CSV = OUT_DIR / "results.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
G, pad, um, W = 20, 4, 3.0, 1.0
du = 2 * um / (G - 1); Gf = G + 2 * pad
u = np.array([-um + (q - pad) * du for q in range(Gf)]); lo, hi = pad, pad + G
ui = u[lo:hi]


def deficit(P, tau):
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij"); T = tau * (U1 + U2 + U3)
    f1 = np.exp(-tau / 2 * (ui - 0.5) ** 2); f0 = np.exp(-tau / 2 * (ui + 0.5) ** 2)
    F1 = f1[:, None, None] * f1[None, :, None] * f1[None, None, :]
    F0 = f0[:, None, None] * f0[None, :, None] * f0[None, None, :]
    Wt = 0.5 * (F1 + F0); m = (P > 1e-4) & (P < 1 - 1e-4)
    t = T[m]; L = np.log(P[m] / (1 - P[m])); w = Wt[m]; w = w / w.sum(); lm = np.average(L, weights=w)
    c1 = np.polyfit(t, L, 1, w=np.sqrt(w)); c5 = np.polyfit(t, L, 5, w=np.sqrt(w))
    lin = np.average((L - np.polyval(c1, t)) ** 2, weights=w) / np.average((L - lm) ** 2, weights=w)
    scat = np.average((L - np.polyval(c5, t)) ** 2, weights=w) / np.average((L - lm) ** 2, weights=w)
    return float(lin), float(scat)


def solve_one(gamma, tau, tol=1e-10, mi=4000):
    tv, gv, Wv = np.full(3, tau), np.full(3, gamma), np.full(3, W)
    halo = init_no_learning_K3(u, tv, gv, Wv); h = max(0.005, 0.05 * du)
    def phi(P):
        return extract_inner(phi_K3_halo_smooth(replace_inner(halo, P, lo, hi), u, lo, hi, tv, gv, Wv, h), lo, hi)
    P, res = anderson_solve(phi, extract_inner(halo, lo, hi), tol=tol, max_iter=mi, m=8)
    lin, scat = deficit(P, tau)
    return P, lin, scat, res


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
    w = csv.DictWriter(fh, fieldnames=["gamma", "tau", "id", "G", "h", "1-R2_lin", "scatter", "residual", "wall_s"])
    if first: w.writeheader()
    for gamma, tau, tid in tasks:
        if (float(gamma), float(tau)) in skip:
            continue
        if time.time() - t0 > a.budget:
            print(f"[G20] budget exhausted; solved {done_cnt}", flush=True); break
        print(f"[G20] start g={gamma} tau={tau} {tid}", flush=True); t1 = time.time()
        try:
            P, lin, scat, res = solve_one(float(gamma), float(tau))
        except Exception as e:
            print(f"[G20] FAIL g={gamma} tau={tau}: {e}", flush=True); continue
        dt = time.time() - t1
        np.savez_compressed(OUT_DIR / f"{tid}.npz", P_inner=P, u_grid_inner=ui, gamma=float(gamma), tau=float(tau), G=G, h=max(0.005, 0.05 * du))
        w.writerow({"gamma": gamma, "tau": tau, "id": tid, "G": G, "h": f"{max(0.005, 0.05*du):.5f}",
                    "1-R2_lin": f"{lin:.6f}", "scatter": f"{scat:.6f}", "residual": f"{res:.2e}", "wall_s": f"{dt:.0f}"})
        fh.flush()
        print(f"[G20] DONE g={gamma} tau={tau} 1-R2={lin:.5f} scat={scat:.5f} res={res:.1e} ({dt:.0f}s)", flush=True)
        done_cnt += 1
    fh.close()
    print(f"[G20] chunk done: {done_cnt} solved in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
