#!/usr/bin/env python3
"""
solve.py — thin local driver for the ree_K3 problem (MIWN).

Imports the shared symmetric-K REE solver from the Standards submodule
(standards/methods/solver/contour_KN_sym.py), runs ONE local fixed-point solve
for a given (gamma, tau), and writes an immutable solution version to
solutions/pool/ree_K3/vNNNN/ with data/, figure/, and meta.json.

The solver internals live in Standards — this file only supplies this paper's
params and handles the pool layout + provenance. Run a tiny grid (small --G) to
confirm wiring; the production high-precision path is the mpmath-Newton polish in
standards/methods/solver/solve.py.

Usage:
    python3 numerics/ree_K3/solve.py --gamma 0.5 --tau 2.0 --G 5
    python3 numerics/ree_K3/solve.py --gamma 0.5 --tau 2.0 --G 5 --version v0001
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths: put the Standards submodule's solver on sys.path (import, don't copy).
# ---------------------------------------------------------------------------
PROBLEM = "ree_K3"
HERE = Path(__file__).resolve().parent          # numerics/ree_K3
REPO = HERE.parents[1]                           # MIWN repo root
SOLVER = REPO / "standards" / "methods" / "solver"

if not SOLVER.exists():
    raise SystemExit(
        f"[solve] {SOLVER} not found — initialise the submodule:\n"
        f"        git submodule update --init --recursive"
    )
sys.path.insert(0, str(SOLVER))

from contour_KN_sym import (  # noqa: E402  (shared methods, from the submodule)
    SymGrid, sym_init_no_learning, sym_phi, sym_weighted_R2,
)


def standards_sha() -> str:
    """git sha of the standards submodule checkout (recorded in meta.json)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO / "standards"), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception as e:  # pragma: no cover
        print(f"[solve] WARN: could not read standards sha ({e})", flush=True)
        return "unknown"


def next_version(problem: str) -> str:
    """Lowest unused vNNNN under solutions/pool/<problem>/ (monotonic)."""
    pool = REPO / "solutions" / "pool" / problem
    existing = []
    if pool.exists():
        for d in pool.iterdir():
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit():
                existing.append(int(d.name[1:]))
    return f"v{(max(existing) + 1) if existing else 1:04d}"


class _SolveTimeout(Exception):
    pass


def solve_fixed_point(sg: SymGrid, u_grid, gamma, tau, W, f_tol, max_evals,
                      anderson_m, max_seconds=120.0):
    """Anderson-accelerated fixed point F(P)=Phi(P)-P=0, tracking the best iterate.

    Wall-capped: if a point won't converge within max_seconds it returns the best
    iterate found so far instead of hanging — so one stuck root-find never blocks a
    worker draining the queue.
    """
    import time  # noqa: PLC0415
    from scipy.optimize import anderson, NoConvergence  # noqa: PLC0415

    P0 = sym_init_no_learning(sg, u_grid, tau, gamma, W)
    best = {"P": P0.copy(), "F": float("inf"), "n": 0}
    t0 = time.perf_counter()

    def residual(P):
        F = sym_phi(P, sg, u_grid, tau, gamma, W) - P
        f = float(np.max(np.abs(F)))
        best["n"] += 1
        if f < best["F"]:
            best["P"], best["F"] = P.copy(), f
        if time.perf_counter() - t0 > max_seconds:
            raise _SolveTimeout()
        return F

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            anderson(residual, P0, f_tol=f_tol, maxiter=max_evals,
                     M=anderson_m, line_search="armijo", verbose=False)
        except (NoConvergence, _SolveTimeout):
            pass  # best-iterate tracking handles non-convergence / timeout

    P = best["P"]
    F_inf = float(np.max(np.abs(sym_phi(P, sg, u_grid, tau, gamma, W) - P)))
    return P, F_inf, best["n"]


def make_figure(path: Path, P, sg, u_grid, tau, metrics) -> None:
    """logit(price) vs sufficient statistic T* = tau*sum(u_k) — the informativeness fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    eps = 1e-12
    Tstar = np.array([tau * float(u_grid[sg.tuples[s]].sum()) for s in range(sg.n)])
    p = np.clip(P, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    slope, intercept = metrics["slope"], metrics["intercept"]

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.scatter(Tstar, logit, s=10, alpha=0.6, label="cells")
    xs = np.linspace(Tstar.min(), Tstar.max(), 50)
    ax.plot(xs, slope * xs + intercept, "r-", lw=1.2, label=f"fit (1-R²={metrics['1-R2']:.2e})")
    ax.set_xlabel(r"$T^* = \tau \sum_k u_k$")
    ax.set_ylabel(r"$\mathrm{logit}(P)$")
    ax.set_title(f"ree_K3  γ={metrics['gamma']}  τ={metrics['tau']}  G={sg.G}")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="ree_K3 local solve driver (MIWN)")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--G", type=int, default=5, help="symmetric grid points (tiny for wiring check)")
    ap.add_argument("--u-max", type=float, default=3.0)
    ap.add_argument("--W", type=float, default=1.0)
    ap.add_argument("--f-tol", type=float, default=1e-9)
    ap.add_argument("--max-evals", type=int, default=4000)
    ap.add_argument("--anderson-m", type=int, default=10)
    ap.add_argument("--max-seconds", type=float, default=120.0,
                    help="wall cap per solve; returns best iterate if exceeded (never hangs)")
    ap.add_argument("--version", default=None, help="vNNNN (default: next free)")
    ap.add_argument("--task-id", default=None)
    args = ap.parse_args()

    gamma, tau, G, W = args.gamma, args.tau, args.G, args.W
    task_id = args.task_id or f"{PROBLEM}_g{int(round(gamma*100)):04d}_t{int(round(tau*100)):04d}"
    version = args.version or next_version(PROBLEM)

    u_grid = np.linspace(-args.u_max, args.u_max, G)
    sg = SymGrid.build(G, 3)
    print(f"[solve] {PROBLEM} {version} γ={gamma} τ={tau} G={G} (n={sg.n} cells)", flush=True)

    P, F_inf, n_evals = solve_fixed_point(
        sg, u_grid, gamma, tau, W, args.f_tol, args.max_evals, args.anderson_m,
        max_seconds=args.max_seconds,
    )
    m = sym_weighted_R2(P, sg, u_grid, tau)
    m["gamma"], m["tau"] = gamma, tau
    print(f"[solve] done  evals={n_evals}  ||F||={F_inf:.3e}  1-R²={m['1-R2']:.6e}", flush=True)

    # ---- branch guard --------------------------------------------------------
    # Cold-start collapses onto the fully-revealing root (1-R²->0), not the
    # strong price-revelation branch. Reject such a solve (exit nonzero so the
    # runner parks the task) rather than committing wrong-branch data.
    P0 = sym_init_no_learning(sg, u_grid, tau, gamma, W)
    nl = float(sym_weighted_R2(P0, sg, u_grid, tau)["1-R2"])
    if m["1-R2"] < 0.5 * nl:
        print(f"[solve] REJECT: 1-R²={m['1-R2']:.3e} << no-learning {nl:.3e} "
              f"(fully-revealing collapse). Needs PR-branch continuation from the seed.",
              flush=True)
        sys.exit(3)

    # ---- write the immutable solution version --------------------------------
    vdir = REPO / "solutions" / "pool" / PROBLEM / version
    (vdir / "data").mkdir(parents=True, exist_ok=True)
    (vdir / "figure").mkdir(parents=True, exist_ok=True)

    data_path = vdir / "data" / "solution.npz"
    np.savez_compressed(
        data_path,
        P_sorted=P, tuples=sg.tuples, u_grid=u_grid,
        K=3, G=G, gamma=gamma, tau=tau, W=W,
        F_inf=F_inf, one_minus_R2=m["1-R2"], slope=m["slope"], intercept=m["intercept"],
    )
    fig_path = vdir / "figure" / "fig_knife_edge.pdf"
    try:
        make_figure(fig_path, P, sg, u_grid, tau, m)
        figures = ["figure/fig_knife_edge.pdf"]
    except Exception as e:   # matplotlib not installed on the runner — data is the deliverable
        print(f"[solve] figure skipped ({e})", flush=True)
        figures = []

    sha = standards_sha()
    meta = {
        "problem": PROBLEM,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "standards_methods_sha": sha,
        "task_id": task_id,
        "params": {"gamma": gamma, "tau": tau, "K": 3, "G": G, "W": W},
        "metrics": {"F_inf": f"{F_inf:.3e}", "one_minus_R2": round(m["1-R2"], 8)},
        "data": ["data/solution.npz"],
        "figures": figures,
        "solver": {
            "engine": "symmetric-anderson (contour_KN_sym)",
            "precision": "float64",
            "entrypoint": "numerics/ree_K3/solve.py",
            "f_tol": f"{args.f_tol:.0e}",
            "n_evals": n_evals,
        },
    }
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[solve] wrote {vdir.relative_to(REPO)}/ (meta.json, data/, figure/)", flush=True)

    update_registry(PROBLEM, version, task_id, sha, m, F_inf)


def update_registry(problem, version, task_id, sha, metrics, F_inf) -> None:
    reg_path = REPO / "solutions" / "REGISTRY.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {
        "registry_version": 1, "solutions": []
    }
    reg["solutions"] = [s for s in reg.get("solutions", [])
                        if not (s.get("problem") == problem and s.get("version") == version)]
    reg["solutions"].append({
        "problem": problem,
        "version": version,
        "task_id": task_id,
        "standards_methods_sha": sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "one_minus_R2": round(metrics["1-R2"], 8),
        "F_inf": f"{F_inf:.3e}",
    })
    reg["updated_at"] = datetime.now(timezone.utc).isoformat()
    reg_path.write_text(json.dumps(reg, indent=2) + "\n")
    print(f"[solve] registry: {problem}/{version} recorded", flush=True)


if __name__ == "__main__":
    main()
