#!/usr/bin/env python3
"""
solve.py — double-double driver for the ree_K3 problem (MIWN).

Solves the K=3 REE fixed point with the shared halo + mpmath-Newton method from
the Standards submodule, at the Standards precision policy: double-double working
precision (dps=32) with a minimum convergence threshold of ||F||inf < 1e-20
(see standards/methods/PRECISION_POLICY.md). Writes an immutable solution version
to solutions/pool/ree_K3/vNNNN/.

Flow: no-learning init (or warm-start) -> float64 Anderson handoff (~1e-4)
      -> mpmath-Newton polish at dps=32 to ||F|| < 1e-20 -> write pool version.

Usage:
    python3 numerics/ree_K3/solve.py --gamma 0.5 --tau 2.0 --G 8
    python3 numerics/ree_K3/solve.py --gamma 0.5 --tau 2.0 --G 8 --version v0008
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROBLEM = "ree_K3"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOLVER = REPO / "standards" / "methods" / "solver"
if not SOLVER.exists():
    raise SystemExit(f"[solve] {SOLVER} not found — run: git submodule update --init --recursive")
sys.path.insert(0, str(SOLVER))

from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth  # noqa: E402
from code.halo import extract_inner, replace_inner                       # noqa: E402
from code.f128 import revelation_deficit_f128                            # noqa: E402
from phi_mp import phi_newton_mp                                         # noqa: E402
from precision import WORKING_DPS, TOL_STR, DONE_THRESHOLD, BAIL_THRESHOLD  # noqa: E402


def standards_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO / "standards"), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def next_version(problem: str) -> str:
    pool = REPO / "solutions" / "pool" / problem
    nums = [int(d.name[1:]) for d in pool.iterdir()
            if pool.exists() and d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()] if pool.exists() else []
    return f"v{(max(nums) + 1) if nums else 1:04d}"


def build_grid(G_inner: int, pad: int, u_inner_max: float):
    G_full = G_inner + 2 * pad
    du = (2.0 * u_inner_max) / (G_inner - 1)
    u_full = np.array([-u_inner_max + (q - pad) * du for q in range(G_full)], dtype=np.float64)
    inner_lo, inner_hi = pad, pad + G_inner
    return u_full, inner_lo, inner_hi, max(0.005, 0.05 * du)


def load_warm_start(path: Path, inner_lo: int, inner_hi: int):
    """Return (P_inner_f64, P_inner_mp_str|None) from a halo-format checkpoint, else None."""
    if not path or not path.exists():
        return None
    try:
        d = np.load(path, allow_pickle=True)
        if "P_inner" in d:
            mp_str = d["P_inner_mp_str"] if "P_inner_mp_str" in d else None
            return d["P_inner"].astype(np.float64), mp_str
    except Exception as e:
        print(f"[solve] warm-start load failed ({e}); cold start", flush=True)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="ree_K3 double-double solve driver (MIWN)")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--G", "--G-inner", type=int, default=8, dest="G_inner",
                    help="inner grid points (halo adds pad on each side)")
    ap.add_argument("--pad", type=int, default=4)
    ap.add_argument("--u-inner-max", "--u-max", type=float, default=3.0, dest="u_inner_max")
    ap.add_argument("--max-seconds", type=float, default=900.0, help="wall cap for the mp-Newton phase")
    ap.add_argument("--anderson-tol", type=float, default=1e-4, help="float64 handoff tolerance")
    ap.add_argument("--warm-start", default=None, help="halo-format .npz to warm-start from")
    ap.add_argument("--version", default=None)
    ap.add_argument("--task-id", default=None)
    ap.add_argument("--worker-id", default=os.environ.get("WORKER_ID", "solve-1"))
    ap.add_argument("--progress-rel", default=None,
                    help="repo-relative progress dir (e.g. todo/progress) — enables live telemetry")
    args = ap.parse_args()

    gamma, tau, K = args.gamma, args.tau, 3
    G_inner, pad = args.G_inner, args.pad
    task_id = args.task_id or f"{PROBLEM}_g{int(round(gamma*100)):04d}_t{int(round(tau*100)):04d}"
    version = args.version or next_version(PROBLEM)

    u_full, inner_lo, inner_hi, kernel_h = build_grid(G_inner, pad, args.u_inner_max)
    u_grid_inner = u_full[inner_lo:inner_hi].copy()
    gv, tv, Wv = np.full(K, gamma), np.full(K, tau), np.ones(K)
    print(f"[solve] {PROBLEM} {version} γ={gamma} τ={tau} G_inner={G_inner} pad={pad} "
          f"dps={WORKING_DPS} target={TOL_STR}", flush=True)

    # live progress (enabled when the runner passes --progress-rel)
    reporter = None
    if args.progress_rel:
        try:
            sys.path.insert(0, str(REPO / "standards" / "runner"))
            from progress import ProgressReporter  # noqa: PLC0415
            reporter = ProgressReporter(
                project="MIWN", task_id=task_id, worker_id=args.worker_id,
                branch=os.environ.get("BRANCH", "main"), interval=15,
                repo_root=REPO, progress_rel=args.progress_rel)
            reporter.start()
        except Exception as e:
            print(f"[solve] progress reporter off ({e})", flush=True); reporter = None

    def _finish(code=None):
        if reporter:
            try: reporter.stop(delete=True)
            except Exception: pass
        if code is not None:
            sys.exit(code)

    halo = init_no_learning_K3(u_full, tv, gv, Wv)

    def phi_full(Pf):
        return phi_K3_halo_smooth(Pf, u_full, inner_lo, inner_hi, tv, gv, Wv, kernel_h)

    # warm-start (halo-format) or cold no-learning init
    ws = load_warm_start(Path(args.warm_start) if args.warm_start else None, inner_lo, inner_hi)
    P_inner_seed = ws[0] if ws else extract_inner(halo, inner_lo, inner_hi)
    P_inner_mp_str_warm = ws[1] if ws else None

    # ---- float64 Anderson handoff (~1e-4) ----
    from scipy.optimize import anderson, NoConvergence  # noqa: PLC0415
    P_full = replace_inner(halo, P_inner_seed, inner_lo, inner_hi)
    best = [P_full.copy(), float("inf"), 0]

    def res(Pf):
        out = phi_full(Pf)
        f = float(np.max(np.abs(extract_inner(out, inner_lo, inner_hi)
                                - extract_inner(Pf, inner_lo, inner_hi))))
        best[2] += 1
        if f < best[1]:
            best[0], best[1] = Pf.copy(), f
        if reporter:
            reporter.update(iter=best[2], ftol=f, phase="anderson")
        return out - Pf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            P_full = anderson(res, P_full, f_tol=args.anderson_tol, maxiter=4000,
                              M=20, line_search="armijo")
        except NoConvergence:
            P_full = best[0]
    P_inner_final = extract_inner(P_full, inner_lo, inner_hi)
    print(f"[solve] float64 handoff ||F||={best[1]:.2e}", flush=True)

    # ---- mpmath-Newton polish at double-double precision -> ||F|| < 1e-20 ----
    P_inner_mp, F_inf, n_mp, mp_str = phi_newton_mp(
        P_inner_final, halo, u_full, inner_lo, inner_hi, tv, gv, Wv, kernel_h,
        phi_float64_fn=phi_full, dps=WORKING_DPS, tol_str=TOL_STR,
        max_newton=50, lgmres_tol=1e-10, lgmres_inner_m=30, lgmres_outer=10,
        reporter=reporter, P_inner_mp_str=P_inner_mp_str_warm, max_wall_s=args.max_seconds,
    )
    one_minus_r2 = float(revelation_deficit_f128(P_inner_mp, u_grid_inner, tv, K))
    print(f"[solve] mp-Newton dps={WORKING_DPS}  ||F||={F_inf:.3e}  iters={n_mp}  "
          f"1-R²={one_minus_r2:.6e}", flush=True)

    # ---- acceptance per Standards policy ----
    if F_inf > BAIL_THRESHOLD:
        print(f"[solve] BAIL: ||F||={F_inf:.3e} > {BAIL_THRESHOLD:.0e}", flush=True)
        _finish(2)
    nl = float(revelation_deficit_f128(extract_inner(halo, inner_lo, inner_hi), u_grid_inner, tv, K))
    if one_minus_r2 < 0.3 * nl:
        print(f"[solve] REJECT: 1-R²={one_minus_r2:.3e} << no-learning {nl:.3e} "
              f"(fully-revealing collapse)", flush=True)
        _finish(3)
    if F_inf > DONE_THRESHOLD:
        print(f"[solve] PARTIAL: ||F||={F_inf:.3e} > policy {DONE_THRESHOLD:.0e} "
              f"(not accepted; retry with more iters/precision)", flush=True)
        _finish(4)

    # ---- write immutable solution version ----
    vdir = REPO / "solutions" / "pool" / PROBLEM / version
    (vdir / "data").mkdir(parents=True, exist_ok=True)
    (vdir / "figure").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        vdir / "data" / "solution.npz",
        P_inner=P_inner_mp.astype(np.float64),
        P_inner_mp_str=mp_str if mp_str is not None else np.array([]),
        u_full=u_full, u_grid_inner=u_grid_inner,
        gamma=gamma, tau=tau, K=K, G_inner=G_inner, pad=pad,
        F_inf=F_inf, one_minus_R2=one_minus_r2,
    )
    sha = standards_sha()
    meta = {
        "problem": PROBLEM, "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "standards_methods_sha": sha, "task_id": task_id,
        "params": {"gamma": gamma, "tau": tau, "K": K, "G_inner": G_inner, "pad": pad,
                   "dps": WORKING_DPS},
        "metrics": {"F_inf": f"{F_inf:.3e}", "one_minus_R2": round(one_minus_r2, 8)},
        "data": ["data/solution.npz"], "figures": [],
        "solver": {"engine": "halo + mpmath-newton",
                   "precision": f"double-double (dps={WORKING_DPS})",
                   "entrypoint": "numerics/ree_K3/solve.py", "n_newton": int(n_mp)},
    }
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[solve] wrote {vdir.relative_to(REPO)}/", flush=True)
    update_registry(PROBLEM, version, task_id, sha, one_minus_r2, F_inf)
    _finish()


def update_registry(problem, version, task_id, sha, one_minus_r2, F_inf) -> None:
    reg_path = REPO / "solutions" / "REGISTRY.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"registry_version": 1, "solutions": []}
    reg["solutions"] = [s for s in reg.get("solutions", [])
                        if not (s.get("problem") == problem and s.get("version") == version)]
    reg["solutions"].append({
        "problem": problem, "version": version, "task_id": task_id,
        "standards_methods_sha": sha, "created_at": datetime.now(timezone.utc).isoformat(),
        "one_minus_R2": round(one_minus_r2, 8), "F_inf": f"{F_inf:.3e}",
    })
    reg["counters"] = reg.get("counters", {})
    reg["counters"][problem] = max(reg["counters"].get(problem, 0), int(version[1:]))
    reg["updated_at"] = datetime.now(timezone.utc).isoformat()
    reg_path.write_text(json.dumps(reg, indent=2) + "\n")
    print(f"[solve] registry: {problem}/{version} recorded", flush=True)


if __name__ == "__main__":
    main()
