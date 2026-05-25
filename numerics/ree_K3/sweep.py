"""
sweep.py — double-double ODE-continuation sweep over gamma (REE K=3).

Tracks the fixed-point branch P*(gamma) at fixed tau by parameter continuation:

  predictor   RK4 integration of the tangent ODE (I - dphi/dP) v = dphi/dgamma,
              solved Jacobian-free via GMRES (one phi-eval per matvec).
              -> 4th-order accurate guess at the next gamma.
  corrector   float64 Anderson acceleration snaps the RK4 guess back onto the
              manifold to ~1e-6.
  polish      the project's quadratic mpmath Newton (phi_newton_mp) drives the
              residual to the Standards precision policy: double-double
              (dps=32), accept only at ||F|| < 1e-20.

The RK4 predictor reuses standards/methods/solver/ode_sweep_rk4.py; the dd polish
reuses phi_newton_mp so swept points satisfy the same policy as cold-start solves.

Usage:
    python3 numerics/ree_K3/sweep.py \
        --anchor solutions/pool/ree_K3/v0010/data/solution.npz \
        --gammas 3.2,4.0,5.0            # tau is read from the anchor

Output: per-gamma ||F|| and 1-R^2, plus an optional --out JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SOLVER = REPO / "standards" / "methods" / "solver"
sys.path.insert(0, str(SOLVER))

from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth  # noqa: E402
from code.halo import extract_inner, replace_inner                       # noqa: E402
from code.f128 import revelation_deficit_f128                            # noqa: E402
from phi_mp import phi_newton_mp                                         # noqa: E402
from precision import WORKING_DPS, TOL_STR, DONE_THRESHOLD               # noqa: E402
from ode_sweep import anderson_solve                                    # noqa: E402
from ode_sweep_rk4 import _rk4_predict                                  # noqa: E402

K = 3


def load_anchor(path: Path):
    d = np.load(path, allow_pickle=True)
    if "P_inner" not in d:
        raise SystemExit(f"{path}: not a K3 dd solution (no P_inner)")
    G_inner, pad = int(d["G_inner"]), int(d["pad"])
    u_full = d["u_full"].astype(np.float64)
    inner_lo, inner_hi = pad, pad + G_inner
    du = float(u_full[1] - u_full[0])
    kernel_h = max(0.005, 0.05 * du)
    gamma0, tau = float(d["gamma"]), float(d["tau"])
    P_inner = d["P_inner"].astype(np.float64)
    return dict(u_full=u_full, inner_lo=inner_lo, inner_hi=inner_hi,
                kernel_h=kernel_h, gamma0=gamma0, tau=tau, P_inner=P_inner,
                G_inner=G_inner, pad=pad)


def main() -> None:
    ap = argparse.ArgumentParser(description="dd ODE-continuation sweep over gamma")
    ap.add_argument("--anchor", default="solutions/pool/ree_K3/v0010/data/solution.npz")
    ap.add_argument("--gammas", default="3.2,4.0,5.0",
                    help="comma list of gamma values to sweep (tau from anchor)")
    ap.add_argument("--corr-tol", type=float, default=1e-6, help="float64 corrector target")
    ap.add_argument("--max-wall-s", type=float, default=300.0, help="dd polish wall cap / point")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    a = load_anchor((REPO / args.anchor) if not Path(args.anchor).is_absolute() else Path(args.anchor))
    u_full, inner_lo, inner_hi = a["u_full"], a["inner_lo"], a["inner_hi"]
    kernel_h, tau = a["kernel_h"], a["tau"]
    u_inner = u_full[inner_lo:inner_hi]
    tv, Wv = np.full(K, tau), np.ones(K)

    grid = [float(x) for x in args.gammas.split(",")]
    anchor_idx = int(np.argmin([abs(g - a["gamma0"]) for g in grid]))
    print(f"[sweep] anchor gamma={a['gamma0']} tau={tau} G_inner={a['G_inner']} "
          f"pad={a['pad']} | grid={grid} anchor_idx={anchor_idx} "
          f"| policy dps={WORKING_DPS} target={TOL_STR}", flush=True)

    def phi_f64_factory(g):
        gv = np.full(K, g)
        def phi(P):
            return phi_K3_halo_smooth(P, u_full, inner_lo, inner_hi, tv, gv, Wv, kernel_h)
        return phi

    # anchor P_full = no-learning halo with the converged inner block dropped in
    halo0 = init_no_learning_K3(u_full, tv, np.full(K, a["gamma0"]), Wv)
    P_anchor_full = replace_inner(halo0, a["P_inner"], inner_lo, inner_hi)

    n = len(grid)
    results = [None] * n
    r2_anchor = float(revelation_deficit_f128(a["P_inner"], u_inner, tv, K))
    results[anchor_idx] = dict(gamma=grid[anchor_idx], F_inf=None, one_minus_R2=r2_anchor,
                               newton=0, note="anchor (given)")

    for direction, idxs in [("->", range(anchor_idx, n)),
                            ("<-", range(anchor_idx - 1, -1, -1))]:
        P_prev = P_anchor_full.copy()
        g_prev = grid[anchor_idx]
        for idx in idxs:
            if results[idx] is not None:
                P_prev, g_prev = P_anchor_full.copy(), grid[idx]
                continue
            g = grid[idx]
            t0 = time.time()
            print(f"\n[sweep] {direction} gamma={g:.4f} (from {g_prev:.4f})", flush=True)

            # 1. RK4 tangent predictor
            try:
                P_pred, res_pred = _rk4_predict(
                    phi_f64_factory, P_prev, g_prev, g,
                    eps_gamma=1e-5, gmres_tol=1e-5, gmres_restart=50,
                    gmres_maxiter=5, verbose=False)
                print(f"        RK4 predict ||F||={res_pred:.3e}", flush=True)
            except Exception as exc:
                print(f"        RK4 failed ({exc}); using previous point", flush=True)
                P_pred = P_prev.copy()

            # 2. float64 Anderson corrector
            P_corr, res_f64 = anderson_solve(phi_f64_factory(g), P_pred,
                                             tol=args.corr_tol, max_iter=200, m=5)
            print(f"        corrector   ||F||={res_f64:.3e}", flush=True)

            # 3. dd Newton polish to the 1e-20 policy
            halo = init_no_learning_K3(u_full, tv, np.full(K, g), Wv)
            P_inner_seed = extract_inner(P_corr, inner_lo, inner_hi)
            P_inner_mp, F_inf, n_mp, _ = phi_newton_mp(
                P_inner_seed, halo, u_full, inner_lo, inner_hi, tv, np.full(K, g), Wv,
                kernel_h, phi_float64_fn=phi_f64_factory(g),
                dps=WORKING_DPS, tol_str=TOL_STR, max_newton=50,
                lgmres_tol=1e-10, lgmres_inner_m=30, lgmres_outer=10,
                max_wall_s=args.max_wall_s)
            r2 = float(revelation_deficit_f128(P_inner_mp, u_inner, tv, K))
            ok = "OK" if F_inf < DONE_THRESHOLD else "PARTIAL"
            print(f"        dd Newton   ||F||={F_inf:.3e} ({ok})  1-R2={r2:.6f}  "
                  f"newton={n_mp}  t={time.time()-t0:.0f}s", flush=True)

            results[idx] = dict(gamma=g, F_inf=f"{F_inf:.3e}", one_minus_R2=r2,
                                newton=n_mp, status=ok)
            P_prev = replace_inner(halo, P_inner_mp, inner_lo, inner_hi)
            g_prev = g

    print("\n[sweep] === summary (tau={:.3g}) ===".format(tau), flush=True)
    print(f"{'gamma':>8} {'||F||':>12} {'1-R^2':>12} {'newton':>7}  status", flush=True)
    for r in results:
        print(f"{r['gamma']:>8.4f} {str(r['F_inf']):>12} {r['one_minus_R2']:>12.6f} "
              f"{r['newton']:>7}  {r.get('status', r.get('note',''))}", flush=True)

    if args.out:
        outp = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(
            dict(tau=tau, anchor=str(args.anchor), policy=dict(dps=WORKING_DPS, target=TOL_STR),
                 results=results), indent=2))
        print(f"[sweep] wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
