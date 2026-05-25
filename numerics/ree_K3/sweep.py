"""
sweep.py — solve ONE fixed point by ODE continuation (REE K=3, double-double).

There is a single fixed point P*(theta) that deforms continuously with the model
parameters theta=(gamma,tau). It satisfies F(P,theta)=phi(P,theta)-P=0, and
differentiating implicitly gives the tangent ODE, defined *at* the fixed point:

    (I - dphi/dP) . dP*/dtheta = dphi/dtheta .

This is a SOLVER for one target fixed point (not a grid generator). Given a
converged anchor P*(theta_anchor) it integrates that ODE along a path to the
target and lands on the single target fixed point:

  path (cheap)   RK4 integration of the tangent ODE in float64, Jacobian-free via
                 GMRES, with a short Anderson corrector after each substep so the
                 tangent stays on the manifold. Intermediate substeps are the
                 integration path only — NOT outputs.
  target (dd)    one quadratic mpmath Newton polish (phi_newton_mp) at the target
                 drives ||F|| to the Standards policy: double-double (dps=32),
                 accept only ||F|| < 1e-20.

A target that differs in both gamma and tau is reached in two legs (walk gamma at
fixed tau, then tau at fixed gamma); the dd polish happens once, at the target.

`continue_solve(...)` is importable (used by sweep2d.py). The CLI solves one point.

Usage:
    python3 numerics/ree_K3/sweep.py \
        --anchor solutions/pool/ree_K3/v0010/data/solution.npz --gamma 6.5 --tau 2
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


def load_anchor(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    if "P_inner" not in d:
        raise SystemExit(f"{path}: not a K3 dd solution (no P_inner)")
    G_inner, pad = int(d["G_inner"]), int(d["pad"])
    u_full = d["u_full"].astype(np.float64)
    du = float(u_full[1] - u_full[0])
    ctx = dict(u_full=u_full, inner_lo=pad, inner_hi=pad + G_inner,
               kernel_h=max(0.005, 0.05 * du), G_inner=G_inner, pad=pad)
    halo = init_no_learning_K3(u_full, np.full(K, float(d["tau"])),
                               np.full(K, float(d["gamma"])), np.ones(K))
    P_full = replace_inner(halo, d["P_inner"].astype(np.float64), pad, pad + G_inner)
    return dict(ctx=ctx, gamma=float(d["gamma"]), tau=float(d["tau"]), P_full=P_full)


def _phi_factory(ctx, which: str, fixed: float):
    """Return f(p)->phi(P), where p is the continuation parameter (`which`)."""
    u_full, lo, hi, h = ctx["u_full"], ctx["inner_lo"], ctx["inner_hi"], ctx["kernel_h"]
    W = np.ones(K)
    def factory(p):
        gv = np.full(K, p if which == "gamma" else fixed)
        tv = np.full(K, p if which == "tau" else fixed)
        def phi(P):
            return phi_K3_halo_smooth(P, u_full, lo, hi, tv, gv, W, h)
        return phi
    return factory


def _walk(ctx, P, which, fixed, p_from, p_to, max_dp, corr_tol, log):
    """Continuation walk of one parameter (float64 predictor+corrector). Returns P."""
    if abs(p_to - p_from) < 1e-12:
        return P
    fac = _phi_factory(ctx, which, fixed)
    nsteps = max(1, int(np.ceil(abs(p_to - p_from) / max_dp)))
    path = list(np.linspace(p_from, p_to, nsteps + 1))
    for s in range(nsteps):
        pp, pn = path[s], path[s + 1]
        try:
            P_pred, res_pred = _rk4_predict(fac, P, pp, pn, eps_gamma=1e-5,
                                            gmres_tol=1e-5, gmres_restart=50,
                                            gmres_maxiter=5, verbose=False)
        except Exception as exc:
            log(f"    RK4 {which}={pp:.3f}->{pn:.3f} failed ({exc}); plain step")
            P_pred = P.copy()
        P, res_c = anderson_solve(fac(pn), P_pred, tol=corr_tol, max_iter=200, m=5)
        log(f"    {which}={pn:.4f}  RK4 ||F||={res_pred:.2e} -> corr ||F||={res_c:.2e}")
    return P


def continue_solve(ctx, P_start, g_from, t_from, g_to, t_to,
                   max_dparam=1.0, corr_tol=1e-6, max_wall_s=600.0,
                   verbose=True) -> dict:
    """Solve the single fixed point at (g_to,t_to) by continuation from
    (g_from,t_from). Walk gamma then tau (float64 path), dd-polish at the target."""
    log = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    lo, hi, u_full, h = ctx["inner_lo"], ctx["inner_hi"], ctx["u_full"], ctx["kernel_h"]
    u_inner = u_full[lo:hi]
    t0 = time.time()
    P = P_start.copy()
    P = _walk(ctx, P, "gamma", t_from, g_from, g_to, max_dparam, corr_tol, log)  # leg 1
    P = _walk(ctx, P, "tau",   g_to,   t_from, t_to, max_dparam, corr_tol, log)  # leg 2

    tv, gv = np.full(K, t_to), np.full(K, g_to)
    halo = init_no_learning_K3(u_full, tv, gv, np.ones(K))
    P_inner_mp, F_inf, n_mp, mp_str = phi_newton_mp(
        extract_inner(P, lo, hi), halo, u_full, lo, hi, tv, gv, np.ones(K), h,
        phi_float64_fn=_phi_factory(ctx, "gamma", t_to)(g_to),
        dps=WORKING_DPS, tol_str=TOL_STR, max_newton=50,
        lgmres_tol=1e-10, lgmres_inner_m=30, lgmres_outer=10, max_wall_s=max_wall_s)
    r2 = float(revelation_deficit_f128(P_inner_mp, u_inner, tv, K))
    return dict(gamma=g_to, tau=t_to, F_inf=float(F_inf), one_minus_R2=r2,
                newton=n_mp, accepted=bool(F_inf < DONE_THRESHOLD),
                wall_s=round(time.time() - t0, 1),
                P_full=replace_inner(halo, P_inner_mp, lo, hi))


def main() -> None:
    ap = argparse.ArgumentParser(description="solve one fixed point by ODE continuation")
    ap.add_argument("--anchor", default="solutions/pool/ree_K3/v0010/data/solution.npz")
    ap.add_argument("--gamma", type=float, required=True, help="TARGET gamma")
    ap.add_argument("--tau", type=float, default=None, help="TARGET tau (default: anchor's)")
    ap.add_argument("--max-dparam", type=float, default=1.0, help="max gamma/tau per substep")
    ap.add_argument("--max-wall-s", type=float, default=600.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a = load_anchor((REPO / args.anchor) if not Path(args.anchor).is_absolute() else Path(args.anchor))
    tau = a["tau"] if args.tau is None else args.tau
    print(f"[sweep] solve ONE fixed point  target (gamma={args.gamma}, tau={tau}) "
          f"from anchor (gamma={a['gamma']}, tau={a['tau']}) | policy dps={WORKING_DPS} "
          f"target={TOL_STR}", flush=True)
    r = continue_solve(a["ctx"], a["P_full"], a["gamma"], a["tau"], args.gamma, tau,
                       max_dparam=args.max_dparam, max_wall_s=args.max_wall_s)
    print(f"\n[sweep] === single fixed point (gamma={r['gamma']}, tau={r['tau']}) ===", flush=True)
    print(f"  ||F||   = {r['F_inf']:.3e}   ({'ACCEPT <1e-20' if r['accepted'] else 'PARTIAL'})", flush=True)
    print(f"  1-R^2   = {r['one_minus_R2']:.6f}", flush=True)
    print(f"  newton  = {r['newton']} steps   wall = {r['wall_s']}s", flush=True)
    if args.out:
        outp = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        rec = {k: v for k, v in r.items() if k != "P_full"}
        rec.update(anchor=str(args.anchor), policy=dict(dps=WORKING_DPS, target=TOL_STR))
        outp.write_text(json.dumps(rec, indent=2))
        print(f"[sweep] wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
