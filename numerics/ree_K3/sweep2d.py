"""
sweep2d.py — map the fixed-point branch P*(gamma,tau) across a 2D grid by
continuation (REE K=3, double-double).

A 2D sweep is NOT a set of independent solves: it traces the *one* branch across
(gamma,tau). Starting from a converged anchor it walks a gamma-spine at the
anchor's tau, then walks a tau-column from each spine node — every node reached by
the tangent-ODE continuation in sweep.continue_solve and dd-polished to the
Standards policy (dps=32, ||F|| < 1e-20).

Output: a (gamma x tau) table of 1-R^2 and ||F||, plus JSON.

Usage:
    python3 numerics/ree_K3/sweep2d.py \
        --anchor solutions/pool/ree_K3/v0010/data/solution.npz \
        --gammas 2.5,4,6.5 --taus 1,2,4 --out solutions/derived/ree_K3/sweep2d.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sweep import load_anchor, continue_solve, REPO  # noqa: E402


def nearest(value, solved_keys):
    return min(solved_keys, key=lambda k: abs(k - value))


def main() -> None:
    ap = argparse.ArgumentParser(description="2D (gamma,tau) continuation sweep")
    ap.add_argument("--anchor", default="solutions/pool/ree_K3/v0010/data/solution.npz")
    ap.add_argument("--gammas", default="2.5,4,6.5")
    ap.add_argument("--taus", default="1,2,4")
    ap.add_argument("--max-dparam", type=float, default=1.0)
    ap.add_argument("--max-wall-s", type=float, default=600.0)
    ap.add_argument("--out", default="solutions/derived/ree_K3/sweep2d.json")
    args = ap.parse_args()

    a = load_anchor((REPO / args.anchor) if not Path(args.anchor).is_absolute() else Path(args.anchor))
    ctx, g0, t0, P0 = a["ctx"], a["gamma"], a["tau"], a["P_full"]
    gammas = sorted({float(x) for x in args.gammas.split(",")} | {g0})
    taus = sorted({float(x) for x in args.taus.split(",")} | {t0})
    print(f"[2d] anchor (gamma={g0}, tau={t0})  grid gammas={gammas} taus={taus}", flush=True)

    grid = {}          # (g,t) -> result dict (with P_full)
    grid[(g0, t0)] = dict(gamma=g0, tau=t0, F_inf=None, one_minus_R2=None,
                          newton=0, accepted=True, wall_s=0.0, P_full=P0, note="anchor")
    t_start = time.time()

    # ── gamma-spine at tau = t0 (walk out from the anchor gamma) ──
    spine_order = sorted(gammas, key=lambda g: abs(g - g0))   # nearest first
    for g in spine_order:
        if (g, t0) in grid:
            continue
        src_g = nearest(g, [k[0] for k in grid if k[1] == t0])
        print(f"\n[2d] spine gamma={g} (tau={t0}) from gamma={src_g}", flush=True)
        r = continue_solve(ctx, grid[(src_g, t0)]["P_full"], src_g, t0, g, t0,
                           max_dparam=args.max_dparam, max_wall_s=args.max_wall_s)
        grid[(g, t0)] = r
        print(f"    -> ||F||={r['F_inf']:.2e}  1-R2={r['one_minus_R2']:.6f}  "
              f"({'OK' if r['accepted'] else 'PARTIAL'})  {r['wall_s']}s", flush=True)

    # ── tau-columns from each spine node ──
    for g in gammas:
        col_order = sorted(taus, key=lambda t: abs(t - t0))
        for t in col_order:
            if (g, t) in grid:
                continue
            src_t = nearest(t, [k[1] for k in grid if k[0] == g])
            print(f"\n[2d] column gamma={g} tau={t} from tau={src_t}", flush=True)
            r = continue_solve(ctx, grid[(g, src_t)]["P_full"], g, src_t, g, t,
                               max_dparam=args.max_dparam, max_wall_s=args.max_wall_s)
            grid[(g, t)] = r
            print(f"    -> ||F||={r['F_inf']:.2e}  1-R2={r['one_minus_R2']:.6f}  "
                  f"({'OK' if r['accepted'] else 'PARTIAL'})  {r['wall_s']}s", flush=True)

    # ── tables ──
    def cell(g, t, key, fmt):
        v = grid[(g, t)].get(key)
        return fmt.format(v) if isinstance(v, (int, float)) else "   anchor "

    print("\n[2d] === 1-R^2  (rows gamma, cols tau) ===", flush=True)
    print("gamma\\tau " + "".join(f"{t:>11.3g}" for t in taus), flush=True)
    for g in gammas:
        print(f"{g:>8.3g} " + "".join(cell(g, t, "one_minus_R2", "{:>11.6f}") for t in taus), flush=True)
    print("\n[2d] === ||F||  (rows gamma, cols tau) ===", flush=True)
    print("gamma\\tau " + "".join(f"{t:>11.3g}" for t in taus), flush=True)
    for g in gammas:
        print(f"{g:>8.3g} " + "".join(
            (("{:>11.2e}".format(grid[(g, t)]["F_inf"]) if grid[(g, t)]["F_inf"] is not None
              else "   anchor ")) for t in taus), flush=True)

    print(f"\n[2d] {len(grid)} nodes in {time.time()-t_start:.0f}s "
          f"({sum(1 for r in grid.values() if r.get('accepted') and r['F_inf'] is not None)} "
          f"accepted <1e-20)", flush=True)

    outp = (REPO / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    nodes = [{k: v for k, v in r.items() if k != "P_full"} for r in grid.values()]
    outp.write_text(json.dumps(dict(anchor=str(args.anchor), gammas=gammas, taus=taus,
                                    nodes=nodes), indent=2))
    print(f"[2d] wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
