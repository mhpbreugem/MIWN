"""Direct test on the paper's actual solutions (FPF checkpoints).

For each stored REZN/FPF solution (weightless-co-area, continuation-selected,
1-R2 ~ 0.10, F_max ~ 1e-51):
  1. recompute its 1-R2 (verify);
  2. residual under the WEIGHTLESS cubic map  (should be ~0: it's its fixed pt);
  3. residual under the WEIGHTED co-area map  (1/|grad P| restored);
  4. iterate the WEIGHTED map from it (damped Picard + Cesaro): does 1-R2 stay
     ~0.10 (deficit branch survives proper weighting) or collapse toward 0
     (it was the weightless-co-area artifact)?
"""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
sys.path.insert(0, "/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import phi_K3_halo_cubic
from code.halo import extract_inner, replace_inner
from coarea import phi_K3_coarea

CKDIR = "/home/user/FIXED-POINT-FACTORY/projects/REZN/checkpoints"
OUT = open("/home/user/MIWN/numerics/ree_K3/checkpoint_test.txt", "w")
def log(s): OUT.write(s + "\n"); OUT.flush(); print(s, flush=True)

def deficit(Pin, ui, tau):
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij"); T = tau * (U1 + U2 + U3)
    f1 = np.exp(-tau / 2 * (ui - 0.5) ** 2); f0 = np.exp(-tau / 2 * (ui + 0.5) ** 2)
    F1 = f1[:, None, None] * f1[None, :, None] * f1[None, None, :]
    F0 = f0[:, None, None] * f0[None, :, None] * f0[None, None, :]
    W = 0.5 * (F1 + F0); m = (Pin > 1e-4) & (Pin < 1 - 1e-4)
    t = T[m]; L = np.log(Pin[m] / (1 - Pin[m])); w = W[m]; w = w / w.sum(); lm = np.average(L, weights=w)
    c1 = np.polyfit(t, L, 1, w=np.sqrt(w)); c5 = np.polyfit(t, L, 5, w=np.sqrt(w))
    lin = np.average((L - np.polyval(c1, t)) ** 2, weights=w) / np.average((L - lm) ** 2, weights=w)
    scat = np.average((L - np.polyval(c5, t)) ** 2, weights=w) / np.average((L - lm) ** 2, weights=w)
    return lin, scat

def run_from(P_full, u, lo, hi, tv, gv, Wv, weighted, n=200, alpha=0.2, ces=100):
    P = extract_inner(P_full, lo, hi).copy(); acc = None; cnt = 0; best = (P.copy(), 1e9)
    for it in range(n):
        Pf = replace_inner(P_full, P, lo, hi)
        Q = extract_inner(phi_K3_coarea(Pf, u, lo, hi, tv, gv, Wv) if weighted
                          else phi_K3_halo_cubic(Pf, u, lo, hi, tv, gv, Wv), lo, hi)
        F = float(np.max(np.abs(Q - P)))
        if F < best[1]: best = (P.copy(), F)
        P = (1 - alpha) * P + alpha * Q; P = np.clip(P, 1e-9, 1 - 1e-9)
        if it >= n - ces: acc = Q.copy() if acc is None else acc + Q; cnt += 1
    return acc / cnt, best[1]

def res_under(P_full, u, lo, hi, tv, gv, Wv, weighted):
    Q = phi_K3_coarea(P_full, u, lo, hi, tv, gv, Wv) if weighted else phi_K3_halo_cubic(P_full, u, lo, hi, tv, gv, Wv)
    inr = (slice(lo, hi),) * 3
    return float(np.max(np.abs(Q[inr] - P_full[inr])))

NAMES = ("g100_t0400", "g100_t0300", "g025_t0300", "g050_t0300")

def load(name):
    a = np.load(f"{CKDIR}/{name}.npz", allow_pickle=True)
    return dict(P_full=a["P_full"].astype(float), u=a["u_full"].astype(float),
               tv=a["tau_vec"].astype(float), gv=a["gamma_vec"].astype(float),
               Wv=a["W_vec"].astype(float), G=int(a["G_inner"]), pad=int(a["pad"]),
               Pin=a["P_inner"].astype(float))

if __name__ == "__main__":
    log("LOOP A: cheap residual diagnostics (which map has which fixed point?)")
    cache = {}
    for name in NAMES:
        try:
            d = load(name); cache[name] = d
        except Exception as e:
            log(f"{name}: load failed {e}"); continue
        P_full, u, tv, gv, Wv = d["P_full"], d["u"], d["tv"], d["gv"], d["Wv"]
        G, pad = d["G"], d["pad"]; lo, hi = pad, pad + G; ui = u[lo:hi]; tau = float(tv[0])
        lin0, scat0 = deficit(d["Pin"], ui, tau)
        # sigma(T*) on the same grid
        U1, U2, U3 = np.meshgrid(u, u, u, indexing="ij"); Psig = 1 / (1 + np.exp(-tau * (U1 + U2 + U3)))
        rsig_wl = res_under(Psig, u, lo, hi, tv, gv, Wv, weighted=False)
        rsig_w = res_under(Psig, u, lo, hi, tv, gv, Wv, weighted=True)
        r_wl = res_under(P_full, u, lo, hi, tv, gv, Wv, weighted=False)
        r_w = res_under(P_full, u, lo, hi, tv, gv, Wv, weighted=True)
        log(f"\n=== {name} gamma={gv[0]} tau={tau} G={G} | stored 1-R2={lin0:.5f} scat={scat0:.5f} ===")
        log(f"   residual of STORED   under weightless={r_wl:.2e}  weighted={r_w:.2e}")
        log(f"   residual of sigma(T*) under weightless={rsig_wl:.2e}  weighted={rsig_w:.2e}")
    log("\nLOOP B: iterate from the stored deficit solution (does deficit survive?)")
    for name in NAMES:
        if name not in cache: continue
        d = cache[name]; P_full, u, tv, gv, Wv = d["P_full"], d["u"], d["tv"], d["gv"], d["Wv"]
        G, pad = d["G"], d["pad"]; lo, hi = pad, pad + G; ui = u[lo:hi]; tau = float(tv[0])
        t0 = time.time()
        Pw, rw = run_from(P_full, u, lo, hi, tv, gv, Wv, weighted=True)
        lw, sw = deficit(Pw, ui, tau)
        Pc, rc = run_from(P_full, u, lo, hi, tv, gv, Wv, weighted=False)
        lc, sc = deficit(Pc, ui, tau)
        log(f"  {name}: WEIGHTED-from-stored 1-R2={lw:.5f} scat={sw:.5f}(res {rw:.0e}) | "
            f"WEIGHTLESS-control 1-R2={lc:.5f}(res {rc:.0e}) ({time.time()-t0:.0f}s)")
    OUT.close()
