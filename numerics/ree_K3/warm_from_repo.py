"""Warm-start the bandwidth-free (h=0) cubic co-area map from each REPO solution
(the finite-h curved branch) instead of no-learning.  If the deficit survives,
the partial-revelation (curved) equilibrium is a real branch; if it collapses to
~1e-3 it was the bandwidth artifact.
"""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
sys.path.insert(0, "/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_cubic
from code.halo import extract_inner, replace_inner
from ode_sweep import anderson_solve

OUT = open("/home/user/MIWN/numerics/ree_K3/warm_results.txt", "w")
def log(s): OUT.write(s + "\n"); OUT.flush(); print(s, flush=True)

def deficit(P, ui, tau):
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij"); S = U1 + U2 + U3
    f1 = np.exp(-tau / 2 * (ui - 0.5) ** 2); f0 = np.exp(-tau / 2 * (ui + 0.5) ** 2)
    F1 = f1[:, None, None] * f1[None, :, None] * f1[None, None, :]
    F0 = f0[:, None, None] * f0[None, :, None] * f0[None, None, :]
    Wt = 0.5 * (F1 + F0); m = (P > 1e-4) & (P < 1 - 1e-4)
    t = S[m]; L = np.log(P[m] / (1 - P[m])); w = Wt[m]; w = w / w.sum()
    def wr2(deg):
        c = np.polyfit(t, L, deg, w=np.sqrt(w)); pr = np.polyval(c, t); lm = np.average(L, weights=w)
        return np.average((L - pr) ** 2, weights=w) / np.average((L - lm) ** 2, weights=w)
    return wr2(1), wr2(5)

def warm(ver, tol=1e-9, mi=5000):
    a = np.load(f"solutions/pool/ree_K3/{ver}/data/solution.npz", allow_pickle=True)
    gamma = float(a["gamma"]); tau = float(a["tau"]); pad = int(a["pad"])
    u = a["u_full"].astype(float); G = int(a["G_inner"]); lo, hi = pad, pad + G
    P_inner0 = a["P_inner"].astype(float)
    tv = np.full(3, tau); gv = np.full(3, gamma); Wv = np.full(3, 1.0)
    halo = init_no_learning_K3(u, tv, gv, Wv)
    stored_lin, stored_scat = deficit(P_inner0, u[lo:hi], tau)
    def phi(P):
        Pf = replace_inner(halo, P, lo, hi)
        return extract_inner(phi_K3_halo_cubic(Pf, u, lo, hi, tv, gv, Wv), lo, hi)
    Ps, res = anderson_solve(phi, P_inner0.copy(), tol=tol, max_iter=mi, m=6)
    lin, scat = deficit(Ps, u[lo:hi], tau)
    # also cold-start for comparison
    Pc, rc = anderson_solve(phi, extract_inner(halo, lo, hi), tol=tol, max_iter=mi, m=6)
    linc, scatc = deficit(Pc, u[lo:hi], tau)
    return gamma, tau, stored_lin, stored_scat, lin, scat, res, linc, scatc

if __name__ == "__main__":
    log("Warm-start bandwidth-free cubic from REPO solutions (G=10).")
    log("  stored = finite-h solution's deficit ;  warm = after bandwidth-free cubic from it ; cold = from no-learning")
    for ver in ("v0011", "v0009", "v0008", "v0010"):
        t = time.time()
        g, ta, sl, ss, lin, scat, res, linc, scatc = warm(ver)
        log(f"  {ver} g={g} tau={ta}: stored 1-R2={sl:.4f}(scat {ss:.4f}) | "
            f"WARM->h0: 1-R2={lin:.4f} scat={scat:.4f} | COLD->h0: 1-R2={linc:.4f} scat={scatc:.4f} | res={res:.1e} ({time.time()-t:.0f}s)")
    OUT.close()
