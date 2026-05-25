"""Bandwidth-free (no h) marching-squares co-area: convergence + deficit."""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
sys.path.insert(0, "/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import init_no_learning_K3
from code.halo import extract_inner, replace_inner
from ode_sweep import anderson_solve
from marching_coarea import phi_K3_marching

tau, gamma, W = 0.5, 1.0, 1.0
tv, gv, Wv = np.full(3, tau), np.full(3, gamma), np.full(3, W)
OUT = "/home/user/MIWN/numerics/ree_K3/marching_results.txt"
fh = open(OUT, "w")
def log(s): fh.write(s + "\n"); fh.flush(); print(s, flush=True)

def grid(G, pad=4, um=3.0):
    du = 2.0 * um / (G - 1); Gf = G + 2 * pad
    u = np.array([-um + (q - pad) * du for q in range(Gf)]); return u, pad, pad + G, du

def deficit(P, ui):
    G = len(ui)
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij")
    T = tau * (U1 + U2 + U3)
    f1 = np.exp(-tau / 2 * (ui - 0.5) ** 2); f0 = np.exp(-tau / 2 * (ui + 0.5) ** 2)
    F1 = f1[:, None, None] * f1[None, :, None] * f1[None, None, :]
    F0 = f0[:, None, None] * f0[None, :, None] * f0[None, None, :]
    Wt = 0.5 * (F1 + F0)
    m = (P > 1e-4) & (P < 1 - 1e-4)
    t = T[m]; L = np.log(P[m] / (1 - P[m])); w = Wt[m]; w = w / w.sum()
    def wr2(deg):
        c = np.polyfit(t, L, deg, w=np.sqrt(w)); pr = np.polyval(c, t)
        lm = np.average(L, weights=w); vt = np.average((L - lm) ** 2, weights=w)
        return np.average((L - pr) ** 2, weights=w) / vt
    return wr2(1), wr2(5)

def solve(G, tol=1e-12, mi=4000):
    u, lo, hi, du = grid(G); halo = init_no_learning_K3(u, tv, gv, Wv)
    def phi(P):
        Pf = replace_inner(halo, P, lo, hi)
        return extract_inner(phi_K3_marching(Pf, u, lo, hi, tv, gv, Wv), lo, hi)
    Ps, res = anderson_solve(phi, extract_inner(halo, lo, hi), tol=tol, max_iter=mi, m=8)
    lin, scat = deficit(Ps, u[lo:hi])
    return lin, scat, res, du

if __name__ == "__main__":
    log("MARCHING-SQUARES co-area (NO bandwidth).  lin=linear 1-R2, scat=scatter(true deficit), Jensen=lin-scat")
    for G in (14, 18, 22, 26):
        t = time.time(); lin, scat, res, du = solve(G)
        log(f"   G={G:2d} du={du:.4f}  1-R2_lin={lin:.6f}  scatter={scat:.6f}  Jensen={lin-scat:.6f}  res={res:.1e} ({time.time()-t:.0f}s)")
    fh.close()
