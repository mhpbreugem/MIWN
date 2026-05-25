"""Tests for strong curvature / partial-revelation branch (bandwidth-free).

A) cold-start, lower gamma (0.1,0.15,0.2) at tau=0.5
B) warm-start from a CURVED (large-h smooth) price, then bandwidth-free cubic:
   if it stays curved (high scatter) a partial-revelation branch exists.
C) tau sweep at gamma=0.3: does signal precision drive the curvature?
"""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
sys.path.insert(0, "/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_cubic, phi_K3_halo_smooth
from code.halo import extract_inner, replace_inner
from ode_sweep import anderson_solve

W = 1.0
OUT = open("/home/user/MIWN/numerics/ree_K3/branch_results.txt", "w")
def log(s): OUT.write(s + "\n"); OUT.flush(); print(s, flush=True)

def grid(G, pad=4, um=3.0):
    du = 2 * um / (G - 1); Gf = G + 2 * pad
    u = np.array([-um + (q - pad) * du for q in range(Gf)]); return u, pad, pad + G, du

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

def run_cubic(G, gamma, tau, P0_inner=None, tol=1e-9, mi=4000):
    u, lo, hi, du = grid(G); gv = np.full(3, gamma); tv = np.full(3, tau); Wv = np.full(3, W)
    halo = init_no_learning_K3(u, tv, gv, Wv)
    if P0_inner is None:
        P0_inner = extract_inner(halo, lo, hi)
    def phi(P):
        Pf = replace_inner(halo, P, lo, hi)
        return extract_inner(phi_K3_halo_cubic(Pf, u, lo, hi, tv, gv, Wv), lo, hi)
    Ps, res = anderson_solve(phi, P0_inner, tol=tol, max_iter=mi, m=6)
    lin, scat = deficit(Ps, u[lo:hi], tau)
    return lin, scat, res, Ps

def curved_init(G, gamma, tau):
    u, lo, hi, du = grid(G); gv = np.full(3, gamma); tv = np.full(3, tau); Wv = np.full(3, W)
    halo = init_no_learning_K3(u, tv, gv, Wv); h = 0.30 * du
    def phi(P):
        Pf = replace_inner(halo, P, lo, hi)
        return extract_inner(phi_K3_halo_smooth(Pf, u, lo, hi, tv, gv, Wv, h), lo, hi)
    Ps, res = anderson_solve(phi, extract_inner(halo, lo, hi), tol=1e-9, max_iter=4000, m=8)
    lin, scat = deficit(Ps, u[lo:hi], tau)
    return Ps, scat

if __name__ == "__main__":
    G = 16
    log("A) COLD-START lower gamma (tau=0.5, bandwidth-free cubic):")
    for gamma in (0.2, 0.15, 0.1):
        t = time.time(); lin, scat, res, _ = run_cubic(G, gamma, 0.5)
        log(f"   gamma={gamma:.2f}  lin={lin:.5f} scatter={scat:.5f} Jensen={lin-scat:.5f} res={res:.1e} ({time.time()-t:.0f}s)")
    log("B) WARM-START from curved (large-h) price, then bandwidth-free cubic (tau=0.5):")
    for gamma in (1.0, 0.3):
        t = time.time(); Pc, scatc = curved_init(G, gamma, 0.5)
        lin, scat, res, _ = run_cubic(G, gamma, 0.5, P0_inner=Pc)
        log(f"   gamma={gamma:.1f}  curved-init scatter={scatc:.4f} -> after cubic: scatter={scat:.5f} lin={lin:.5f} res={res:.1e} ({time.time()-t:.0f}s)")
    log("C) TAU sweep at gamma=0.3 (bandwidth-free cubic, cold-start):")
    for tau in (0.2, 0.5, 1.0, 2.0):
        t = time.time(); lin, scat, res, _ = run_cubic(G, 0.3, tau)
        log(f"   tau={tau:.1f}  lin={lin:.5f} scatter={scat:.5f} Jensen={lin-scat:.5f} res={res:.1e} ({time.time()-t:.0f}s)")
    OUT.close()
