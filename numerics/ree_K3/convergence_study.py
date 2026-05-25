"""Decisive convergence study: is the revelation deficit a grid/bandwidth
artifact, or a real equilibrium property?

TEST 1  fixed physical bandwidth h, refine grid G -> inf:
        if 1-R2 converges to a positive constant, the deficit at THAT h is a
        real (grid-converged) fixed-point property, not a discretisation error.
TEST 2  alpha = h/du fixed (bandwidth shrinks with grid) -> the h->0 limit:
        if 1-R2 -> 0 the bandwidth-free (physical, no-noise) deficit vanishes.
"""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from code.halo import extract_inner, replace_inner
from code.metrics import revelation_deficit
from ode_sweep import anderson_solve

tau, gamma, W = 0.5, 1.0, 1.0
tv, gv, Wv = np.full(3, tau), np.full(3, gamma), np.full(3, W)
OUT = "/home/user/MIWN/numerics/ree_K3/conv_results.txt"
fh = open(OUT, "w")
def log(s):
    fh.write(s + "\n"); fh.flush(); print(s, flush=True)

def grid(G, pad=4, um=3.0):
    du = 2.0 * um / (G - 1); Gf = G + 2 * pad
    u = np.array([-um + (q - pad) * du for q in range(Gf)])
    return u, pad, pad + G, du

def solve(G, h, tol=1e-11, mi=4000):
    u, lo, hi, du = grid(G)
    halo = init_no_learning_K3(u, tv, gv, Wv)
    def phi(P):
        Pf = replace_inner(halo, P, lo, hi)
        return extract_inner(phi_K3_halo_smooth(Pf, u, lo, hi, tv, gv, Wv, h), lo, hi)
    Ps, res = anderson_solve(phi, extract_inner(halo, lo, hi), tol=tol, max_iter=mi, m=8)
    return revelation_deficit(Ps, u[lo:hi], tv, 3), res, du

if __name__ == "__main__":
    log("TEST 1 - fixed physical bandwidth h=0.0333, refine grid:")
    for G in (10, 14, 18, 22, 26):
        t = time.time(); r2, res, du = solve(G, 0.0333)
        log(f"   G={G:2d} du={du:.4f} h/du={0.0333/du:.3f}  1-R2={r2:.6f} res={res:.1e} ({time.time()-t:.0f}s)")
    log("TEST 2 - alpha=h/du=0.05 fixed, refine grid (h->0):")
    for G in (10, 14, 18, 22, 26):
        du0 = 6.0 / (G - 1)
        t = time.time(); r2, res, du = solve(G, 0.05 * du0)
        log(f"   G={G:2d} du={du:.4f} h={0.05*du0:.4f}  1-R2={r2:.6f} res={res:.1e} ({time.time()-t:.0f}s)")
    fh.close()
