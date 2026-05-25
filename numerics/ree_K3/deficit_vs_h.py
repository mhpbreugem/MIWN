"""Grid-converged deficit(h): for each FIXED bandwidth h, refine the grid until
1-R2 stops moving, giving the true smooth-h fixed-point deficit.  Extrapolating
deficit(h) as h->0 gives the physical (bandwidth-free, no-noise) deficit.
"""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from code.halo import extract_inner, replace_inner
from code.metrics import revelation_deficit
from ode_sweep import anderson_solve

tau, gamma, W = 0.5, 1.0, 1.0
tv, gv, Wv = np.full(3, tau), np.full(3, gamma), np.full(3, W)
OUT = "/home/user/MIWN/numerics/ree_K3/deficit_vs_h.txt"
fh = open(OUT, "w")
def log(s): fh.write(s + "\n"); fh.flush(); print(s, flush=True)

def solve(G, h, pad=4, um=3.0, tol=1e-11, mi=5000):
    du = 2.0 * um / (G - 1); Gf = G + 2 * pad
    u = np.array([-um + (q - pad) * du for q in range(Gf)]); lo, hi = pad, pad + G
    halo = init_no_learning_K3(u, tv, gv, Wv)
    def phi(P):
        Pf = replace_inner(halo, P, lo, hi)
        return extract_inner(phi_K3_halo_smooth(Pf, u, lo, hi, tv, gv, Wv, h), lo, hi)
    Ps, res = anderson_solve(phi, extract_inner(halo, lo, hi), tol=tol, max_iter=mi, m=8)
    return revelation_deficit(Ps, u[lo:hi], tv, 3), res, du

if __name__ == "__main__":
    log("deficit(h) grid-converged: each h refined over two grids (h/du ~ 0.08-0.12)")
    # (h, [G grids to confirm convergence])
    plan = [(0.0333, [18, 22]), (0.025, [24, 28]), (0.020, [26, 30]), (0.016, [32, 36])]
    for h, Gs in plan:
        for G in Gs:
            t = time.time(); r2, res, du = solve(G, h)
            log(f"   h={h:.4f} G={G:2d} h/du={h/du:.3f}  1-R2={r2:.6f} res={res:.1e} ({time.time()-t:.0f}s)")
    fh.close()
