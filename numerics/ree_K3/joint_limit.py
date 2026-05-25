"""Proper joint limit at the FPF point gamma=1,tau=4 (and tau=3): fix the
RELATIVE bandwidth h/du (so the kernel always spans the same number of grid
cells = well-resolved) and refine G.  As G grows, h=alpha*du -> 0.  If the
deficit -> 0, the no-noise REE is full revelation even at high tau (the fixed-G
h-shrink earlier was just under-resolution)."""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
sys.path.insert(0, "/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from code.halo import extract_inner, replace_inner
from ode_sweep import anderson_solve

OUT = open("/home/user/MIWN/numerics/ree_K3/joint_limit.txt", "w")
def log(s): OUT.write(s + "\n"); OUT.flush(); print(s, flush=True)

def deficit(Pin, ui, tau):
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij"); T = tau * (U1 + U2 + U3)
    f1 = np.exp(-tau / 2 * (ui - 0.5) ** 2); f0 = np.exp(-tau / 2 * (ui + 0.5) ** 2)
    F1 = f1[:, None, None] * f1[None, :, None] * f1[None, None, :]
    F0 = f0[:, None, None] * f0[None, :, None] * f0[None, None, :]
    W = 0.5 * (F1 + F0); m = (Pin > 1e-4) & (Pin < 1 - 1e-4)
    t = T[m]; L = np.log(Pin[m] / (1 - Pin[m])); w = W[m]; w = w / w.sum(); lm = np.average(L, weights=w)
    c1 = np.polyfit(t, L, 1, w=np.sqrt(w)); c5 = np.polyfit(t, L, 5, w=np.sqrt(w))
    return (np.average((L - np.polyval(c1, t)) ** 2, weights=w) / np.average((L - lm) ** 2, weights=w),
            np.average((L - np.polyval(c5, t)) ** 2, weights=w) / np.average((L - lm) ** 2, weights=w))

def solve(G, tau, alpha, gamma=1.0, pad=4, um=5.0):
    du = 2 * um / (G - 1); Gf = G + 2 * pad
    u = np.array([-um + (q - pad) * du for q in range(Gf)]); lo, hi = pad, pad + G
    tv, gv, Wv = np.full(3, tau), np.full(3, gamma), np.full(3, 1.0)
    halo = init_no_learning_K3(u, tv, gv, Wv); h = alpha * du
    def phi(x):
        return extract_inner(phi_K3_halo_smooth(replace_inner(halo, x, lo, hi), u, lo, hi, tv, gv, Wv, h), lo, hi)
    P, res = anderson_solve(phi, extract_inner(halo, lo, hi), tol=1e-10, max_iter=4000, m=8)
    lin, scat = deficit(P, u[lo:hi], tau)
    return lin, scat, res, du, h

if __name__ == "__main__":
    for tau in (4.0, 3.0):
        for alpha in (0.10, 0.20):
            log(f"\ngamma=1 tau={tau} fixed h/du={alpha}: refine G (h=alpha*du -> 0)")
            for G in (12, 16, 20, 24):
                t0 = time.time(); lin, scat, res, du, h = solve(G, tau, alpha)
                log(f"   G={G:2d} du={du:.3f} h={h:.4f}  1-R2={lin:.5f} scat={scat:.5f} res={res:.1e} ({time.time()-t0:.0f}s)")
    OUT.close()
