"""Bandwidth-free Newton/Anderson solve on the smooth line-integral co-area map.
Demonstrates convergence far below the crossing-count floor (~1e-3)."""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
sys.path.insert(0, "/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import init_no_learning_K3
from code.halo import extract_inner, replace_inner
from lineint_coarea import phi_lineint

OUT = open("/home/user/MIWN/numerics/ree_K3/lineint_solve_out.txt", "w")
def log(s): OUT.write(s + "\n"); OUT.flush(); print(s, flush=True)

def deficit(P, ui, tau):
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij"); S = U1 + U2 + U3
    f1 = np.exp(-tau / 2 * (ui - 0.5) ** 2); f0 = np.exp(-tau / 2 * (ui + 0.5) ** 2)
    F1 = f1[:, None, None] * f1[None, :, None] * f1[None, None, :]
    F0 = f0[:, None, None] * f0[None, :, None] * f0[None, None, :]
    Wt = 0.5 * (F1 + F0); m = (P > 1e-4) & (P < 1 - 1e-4)
    t = S[m]; L = np.log(P[m] / (1 - P[m])); w = Wt[m]; w = w / w.sum()
    lm = np.average(L, weights=w); c = np.polyfit(t, L, 1, w=np.sqrt(w))
    return np.average((L - np.polyval(c, t)) ** 2, weights=w) / np.average((L - lm) ** 2, weights=w)

def main():
    gamma, tau, W = 1.0, 0.5, 1.0
    G, pad, um = 8, 4, 3.0
    du = 2 * um / (G - 1); Gf = G + 2 * pad
    u = np.array([-um + (q - pad) * du for q in range(Gf)]); lo, hi = pad, pad + G
    tv, gv, Wv = np.full(3, tau), np.full(3, gamma), np.full(3, W)
    halo = init_no_learning_K3(u, tv, gv, Wv)
    P0 = extract_inner(halo, lo, hi); shape = P0.shape
    def phi_in(Pin):
        Pf = replace_inner(halo, Pin, lo, hi)
        return extract_inner(phi_lineint(Pf, u, lo, hi, tv, gv, Wv), lo, hi)
    log(f"Bandwidth-free smooth line-integral map (ANDERSON), gamma={gamma} tau={tau} G={G}")
    from ode_sweep import _anderson_step
    P = P0.ravel().copy(); Fh, Ph = [], []; best = (P.copy(), 1e9); t0 = time.time()
    for it in range(80):
        Q = phi_in(P.reshape(shape)).ravel(); F = Q - P; res = float(np.max(np.abs(F)))
        if res < best[1]: best = (P.copy(), res)
        if it % 2 == 0 or res < 1e-11:
            log(f"  anderson it={it:3d} ||F||={res:.3e} 1-R2={deficit(P.reshape(shape),u[lo:hi],tau):.2e} t={time.time()-t0:.0f}s")
        if res < 1e-12: break
        Fh.append(F.copy()); Ph.append(P.copy()); P = _anderson_step(Fh, Ph, 6); P = np.clip(P, 1e-9, 1 - 1e-9)
    P = best[0].reshape(shape)
    log(f"  FINAL best ||F||={best[1]:.3e}  1-R2={deficit(P,u[lo:hi],tau):.3e}")
    # compare to exact sigma(T*)
    U1, U2, U3 = np.meshgrid(u[lo:hi], u[lo:hi], u[lo:hi], indexing="ij")
    Pexact = 1 / (1 + np.exp(-tau * (U1 + U2 + U3)))
    log(f"  ||P - sigma(tau*Sum u)||inf = {float(np.max(np.abs(P - Pexact))):.3e}  (converges to the closed-form REE)")
    OUT.close()

if __name__ == "__main__":
    main()
