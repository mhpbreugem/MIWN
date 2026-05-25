"""deficit(h) -> 0 across the FPF parameter points (kernel map, refine h)."""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
sys.path.insert(0, "/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from code.halo import extract_inner, replace_inner
from ode_sweep import anderson_solve

OUT = open("/home/user/MIWN/numerics/ree_K3/deficit_h_fpf.txt", "w")
def log(s): OUT.write(s + "\n"); OUT.flush(); print(s, flush=True)

def deficit(Pin, ui, tau):
    U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij"); T = tau * (U1 + U2 + U3)
    f1 = np.exp(-tau / 2 * (ui - 0.5) ** 2); f0 = np.exp(-tau / 2 * (ui + 0.5) ** 2)
    F1 = f1[:, None, None] * f1[None, :, None] * f1[None, None, :]
    F0 = f0[:, None, None] * f0[None, :, None] * f0[None, None, :]
    W = 0.5 * (F1 + F0); m = (Pin > 1e-4) & (Pin < 1 - 1e-4)
    t = T[m]; L = np.log(Pin[m] / (1 - Pin[m])); w = W[m]; w = w / w.sum(); lm = np.average(L, weights=w)
    c = np.polyfit(t, L, 1, w=np.sqrt(w))
    return np.average((L - np.polyval(c, t)) ** 2, weights=w) / np.average((L - lm) ** 2, weights=w)

if __name__ == "__main__":
    G, pad, um = 14, 4, 5.0; W = 1.0
    duG = 2 * um / (G - 1); Gf = G + 2 * pad
    u = np.array([-um + (q - pad) * duG for q in range(Gf)]); lo, hi = pad, pad + G
    log(f"kernel deficit(h) refine, G={G} um={um}  (FPF stored values for reference)")
    refs = {(0.25, 3.0): 0.1028, (0.5, 3.0): 0.1086, (1.0, 3.0): 0.0967}
    for (gamma, tau), ref in refs.items():
        tv, gv, Wv = np.full(3, tau), np.full(3, gamma), np.full(3, W)
        halo = init_no_learning_K3(u, tv, gv, Wv); P = extract_inner(halo, lo, hi)
        log(f"\n gamma={gamma} tau={tau} (FPF G=21 stored 1-R2={ref}):")
        for h in (0.05*duG, 0.025*duG, 0.012*duG, 0.006*duG):
            def phi(x, h=h):
                return extract_inner(phi_K3_halo_smooth(replace_inner(halo, x, lo, hi), u, lo, hi, tv, gv, Wv, h), lo, hi)
            t0 = time.time(); P, res = anderson_solve(phi, P, tol=1e-10, max_iter=3000, m=8)
            log(f"   h={h:.4f} (h/du={h/duG:.3f})  1-R2={deficit(P,u[lo:hi],tau):.5f}  res={res:.1e} ({time.time()-t0:.0f}s)")
    OUT.close()
