"""Gamma sweep with live iter/||F|| reporting (~every 15s wall clock).
Bandwidth-free cubic co-area map; reports scatter (true deficit vs g(Sigma u))."""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
sys.path.insert(0, "/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_cubic
from code.halo import extract_inner, replace_inner
from ode_sweep import _anderson_step

tau, W = 0.5, 1.0
G = 16

def grid(Gi, pad=4, um=3.0):
    du = 2 * um / (Gi - 1); Gf = Gi + 2 * pad
    u = np.array([-um + (q - pad) * du for q in range(Gf)]); return u, pad, pad + Gi, du

def deficit(P, ui):
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

def solve_mon(gamma, tol=1e-9, max_iter=6000, m=6, report=15.0):
    u, lo, hi, du = grid(G); gv = np.full(3, gamma); tv = np.full(3, tau); Wv = np.full(3, W)
    halo = init_no_learning_K3(u, tv, gv, Wv)
    def phi_in(P):
        Pf = replace_inner(halo, P, lo, hi)
        return extract_inner(phi_K3_halo_cubic(Pf, u, lo, hi, tv, gv, Wv), lo, hi)
    shape = (G, G, G)
    P = extract_inner(halo, lo, hi).ravel()
    Fh, Ph = [], []; best = (P.copy(), 1e9); t0 = time.time(); last = 0.0; nimp = 0; lastmean = 1e9
    for it in range(max_iter):
        Phi = phi_in(P.reshape(shape)).ravel(); F = Phi - P; res = float(np.max(np.abs(F)))
        if res < best[1]:
            best = (P.copy(), res)
        if res < lastmean * 0.999:          # meaningful (>=0.1%) improvement
            lastmean = res; nimp = 0
        else:
            nimp += 1
        el = time.time() - t0
        if el - last >= report or res < tol or it == 0:
            print(f"[g={gamma}] iter={it:4d} ||F||={res:.3e} best={best[1]:.3e} t={el:5.0f}s", flush=True); last = el
        if res < tol:
            break
        if best[1] < 2e-3 and nimp > 250:
            print(f"[g={gamma}] plateau-stop iter={it} best||F||={best[1]:.3e}", flush=True); break
        Fh.append(F.copy()); Ph.append(P.copy()); P = _anderson_step(Fh, Ph, m); P = np.clip(P, 1e-9, 1 - 1e-9)
    Ps = best[0].reshape(shape); lin, scat = deficit(Ps, u[lo:hi])
    print(f"[g={gamma}] DONE 1-R2_lin={lin:.5f} scatter={scat:.5f} Jensen={lin-scat:.5f} res={best[1]:.2e}", flush=True)
    return Ps, u[lo:hi]

if __name__ == "__main__":
    slices = {}
    for gamma in (0.3, 0.5, 1.0, 2.0):
        Ps, ui = solve_mon(gamma); slices[f"g{gamma}"] = Ps[:, :, G // 2]; slices["ui"] = ui
    np.savez("/home/user/MIWN/numerics/ree_K3/gamma_slices.npz", **slices)
    print("ALLDONE", flush=True)
