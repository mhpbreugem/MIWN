"""Confirm the stored FPF/REZN deficit solutions are KERNEL-bandwidth fixed
points, and that their deficit -> 0 as the bandwidth h -> 0.

(1) residual of each stored solution under the smooth KERNEL map at its h
    (= max(0.005,0.05*du))  -> should be small (it IS a kernel fixed point);
    residual of sigma(T*) under the kernel -> larger (kernel induces deficit).
(2) deficit(h) at the FPF point (gamma=1,tau=4): solve the kernel map for a
    ladder of shrinking h (warm-started), show 1-R2 -> 0 as h -> 0.
"""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
sys.path.insert(0, "/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_smooth
from code.halo import extract_inner, replace_inner
from ode_sweep import anderson_solve

CKDIR = "/home/user/FIXED-POINT-FACTORY/projects/REZN/checkpoints"
OUT = open("/home/user/MIWN/numerics/ree_K3/kernel_verify.txt", "w")
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

def res_kernel(P_full, u, lo, hi, tv, gv, Wv, h):
    Q = phi_K3_halo_smooth(P_full, u, lo, hi, tv, gv, Wv, h); inr = (slice(lo, hi),) * 3
    return float(np.max(np.abs(Q[inr] - P_full[inr])))

if __name__ == "__main__":
    log("(1) Is each stored solution a KERNEL fixed point?  (du, h=0.05*du)")
    for name in ("g100_t0400", "g100_t0300", "g025_t0300", "g050_t0300"):
        a = np.load(f"{CKDIR}/{name}.npz", allow_pickle=True)
        P_full = a["P_full"].astype(float); u = a["u_full"].astype(float)
        tv = a["tau_vec"].astype(float); gv = a["gamma_vec"].astype(float); Wv = a["W_vec"].astype(float)
        G = int(a["G_inner"]); pad = int(a["pad"]); lo, hi = pad, pad + G
        du = u[lo + 1] - u[lo]; h = max(0.005, 0.05 * du)
        U1, U2, U3 = np.meshgrid(u, u, u, indexing="ij"); Psig = 1 / (1 + np.exp(-tv[0] * (U1 + U2 + U3)))
        rs = {hh: res_kernel(P_full, u, lo, hi, tv, gv, Wv, hh) for hh in (h, 2*h, 0.5*h)}
        rsig = res_kernel(Psig, u, lo, hi, tv, gv, Wv, h)
        log(f"  {name} g={gv[0]} tau={tv[0]} du={du:.3f} h={h:.4f}: "
            f"res(stored)@h={rs[h]:.2e} @2h={rs[2*h]:.2e} @h/2={rs[0.5*h]:.2e} | res(sigmaT*)@h={rsig:.2e}")

    log("\n(2) deficit(h) at gamma=1,tau=4 (FPF point), kernel map, refine h->0:")
    gamma, tau, W = 1.0, 4.0, 1.0; G, pad, um = 14, 4, 5.0
    duG = 2 * um / (G - 1); Gf = G + 2 * pad
    u = np.array([-um + (q - pad) * duG for q in range(Gf)]); lo, hi = pad, pad + G
    tv, gv, Wv = np.full(3, tau), np.full(3, gamma), np.full(3, W)
    halo = init_no_learning_K3(u, tv, gv, Wv); P = extract_inner(halo, lo, hi)
    for h in (0.05*duG, 0.025*duG, 0.012*duG, 0.006*duG):
        def phi(x):
            return extract_inner(phi_K3_halo_smooth(replace_inner(halo, x, lo, hi), u, lo, hi, tv, gv, Wv, h), lo, hi)
        P, res = anderson_solve(phi, P, tol=1e-10, max_iter=3000, m=8)
        log(f"   h={h:.4f} (h/du={h/duG:.3f})  1-R2={deficit(P,u[lo:hi],tau):.5f}  res={res:.1e}")
    OUT.close()
