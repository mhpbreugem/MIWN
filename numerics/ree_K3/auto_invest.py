"""Auto-investigation: does the partial-revelation (deficit) branch survive when
the co-area Jacobian 1/|grad P| is restored?

For a chosen map (weightless cubic = paper's method, OR weighted co-area),
run a tau-ladder CONTINUATION at fixed gamma (warm-start each tau from the
previous), using damped Picard + Cesaro averaging (as REZN did).  Report 1-R2
and scatter vs tau.  Also a COLD solve at each tau, to expose multiplicity
(cold vs continued).

usage: auto_invest.py {weightless|weighted} {gamma}
"""
import sys, numpy as np, time
sys.path.insert(0, "/home/user/MIWN/standards/methods/solver")
sys.path.insert(0, "/home/user/MIWN/numerics/ree_K3")
from code.contour_K3_halo import init_no_learning_K3, phi_K3_halo_cubic
from code.halo import extract_inner, replace_inner
from coarea import phi_K3_coarea

MAP = sys.argv[1] if len(sys.argv) > 1 else "weightless"
GAMMA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
G, pad, um = 12, 4, 5.0
TAUS = [0.5, 0.75, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.5, 4.0]
NIT, ALPHA, CES = 260, 0.18, 130
OUT = open(f"/home/user/MIWN/numerics/ree_K3/auto_{MAP}_g{int(GAMMA*100):03d}.txt", "w")
def log(s): OUT.write(s + "\n"); OUT.flush(); print(s, flush=True)

du = 2 * um / (G - 1); Gf = G + 2 * pad
u = np.array([-um + (q - pad) * du for q in range(Gf)]); lo, hi = pad, pad + G
gv, Wv = np.full(3, GAMMA), np.full(3, 1.0)

def call(map_name, Pf, tv):
    if map_name == "weightless":
        return phi_K3_halo_cubic(Pf, u, lo, hi, tv, gv, Wv)
    return phi_K3_coarea(Pf, u, lo, hi, tv, gv, Wv)

def deficit(Pin, tau):
    ui = u[lo:hi]; U1, U2, U3 = np.meshgrid(ui, ui, ui, indexing="ij"); T = tau * (U1 + U2 + U3)
    f1 = np.exp(-tau / 2 * (ui - 0.5) ** 2); f0 = np.exp(-tau / 2 * (ui + 0.5) ** 2)
    F1 = f1[:, None, None] * f1[None, :, None] * f1[None, None, :]
    F0 = f0[:, None, None] * f0[None, :, None] * f0[None, None, :]
    W = 0.5 * (F1 + F0); m = (Pin > 1e-4) & (Pin < 1 - 1e-4)
    t = T[m]; L = np.log(Pin[m] / (1 - Pin[m])); w = W[m]; w = w / w.sum()
    lm = np.average(L, weights=w)
    c1 = np.polyfit(t, L, 1, w=np.sqrt(w)); c5 = np.polyfit(t, L, 5, w=np.sqrt(w))
    lin = np.average((L - np.polyval(c1, t)) ** 2, weights=w) / np.average((L - lm) ** 2, weights=w)
    scat = np.average((L - np.polyval(c5, t)) ** 2, weights=w) / np.average((L - lm) ** 2, weights=w)
    return lin, scat

def solve(tau, P0):
    tv = np.full(3, tau); halo = init_no_learning_K3(u, tv, gv, Wv)
    P = P0.copy(); acc = None; cnt = 0; best = (P.copy(), 1e9)
    for it in range(NIT):
        Q = extract_inner(call(MAP, replace_inner(halo, P, lo, hi), tv), lo, hi)
        F = float(np.max(np.abs(Q - P)))
        if F < best[1]: best = (P.copy(), F)
        P = (1 - ALPHA) * P + ALPHA * Q; P = np.clip(P, 1e-9, 1 - 1e-9)
        if it >= NIT - CES: acc = Q.copy() if acc is None else acc + Q; cnt += 1
    Pc = acc / cnt
    return Pc, best[1]

if __name__ == "__main__":
    log(f"MAP={MAP} gamma={GAMMA} G={G} um={um}  (continuation = warm tau-ladder; cold = no-learning)")
    log(f"{'tau':>5} | {'CONT 1-R2':>10} {'scat':>8} {'res':>9} | {'COLD 1-R2':>10} {'scat':>8} {'res':>9}")
    halo0 = init_no_learning_K3(u, np.full(3, TAUS[0]), gv, Wv)
    Pcont = extract_inner(halo0, lo, hi)
    for tau in TAUS:
        t0 = time.time()
        Pcont, rc = solve(tau, Pcont)           # warm from previous tau
        lin_c, scat_c = deficit(Pcont, tau)
        halo = init_no_learning_K3(u, np.full(3, tau), gv, Wv)
        Pcold, rk = solve(tau, extract_inner(halo, lo, hi))   # cold
        lin_k, scat_k = deficit(Pcold, tau)
        log(f"{tau:5.2f} | {lin_c:10.5f} {scat_c:8.5f} {rc:9.1e} | {lin_k:10.5f} {scat_k:8.5f} {rk:9.1e}  ({time.time()-t0:.0f}s)")
        np.save(f"/home/user/MIWN/numerics/ree_K3/auto_{MAP}_g{int(GAMMA*100):03d}_t{int(tau*100):04d}.npy", Pcont)
    OUT.close()
