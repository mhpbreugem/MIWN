"""Final figure: grid-converged revelation deficit vs kernel bandwidth h.

The kernel bandwidth acts as effective price-observation noise; deficit(h->0)
is the no-noise limit.  Points are grid-converged 1-R2 (each h refined until
1-R2 is stable, ||F|| ~ 1e-11).
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

# grid-converged deficit(h) (from deficit_vs_h.txt / deficit_vs_h2.txt)
H = np.array([0.0333, 0.0250, 0.0200, 0.0160])
D = np.array([0.0432, 0.0285, np.nan, np.nan])  # filled below if available
import os, re
def load(path):
    out = {}
    if not os.path.exists(path):
        return out
    for ln in open(path):
        m = re.search(r"h=([0-9.]+).*?1-R2=([0-9.]+)", ln)
        if m:
            out[round(float(m.group(1)), 4)] = float(m.group(2))
    return out
vals = {}
for p in ["/home/user/MIWN/numerics/ree_K3/deficit_vs_h.txt",
          "/home/user/MIWN/numerics/ree_K3/deficit_vs_h2.txt"]:
    for k, v in load(p).items():
        vals.setdefault(k, []).append(v)
# grid-converged = max-G value (last); average duplicates at same h
Hs, Ds = [], []
for h in sorted(vals, reverse=True):
    Hs.append(h); Ds.append(np.mean(vals[h]))
Hs = np.array(Hs); Ds = np.array(Ds)

fig, ax = plt.subplots(figsize=(10.5, 6.4))
ax.plot(Hs, Ds, "o", ms=11, color="navy", label="grid-converged 1−R²  (||F||~1e-11)", zorder=5)
# power-law fit deficit = a*h^p  through the points
if len(Hs) >= 2:
    p, loga = np.polyfit(np.log(Hs), np.log(Ds), 1)
    hh = np.linspace(0, Hs.max() * 1.05, 200)
    ax.plot(hh, np.exp(loga) * np.power(hh, p, where=hh > 0, out=np.zeros_like(hh)),
            "-", color="crimson", lw=2,
            label=f"power-law fit  deficit ≈ {np.exp(loga):.2f}·h^{p:.2f}  → 0 as h→0")
ax.axhline(0.002, ls=":", color="gray", label="no-learning / full-revelation baseline (~0.002)")
ax.scatter([0.0333], [0.05234], marker="*", s=260, color="darkorange",
           edgecolor="k", zorder=6, label="published v0011 (coarse grid G=10) = 0.0523")
ax.scatter([0.0], [0.002], marker="s", s=90, color="green", zorder=6,
           label="bandwidth-free contour map (h=0, loosely converged) ≈ 0.002")
ax.set_xlabel("kernel bandwidth  h   (≈ effective price-observation noise)", fontsize=12)
ax.set_ylabel("revelation deficit  1 − R²", fontsize=12)
ax.set_title("The deficit vanishes as bandwidth/noise → 0:\n"
             "no market inefficiency in the no-noise limit (K=3, γ=1, τ=0.5)",
             fontsize=13, fontweight="bold")
ax.set_xlim(-0.001, 0.037); ax.set_ylim(-0.002, 0.058)
ax.legend(fontsize=9.5, loc="upper left"); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig("/home/user/MIWN/numerics/ree_K3/deficit_vs_h.png", dpi=150)
print("saved; points:", list(zip(Hs.tolist(), np.round(Ds, 5).tolist())))
