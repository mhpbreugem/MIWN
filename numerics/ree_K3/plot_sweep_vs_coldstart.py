"""
plot_sweep_vs_coldstart.py — overlay ODE-continuation 1-R^2 against the existing
cold-start pool results (REE K=3, double-double).

Two slices through the anchor (gamma=4, tau=2):
  left   1-R^2 vs gamma at tau=2
  right  1-R^2 vs tau   at gamma=4

Cold-start = independent per-point solves (solutions/derived/ree_K3/coldstart_1mR2.json,
extracted from the done task queue). ODE = continuation from the gamma=4,tau=2 anchor
(sweep2d.json + test_runs.json + validate_g6.5_t2.json). All points satisfy ||F||<1e-20.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parents[2] / "solutions/derived/ree_K3"

cold = json.load(open(D / "coldstart_1mR2.json"))["points"]

# Gather ODE-continuation points: (gamma, tau, 1-R2)
ode = []
g = json.load(open(D / "sweep2d.json"))
for n in g["nodes"]:
    if n.get("one_minus_R2") is not None:
        ode.append((n["gamma"], n["tau"], n["one_minus_R2"]))
ode.append((4.0, 2.0, 0.08224460506371453))  # anchor (v0010)
for r in json.load(open(D / "test_runs.json"))["runs"]:
    ode.append((r["gamma"], r["tau"], r["one_minus_R2"]))
v = json.load(open(D / "validate_g6.5_t2.json"))
ode.append((v["gamma"], v["tau"], v["one_minus_R2"]))


def line(points, fix_idx, fix_val, var_idx):
    pts = [(p[var_idx], p[2]) for p in points if abs(p[fix_idx] - fix_val) < 1e-9]
    pts = sorted(set(pts))
    return [x for x, _ in pts], [y for _, y in pts]


def cline(points, fix_key, fix_val, var_key):
    pts = sorted({(p[var_key], p["one_minus_R2"]) for p in points if abs(p[fix_key] - fix_val) < 1e-9})
    return [x for x, _ in pts], [y for _, y in pts]


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

# ── left: 1-R^2 vs gamma at tau=2 ──
gx, gy = cline(cold, "tau", 2.0, "gamma")
ox, oy = line(ode, 1, 2.0, 0)
axL.plot(gx, gy, "o--", color="0.45", mfc="none", ms=6, label="cold-start (independent)")
axL.plot(ox, oy, "s-", color="C3", ms=6, label="ODE continuation (from γ=4)")
axL.axvline(4.0, color="0.8", lw=1, zorder=0)
axL.set_xscale("log"); axL.set_xlabel("γ  (risk aversion)"); axL.set_ylabel("1 − R²")
axL.set_title("τ = 2")
axL.legend(fontsize=9, frameon=False)

# ── right: 1-R^2 vs tau at gamma=4 ──
tx, ty = cline(cold, "gamma", 4.0, "tau")
otx, oty = line(ode, 0, 4.0, 1)
axR.plot(tx, ty, "o--", color="0.45", mfc="none", ms=6, label="cold-start (independent)")
axR.plot(otx, oty, "s-", color="C0", ms=6, label="ODE continuation (from τ=2)")
axR.axvline(2.0, color="0.8", lw=1, zorder=0)
axR.set_xscale("log"); axR.set_xlabel("τ  (signal precision)"); axR.set_ylabel("1 − R²")
axR.set_title("γ = 4")
axR.legend(fontsize=9, frameon=False)

fig.suptitle("ODE continuation vs cold-start  —  REE K=3, double-double (‖F‖<1e-20)", y=1.0)
fig.tight_layout()
out = D / "sweep_vs_coldstart.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
print("tau=2  gamma line  cold:", list(zip(gx, [round(y, 4) for y in gy])))
print("tau=2  gamma line  ode :", list(zip(ox, [round(y, 4) for y in oy])))
print("gamma=4 tau line   cold:", list(zip(tx, [round(y, 4) for y in ty])))
print("gamma=4 tau line   ode :", list(zip(otx, [round(y, 4) for y in oty])))
