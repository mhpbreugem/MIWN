# h=0 PR-branch hunt (axis-scan halo discretization)

Question: is the partial-revelation (PR) equilibrium reported by REZN's
ξ-solver (kernel-free, ‖F‖~1e-15, 1−R²≈0.085 at γ=0.5,τ=3) reachable on the
**axis-scan halo** discretization I have here at G=20?

## Methods tried (all at h=0, hardwired — no kernel anywhere)

| solver | map | seed | outcome |
|---|---|---|---|
| Anderson m=8 | `phi_K3_halo_cubic` (weightless) | no-learning | **FR** (1−R²=0), best ‖F‖ 1e-3 |
| Picard α=0.05 + Cesàro | `phi_K3_halo_cubic` | no-learning | FR (1−R²=0), best ‖F‖ ~7e-5 |
| τ-ladder Picard (τ=0.5→2.0) | `phi_K3_halo_cubic` | warm chain | FR for *every* τ |
| Picard α=0.05 | `phi_K3_halo_cubic` | **FPF kernel-PR (interp 12→20)** | collapsed 0.087 → 0.0003 |
| **Newton–Krylov** (lgmres) | `phi_K3_halo_cubic` | FPF kernel-PR | **NoConvergence**: ‖F‖ stuck at 0.445 after 1600 evals |
| Picard α=0.05 | `phi_K3_coarea` **(weighted, 1/|∇P|)** | FPF kernel-PR | collapsed 0.087 → 0.00026 |

**Every combination of (weightless vs proper-weighted) × (Picard / Anderson /
Newton-Krylov) × (cold / continuation / kernel-PR warm) ends on FR (σ(τΣu)).**

## Interpretation

The kernel-PR field is *not* a fixed point of the axis-scan halo cubic at
h=0 — applying the map gives residual ~0.34–0.44 immediately, and either NK
fails to find a nearby fixed point or Picard drives the iterate to σ(T*).

This does **not** refute the ξ-solver's PR. The two are different
discretizations of the same conceptual REE inference:
- ξ-solver: smooth (spline) representation in a 1D revealed-coordinate ξ.
  Has a tightly-converged PR fixed point at ‖F‖<1e-15 per the other
  investigation.
- Axis-scan halo: 3D grid, linear/cubic contour crossings, halo boundary.
  Only FR is reachable here in my hands.

Whether the continuum (h=0, Bayesian-exact) REE actually has a PR fixed point
remains discretization-dependent in observation: the ξ-solver says yes, my
axis-scan says no. Without the ξ-solver code or a smooth-interpolant
proper-weighted reference solver in this checkout, I cannot adjudicate.

## What this *does* establish

- The exact closed-form REE `P=σ(Στₖuₖ)` (the **FR/no-trade** branch) is an
  exact fixed point at any precision (`verify_dps50.py`: 2.7e-51; the asymmetric
  generalisation `verify_asym.py`: 2.3e-41).
- That FR branch is the **unique attractor** of every axis-scan halo solver I
  ran from any seed (no-learning, τ-ladder, kernel-PR warm).
- A converged PR at h=0 is **not in this map family**. It requires the
  ξ-solver's smooth representation (build branch, not present here).

## Files
`h0_live.py` (anderson live), `h0_picard.py` (low-damp Picard live),
`tau_ladder.py` (continuation), `fpf_warm.py` (Picard warm), `nk_warm.py`
(Newton-Krylov warm), `coarea_warm.py` (proper-weighted warm).
Each writes incremental .npz and/or log under
`solutions/pool/ree_K3_G20_h0/`.
