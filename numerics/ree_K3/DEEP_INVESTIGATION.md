# Deep investigation: is the K=3 homogeneous-CRRA REE inefficient without noise?

Autonomous investigation of whether the reported "revelation deficit"
(1-R^2 of logit price on T*=tau*sum u) is a genuine no-noise market
inefficiency or a numerical artifact. Scripts in this directory; results in the
`*_results.txt` / `*.txt` files; figures `verdict.png`, `deficit_vs_h.png`,
`explainer.png`, `lineint_smooth.png`, `jensen_contours.png`.

## TL;DR

**The no-noise REE is exactly fully revealing: P = sigma(tau * sum u_k), with
1-R^2 = 0, for every (gamma, tau).** Proven analytically and verified to 50
digits. Every stored "deficit" solution across MIWN, REZN and
fixed-point-factory (1-R^2 ~ 0.05-0.26) is a fixed point of the **smooth-kernel**
map at bandwidth h = 0.05*du; the kernel bandwidth acts as price-observation
noise. Remove the kernel and the deficit collapses to ~0.

## The exact result

With zero net supply the market clears where excess demand is zero. If the price
is fully revealing, every agent infers T* = tau*sum u, forms the **same**
posterior sigma(T*), so there is no trade and the price equals that common
belief: P = sigma(T*). Then logit P = T* exactly => 1-R^2 = 0.

The contour Bayesian update reproduces this: the likelihood ratio factorises,
  f1(u_k)/f0(u_k) * phi1(sum_{-k})/phi0(sum_{-k}) = exp(tau u_k)*exp(tau sum_{-k})
  = exp(tau sum u) = exp(T*),
so every agent's posterior is sigma(T*) regardless of gamma.

`verify_dps50.py` (mpmath dps=50, exact quadrature, no grid/kernel): for
(u1,u2,u3) across tau in {0.5,2,11} and gamma in {0.3,1,4},
  |posterior - price|      <= 2.7e-51
  |clearing - sigma(T*)|   <= 2.7e-51
  logit(sigma(tau*S)) - T* == 0.

So 1-R^2 = 0 is structural, not a precision artifact (float64 gives 2e-16,
dps=50 gives 2.7e-51 -- the precision floor tracks, never revealing a deficit).

## What the stored "deficit" solutions actually are

Inventory of positive-1-R^2 solutions found:
- **MIWN** pool v0008-v0011: 1-R^2 = 0.05-0.08 (G=10, dps=32).
- **REZN** results/full_ree/fig4B (G=20): 1-R^2 = 0.016-0.257 across gamma.
- **fixed-point-factory** checkpoints (the paper's data, G=21): 1-R^2 =
  0.095-0.109 at F_max ~ 1e-51.

`kernel_verify.py` (Part 1): each FPF solution has residual **3.6e-15 under the
smooth-kernel map at h = 0.05*du**, but 0.05-0.35 at 2h or h/2 -- they are
kernel fixed points tied to that exact bandwidth. sigma(T*) has residual
0.03-0.04 under the kernel (the kernel pushes the price off full revelation).

`checkpoint_test.py`:
- LOOP A: stored solutions have residual ~0.2-0.37 under the **bandwidth-free**
  contour maps; sigma(T*) has residual 1e-12 -- 1e-15 under both. So sigma(T*)
  is the bandwidth-free fixed point; the stored deficits are not.
- LOOP B: iterating the bandwidth-free map from each stored solution collapses
  the deficit:
    g100_t0400  0.0956 -> 0.0031
    g100_t0300  0.0967 -> 0.00001
    g025_t0300  0.1028 -> 0.0004
    g050_t0300  0.1086 -> 0.0004
  (both the weighted 1/|grad P| and the weightless co-area collapse it.)

## The kernel bandwidth is price-observation noise

The smooth map replaces the exact conditional density delta(P-p) by a Gaussian
kernel of width h in price units: the agent cannot resolve the price finer than
h, i.e. observes a band of width h == a price corrupted by noise ~h. Grid-
converged deficit(h) (`deficit_vs_h.py`, gamma=1, tau=0.5): 0.043 (h=0.033),
0.0285 (h=0.025), 0.0197 (h=0.020) ~ h^1.5 -> 0. So deficit -> 0 as the noise
-> 0; the published 0.052 is the value at the finite numerical bandwidth.

## The metric also over-counts (Jensen gap)

1-R^2 uses a *linear* fit of logit P on T*. Decomposed (scatter vs flexible
g(T*)), a chunk of the finite-h deficit is curvature (a fully-revealing but
nonlinear price), not informational scatter -- e.g. at gamma=1, tau=0.5,
h=0.033: 0.043 = 0.026 scatter + 0.017 Jensen. At the no-noise REE both are 0.

## Methods built (paper-side; back-port candidates)

- `dd_numba.py` numba double-double arithmetic (validated vs mpmath ~1e-28).
- `dd_phi.py` numba-DD smooth Phi (fast policy-grade precision).
- `coarea.py` bandwidth-free co-area with the 1/|grad P| weight the vendored
  maps drop.
- `marching_coarea.py` segment-based co-area (smooth in principle).
- `lineint_coarea.py` direct contour line integral with a C2 spline per scan
  line: sigma(T*) is a fixed point to 2e-16 (gamma=1,tau=0.5).

## Caveats / open edges

- The bandwidth-free contour maps are non-smooth in max-norm (worst cell floors
  ~1e-3 at moderate G; REZN's F_max was 7e-9 at G=20), so ||F|| < 1e-20 is only
  reachable with the (kernel-smoothed) map or a fully C2-spline co-area. This is
  why the production pipeline used the kernel -- and why its result carries the
  bandwidth.
- sigma(T*) is proven to be *an* exact REE for all (gamma,tau); uniqueness is
  not proven analytically, but iterating the bandwidth-free map from the paper's
  deficit solutions collapses them to it (no stable partial-revelation branch
  found at h=0).
- Conclusion is for K=3 homogeneous CRRA, zero net supply, Gaussian binary-state
  signals.
