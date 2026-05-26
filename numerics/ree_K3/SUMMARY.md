# Summary: is the K=3 CRRA market inefficient without noise?

## Question
The paper reports a "revelation deficit" `1-R^2` (of `logit P` on the sufficient
statistic) ~0.05-0.26 in a no-noise homogeneous-CRRA REE. Goal: pin that number
and confirm it is a real no-noise inefficiency rather than a numerical artifact.

## Headline
**The no-noise REE is exactly fully revealing: `P = sigma(sum_k tau_k u_k)`, with
`1-R^2 = 0`.** Proven analytically, verified to 40-50 digits, and shown to hold
for **arbitrary heterogeneous (gamma_k, tau_k)**. Every stored "deficit" solution
across MIWN / REZN / fixed-point-factory is a fixed point of the **smooth-kernel**
map; the kernel bandwidth acts as price-observation noise. Remove it and the
deficit collapses to ~0.

## Evidence chain
1. **Exact REE.** With the price contour `{P=p}` equal to the iso-likelihood-ratio
   set, every agent's contour-Bayes posterior equals the price; at that (fair)
   price every risk-averse agent holds zero, so the market clears with no trade at
   `P=sigma(T*)`. `logit P = T*` exactly => `1-R^2 = 0`. `verify_dps50.py`:
   `|posterior-price|, |clearing-sigma(T*)| <= 2.7e-51`.
2. **Stored deficits are kernel fixed points.** `kernel_verify.py`: each FPF G=21
   solution has residual `3.6e-15` under the kernel at `h=0.05*du`, but `0.05-0.35`
   at `2h` or `h/2`. `sigma(T*)` is NOT a kernel fixed point (res ~0.03).
3. **Remove the kernel -> collapse.** `checkpoint_test.py` (iterate the
   bandwidth-free map from each paper solution): `0.0956->0.0031`,
   `0.0967->1e-5`, `0.1028->4e-4`, `0.1086->4e-4`. `sigma(T*)` is the
   bandwidth-free fixed point (res `1e-12`-`1e-15`).
4. **Kernel = noise; deficit(h) -> 0.** Grid-converged `deficit(h)` (gamma=1,
   tau=0.5): `0.043, 0.0285, 0.0197` at `h=0.033, 0.025, 0.020` ~ `h^1.5 -> 0`.
   Joint limit at tau=4 (fixed `h/du`, refine G): `0.114->0.105->0.101` -> 0 (slow
   at high tau; exactly 0 at `h=0` by the proof).

## Why the published ~0.05 appears: three stacked artifacts
- **Bandwidth (noise):** the smooth kernel regularises `delta(P-p)` with width `h`
  in price units = the agent sees a price band of width `h` = noise; `h=0.05*du`.
- **Coarse grid:** at fixed `h` the deficit is grid-sensitive (v0011's 0.052 at
  G=10 is 0.043 grid-converged).
- **Jensen / metric:** `1-R^2` is a *linear* fit; part of the finite-h value is
  curvature of a fully-revealing-but-nonlinear price, not informational scatter
  (gamma=1,tau=0.5,h=0.033: 0.043 = 0.026 scatter + 0.017 Jensen).

## Asymmetric agents
`verify_asym.py` (dps=40, gamma=[0.3,1,4], tau=[0.5,2,5]): `P=sigma(sum tau_k u_k)`
is still an exact REE -- posteriors equal the price (`2e-41`), zero excess demand
(`6e-38`, no trade). Heterogeneity only makes the sufficient statistic
**precision-weighted** (`sum tau_k u_k`). Measuring against the wrong `T*` (e.g.
`tau*sum u`) manufactures a spurious deficit -- a fourth, metric-misspecification
artifact.

## Grossman nuance
`sigma(T*)` is a **no-trade** equilibrium; the paper's solutions have trade
(TV=0.625) and partial revelation, which exist only at `h>0`. Classic result:
without noise the competitive REE is fully revealing and trade-free; a deficit
needs a friction (the kernel here). Genuine inefficiency would require
heterogeneous **priors**, **aggregate risk / nonzero net supply** (-> a
Jensen-curved but still-revealing price), or real **noise** -- not (gamma,tau)
heterogeneity.

## Caveats
- `sigma(T*)` proven to be *an* exact REE; analytic uniqueness not proven, but the
  bandwidth-free map collapses the paper's solutions onto it (no stable
  partial-revelation branch found at `h=0`).
- Whether the no-trade FR or a with-trade equilibrium is the economically intended
  object is interpretive; numerically the deficit is carried by `h>0`.
- Results are for K=3, common-value, common-prior, riskless homogeneous endowment,
  zero net supply.

## Artifacts (branch claude/repo-reorganization-gVq7w, numerics/ree_K3/)
- Report: `DEEP_INVESTIGATION.md`, this `SUMMARY.md`. Figures: `verdict.png`,
  `deficit_vs_h.png`, `explainer.png`, `jensen_contours.png`, `lineint_smooth.png`,
  `gamma_contours.png`.
- Proofs/harnesses: `verify_dps50.py`, `verify_asym.py`, `kernel_verify.py`,
  `checkpoint_test.py`, `deficit_vs_h.py`, `joint_limit.py`, `convergence_study.py`.
- Bandwidth-free / precision solvers: `coarea.py`, `marching_coarea.py`,
  `lineint_coarea.py`, `dd_numba.py`, `dd_phi.py`.
