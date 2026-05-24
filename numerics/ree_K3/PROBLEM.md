# Problem: ree_K3

Symmetric K=3 rational-expectations equilibrium (REE) in a homogeneous-CRRA
market with private signals.

## Model

- `K = 3` agents, identical CRRA risk aversion `γ` and signal precision `τ`,
  endowment `W`.
- A binary fundamental `v ∈ {0,1}`. Each agent `k` observes a private Gaussian
  signal `u_k` with precision `τ`, mean `±0.5` depending on `v`.
- Agents form Bayesian posteriors and trade a CRRA-optimal demand; the market
  clears at price `P(u_1,…,u_K)`.
- Homogeneity ⇒ `P` is permutation-invariant, so it is stored on the manifold of
  **sorted index tuples** (`C(G+K-1, K)` cells on a `G`-point grid).

## Fixed point

Solve `P = Φ(P)`, where `Φ` performs, per state: a contour-integral Bayesian
update for each agent (conditioning on the equilibrium price) followed by CRRA
market clearing. The shared map is `sym_phi` in
`standards/methods/solver/contour_KN_sym.py`.

We solve `F(P) = Φ(P) − P = 0` with Anderson acceleration (float64), tracking the
best iterate. (The mpmath-Newton high-precision polish in
`standards/methods/solver/solve.py` / `phi_mp.py` is the production path for tight
`||F||`; the local driver here runs the float64 fixed point to confirm wiring.)

## Metric

Price informativeness as `1 − R²` of `logit(P)` regressed on the sufficient
statistic `T* = τ · Σ_k u_k`, weighted by the ex-ante state density
`0.5·(Πf₀ + Πf₁)` with permutation multiplicity (`sym_weighted_R2`). As the market
approaches full revelation, `1 − R² → 0`; the "knife-edge" is the regime where it
turns sharply.

## Provenance

Model, parameters, and the numerical core follow REZN @ `7f03509`. The core
(`contour_K3_halo`, `halo`, `staggered`, `metrics`, …) is vendored in
`standards/methods/solver/code/`; the symmetric-K driver is
`standards/methods/solver/contour_KN_sym.py`. Do not copy that code here — import it
from the submodule.

## Grid & runner

`spec.json` defines the (γ, τ) grid and solver settings. `todo/TASK_QUEUE.json` is
generated from it (one task per (γ, τ)); the Standards runner claims tasks and
writes solutions to `solutions/pool/ree_K3/vNNNN/`.
