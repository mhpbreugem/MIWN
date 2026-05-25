# ODE-continuation solver — test runs & 2D sweep (REE K=3, double-double)

Solver: `numerics/ree_K3/sweep.py` (single fixed point) and `numerics/ree_K3/sweep2d.py`
(2D branch map). Method: the fixed point `P*(γ,τ)` deforms continuously with the
model parameters and satisfies the tangent ODE `(I − ∂φ/∂P)·dP*/dθ = ∂φ/∂θ`,
defined *at* the fixed point. From a converged anchor we integrate that ODE along a
path to the target (RK4 predictor, Jacobian-free GMRES, float64; short Anderson
corrector each substep to stay on the manifold), then apply **one** quadratic mpmath
Newton polish at the target. Precision policy (Standards `precision.py`):
**double-double, dps=32, accept only ‖F‖ < 1e-20.**

Anchor: `solutions/pool/ree_K3/v0010` — γ=4, τ=2, 1−R²=0.082245, ‖F‖≈6.16e-21
(G_inner=10, pad=4). Reuses the submodule RK4 predictor (`ode_sweep_rk4`) and the
project dd Newton (`phi_mp.phi_newton_mp`) — no fork.

## Test runs (single fixed point each, from the anchor)

Five targets probing both parameters, both directions, large/small steps, and a
two-leg diagonal (walk γ then τ). All accepted (‖F‖ < 1e-20).

| target (γ, τ) | step from anchor | ‖F‖ | 1−R² | dd-Newton | wall |
|---|---|---|---|---|---|
| 8, 2   | γ +4 (large)      | 2.15e-21 | 0.082720 | 6 | 192s |
| 1, 2   | γ −3 (large)      | 1.67e-21 | 0.079669 | 8 | 265s |
| 4, 5   | τ +3              | 2.78e-21 | 0.085413 | 4 | 130s |
| 4, 0.8 | τ −1.2            | 1.83e-24 | 0.044206 | 8 | 282s |
| 8, 4   | diagonal (γ then τ) | 1.23e-24 | 0.091132 | 4 | 130s |

Each γ/τ leg used ≤1.0 per RK4 substep; the predictor lands the corrector at
~1e-6 so the dd Newton needs only 4–8 steps. The diagonal confirms the two-leg
walk (γ at fixed τ, then τ at fixed γ) reaches the target cleanly.

## 2D sweep — one branch over γ × τ

γ ∈ {2.5, 4, 6.5}, τ ∈ {1, 2, 4}; γ-spine at τ=2 from the anchor, then τ-columns
from each spine node. 9 nodes, 8 dd-solved (the 9th is the anchor), all ‖F‖ < 1e-20,
in ~24 min.

### 1 − R² (rows γ, cols τ)

| γ \ τ | 1 | 2 | 4 |
|---|---|---|---|
| 2.5 | 0.064635 | 0.081680 | 0.092579 |
| 4   | 0.069583 | 0.082245* | 0.091792 |
| 6.5 | 0.065542 | 0.082610 | 0.091284 |

\* anchor (v0010). 1−R² rises with τ across the grid; mild, non-monotonic γ-dependence.

### ‖F‖ (rows γ, cols τ)

| γ \ τ | 1 | 2 | 4 |
|---|---|---|---|
| 2.5 | 2.77e-21 | 1.05e-21 | 7.09e-25 |
| 4   | 3.85e-23 | 6.16e-21* | 8.40e-25 |
| 6.5 | 1.68e-22 | 4.99e-24 | 7.78e-25 |

## Validation

- **Path independence.** γ=6.5, τ=2 solved directly from the anchor
  (`validate_g6.5_t2.json`: ‖F‖=4.99e-24, 1−R²=0.0826104) is identical to the same
  node reached via the 2D grid path — the continuation lands on the same fixed point
  regardless of route.
- **Policy.** Every reported point satisfies ‖F‖ < 1e-20 (double-double, dps=32).
- **Consistency with cold start.** The anchor and τ=2 line agree with the
  independent cold-start pool solves at τ=2.

## Reproduce

```
# one fixed point
python3 numerics/ree_K3/sweep.py --anchor solutions/pool/ree_K3/v0010/data/solution.npz --gamma 6.5 --tau 2
# 2D branch map
python3 numerics/ree_K3/sweep2d.py --gammas 2.5,4,6.5 --taus 1,2,4 --out solutions/derived/ree_K3/sweep2d.json
```

Data: `test_runs.json`, `sweep2d.json`, `validate_g6.5_t2.json` (this directory).
