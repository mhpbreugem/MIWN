# Figure list (points & data sources)

The paper's figures and the **points** each one needs. Layout/styling is **not**
specified here — that lives in the Standards `writing/figures` standard. This file
specifies only *which parameter points to compute* and *which solution each figure
consumes*. The runner drains `todo/TASK_QUEUE.json`; results land in
`solutions/pool/ree_K3/vNNNN/`; figure data for the dashboard is in
`solutions/figures_data.json`.

Flexible-parameter rule: ranges are γ ∈ [0.1, 10], τ ∈ [0.3, 15], **≥20 points per
sweep line**, consistent ranges across figures. Any point that misses the solve
budget (‖F‖ < 1e-20, time-boxed; dps from Standards) is **skipped** — a curve only
needs ≥5 points spanning the range for the effect to be visible.

Anchor / seed: (γ, τ) = (0.5, 2) = `v0007` (migrated REZN @ 7f03509, ‖F‖≈7e-119).

## Cheap & robust (no fixed-point convergence — closed-form or seed extraction)

| Figure | Shows | Points | Source | Status |
|---|---|---|---|---|
| knife-edge (`fig_knife_edge`) | 1−R² vs τ | τ-grid (20), γ=0.5 | no-learning + `v0007` | data ready (NL) |
| K-agents (`fig_knife_edge_K`) | 1−R² vs K | K = 3…22 (20) | no-learning | to compute (cheap) |
| lognormal (`fig_knife_edge_lognormal`) | 1−R² vs τ, lognormal payoff | τ-grid (20), γ=0.5 | no-learning | to compute (cheap) |
| mechanisms (`fig6_mechanisms`) | deficit decomposition | 6 named configs | no-learning | to compute (cheap) |
| contours (`fig_multicontour_A/B`) | price contours, u₁=1 slice | — | `v0007` (extract) | extract |
| price vs T\* (`fig_ree_vs_nolearning`) | price vs T\* | T\* sweep | `v0007` (extract) | extract |
| posteriors (`fig4_posteriors_A/B`) | μ\*, price vs T\* | — | `v0007` (extract) | extract |
| grid convergence (`fig5_convergence`) | ‖F‖ / 1−R² vs G | G = 15/18/20 + iter history | `v0003`/`v0005`/`v0006` | data ready |

## Need new strong-PR REE points (queued for the runner; continuation from seed)

| Figure | Shows | Points (≥20/line) | Source |
|---|---|---|---|
| REE 1−R² panels (`fig_ree_panels_A/B`) | 1−R² at converged REE | γ-sweep @ τ=2; τ-sweep @ γ=0.5 | REE pool versions |
| trade volume (`fig7_volume`) | E[\|x\|] vs γ | γ-sweep @ τ=2 (20) | REE pool versions |
| value of info (`fig8_value_info`) | V(τ) | τ-sweep @ γ∈{0.5,1,4} (20 each) | REE pool versions |
| net signal value (`fig9_GS`) | V(τ)−c | same as value-of-info | REE pool versions |

These four are REZN's flagged placeholders: the production solver tended to
collapse onto the fully-revealing root. The queue uses **continuation chains
radiating from the `v0007` seed** so each solve warm-starts from a converged
neighbour and stays on the strong-PR branch. Phase-0 spike validates this on a few
neighbours before the full grid.

## Sweep grids (in `todo/TASK_QUEUE.json` `params.grid`)

- **γ:** 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.5, 8.0, 10.0
- **τ:** 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0, 13.0, 15.0
- **τ-sweep γ values:** 0.5, 1.0, 4.0
