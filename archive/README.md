# archive — obsolete material

Kept for reference only. Not part of the active pipeline.

## `ree_K3_solve_symmetric.py`
The original `numerics/ree_K3/solve.py`, which drove the **symmetric float64**
solver (`standards/methods/solver/contour_KN_sym`). Retired because:
- it **collapsed to the fully-revealing branch** (1−R²≈1e-4 at (γ,τ)=(0.5,2),
  vs the strong-PR value ≈0.1), and
- float64 bottoms out near 1e-15, so it **cannot meet the Standards precision
  policy** (double-double, ‖F‖ < 1e-20; see `standards/methods/PRECISION_POLICY.md`).

Superseded by the current `numerics/ree_K3/solve.py`, which uses the shared
**halo + mpmath-Newton** method at double-double precision and converges to the
strong-PR branch (verified: ‖F‖≈1.9e-23, 1−R²≈0.106 at (0.5,2)).

## `v0002_symmetric_demo/`
The symmetric solver's fully-revealing-branch demo solution (G=5). Obsolete for
the same reason; moved out of `solutions/pool/` so the pool holds only
double-double / policy-compliant results.
