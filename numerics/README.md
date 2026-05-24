# numerics

Per-problem definitions and thin local drivers. Each problem is a folder:

```
numerics/<problem>/
├── PROBLEM.md   what the fixed point is, the model, the metric
├── spec.json    parameter grid (γ/τ) + solver settings — drives todo/TASK_QUEUE.json
└── solve.py     thin local driver: imports shared methods, runs ONE solve, writes a pool version
```

`solve.py` is deliberately thin: it puts `standards/methods/solver` on `sys.path`,
imports the shared K-symmetric pieces, supplies this paper's Φ/params, runs a
single local solve, and writes a `solutions/pool/<problem>/vNNNN/` version with a
`meta.json`. The solver internals stay in Standards — import, don't copy.

Current problems:

- `ree_K3/` — symmetric K=3 rational-expectations equilibrium (homogeneous CRRA).
