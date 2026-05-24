# MIWN — Market Inefficiency Without Noise

Financial-economics paper repo. The paper studies a rational-expectations
equilibrium (REE) in a homogeneous-CRRA market with private signals, and the
"knife-edge" at which the price stops revealing the fundamental.

This repo holds **only** what is specific to this paper: the math, the task
queue, the numerical drivers, the solutions, and the manuscript. Everything
reusable — writing standards, numerical methods, the distributed runner — lives
in the shared **Standards** hub, pulled in here as the `standards/` submodule.

## Flow

```
spec.json ──► todo/TASK_QUEUE.json ──► runner (in Standards) ──► solutions/pool/<problem>/vNNNN/
   (grid)         (one task per (γ,τ))      claims & solves           data/ + figure/ + meta.json
                                                                              │
tex/*.tex ──► solutions/by-tex/<stem>/lock.json ◄── pins the exact versions ─┘
                         │
                         └──► make publish ──► releases/<label>/ (light) + GitHub Release asset (heavy)
```

- **numerics/** — per-problem definition (`PROBLEM.md`, `spec.json`) and a thin
  local `solve.py` that imports the shared methods and writes one solution.
- **todo/** — the runner's task queue (`TASK_QUEUE.json`) + `runner.config.json`.
- **solutions/** — immutable `pool/<problem>/vNNNN/` versions; `by-tex/` pins which
  versions each figure uses.
- **tex/** — `paper.tex` + per-figure `.tex`.
- **releases/** — light in-git snapshots; heavy bundles ship as GitHub Release assets.
- **scripts/** — `stale.py` (flags solutions built against an out-of-date methods pin).

## Standards hub

Shared standards/methods/runner live in
[`github.com/mhpbreugem/standards`](https://github.com/mhpbreugem/standards),
pinned here as the `standards/` submodule.

- **Writing:** before a figure/table/manuscript edit, consult `standards/writing/`.
- **Methods:** import from `standards/methods/` (single source of truth — see its
  `MAP.md`); never keep private edits, back-port instead.
- **Runner:** wire this paper's queue/solver onto `standards/runner/`.

Clone with submodules:

```sh
git clone --recurse-submodules https://github.com/mhpbreugem/MIWN.git
# or, after a plain clone:
git submodule update --init --recursive
```

## Common tasks

```sh
make help     # list targets
make stale    # flag solutions built against a stale methods pin
make check    # stale --strict (nonzero exit if anything is stale) — for CI
make publish VERSION=<label>   # assemble a release snapshot
```
