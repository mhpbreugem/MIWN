# todo

The distributed task queue for MIWN and its runner config.

- `TASK_QUEUE.json` — single source of truth for task state and the distributed
  lock. Generated from `numerics/<problem>/spec.json` (one task per (γ, τ)).
  Schema and lifecycle: `standards/runner/TASK_SCHEMA.md`.
- `runner.config.json` — tells the Standards runner where this project's queue,
  problems, and output pool live (`queue_path`, `problems_dir`, `output_pool`), so
  it can operate on MIWN cross-repo.
- `TODO.md` — human-facing checklist of what's next.

## How runs happen

The runner lives in Standards (`standards/runner/`). Pointed at a MIWN checkout
(with the submodule initialized), it claims a `ready` task via the git-race lock
(`claim_task.py`), solves it with the shared methods, writes a `vNNNN` solution +
`meta.json` into `solutions/pool/`, and flips the task to `done`.

> **Cross-repo runner (T6):** the MIWN side is wired —
> `.github/workflows/solve.yml` drains this queue via
> `standards/runner/run_task.py` (claim → solve → write `vNNNN` back). It is
> **inert until** the Standards-side change (`QUEUE_REL`/`runner.config.json`
> support + `run_task.py`) is merged and the `standards/` submodule is bumped to a
> commit that contains it. Until then, drive a solve locally with
> `numerics/ree_K3/solve.py`, or dry-run the driver:
> `REPO_ROOT=$PWD python3 standards/runner/run_task.py --config todo/runner.config.json --local --G 5`.
