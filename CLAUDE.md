# MIWN — agent guide

Paper repo for *Market Inefficiency Without Noise* (REE, homogeneous CRRA). Read
this before working here.

## Standards hub (single source of truth)

Shared standards/methods/runner live in `github.com/mhpbreugem/standards`, pinned
here as the **`standards/` submodule**.

- **Writing:** before a figure/table/manuscript edit, consult `standards/writing/`
  and apply the relevant checklist.
- **Methods:** import from `standards/methods/` (see its `MAP.md`). This is the
  single source of truth — see the methods-sync rule below.
- **Distributed runs:** wire this paper's queue/solver onto `standards/runner/`
  per `standards/runner/README.md`.

Paper-specific math, the task queue, the numerical drivers, and glue stay here.

## Methods-sync rule — back-port, don't fork

Never edit code inside `standards/`. If a shared method needs a change:

1. make the change in the **Standards** repo on a feature branch and open a PR
   (Standards `main` is protected; the owner merges — never self-merge);
2. once merged, **bump the submodule pin** here:
   `git -C standards fetch && git -C standards checkout <new-sha>`, then commit the
   updated submodule pointer;
3. re-run affected solves; `make stale` flags any solution whose
   `standards_methods_sha` no longer matches the pin.

The submodule commit is recorded in every solution's `meta.json`
(`standards_methods_sha`) so results are auditable against the methods that built
them.

## Layout

See `README.md` for the tree and flow. Key invariants:

- Solutions are **immutable** `solutions/pool/<problem>/vNNNN/` versions; never
  overwrite a version. New run → new `vNNNN`.
- Arrays are stored as **compressed binary `.npz`** (`np.savez_compressed`), never
  4-byte string arrays. Move a version's `data/` to Git LFS / external storage only
  if a single artifact exceeds ~25 MB.
- Each `tex/*.tex` figure has a `solutions/by-tex/<stem>/lock.json` pinning the
  exact pool versions it uses; the small ready PDFs are copied into
  `by-tex/<stem>/figures/` (don't duplicate the heavy pool data).
- `releases/` snapshots are intentionally frozen — excluded from `make stale`.

## Change control

This is a research repo; keep commits small and descriptive. Do not commit secrets
or multi-hundred-MB bundles (those ship as GitHub Release assets — see
`releases/README.md`).
