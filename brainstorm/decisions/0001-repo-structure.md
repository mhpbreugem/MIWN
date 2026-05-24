# 0001 — Repo structure

- **Status:** accepted
- **Date:** 2026-05-24
- **Context:** Standing up MIWN as a paper repo that consumes the shared Standards
  hub (writing standards, numerical methods, distributed runner).

## Decision

1. **Multi-repo.** Standards is its own repo; MIWN is its own repo. Chosen over a
   monorepo or branches-as-projects: independent history, independent access
   control, and a clean "library vs. consumer" boundary.

2. **Solution storage:** plain git + **compressed binary `.npz`**
   (`np.savez_compressed`), never 4-byte string arrays. Move a problem's
   `pool/<v>/data/` to Git LFS or an external store only if a single artifact
   exceeds ~25 MB.

3. **Versioning:** monotonic `vNNNN` per problem. Solutions are immutable — a new
   run is a new version, never an overwrite. Every solution's `meta.json` records
   params, metrics, and the **Standards methods commit** (`standards_methods_sha`)
   it was built with.

4. **Per-tex figures:** copy the small ready PDFs into
   `solutions/by-tex/<tex-stem>/figures/`; reference the heavy data in the pool via
   `lock.json`. Do not duplicate pool data.

5. **Runner lives in Standards**, reaches into MIWN via `todo/runner.config.json`:
   it claims from `todo/TASK_QUEUE.json` and writes solutions back to
   `solutions/pool/`.

6. **`tex/` granularity:** one `paper.tex` plus per-figure `.tex`; each `.tex` gets
   a `solutions/by-tex/<stem>/` folder.

7. **Methods consumption:** Standards as a **pinned git submodule** at
   `MIWN/standards`. Shared methods are edited only in Standards (via PR), then the
   submodule pin is bumped here — never fork them into MIWN ("back-port, don't
   fork").

8. **Releases/archival:** `releases/` holds light, in-git per-version snapshots
   (`MANIFEST.json` + `paper.pdf` + `README.md`); the heavy frozen bundle (vendored
   methods + used results) ships as a **GitHub Release asset** on a matching git
   tag, optionally archived to Zenodo for a DOI. Built by `make publish`. Excluded
   from `make stale` (snapshots are intentionally frozen).

## Consequences

- Reproducibility is auditable: `meta.json` ties each result to the exact methods
  commit; `make stale` flags drift after a submodule bump.
- The runner stays project-agnostic in Standards; MIWN supplies only config and a
  thin local driver.
- Curated predecessor (REZN) content can migrate into this structure later without
  copying REZN's `code/` (which now lives in `standards/methods/solver/code/`).
