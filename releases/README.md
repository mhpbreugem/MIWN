# releases

Immutable, in-git **light** snapshots of the paper at a labelled version. The
**heavy** frozen bundle (vendored methods + used results) ships as a GitHub Release
asset, not in git. Built by `make publish VERSION=<label>`.

```
releases/<label>/
├── MANIFEST.json   miwn_commit, standards_commit, solutions_used, bundle_asset, doi
├── paper.pdf       the compiled manuscript at this version
└── README.md       one-command reproduce steps
```

## What `make publish` does

1. Resolves every `solutions/by-tex/*/lock.json` → the exact pool versions the
   paper uses.
2. Compiles `tex/paper.tex` → `paper.pdf`.
3. Writes the **light** part into `releases/<label>/` (committed): `paper.pdf`,
   `MANIFEST.json`, `README.md`.
4. Assembles the **heavy** bundle `miwn-<label>.zip` in a temp dir (NOT committed):
   frozen `tex/`, only-used pool results, a vendored copy of `standards/methods` at
   the submodule commit (`git -C standards archive`), `requirements.txt`, plus the
   manifest + reproduce README.
5. Prints the `gh release create` command for the heavy bundle (and a note to
   connect Zenodo for a DOI). It does not run it.

## Rules

- Snapshots are **frozen** — excluded from `make stale`. They intentionally pin an
  older `standards_methods_sha`.
- Never `git add` the multi-hundred-MB `miwn-<label>.zip`; it lives only as a
  Release asset (`.gitignore` already excludes `miwn-*.zip`).
