# TODO

## Now
- [x] Scaffold MIWN structure + Standards submodule pin.
- [x] Wire `numerics/ree_K3/solve.py` to the shared methods; one local solve →
      `solutions/pool/ree_K3/v0001/`.
- [x] Generate `TASK_QUEUE.json` from `numerics/ree_K3/spec.json`.
- [x] Tooling: `scripts/stale.py`, `Makefile` (help/stale/check/publish), `releases/`.

## Next
- [~] T6 — cross-repo runner. MIWN side done (`.github/workflows/solve.yml` +
      `run_task.py` driver, dry-run verified). **Pending:** merge the Standards
      PR (`QUEUE_REL`/`run_task.py`), then bump the `standards/` submodule here to
      that commit so the workflow goes live. Standards `main` via PR only.
- [ ] Run the full (γ, τ) grid at production `G`/precision via the runner.
- [x] Migrate curated REZN content (manuscript, 5 converged solutions → pool
      v0003–v0007, 15 paper figures, proofs/notes) per blueprint §D.
- [ ] Confirm the best-effort figure→solution pins in `solutions/by-tex/*/lock.json`
      against each figure's source (migration set them all to the canonical v0006,
      except fig5_convergence → G-ladder and fig_multicontour_B → v0007).
- [ ] Follow-ups: update Standards `methods/MAP.md` provenance to the archived REZN
      commit (Standards PR), then archive the REZN repo (owner, GitHub Settings).
- [ ] Wire `make publish` end-to-end once `tex/paper.tex` compiles with real figures.
