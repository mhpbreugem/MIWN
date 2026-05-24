# TODO

## Now
- [x] Scaffold MIWN structure + Standards submodule pin.
- [x] Wire `numerics/ree_K3/solve.py` to the shared methods; one local solve →
      `solutions/pool/ree_K3/v0001/`.
- [x] Generate `TASK_QUEUE.json` from `numerics/ree_K3/spec.json`.
- [x] Tooling: `scripts/stale.py`, `Makefile` (help/stale/check/publish), `releases/`.

## Next
- [ ] T6 — cross-repo runner: Standards-side `QUEUE_REL`/`runner.config.json`
      support + a MIWN `.github/workflows/solve.yml`. Design first, confirm, then
      Standards branch + PR. Do not push Standards `main`.
- [ ] Run the full (γ, τ) grid at production `G`/precision via the runner.
- [ ] Migrate curated REZN content (tex, converged solutions, proofs) per the
      blueprint §D — a separate focused pass after the scaffold lands.
- [ ] Wire `make publish` end-to-end once `tex/paper.tex` compiles with real figures.
